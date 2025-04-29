import concurrent.futures
import copy
import json
import os
import re
import shutil
import time
from collections.abc import Generator
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from enum import Enum
from typing import Any
from typing import cast
from typing import Optional

import tree_sitter
from github import Github
from github import RateLimitExceededException
from github import Repository
from github.ContentFile import ContentFile
from github.GithubException import GithubException
from github.Issue import Issue
from github.PaginatedList import PaginatedList
from github.PullRequest import PullRequest
from github.Requester import Requester
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from tree_sitter import Language
from tree_sitter import Query
from typing_extensions import override

from onyx.configs.app_configs import GITHUB_CONNECTOR_BASE_URL
from onyx.configs.constants import DocumentSource
from onyx.connectors.exceptions import ConnectorValidationError
from onyx.connectors.exceptions import CredentialExpiredError
from onyx.connectors.exceptions import InsufficientPermissionsError
from onyx.connectors.exceptions import UnexpectedValidationError
from onyx.connectors.interfaces import CheckpointedConnector
from onyx.connectors.interfaces import CheckpointOutput
from onyx.connectors.interfaces import ConnectorCheckpoint
from onyx.connectors.interfaces import ConnectorFailure
from onyx.connectors.interfaces import SecondsSinceUnixEpoch
from onyx.connectors.models import ConnectorMissingCredentialError
from onyx.connectors.models import Document
from onyx.connectors.models import DocumentFailure
from onyx.connectors.models import TextSection
from onyx.utils.logger import setup_logger

# from git import Repo, GitCommandError

logger = setup_logger()

ITEMS_PER_PAGE = 100

_MAX_NUM_RATE_LIMIT_RETRIES = 5


def _sleep_after_rate_limit_exception(github_client: Github) -> None:
    sleep_time = github_client.get_rate_limit().core.reset.replace(
        tzinfo=timezone.utc
    ) - datetime.now(tz=timezone.utc)
    sleep_time += timedelta(minutes=1)  # add an extra minute just to be safe
    logger.notice(f"Ran into Github rate-limit. Sleeping {sleep_time.seconds} seconds.")
    time.sleep(sleep_time.seconds)


def _get_batch_rate_limited(
    git_objs: PaginatedList, page_num: int, github_client: Github, attempt_num: int = 0
) -> list[PullRequest | Issue]:
    if attempt_num > _MAX_NUM_RATE_LIMIT_RETRIES:
        raise RuntimeError(
            "Re-tried fetching batch too many times. Something is going wrong with fetching objects from Github"
        )

    try:
        objs = list(git_objs.get_page(page_num))
        # fetch all data here to disable lazy loading later
        # this is needed to capture the rate limit exception here (if one occurs)
        for obj in objs:
            if hasattr(obj, "raw_data"):
                getattr(obj, "raw_data")
        return objs
    except RateLimitExceededException:
        _sleep_after_rate_limit_exception(github_client)
        return _get_batch_rate_limited(
            git_objs, page_num, github_client, attempt_num + 1
        )


def _convert_pr_to_document(pull_request: PullRequest) -> Document:
    return Document(
        id=pull_request.html_url,
        sections=[
            TextSection(link=pull_request.html_url, text=pull_request.body or "")
        ],
        source=DocumentSource.GITHUB,
        semantic_identifier=pull_request.title,
        # updated_at is UTC time but is timezone unaware, explicitly add UTC
        # as there is logic in indexing to prevent wrong timestamped docs
        # due to local time discrepancies with UTC
        doc_updated_at=(
            pull_request.updated_at.replace(tzinfo=timezone.utc)
            if pull_request.updated_at
            else None
        ),
        metadata={
            "merged": str(pull_request.merged),
            "state": pull_request.state,
        },
    )


def _fetch_issue_comments(issue: Issue) -> str:
    comments = issue.get_comments()
    return "\nComment: ".join(comment.body for comment in comments)


def _convert_issue_to_document(issue: Issue) -> Document:
    return Document(
        id=issue.html_url,
        sections=[TextSection(link=issue.html_url, text=issue.body or "")],
        source=DocumentSource.GITHUB,
        semantic_identifier=issue.title,
        # updated_at is UTC time but is timezone unaware
        doc_updated_at=issue.updated_at.replace(tzinfo=timezone.utc),
        metadata={
            "state": issue.state,
        },
    )


class SerializedRepository(BaseModel):
    # id is part of the raw_data as well, just pulled out for convenience
    id: int
    headers: dict[str, str | int]
    raw_data: dict[str, Any]

    def to_Repository(self, requester: Requester) -> Repository.Repository:
        return Repository.Repository(
            requester, self.headers, self.raw_data, completed=True
        )


class GithubConnectorStage(Enum):
    START = "start"
    PRS = "prs"
    ISSUES = "issues"
    FILES = "files"


class GithubConnectorCheckpoint(ConnectorCheckpoint):
    stage: GithubConnectorStage
    curr_page: int
    directory_stack: list[str] | None = None
    cached_repo_ids: list[int] | None = None
    cached_repo: SerializedRepository | None = None

    # This is a workaround to allow arbitrary types in the model
    # TODO: Remove this once we have a better solution
    # class Config:
    #     arbitrary_types_allowed = True


@dataclass
class CodeChunk:
    """Data class representing a chunk of code with metadata."""

    text: str
    file_path: str
    chunk_id: str
    repository: str
    repo_url: str
    file_type: str
    parent_class: Optional[str] = None
    parent_function: Optional[str] = None
    namespace: Optional[str] = None
    called_functions: list[str] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    start_line: int = 0
    end_line: int = 0
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_document(self) -> Document:
        """Convert the chunk to a document for indexing."""
        return Document(
            id=self.chunk_id,
            sections=[
                TextSection(link=self.repo_url + self.file_path, text=self.text or "")
            ],
            source=DocumentSource.GITHUB_SOURCE,
            semantic_identifier=self.chunk_id,
            # updated_at is UTC time but is timezone unaware
            doc_updated_at=self.updated_at.replace(tzinfo=timezone.utc),
            metadata={
                "repository": self.repository,
                "repo_url": self.repo_url,
                "file_path": self.file_path,
                "file_type": self.file_type,
                "parent_class": self.parent_class or "",
                "parent_function": self.parent_function or "",
                "namespace": self.namespace or "",
                "called_functions": ",".join(self.called_functions) or "",
                "imports": ",".join(self.imports) or "",
                "start_line": str(self.start_line),
                "end_line": str(self.end_line),
                "source": "github",
            },
        )


class TreeSitterChunker:
    """Handle code parsing using tree-sitter for better code understanding."""

    def __init__(self, language_dir: str = "./tree-sitter-grammars"):
        """
        Initialize the TreeSitterChunker.

        Args:
            language_dir: Directory containing compiled tree-sitter language libraries
        """
        # Convert relative path to absolute path
        if not os.path.isabs(language_dir):
            # Get the directory where the connector script is located
            current_dir = os.path.dirname(os.path.abspath(__file__))
            language_dir = os.path.abspath(os.path.join(current_dir, language_dir))

        self.language_dir = language_dir
        self.parsers = {}

        # Debugging: Log the absolute path
        logger.info(f"Tree-sitter language directory path: {self.language_dir}")

        self._init_parsers()

    def _init_parsers(self):
        """Initialize parsers for supported languages."""
        try:
            from tree_sitter import Parser

            # Use proper Python imports to get the languages
            languages = {}

            # Import available language packages - use try/except to gracefully handle missing ones
            try:
                from tree_sitter_c_sharp import language as cs_language

                languages[".cs"] = Language(cs_language())
                logger.info("Loaded C# language parser")
            except ImportError:
                logger.debug("C# language parser not available")

            try:
                from tree_sitter_python import language as py_language

                languages[".py"] = Language(py_language())
                logger.info("Loaded Python language parser")
            except ImportError:
                logger.debug("Python language parser not available")

            try:
                from tree_sitter_markdown import language as md_language

                languages[".md"] = Language(md_language())
                logger.info("Loaded Markdown language parser")
            except ImportError:
                logger.debug("Markdown language parser not available")

            try:
                from tree_sitter_html import language as html_language

                languages[".html"] = Language(html_language())
                languages[".htm"] = Language(html_language())
                logger.info("Loaded HTML language parser")
            except ImportError:
                logger.debug("HTML language parser not available")
            except Exception as e:
                logger.warning(f"Failed to load HTML parser: {e}")

            try:
                from tree_sitter_ruby import language as ruby_language

                languages[".rb"] = Language(ruby_language())
                languages[".rake"] = Language(ruby_language())
                logger.info("Loaded Ruby language parser")
            except ImportError:
                logger.debug("Ruby language parser not available")
            except Exception as e:
                logger.warning(f"Failed to load Ruby parser: {e}")

            try:
                from tree_sitter_scss import language as scss_language

                languages[".scss"] = Language(scss_language())
                languages[".sass"] = Language(scss_language())
                logger.info("Loaded SCSS language parser")
            except ImportError:
                logger.debug("SCSS language parser not available")
            except Exception as e:
                logger.warning(f"Failed to load SCSS parser: {e}")

            try:
                from tree_sitter_sql import language as sql_language

                languages[".sql"] = Language(sql_language())
                logger.info("Loaded SQL language parser")
            except ImportError:
                logger.debug("SQL language parser not available")
            except Exception as e:
                logger.warning(f"Failed to load SQL parser: {e}")

            try:
                from tree_sitter_xml import language_xml

                languages[".xml"] = Language(language_xml())
                languages[".svg"] = Language(language_xml())
                languages[".xsd"] = Language(language_xml())
                logger.info("Loaded XML language parser")
            except ImportError:
                logger.debug("XML language parser not available")
            except Exception as e:
                logger.warning(f"Failed to load XML parser: {e}")

            try:
                from tree_sitter_yaml import language as yaml_language

                languages[".yaml"] = Language(yaml_language())
                languages[".yml"] = Language(yaml_language())
                logger.info("Loaded YAML language parser")
            except ImportError:
                logger.debug("YAML language parser not available")
            except Exception as e:
                logger.warning(f"Failed to load YAML parser: {e}")

            try:
                from tree_sitter_json import language as json_language

                languages[".json"] = Language(json_language())
                logger.info("Loaded JSON language parser")
            except ImportError:
                logger.debug("JSON language parser not available")
            except Exception as e:
                logger.warning(f"Failed to load JSON parser: {e}")

            try:
                from tree_sitter_javascript import language as js_language

                languages[".js"] = Language(js_language())
                languages[".jsx"] = Language(js_language())
                logger.info("Loaded JavaScript language parser")
            except ImportError:
                logger.debug("JavaScript language parser not available")

            try:
                from tree_sitter_typescript import language_typescript
                from tree_sitter_typescript import language_tsx

                languages[".ts"] = Language(language_typescript())
                languages[".tsx"] = Language(language_tsx())
                logger.info("Loaded TypeScript language parser")
            except ImportError:
                logger.debug("TypeScript language parser not available")

            try:
                from tree_sitter_c import language as c_language

                languages[".c"] = Language(c_language())
                logger.info("Loaded C language parser")
            except ImportError:
                logger.debug("C language parser not available")

            try:
                from tree_sitter_cpp import language as cpp_language

                languages[".cpp"] = Language(cpp_language())
                languages[".hpp"] = Language(cpp_language())
                logger.info("Loaded C++ language parser")
            except ImportError:
                logger.debug("C++ language parser not available")

            # Create parsers for each language
            for ext, language in languages.items():
                parser = Parser(language)
                self.parsers[ext] = parser
                logger.info(f"Created parser for {ext}")

        except ImportError as e:
            logger.warning(f"Could not initialize tree-sitter: {e}")
        except Exception as e:
            logger.error(f"Error initializing tree-sitter parsers: {e}")

    def has_parser(self, file_ext: str) -> bool:
        """Check if a parser exists for the given file extension."""
        return file_ext in self.parsers

    def extract_metadata(self, code: str, file_path: str) -> dict[str, Any]:
        """
        Extract metadata from code using tree-sitter.

        Args:
            code: Source code
            file_path: Path to the file

        Returns:
            dictionary of extracted metadata
        """
        _, ext = os.path.splitext(file_path)

        metadata = {
            "imports": [],
            "classes": [],
            "functions": [],
            "called_functions": [],
            "decorators": [],
            "namespaces": [],
            "jsx_elements": [],
        }

        if ext not in self.parsers:
            return metadata

        try:
            parser = self.parsers[ext]
            tree = parser.parse(bytes(code, "utf8"))

            # Extract imports, classes, functions based on language
            if ext == ".cs":
                metadata = self._extract_csharp_metadata(tree)
            elif ext == ".py":
                metadata = self._extract_python_metadata(tree)
            elif ext in (".ts", ".js"):
                metadata = self._extract_js_ts_metadata(tree)
            elif ext in (".tsx", ".jsx"):
                metadata = self._extract_tsx_jsx_metadata(tree)
            elif ext in (".c"):
                metadata = self._extract_c_metadata(tree)
            elif ext in (".cpp", ".hpp"):
                metadata = self._extract_cpp_metadata(tree)

            # elif ext in (".html", ".htm"):
            #     metadata = self._extract_html_metadata(tree)
            # elif ext in (".rb", ".rake"):
            #     metadata = self._extract_ruby_metadata(tree)
            # elif ext in (".scss", ".sass"):
            #     metadata = self._extract_scss_metadata(tree)
            # elif ext == ".sql":
            #     metadata = self._extract_sql_metadata(tree)
            # elif ext in (".xml", ".svg", ".xsd"):
            #     metadata = self._extract_xml_metadata(tree)
            # elif ext in (".yaml", ".yml"):
            #     metadata = self._extract_yaml_metadata(parser, code)
            # # elif ext == ".json":
            # #     metadata = self._extract_json_metadata(tree)
            # Add more language-specific extractors as needed

        except Exception as e:
            logger.warning(f"Error extracting metadata from {file_path}: {e}")

        return metadata

    def _extract_c_metadata(self, tree: tree_sitter.Tree) -> dict[str, Any]:
        query = """
        (preproc_include
            (string) @import)

        (function_definition
            declarator: (function_declarator
                declarator: (identifier) @function_name))

        (call_expression
            function: (identifier) @called_function)
        """
        return self._extract_metadata_with_query(tree, query)

    def _extract_cpp_metadata(self, tree: tree_sitter.Tree) -> dict[str, Any]:
        query = """
        (preproc_include
            (string) @import)

        (namespace_definition
            name: (identifier) @namespace_name)

        (class_specifier
            name: (type_identifier) @class_name)

        (function_definition
            declarator: (function_declarator
                declarator: (identifier) @function_name))

        (call_expression
            function: (identifier) @called_function)
        """
        return self._extract_metadata_with_query(tree, query)

    def _extract_python_metadata(self, tree: tree_sitter.Tree) -> dict[str, Any]:
        query = """
        (import_statement
            name: (dotted_name (identifier) @import))

        (import_from_statement
            module_name: (dotted_name (identifier) @import))

        (class_definition
            name: (identifier) @class_name)

        (function_definition
            name: (identifier) @function_name)

        (call
            function: (identifier) @called_function)
        """
        return self._extract_metadata_with_query(tree, query)

    def _extract_tsx_jsx_metadata(self, tree: tree_sitter.Tree) -> dict[str, Any]:
        query = """
        ;; IMPORTS
        (import_statement (import_clause) @import)

        ;; CLASSES
        (class_declaration name: (type_identifier) @class_name)

        ;; FUNCTIONS
        (method_definition name: (property_identifier) @function_name)
        (function_declaration name: (identifier) @function_name)
        (lexical_declaration
        (variable_declarator
            name: (identifier) @function_name
            value: (arrow_function)))

        ;; FUNCTION CALLS
        (call_expression function: (identifier) @called_function)
        (call_expression function: (member_expression property: (property_identifier) @called_function))

        ;; JSX ELEMENTS
        (jsx_opening_element name: (identifier) @jsx_element)
        (jsx_self_closing_element name: (identifier) @jsx_element)
        """
        return self._extract_metadata_with_query(tree, query)

    def _extract_js_ts_metadata(self, tree: tree_sitter.Tree) -> dict[str, Any]:
        query = """
        ;; IMPORTS
        (import_statement (import_clause) @import)

        ;; CLASSES
        (class_declaration name: (type_identifier) @class_name)

        ;; FUNCTIONS
        (method_definition name: (property_identifier) @function_name)
        (function_declaration name: (identifier) @function_name)

        ;; CALLED FUNCTIONS
        (call_expression function: (identifier) @called_function)
        (call_expression function: (member_expression property: (property_identifier) @called_function))

        ;; DECORATORS (Angular-specific)
        (decorator (call_expression function: (identifier) @decorator_name))
        """
        return self._extract_metadata_with_query(tree, query)

    def _extract_csharp_metadata(self, tree: tree_sitter.Tree) -> dict[str, Any]:
        query = """
        (using_directive
        (identifier) @import)

        (namespace_declaration
            name: (identifier) @namespace_name)

        (class_declaration
            name: (identifier) @class_name)

        (method_declaration
            name: (identifier) @function_name)

        (invocation_expression
            function: (identifier) @called_function)
        """
        return self._extract_metadata_with_query(tree, query)

    def _extract_metadata_with_query(
        self, tree: tree_sitter.Tree, query: str
    ) -> dict[str, Any]:
        """
        Extract metadata from code using a tree-sitter query.

        Args:
            tree: The parsed tree-sitter syntax tree.
            query: The tree-sitter query string.

        Returns:
            A dictionary containing extracted metadata.
        """
        metadata = {
            "imports": [],
            "classes": [],
            "functions": [],
            "called_functions": [],
            "decorators": [],
            "namespaces": [],
            "jsx_elements": [],
        }

        try:
            root_node = tree.root_node
            parser_query = Query(tree.language, query)
            captures = parser_query.captures(root_node)

            for capture, nodes in captures.items():
                for node in nodes:
                    if capture == "import":
                        metadata["imports"].append(node.text.decode("utf-8"))
                    elif capture == "class_name":
                        metadata["classes"].append(node.text.decode("utf-8"))
                    elif capture == "function_name":
                        metadata["functions"].append(node.text.decode("utf-8"))
                    elif capture == "called_function":
                        metadata["called_functions"].append(node.text.decode("utf-8"))
                    elif capture == "decorator_name":
                        metadata["decorators"].append(node.text.decode("utf-8"))
                    elif capture == "jsx_element":
                        metadata["jsx_elements"].append(node.text.decode("utf-8"))
                    elif capture == "namespace_name":
                        metadata["namespaces"].append(node.text.decode("utf-8"))

        except Exception as e:
            logger.warning(f"Error extracting metadata: {e}")

        return metadata


class RecursiveCodeChunker:
    """
    Handles the recursive code chunking process according to the specified strategy.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        tree_sitter_chunker: Optional[TreeSitterChunker] = None,
    ):
        """
        Initialize the RecursiveCodeChunker.

        Args:
            chunk_size: Size of chunks in characters (default: 1000)
            chunk_overlap: Overlap between chunks in characters (default: 100)
            tree_sitter_chunker: Optional TreeSitterChunker for enhanced parsing
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tree_sitter_chunker = tree_sitter_chunker or TreeSitterChunker()

        # Initialize language-specific splitters
        self.splitters = self._init_splitters()

    def _init_splitters(self) -> dict[str, RecursiveCharacterTextSplitter]:
        """Initialize language-specific text splitters."""
        splitters = {}

        # Create a default splitter for languages without specific support
        default_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

        # Python splitter
        splitters[".py"] = RecursiveCharacterTextSplitter.from_language(
            language="python",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        # JavaScript/TypeScript splitters
        splitters[".js"] = RecursiveCharacterTextSplitter.from_language(
            language="js", chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        splitters[".jsx"] = splitters[".js"]
        splitters[".ts"] = RecursiveCharacterTextSplitter.from_language(
            language="ts", chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        splitters[".tsx"] = splitters[".ts"]

        # Java splitter
        splitters[".java"] = RecursiveCharacterTextSplitter.from_language(
            language="java",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        # C# splitter
        splitters[".cs"] = RecursiveCharacterTextSplitter.from_language(
            language="csharp",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        # HTML splitter (XML has to use default)
        splitters[".html"] = RecursiveCharacterTextSplitter.from_language(
            language="html",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        splitters[".htm"] = splitters[".html"]

        # XML and related formats - use HTML as closest alternative or default
        try:
            # Try to use HTML as a fallback for XML formats
            xml_splitter = RecursiveCharacterTextSplitter.from_language(
                language="html",  # Use HTML as proxy for XML-like languages
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            splitters[".xml"] = xml_splitter
            splitters[".svg"] = xml_splitter
            splitters[".xsd"] = xml_splitter
        except ValueError:
            # If HTML isn't supported either, use default
            splitters[".xml"] = default_splitter
            splitters[".svg"] = default_splitter
            splitters[".xsd"] = default_splitter
            logger.warning(
                "Using default splitter for XML documents (no XML/HTML support in langchain)"
            )

        # Ruby splitter
        splitters[".rb"] = RecursiveCharacterTextSplitter.from_language(
            language="ruby",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        splitters[".rake"] = splitters[".rb"]

        # SCSS/CSS splitters - use default as no direct support
        splitters[".scss"] = default_splitter
        splitters[".sass"] = default_splitter
        splitters[".css"] = default_splitter
        logger.info("Using default splitter for CSS/SCSS documents")

        # SQL splitter - no direct support, use default
        splitters[".sql"] = default_splitter
        logger.info("Using default splitter for SQL documents")

        # YAML splitter - no direct support, use default
        splitters[".yaml"] = default_splitter
        splitters[".yml"] = default_splitter
        logger.info("Using default splitter for YAML documents")

        # JSON splitter - no direct support, use default
        splitters[".json"] = default_splitter
        logger.info("Using default splitter for JSON documents")

        # C/C++ splitters
        splitters[".c"] = RecursiveCharacterTextSplitter.from_language(
            language="c",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        splitters[".cpp"] = RecursiveCharacterTextSplitter.from_language(
            language="cpp",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        splitters[".hpp"] = splitters[".cpp"]

        # Markdown splitter
        splitters[".md"] = RecursiveCharacterTextSplitter.from_language(
            language="markdown",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        splitters[".markdown"] = splitters[".md"]

        # Go splitter
        splitters[".go"] = RecursiveCharacterTextSplitter.from_language(
            language="go",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        # PHP splitter
        splitters[".php"] = RecursiveCharacterTextSplitter.from_language(
            language="php",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        # Rust splitter
        splitters[".rs"] = RecursiveCharacterTextSplitter.from_language(
            language="rust",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        # Default text splitter for all other file types
        splitters["default"] = default_splitter

        return splitters

    def get_splitter_for_file(self, file_path: str) -> RecursiveCharacterTextSplitter:
        """Get the appropriate splitter for a file based on its extension."""
        _, ext = os.path.splitext(file_path)
        return self.splitters.get(ext.lower(), self.splitters["default"])

    def chunk_content(self, content: ContentFile) -> list[CodeChunk]:
        """
        Chunk a file recursively using the appropriate splitter.

        Args:
            content: Content of the file

        Returns:
            list of CodeChunk objects
        """
        file_path = content.path
        _, ext = os.path.splitext(file_path)
        file_type = ext.lstrip(".")
        splitter = self.get_splitter_for_file(file_path)

        # Extract code metadata if possible
        namespace = None
        parent_class = None
        parent_function = None
        called_functions = []
        imports = []

        if content.encoding != "base64":
            logger.warning(
                f"File {file_path} is not base64 encoded. Skipping chunking."
            )
            return []

        file_content = content.decoded_content.decode("utf-8")

        if self.tree_sitter_chunker.has_parser(ext):
            try:
                code_metadata = self.tree_sitter_chunker.extract_metadata(
                    file_content, file_path
                )
                imports = code_metadata.get("imports", [])
                classes = code_metadata.get("classes", [])
                functions = code_metadata.get("functions", [])
                parent_class = classes[0] if classes else None
                parent_function = functions[0] if functions else None
                called_functions = code_metadata.get("called_functions", [])

                # If namespace is explicitly provided in metadata
                if "namespace" in code_metadata:
                    namespace = code_metadata["namespace"]
                # Try to extract namespace from imports or file structure
                elif imports:
                    # Simple heuristic: use the first import's package as namespace
                    first_import = imports[0]
                    if isinstance(first_import, str):
                        match = re.search(
                            r"import\s+([a-zA-Z0-9_.]+)|from\s+([a-zA-Z0-9_.]+)|package\s+([a-zA-Z0-9_.]+)|namespace\s+([a-zA-Z0-9_.]+)",
                            first_import,
                        )
                        if match:
                            # Take the first matching group that isn't None
                            namespace = next(
                                (g for g in match.groups() if g is not None), None
                            )
                            if namespace:
                                namespace = namespace.split(".")[0]

                if not namespace:
                    # Use directory structure for namespace
                    dir_parts = os.path.dirname(file_path).split(os.path.sep)
                    if len(dir_parts) > 1 and dir_parts[-1]:
                        namespace = dir_parts[-1]
                    elif len(dir_parts) > 2:
                        namespace = dir_parts[-2]
            except Exception as e:
                logger.warning(f"Error extracting metadata from {file_path}: {e}")

        # Chunk the text
        chunks = []
        try:
            # Split the text into chunks
            text_chunks = splitter.split_text(file_content)

            # Create CodeChunk objects
            for i, chunk_text in enumerate(text_chunks):
                # Create a unique chunk ID
                chunk_id = f"{content.repository.full_name}:{file_path}:{i}"

                # Extract line numbers
                start_line = (
                    file_content.count(
                        "\n", 0, file_content.find(chunk_text.strip()[:50])
                    )
                    + 1
                )
                end_line = start_line + chunk_text.count("\n")

                # Create chunk
                chunk = CodeChunk(
                    text=chunk_text,
                    file_path=file_path,
                    chunk_id=chunk_id,
                    repository=content.repository.name,
                    repo_url=content.repository.html_url,  # Use html_url for browser URL
                    file_type=file_type,
                    parent_class=parent_class,
                    parent_function=parent_function,
                    namespace=namespace,
                    called_functions=called_functions,
                    imports=imports,
                    start_line=start_line,
                    end_line=end_line,
                    updated_at=content.last_modified_datetime
                    or datetime.now(timezone.utc),
                )
                chunks.append(chunk)

        except Exception as e:
            logger.error(f"Error chunking file {file_path}: {e}")
            # If chunking fails, create a single chunk with the entire file
            chunk_id = f"{content.repository.full_name}:{file_path}:0"
            chunk = CodeChunk(
                text=file_content,
                file_path=file_path,
                chunk_id=chunk_id,
                repository=content.repository.name,
                repo_url=content.repository.html_url,
                file_type=file_type,
                parent_class=parent_class,
                parent_function=parent_function,
                namespace=namespace,
                called_functions=called_functions,
                imports=imports,
                start_line=1,
                end_line=file_content.count("\n") + 1,
                updated_at=content.last_modified_datetime or datetime.now(timezone.utc),
            )
            chunks.append(chunk)

        return chunks

    def chunk_file(
        self, file_path: str, file_content: str, repo_name: str, repo_url: str
    ) -> list[CodeChunk]:
        """
        Chunk a file recursively using the appropriate splitter.

        Args:
            file_path: Path to the file
            file_content: Content of the file
            repo_name: Name of the repository
            repo_url: URL of the repository

        Returns:
            list of CodeChunk objects
        """
        _, ext = os.path.splitext(file_path)
        file_type = ext.lstrip(".")
        splitter = self.get_splitter_for_file(file_path)

        # Extract code metadata if possible
        namespace = None
        parent_class = None
        called_functions = []
        imports = []

        if self.tree_sitter_chunker.has_parser(ext):
            try:
                code_metadata = self.tree_sitter_chunker.extract_metadata(
                    file_content, file_path
                )
                imports = code_metadata.get("imports", [])
                classes = code_metadata.get("classes", [])
                parent_class = classes[0] if classes else None
                called_functions = code_metadata.get("called_functions", [])

                # Try to extract namespace from imports or file structure
                if imports:
                    # Simple heuristic: use the first import's package as namespace
                    first_import = imports[0]
                    match = re.search(r"import\s+([a-zA-Z0-9_.]+)", first_import)
                    if match:
                        namespace = match.group(1).split(".")[0]

                if not namespace:
                    # Use directory structure for namespace
                    dir_parts = os.path.dirname(file_path).split(os.path.sep)
                    if len(dir_parts) > 1 and dir_parts[-1]:
                        namespace = dir_parts[-1]
                    elif len(dir_parts) > 2:
                        namespace = dir_parts[-2]
            except Exception as e:
                logger.warning(f"Error extracting metadata from {file_path}: {e}")

        # Chunk the text
        chunks = []
        try:
            # Split the text into chunks
            text_chunks = splitter.split_text(file_content)

            # Create CodeChunk objects
            for i, chunk_text in enumerate(text_chunks):
                # Create a unique chunk ID
                chunk_id = f"{repo_name}:{file_path}:{i}"

                # Extract line numbers
                start_line = (
                    file_content.count(
                        "\n", 0, file_content.find(chunk_text.strip()[:50])
                    )
                    + 1
                )
                end_line = start_line + chunk_text.count("\n")

                # Create chunk
                chunk = CodeChunk(
                    text=chunk_text,
                    file_path=file_path,
                    chunk_id=chunk_id,
                    repository=repo_name,
                    repo_url=repo_url,
                    file_type=file_type,
                    parent_class=parent_class,
                    namespace=namespace,
                    called_functions=called_functions,
                    imports=imports,
                    start_line=start_line,
                    end_line=end_line,
                )
                chunks.append(chunk)

        except Exception as e:
            logger.error(f"Error chunking file {file_path}: {e}")
            # If chunking fails, create a single chunk with the entire file
            chunk_id = f"{repo_name}:{file_path}:0"
            chunk = CodeChunk(
                text=file_content,
                file_path=file_path,
                chunk_id=chunk_id,
                repository=repo_name,
                repo_url=repo_url,
                file_type=file_type,
                parent_class=parent_class,
                namespace=namespace,
                called_functions=called_functions,
                imports=imports,
                start_line=1,
                end_line=file_content.count("\n") + 1,
            )
            chunks.append(chunk)

        return chunks


class GithubSourceConnector(CheckpointedConnector[GithubConnectorCheckpoint]):
    def __init__(
        self,
        repo_owner: str,
        repositories: str | None = None,
        state_filter: str = "all",
        include_prs: bool = True,
        include_issues: bool = False,
        include_files: bool = False,  # New flag to include source files,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        max_workers: int = 5,
        excluded_extensions: Optional[list[str]] = None,
        excluded_directories: Optional[list[str]] = None,
    ) -> None:
        """
        Initialize the Onyx GitHub Connector.

        Args:
            repo_owner (str): The owner of the GitHub repository.
            repo_name (str): The name of the GitHub repository.
            state_filter (str): The filter for the state of issues and pull requests. Defaults to "all".
            include_prs (bool): Whether to include pull requests in the processing. Defaults to True.
            include_issues (bool): Whether to include issues in the processing. Defaults to False.
            include_files (bool): Whether to include source files in the processing. Defaults to False.
            chunk_size (int): Size of chunks in characters for processing source files. Defaults to 1000.
            chunk_overlap (int): Overlap between chunks in characters for processing source files. Defaults to 150.
            max_workers (int): Maximum number of concurrent workers for processing. Defaults to 5.
            excluded_extensions (Optional[list[str]]): list of file extensions to exclude. Defaults to common
            binary and media file types.
            excluded_directories (Optional[list[str]]): list of directory names to exclude.
            Defaults to common ignored directories.
        """
        self.repo_owner = repo_owner
        self.repositories = repositories
        self.state_filter = state_filter
        self.include_prs = include_prs
        self.include_issues = include_issues
        self.include_files = include_files  # Initialize the new flag
        self.github_client: Github | None = None
        self.excluded_extensions = excluded_extensions or [
            # Image formats
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".bmp",
            ".tiff",
            ".ico",
            ".webp",
            ".svg",
            # Video formats
            ".mp4",
            ".avi",
            ".mov",
            ".wmv",
            ".flv",
            ".mkv",
            ".webm",
            # Audio formats
            ".mp3",
            ".wav",
            ".ogg",
            ".flac",
            ".aac",
            # Archive formats
            ".zip",
            ".tar",
            ".gz",
            ".rar",
            ".7z",
            ".bz2",
            ".xz",
            # Binary/executable formats
            ".exe",
            ".dll",
            ".so",
            ".dylib",
            ".bin",
            ".dat",
            # Document formats (that aren't plain text)
            ".pdf",
            ".doc",
            ".docx",
            ".ppt",
            ".pptx",
            ".xls",
            ".xlsx",
            # Database and large data files
            ".db",
            ".sqlite",
            ".mdb",
            ".accdb",
            ".csv",
            ".tsv",
            # Font files
            ".ttf",
            ".otf",
            ".woff",
            ".woff2",
            ".eot",
            # Other binary formats
            ".pyc",
            ".pyd",
            ".class",
            ".o",
            ".obj",
        ]
        self.excluded_directories = excluded_directories or [
            ".git",
            "node_modules",
            "__pycache__",
            "venv",
            ".env",
            "dist",
            "build",
        ]
        self.max_workers = max_workers

        # Initialize chunker
        tree_sitter_chunker = TreeSitterChunker(language_dir="./tree-sitter-grammars")
        self.code_chunker = RecursiveCodeChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            tree_sitter_chunker=tree_sitter_chunker,
        )

    def clone_repository(self, repo_url: str, branch: str = "main") -> str:
        """
        Clone a GitHub repository to a temporary directory.

        Args:
            repo_url: URL of the GitHub repository
            branch: Branch name to clone (default: main)

        Returns:
            Path to the cloned repository
        """
        logger.info("#######clone_repository commented out########")
        # temp_dir = tempfile.mkdtemp()
        # logger.info(f"Cloning repository {repo_url} to {temp_dir}")

        # clone_url = repo_url
        # if self.github_token and 'github.com' in repo_url:
        #     # Insert token for authentication if it's a GitHub repo
        #     if repo_url.startswith('https://'):
        #         clone_url = repo_url.replace('https://', f'https://{self.github_token}@')

        # try:
        #     Repo.clone_from(clone_url, temp_dir, branch=branch)
        #     logger.info(f"Successfully cloned repository to {temp_dir}")
        #     return temp_dir
        # except GitCommandError as e:
        #     logger.error(f"Failed to clone repository: {e}")
        #     shutil.rmtree(temp_dir, ignore_errors=True)
        #     raise

    def should_process_file(self, file_path: str) -> bool:
        """
        Determine if a file should be processed based on exclusion rules.

        Args:
            file_path: Path to the file

        Returns:
            Boolean indicating if the file should be processed
        """
        # Check file extension
        _, ext = os.path.splitext(file_path)
        if ext.lower() in self.excluded_extensions:
            return False

        # Check if file is in excluded directory
        parts = file_path.split(os.path.sep)
        for part in parts:
            if part in self.excluded_directories:
                return False

        return True

    def read_file_content(self, file_path: str) -> Optional[str]:
        """
        Read content of a file, handling encoding issues.

        Args:
            file_path: Path to the file

        Returns:
            File content as string or None if file can't be read
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            try:
                # Try with a different encoding
                with open(file_path, "r", encoding="latin-1") as f:
                    return f.read()
            except Exception as e:
                logger.warning(f"Failed to read file {file_path}: {e}")
                return None
        except Exception as e:
            logger.warning(f"Failed to read file {file_path}: {e}")
            return None

    def index_chunk(self, chunk: CodeChunk) -> bool:
        """
        Index a single code chunk into Onyx.

        Args:
            chunk: CodeChunk object to index

        Returns:
            Boolean indicating if indexing was successful
        """
        chunk.to_document()

        # headers = {
        #     "Content-Type": "application/json",
        #     "Authorization": f"Bearer {self.onyx_api_key}",
        # }

        # try:
        #     url = f"{self.onyx_api_url}/indexes/{self.index_name}/documents"
        #     response = requests.post(url, headers=headers, json=document)

        #     if response.status_code in (200, 201):
        #         logger.debug(f"Successfully indexed {chunk.chunk_id}")
        #         return True
        #     else:
        #         logger.error(
        #             f"Failed to index {chunk.chunk_id}: {response.status_code} - {response.text}"
        #         )
        #         return False
        # except Exception as e:
        #     logger.error(f"Exception during indexing of {chunk.chunk_id}: {e}")
        #     return False

    def index_chunks_batch(self, chunks: list[CodeChunk]) -> list[Document]:
        """
        Index a batch of code chunks into Onyx.

        Args:
            chunks: list of CodeChunk objects to index

        Returns:
            Tuple of (chunks indexed, failures)
        """
        if not chunks:
            return 0, 0

        documents = [chunk.to_document() for chunk in chunks]

        return documents

    def process_content_into_documents(self, content: ContentFile) -> list[Document]:
        """
        Process a single file and index its chunks.

        Args:
            file_path: Path to the file
            repo_name: Name of the repository
            repo_url: URL of the repository

        Returns:
            Statistics about the processing
        """
        file_path = content.path
        stats = {
            "file": os.path.basename(file_path),
            "chunks_created": 0,
            "chunks_indexed": 0,
            "errors": 0,
        }

        # Chunk the file
        chunks = self.code_chunker.chunk_content(content)
        stats["chunks_created"] = len(chunks)

        # Index the chunks
        documents = self.index_chunks_batch(chunks)
        stats["chunks_indexed"] = len(documents)
        # stats["errors"] = failures
        # stats["status"] = "success" if failures == 0 else "partial_failure"

        return documents

    def process_file(
        self, file_path: str, repo_name: str, repo_url: str
    ) -> dict[str, Any]:
        """
        Process a single file and index its chunks.

        Args:
            file_path: Path to the file
            repo_name: Name of the repository
            repo_url: URL of the repository

        Returns:
            Statistics about the processing
        """
        stats = {
            "file": os.path.basename(file_path),
            "chunks_created": 0,
            "chunks_indexed": 0,
            "errors": 0,
        }

        if not self.should_process_file(file_path):
            stats["status"] = "skipped"
            return stats

        content = self.read_file_content(file_path)
        if content is None:
            stats["status"] = "error"
            stats["errors"] = 1
            return stats

        # Skip empty files
        if not content.strip():
            stats["status"] = "empty"
            return stats

        # Chunk the file
        chunks = self.code_chunker.chunk_file(file_path, content, repo_name, repo_url)
        stats["chunks_created"] = len(chunks)

        # Index the chunks
        documents = self.index_chunks_batch(chunks)
        stats["chunks_indexed"] = len(documents)
        # stats["errors"] = failures
        # stats["status"] = "success" if failures == 0 else "partial_failure"

        return stats

    def process_directory(
        self, dir_path: str, repo_name: str, repo_url: str
    ) -> dict[str, Any]:
        """
        Process a directory recursively and index all valid files.

        Args:
            dir_path: Path to the directory
            repo_name: Name of the repository
            repo_url: URL of the repository

        Returns:
            Statistics about the processing
        """
        stats = {
            "files_processed": 0,
            "files_skipped": 0,
            "chunks_created": 0,
            "chunks_indexed": 0,
            "errors": 0,
            "file_stats": {},
        }

        file_paths = []

        # Collect all files
        for root, dirs, files in os.walk(dir_path):
            # Filter out excluded directories
            for excluded_dir in self.excluded_directories:
                if excluded_dir in dirs:
                    dirs.remove(excluded_dir)

            for file in files:
                file_path = os.path.join(root, file)
                if self.should_process_file(file_path):
                    file_paths.append(file_path)
                else:
                    stats["files_skipped"] += 1

        logger.info(f"Found {len(file_paths)} files to process in {repo_name}")

        # Process files in parallel
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            future_to_file = {
                executor.submit(
                    self.process_file, file_path, repo_name, repo_url
                ): file_path
                for file_path in file_paths
            }

            for future in concurrent.futures.as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    file_stats = future.result()
                    stats["files_processed"] += 1
                    stats["chunks_created"] += file_stats["chunks_created"]
                    stats["chunks_indexed"] += file_stats["chunks_indexed"]
                    stats["errors"] += file_stats["errors"]

                    # Store individual file stats
                    rel_path = os.path.relpath(file_path, dir_path)
                    stats["file_stats"][rel_path] = file_stats

                    if stats["files_processed"] % 50 == 0:
                        logger.info(
                            f"Processed {stats['files_processed']} files so far..."
                        )

                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")
                    stats["errors"] += 1
                    rel_path = os.path.relpath(file_path, dir_path)
                    stats["file_stats"][rel_path] = {
                        "status": "error",
                        "errors": 1,
                        "chunks_created": 0,
                        "chunks_indexed": 0,
                    }

        return stats

    def ingest_repository(self, repo_url: str, branch: str = "main") -> dict[str, Any]:
        """
        Ingest a GitHub repository into the Onyx index.

        Args:
            repo_url: URL of the GitHub repository
            branch: Branch name to ingest (default: main)

        Returns:
            Statistics about the ingestion process
        """
        # Extract repo name from URL
        repo_name = repo_url.rstrip("/").split("/")[-1]
        if repo_name.endswith(".git"):
            repo_name = repo_name[:-4]

        # Clone the repository
        temp_dir = None
        try:
            temp_dir = self.clone_repository(repo_url, branch)

            # Process the repository
            stats = self.process_directory(temp_dir, repo_name, repo_url)
            stats["repository"] = repo_name
            stats["branch"] = branch

            logger.info(
                f"""Repository {repo_name} processing completed:
                {json.dumps({k: v for k, v in stats.items() if k != 'file_stats'})}"""
            )
            return stats

        finally:
            # Clean up temporary directory
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
                logger.info(f"Cleaned up temporary directory {temp_dir}")

    def ingest_repositories(
        self, repo_list: list[dict[str, str]]
    ) -> list[dict[str, Any]]:
        """
        Ingest multiple GitHub repositories into the Onyx index.

        Args:
            repo_list: list of dictionaries with 'url' and optional 'branch' keys

        Returns:
            list of statistics for each repository
        """
        results = []

        for repo_info in repo_list:
            repo_url = repo_info["url"]
            branch = repo_info.get("branch", "main")

            try:
                stats = self.ingest_repository(repo_url, branch)
                results.append(stats)
            except Exception as e:
                logger.error(f"Failed to ingest repository {repo_url}: {e}")
                results.append(
                    {
                        "repository": repo_url.rstrip("/").split("/")[-1],
                        "branch": branch,
                        "error": str(e),
                        "status": "failed",
                    }
                )

        return results

    def load_credentials(self, credentials: dict[str, Any]) -> dict[str, Any] | None:
        # defaults to 30 items per page, can be set to as high as 100
        self.github_client = (
            Github(
                credentials["github_access_token"],
                base_url=GITHUB_CONNECTOR_BASE_URL,
                per_page=ITEMS_PER_PAGE,
            )
            if GITHUB_CONNECTOR_BASE_URL
            else Github(credentials["github_access_token"], per_page=ITEMS_PER_PAGE)
        )
        return None

    def _get_github_repo(
        self, github_client: Github, attempt_num: int = 0
    ) -> Repository.Repository:
        if attempt_num > _MAX_NUM_RATE_LIMIT_RETRIES:
            raise RuntimeError(
                "Re-tried fetching repo too many times. Something is going wrong with fetching objects from Github"
            )

        try:
            return github_client.get_repo(f"{self.repo_owner}/{self.repositories}")
        except RateLimitExceededException:
            _sleep_after_rate_limit_exception(github_client)
            return self._get_github_repo(github_client, attempt_num + 1)

    def _get_github_repos(
        self, github_client: Github, attempt_num: int = 0
    ) -> list[Repository.Repository]:
        """Get specific repositories based on comma-separated repo_name string."""
        if attempt_num > _MAX_NUM_RATE_LIMIT_RETRIES:
            raise RuntimeError(
                "Re-tried fetching repos too many times. Something is going wrong with fetching objects from Github"
            )

        try:
            repos = []
            # Split repo_name by comma and strip whitespace
            repo_names = [
                name.strip() for name in (cast(str, self.repositories)).split(",")
            ]

            for repo_name in repo_names:
                if repo_name:  # Skip empty strings
                    try:
                        repo = github_client.get_repo(f"{self.repo_owner}/{repo_name}")
                        repos.append(repo)
                    except GithubException as e:
                        logger.warning(
                            f"Could not fetch repo {self.repo_owner}/{repo_name}: {e}"
                        )

            return repos
        except RateLimitExceededException:
            _sleep_after_rate_limit_exception(github_client)
            return self._get_github_repos(github_client, attempt_num + 1)

    def _get_all_repos(
        self, github_client: Github, attempt_num: int = 0
    ) -> list[Repository.Repository]:
        if attempt_num > _MAX_NUM_RATE_LIMIT_RETRIES:
            raise RuntimeError(
                "Re-tried fetching repos too many times. Something is going wrong with fetching objects from Github"
            )

        try:
            # Try to get organization first
            try:
                org = github_client.get_organization(self.repo_owner)
                return list(org.get_repos())
            except GithubException:
                # If not an org, try as a user
                user = github_client.get_user(self.repo_owner)
                return list(user.get_repos())
        except RateLimitExceededException:
            _sleep_after_rate_limit_exception(github_client)
            return self._get_all_repos(github_client, attempt_num + 1)

    def _fetch_from_github(
        self,
        checkpoint: GithubConnectorCheckpoint,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> Generator[Document | ConnectorFailure, None, GithubConnectorCheckpoint]:
        if self.github_client is None:
            raise ConnectorMissingCredentialError("GitHub")

        checkpoint = copy.deepcopy(checkpoint)

        # First run of the connector, fetch all repos and store in checkpoint
        if checkpoint.cached_repo_ids is None:
            repos = []
            if self.repositories:
                if "," in self.repositories:
                    # Multiple repositories specified
                    repos = self._get_github_repos(self.github_client)
                else:
                    # Single repository (backward compatibility)
                    repos = [self._get_github_repo(self.github_client)]
            else:
                # All repositories
                repos = self._get_all_repos(self.github_client)
            if not repos:
                checkpoint.has_more = False
                return checkpoint

            checkpoint.cached_repo_ids = sorted([repo.id for repo in repos])
            checkpoint.cached_repo = SerializedRepository(
                id=checkpoint.cached_repo_ids[0],
                headers=repos[0].raw_headers,
                raw_data=repos[0].raw_data,
            )
            checkpoint.stage = GithubConnectorStage.PRS
            checkpoint.curr_page = 0
            # save checkpoint with repo ids retrieved
            return checkpoint

        assert checkpoint.cached_repo is not None, "No repo saved in checkpoint"

        # Try to access the requester - different PyGithub versions may use different attribute names
        try:
            # Try direct access to a known attribute name first
            if hasattr(self.github_client, "_requester"):
                requester = self.github_client._requester
            elif hasattr(self.github_client, "_Github__requester"):
                requester = self.github_client._Github__requester
            else:
                # If we can't find the requester attribute, we need to fall back to recreating the repo
                raise AttributeError("Could not find requester attribute")

            repo = checkpoint.cached_repo.to_Repository(requester)
        except Exception as e:
            # If all else fails, re-fetch the repo directly
            logger.warning(
                f"Failed to deserialize repository: {e}. Attempting to re-fetch."
            )
            repo_id = checkpoint.cached_repo.id
            repo = self.github_client.get_repo(repo_id)

        if self.include_prs and checkpoint.stage == GithubConnectorStage.PRS:
            logger.info(f"Fetching PRs for repo: {repo.name}")
            pull_requests = repo.get_pulls(
                state=self.state_filter, sort="updated", direction="desc"
            )

            doc_batch: list[Document] = []
            pr_batch = _get_batch_rate_limited(
                pull_requests, checkpoint.curr_page, self.github_client
            )
            checkpoint.curr_page += 1
            done_with_prs = False
            for pr in pr_batch:
                # we iterate backwards in time, so at this point we stop processing prs
                if (
                    start is not None
                    and pr.updated_at
                    and pr.updated_at.replace(tzinfo=timezone.utc) < start
                ):
                    yield from doc_batch
                    done_with_prs = True
                    break
                # Skip PRs updated after the end date
                if (
                    end is not None
                    and pr.updated_at
                    and pr.updated_at.replace(tzinfo=timezone.utc) > end
                ):
                    continue
                try:
                    doc_batch.append(_convert_pr_to_document(cast(PullRequest, pr)))
                except Exception as e:
                    error_msg = f"Error converting PR to document: {e}"
                    logger.exception(error_msg)
                    yield ConnectorFailure(
                        failed_document=DocumentFailure(
                            document_id=str(pr.id), document_link=pr.html_url
                        ),
                        failure_message=error_msg,
                        exception=e,
                    )
                    continue

            # if we found any PRs on the page, yield any associated documents and return the checkpoint
            if not done_with_prs and len(pr_batch) > 0:
                yield from doc_batch
                return checkpoint

            # if we went past the start date during the loop or there are no more
            # prs to get, we move on to issues
            checkpoint.stage = GithubConnectorStage.ISSUES
            checkpoint.curr_page = 0

        checkpoint.stage = GithubConnectorStage.ISSUES

        if self.include_issues and checkpoint.stage == GithubConnectorStage.ISSUES:
            logger.info(f"Fetching issues for repo: {repo.name}")
            issues = repo.get_issues(
                state=self.state_filter, sort="updated", direction="desc"
            )

            doc_batch = []
            issue_batch = _get_batch_rate_limited(
                issues, checkpoint.curr_page, self.github_client
            )
            checkpoint.curr_page += 1
            done_with_issues = False
            for issue in cast(list[Issue], issue_batch):
                # we iterate backwards in time, so at this point we stop processing prs
                if (
                    start is not None
                    and issue.updated_at.replace(tzinfo=timezone.utc) < start
                ):
                    yield from doc_batch
                    done_with_issues = True
                    break
                # Skip PRs updated after the end date
                if (
                    end is not None
                    and issue.updated_at.replace(tzinfo=timezone.utc) > end
                ):
                    continue

                if issue.pull_request is not None:
                    # PRs are handled separately
                    continue

                try:
                    doc_batch.append(_convert_issue_to_document(issue))
                except Exception as e:
                    error_msg = f"Error converting issue to document: {e}"
                    logger.exception(error_msg)
                    yield ConnectorFailure(
                        failed_document=DocumentFailure(
                            document_id=str(issue.id),
                            document_link=issue.html_url,
                        ),
                        failure_message=error_msg,
                        exception=e,
                    )
                    continue

            # if we found any issues on the page, yield them and return the checkpoint
            if not done_with_issues and len(issue_batch) > 0:
                yield from doc_batch
                return checkpoint

            # if we went past the start date during the loop or there are no more
            # issues to get, we move on to the FILES stage
            checkpoint.stage = GithubConnectorStage.FILES
            checkpoint.curr_page = 0

        checkpoint.stage = GithubConnectorStage.FILES

        if self.include_files and checkpoint.stage == GithubConnectorStage.FILES:
            logger.info(f"Fetching source files for repo: {repo.name}")
            doc_batch = []
            # Initialize the stack with the root directory if not already in the checkpoint
            if (
                not hasattr(checkpoint, "directory_stack")
                or checkpoint.directory_stack is None
            ):
                checkpoint.directory_stack = [""]

            while checkpoint.directory_stack:
                current_path = (
                    checkpoint.directory_stack.pop()
                )  # Get the current directory path
                contents = repo.get_contents(
                    current_path
                )  # Fetch contents of the directory

                checkpoint.curr_page += 1

                logger.info(
                    f"Processing directory: {current_path}, stack size: {len(checkpoint.directory_stack)}"
                )
                logger.info(f"Found {len(contents)} items in directory")

                for content in contents:
                    # Skip files updated before the start date
                    if start is not None and content.last_modified_datetime < start:
                        # yield from doc_batch
                        # done_with_contents = True
                        # break
                        continue
                    # Skip files updated after the end date
                    if end is not None and content.last_modified_datetime > end:
                        continue

                    if content.type == "dir":
                        # Add the directory's path to the stack
                        checkpoint.directory_stack.append(content.path)
                    elif self.should_process_file(content.path) is False:
                        continue
                    else:
                        try:
                            doc_batch.extend(
                                self.process_content_into_documents(content)
                            )
                        except Exception as e:
                            error_msg = f"Error converting content to document: {e}"
                            logger.exception(error_msg)
                            yield ConnectorFailure(
                                failed_document=DocumentFailure(
                                    document_id=str(content.name),
                                    document_link=content.path,
                                ),
                                failure_message=error_msg,
                                exception=e,
                            )
                            continue

                # if we found any files on the current directory,
                # yield them and return the checkpoint
                if doc_batch and len(checkpoint.directory_stack) > 0:
                    yield from doc_batch
                    doc_batch = []  # Clear the batch after yielding
                    # Only return checkpoint if we have items to process
                    return checkpoint

            # After the directory traversal loop ends
            if doc_batch:  # If we have any remaining documents
                yield from doc_batch

            # if we went past the start date during the loop or there are no more
            # issues to get, we move on to the next repo
            checkpoint.stage = GithubConnectorStage.PRS
            checkpoint.curr_page = 0

        checkpoint.has_more = len(checkpoint.cached_repo_ids) > 1
        if checkpoint.cached_repo_ids:
            next_id = checkpoint.cached_repo_ids.pop()
            next_repo = self.github_client.get_repo(next_id)
            checkpoint.cached_repo = SerializedRepository(
                id=next_id,
                headers=next_repo.raw_headers,
                raw_data=next_repo.raw_data,
            )

        return checkpoint

    @override
    def load_from_checkpoint(
        self,
        start: SecondsSinceUnixEpoch,
        end: SecondsSinceUnixEpoch,
        checkpoint: GithubConnectorCheckpoint,
    ) -> CheckpointOutput[GithubConnectorCheckpoint]:
        start_datetime = datetime.fromtimestamp(start, tz=timezone.utc)
        end_datetime = datetime.fromtimestamp(end, tz=timezone.utc)

        # Move start time back by 3 hours, since some Issues/PRs are getting dropped
        # Could be due to delayed processing on GitHub side
        # The non-updated issues since last poll will be shortcut-ed and not embedded
        adjusted_start_datetime = start_datetime - timedelta(hours=3)

        epoch = datetime.fromtimestamp(0, tz=timezone.utc)
        if adjusted_start_datetime < epoch:
            adjusted_start_datetime = epoch

        return self._fetch_from_github(
            checkpoint, start=adjusted_start_datetime, end=end_datetime
        )

    def validate_connector_settings(self) -> None:
        if self.github_client is None:
            raise ConnectorMissingCredentialError("GitHub credentials not loaded.")

        if not self.repo_owner:
            raise ConnectorValidationError(
                "Invalid connector settings: 'repo_owner' must be provided."
            )

        try:
            if self.repositories:
                if "," in self.repositories:
                    # Multiple repositories specified
                    repo_names = [name.strip() for name in self.repositories.split(",")]
                    if not repo_names:
                        raise ConnectorValidationError(
                            "Invalid connector settings: No valid repository names provided."
                        )

                    # Validate at least one repository exists and is accessible
                    valid_repos = False
                    validation_errors = []

                    for repo_name in repo_names:
                        if not repo_name:
                            continue

                        try:
                            test_repo = self.github_client.get_repo(
                                f"{self.repo_owner}/{repo_name}"
                            )
                            test_repo.get_contents("")
                            valid_repos = True
                            # If at least one repo is valid, we can proceed
                            break
                        except GithubException as e:
                            validation_errors.append(
                                f"Repository '{repo_name}': {e.data.get('message', str(e))}"
                            )

                    if not valid_repos:
                        error_msg = (
                            "None of the specified repositories could be accessed: "
                        )
                        error_msg += ", ".join(validation_errors)
                        raise ConnectorValidationError(error_msg)
                else:
                    # Single repository (backward compatibility)
                    test_repo = self.github_client.get_repo(
                        f"{self.repo_owner}/{self.repositories}"
                    )
                    test_repo.get_contents("")
            else:
                # Try to get organization first
                try:
                    org = self.github_client.get_organization(self.repo_owner)
                    org.get_repos().totalCount  # Just check if we can access repos
                except GithubException:
                    # If not an org, try as a user
                    user = self.github_client.get_user(self.repo_owner)
                    user.get_repos().totalCount  # Just check if we can access repos

        except RateLimitExceededException:
            raise UnexpectedValidationError(
                "Validation failed due to GitHub rate-limits being exceeded. Please try again later."
            )

        except GithubException as e:
            if e.status == 401:
                raise CredentialExpiredError(
                    "GitHub credential appears to be invalid or expired (HTTP 401)."
                )
            elif e.status == 403:
                raise InsufficientPermissionsError(
                    "Your GitHub token does not have sufficient permissions for this repository (HTTP 403)."
                )
            elif e.status == 404:
                if self.repositories:
                    if "," in self.repositories:
                        raise ConnectorValidationError(
                            f"None of the specified GitHub repositories could be found for owner: {self.repo_owner}"
                        )
                    else:
                        raise ConnectorValidationError(
                            f"GitHub repository not found with name: {self.repo_owner}/{self.repositories}"
                        )
                else:
                    raise ConnectorValidationError(
                        f"GitHub user or organization not found: {self.repo_owner}"
                    )
            else:
                raise ConnectorValidationError(
                    f"Unexpected GitHub error (status={e.status}): {e.data}"
                )

        except Exception as exc:
            raise Exception(
                f"Unexpected error during GitHub settings validation: {exc}"
            )

    def validate_checkpoint_json(
        self, checkpoint_json: str
    ) -> GithubConnectorCheckpoint:
        return GithubConnectorCheckpoint.model_validate_json(checkpoint_json)

    def build_dummy_checkpoint(self) -> GithubConnectorCheckpoint:
        return GithubConnectorCheckpoint(
            stage=GithubConnectorStage.PRS, curr_page=0, has_more=True
        )


if __name__ == "__main__":
    import os

    connector = GithubSourceConnector(
        repo_owner=os.environ["REPO_OWNER"],
        repositories=os.environ["REPOSITORIES"],
        excluded_extensions=["jpg", "jpeg", "png", "gif", "mp4", "avi", "mov"],
    )
    connector.load_credentials(
        {"github_access_token": os.environ["ACCESS_TOKEN_GITHUB"]}
    )
    document_batches = connector.load_from_checkpoint(
        0, time.time(), connector.build_dummy_checkpoint()
    )
    print(next(document_batches))
