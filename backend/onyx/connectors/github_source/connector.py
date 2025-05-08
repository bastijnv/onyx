import copy
import os
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
from tree_sitter import Parser
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
    start_line: int = 0
    end_line: int = 0
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Common metadata fields
    variable: list[str] = field(default_factory=list)
    variable_member: list[str] = field(default_factory=list)
    variable_parameter: list[str] = field(default_factory=list)
    variable_builtin: list[str] = field(default_factory=list)

    function_method: list[str] = field(default_factory=list)
    function_method_call: list[str] = field(default_factory=list)

    _type: list[str] = field(default_factory=list)
    type_builtin: list[str] = field(default_factory=list)
    type_definition: list[str] = field(default_factory=list)

    module: list[str] = field(default_factory=list)
    namespace: list[str] = field(default_factory=list)
    constant: list[str] = field(default_factory=list)
    constant_macro: list[str] = field(default_factory=list)
    constant_builtin: list[str] = field(default_factory=list)

    keyword: list[str] = field(default_factory=list)
    keyword_conditional: list[str] = field(default_factory=list)
    keyword_repeat: list[str] = field(default_factory=list)
    keyword_return: list[str] = field(default_factory=list)
    keyword_operator: list[str] = field(default_factory=list)
    keyword_import: list[str] = field(default_factory=list)
    keyword_modifier: list[str] = field(default_factory=list)
    keyword_directive: list[str] = field(default_factory=list)
    keyword_exception: list[str] = field(default_factory=list)

    attribute: list[str] = field(default_factory=list)
    attribute_builtin: list[str] = field(default_factory=list)

    label: list[str] = field(default_factory=list)
    operator: list[str] = field(default_factory=list)
    _property: list[str] = field(default_factory=list)
    constructor: list[str] = field(default_factory=list)

    def to_document(self) -> Document:
        """Convert the chunk to a document for indexing."""
        # Set maximum lengths for metadata fields
        MAX_TAG_LENGTH = 255

        # Truncate string values
        def truncate_str(value: str) -> str:
            if not value:
                return ""
            if len(value) > MAX_TAG_LENGTH:
                return value[: MAX_TAG_LENGTH - 3] + "..."
            return value

        return Document(
            id=self.chunk_id,
            sections=[TextSection(link=self.repo_url, text=self.text or "")],
            source=DocumentSource.GITHUB_SOURCE,
            semantic_identifier=self.chunk_id,
            doc_updated_at=self.updated_at.replace(tzinfo=timezone.utc),
            metadata={
                "repository": truncate_str(self.repository),
                "repo_url": truncate_str(self.repo_url),
                "file_path": truncate_str(self.file_path),
                "file_type": truncate_str(self.file_type),
                "start_line": str(self.start_line),
                "end_line": str(self.end_line),
                "parent_class": truncate_str(self.parent_class or ""),
                "parent_function": truncate_str(self.parent_function or ""),
                "source": "github",
            },
        )


class TreeSitterChunker:
    """Handle code parsing using tree-sitter for better code understanding."""

    def __init__(self, query_dir: str = "./queries"):
        """
        Initialize the TreeSitterChunker.

        Args:
            language_dir: Directory containing compiled tree-sitter language libraries
        """
        # Convert relative path to absolute path
        if not os.path.isabs(query_dir):
            # Get the directory where the connector script is located
            current_dir = os.path.dirname(os.path.abspath(__file__))
            query_dir = os.path.abspath(os.path.join(current_dir, query_dir))

        self.parsers: dict[str, Parser] = {}
        self.query_dir = query_dir

        # Debugging: Log the absolute path
        logger.info(f"Tree-sitter queries directory path: {self.query_dir}")

        self._init_parsers()

    def _init_parsers(self) -> None:
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

    def _load_query(self, query_name: str) -> str:
        """
        Load a query from a `.scm` file.

        Args:
            query_name: Name of the query file (without extension)

        Returns:
            The query text as a string.
        """
        query_path = os.path.join(self.query_dir, f"{query_name}.scm")
        if not os.path.exists(query_path):
            raise FileNotFoundError(f"Query file not found: {query_path}")

        with open(query_path, "r", encoding="utf-8") as f:
            return f.read()

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

        metadata: dict[str, list[str]] = {
            # Common metadata fields
            "variable": [],  # Basic variables
            "variable.member": [],  # Member variables/properties
            "variable.parameter": [],  # Function parameters
            "variable.builtin": [],  # Built-in variables like this, base
            "function.method": [],  # Method definitions
            "function.method.call": [],  # Method calls
            "type": [],  # Types/classes references
            "type.builtin": [],  # Built-in types
            "type.definition": [],  # Type definitions
            "module": [],  # Modules/namespaces
            "constant": [],  # Constants
            "constant.macro": [],  # Macro constants
            "constant.builtin": [],  # Built-in constants
            "keyword": [],  # General keywords
            "keyword.conditional": [],  # if/else/switch
            "keyword.repeat": [],  # loops
            "keyword.return": [],  # return statements
            "keyword.operator": [],  # operator keywords
            "keyword.import": [],  # import/using keywords
            "keyword.modifier": [],  # access modifiers
            "keyword.directive": [],  # preprocessor directives
            "keyword.exception": [],  # try/catch/throw
            "keyword.type": [],  # type keywords
            "keyword.conditional.ternary": [],  # Ternary conditional operator
            "attribute": [],  # Attributes/decorators
            "attribute.builtin": [],  # Built-in attributes
            # Other metadata
            "label": [],  # Code labels for goto
            "operator": [],  # Operators
            "property": [],  # Properties
            "constructor": [],  # Constructors
            "namespace": [],  # Current namespace
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
            elif ext in (".ts", ".tsx"):
                metadata = self._extract_ts_tsx_metadata(tree)
            elif ext in (".js", ".jsx"):
                metadata = self._extract_js_jsx_metadata(tree)
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
        return self._extract_metadata_with_query(tree, "c_query")

    def _extract_cpp_metadata(self, tree: tree_sitter.Tree) -> dict[str, Any]:
        return self._extract_metadata_with_query(tree, "cpp_query")

    def _extract_python_metadata(self, tree: tree_sitter.Tree) -> dict[str, Any]:
        return self._extract_metadata_with_query(tree, "python_query")

    def _extract_ts_tsx_metadata(self, tree: tree_sitter.Tree) -> dict[str, Any]:
        return self._extract_metadata_with_query(tree, "typescript_query")

    def _extract_js_jsx_metadata(self, tree: tree_sitter.Tree) -> dict[str, Any]:
        return self._extract_metadata_with_query(tree, "javascript_query")

    def _extract_csharp_metadata(self, tree: tree_sitter.Tree) -> dict[str, Any]:
        return self._extract_metadata_with_query(tree, "c_sharp_query")

    def _extract_metadata_with_query(
        self, tree: tree_sitter.Tree, query_name: str
    ) -> dict[str, Any]:
        """
        Extract metadata from code using a tree-sitter query.

        Args:
            tree: The parsed tree-sitter syntax tree.
            query: The tree-sitter query string.

        Returns:
            A dictionary containing extracted metadata.
        """
        metadata: dict[str, list[str]] = {
            # Common metadata fields
            "variable": [],  # Basic variables
            "variable.member": [],  # Member variables/properties
            "variable.parameter": [],  # Function parameters
            "variable.builtin": [],  # Built-in variables like this, base
            "function.method": [],  # Method definitions
            "function.method.call": [],  # Method calls
            "type": [],  # Types/classes references
            "type.builtin": [],  # Built-in types
            "type.definition": [],  # Type definitions
            "module": [],  # Modules/namespaces
            "constant": [],  # Constants
            "constant.macro": [],  # Macro constants
            "constant.builtin": [],  # Built-in constants
            "keyword": [],  # General keywords
            "keyword.conditional": [],  # if/else/switch
            "keyword.repeat": [],  # loops
            "keyword.return": [],  # return statements
            "keyword.operator": [],  # operator keywords
            "keyword.import": [],  # import/using keywords
            "keyword.modifier": [],  # access modifiers
            "keyword.directive": [],  # preprocessor directives
            "keyword.exception": [],  # try/catch/throw
            "keyword.type": [],  # type keywords
            "keyword.conditional.ternary": [],  # Ternary conditional operator
            "attribute": [],  # Attributes/decorators
            "attribute.builtin": [],  # Built-in attributes
            # Other metadata
            "label": [],  # Code labels for goto
            "operator": [],  # Operators
            "property": [],  # Properties
            "constructor": [],  # Constructors
            "namespace": [],  # Current namespace
        }

        try:
            root_node = tree.root_node

            # For debugging any specific parse issues
            if root_node.has_error:
                logger.debug("Tree has parsing errors but continuing with query")
            try:
                query_text = self._load_query(query_name)
                parser_query = Query(tree.language, query_text)
                captures = parser_query.captures(root_node)

                for capture, nodes in captures.items():
                    try:
                        for node in nodes:
                            if node.text is None:
                                continue

                            text = node.text.decode("utf-8")

                            # Variable categories
                            if capture == "variable" or capture.startswith("variable."):
                                metadata[capture].append(text)

                            # Function/method categories
                            elif capture == "function.method":
                                metadata["function.method"].append(text)

                            elif capture == "function.method.call":
                                metadata["function.method.call"].append(text)

                            # Type categories
                            elif capture.startswith("type"):
                                metadata[capture].append(text)

                            # Module/namespace
                            elif capture == "module":
                                metadata["module"].append(text)
                                metadata["namespace"].append(text)

                            # Constant categories
                            elif capture.startswith("constant"):
                                metadata[capture].append(text)

                            # Keyword categories
                            elif capture.startswith("keyword"):
                                metadata[capture].append(text)

                            # Attribute/decorator categories
                            elif capture.startswith("attribute"):
                                metadata[capture].append(text)
                                # Also add to decorators for backward compatibility
                                if "decorators" not in metadata:
                                    metadata["decorators"] = []
                                metadata["decorators"].append(text)

                            # Other specific categories
                            elif capture == "label":
                                metadata["label"].append(text)
                            elif capture == "operator":
                                metadata["operator"].append(text)
                            elif capture == "property":
                                metadata["property"].append(text)
                            elif capture == "constructor":
                                metadata["constructor"].append(text)

                            # else:
                            #    logger.debug(f"Unhandled capture type: {capture} with value: {text}")
                    except Exception as node_err:
                        logger.debug(f"Error processing node: {node_err}")
                        continue

            except Exception as query_err:
                logger.warning(f"Query execution error: {query_err}")
                # Continue with empty metadata rather than failing completely

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

        # Import the Language enum from langchain_text_splitters
        from langchain_text_splitters import Language

        # Create a default splitter for languages without specific support
        default_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

        # Python splitter
        splitters[".py"] = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        # JavaScript/TypeScript splitters
        splitters[".js"] = RecursiveCharacterTextSplitter.from_language(
            language=Language.JS,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        splitters[".jsx"] = splitters[".js"]
        splitters[".ts"] = RecursiveCharacterTextSplitter.from_language(
            language=Language.TS,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        splitters[".tsx"] = splitters[".ts"]

        # Java splitter
        splitters[".java"] = RecursiveCharacterTextSplitter.from_language(
            language=Language.JAVA,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        # C# splitter
        splitters[".cs"] = RecursiveCharacterTextSplitter.from_language(
            language=Language.CSHARP,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        # HTML splitter (XML has to use default)
        splitters[".html"] = RecursiveCharacterTextSplitter.from_language(
            language=Language.HTML,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        splitters[".htm"] = splitters[".html"]

        # XML and related formats - use HTML as closest alternative or default
        try:
            # Try to use HTML as a fallback for XML formats
            xml_splitter = RecursiveCharacterTextSplitter.from_language(
                language=Language.HTML,  # Use HTML as proxy for XML-like languages
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
            language=Language.RUBY,
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
            language=Language.C,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        splitters[".cpp"] = RecursiveCharacterTextSplitter.from_language(
            language=Language.CPP,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        splitters[".hpp"] = splitters[".cpp"]

        # Markdown splitter
        splitters[".md"] = RecursiveCharacterTextSplitter.from_language(
            language=Language.MARKDOWN,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        splitters[".markdown"] = splitters[".md"]

        # Go splitter
        splitters[".go"] = RecursiveCharacterTextSplitter.from_language(
            language=Language.GO,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        # PHP splitter
        splitters[".php"] = RecursiveCharacterTextSplitter.from_language(
            language=Language.PHP,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        # Rust splitter
        splitters[".rs"] = RecursiveCharacterTextSplitter.from_language(
            language=Language.RUST,
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
        Chunk a file recursively using the appropriate splitter with improved function call detection.
        """
        file_path = content.path
        _, ext = os.path.splitext(file_path)
        file_type = ext.lstrip(".")
        splitter = self.get_splitter_for_file(file_path)

        # Extract code metadata if possible
        parent_class = None
        parent_function = None

        # Initialize metadata variables
        variable = []
        variable_member = []
        variable_parameter = []
        variable_builtin = []

        function_method = []
        function_method_call = []

        type_data = []
        type_builtin = []
        type_definition = []

        module = []
        namespace = []

        constant = []
        constant_macro = []
        constant_builtin = []

        keyword = []
        keyword_conditional = []
        keyword_repeat = []
        keyword_return = []
        keyword_operator = []
        keyword_import = []
        keyword_modifier = []
        keyword_directive = []
        keyword_exception = []

        attribute = []
        attribute_builtin = []

        label = []
        operator = []
        property_data = []
        constructor = []

        try:
            if content.encoding != "base64":
                logger.warning(
                    f"File {file_path} is not base64 encoded. Skipping chunking."
                )
                return []

            file_content = content.decoded_content.decode("utf-8")
            code_metadata = {}

            # Parse the full file with tree-sitter to extract metadata
            if self.tree_sitter_chunker.has_parser(ext):
                try:
                    code_metadata = self.tree_sitter_chunker.extract_metadata(
                        file_content, file_path
                    )

                    # Extract metadata using the new structure
                    # Extract metadata fields from the new structure
                    parent_class = None
                    parent_function = None

                    # Extract function and class-related metadata
                    if code_metadata.get("type.definition"):
                        parent_class = (
                            code_metadata["type.definition"][0]
                            if code_metadata["type.definition"]
                            else None
                        )

                    if code_metadata.get("function.method"):
                        parent_function = (
                            code_metadata["function.method"][0]
                            if code_metadata["function.method"]
                            else None
                        )

                    # Extract common metadata fields - unqiue only
                    variable = list(set(code_metadata.get("variable", [])))
                    variable_member = list(
                        set(code_metadata.get("variable.member", []))
                    )
                    variable_parameter = list(
                        set(code_metadata.get("variable.parameter", []))
                    )
                    variable_builtin = list(
                        set(code_metadata.get("variable.builtin", []))
                    )

                    # Extract function-related metadata
                    function_method = list(
                        set(code_metadata.get("function.method", []))
                    )
                    function_method_call = list(
                        set(code_metadata.get("function.method.call", []))
                    )

                    # Extract type-related metadata
                    type_data = list(set(code_metadata.get("type", [])))
                    type_builtin = list(set(code_metadata.get("type.builtin", [])))
                    type_definition = list(
                        set(code_metadata.get("type.definition", []))
                    )

                    # Extract other common metadata
                    module = list(set(code_metadata.get("module", [])))
                    namespace = list(set(code_metadata.get("namespace", [])))

                    constant = list(set(code_metadata.get("constant", [])))
                    constant_macro = list(set(code_metadata.get("constant.macro", [])))
                    constant_builtin = list(
                        set(code_metadata.get("constant.builtin", []))
                    )

                    keyword = list(set(code_metadata.get("keyword", [])))
                    keyword_conditional = list(
                        set(code_metadata.get("keyword.conditional", []))
                    )
                    keyword_repeat = list(set(code_metadata.get("keyword.repeat", [])))
                    keyword_return = list(set(code_metadata.get("keyword.return", [])))
                    keyword_operator = list(
                        set(code_metadata.get("keyword.operator", []))
                    )
                    keyword_import = list(set(code_metadata.get("keyword.import", [])))
                    keyword_modifier = list(
                        set(code_metadata.get("keyword.modifier", []))
                    )
                    keyword_directive = list(
                        set(code_metadata.get("keyword.directive", []))
                    )
                    keyword_exception = list(
                        set(code_metadata.get("keyword.exception", []))
                    )

                    attribute = list(set(code_metadata.get("attribute", [])))
                    attribute_builtin = list(
                        set(code_metadata.get("attribute.builtin", []))
                    )

                    label = list(set(code_metadata.get("label", [])))
                    operator = list(set(code_metadata.get("operator", [])))
                    property_data = list(set(code_metadata.get("property", [])))
                    constructor = list(set(code_metadata.get("constructor", [])))
                except Exception as e:
                    logger.warning(f"Error extracting metadata from {file_path}: {e}")

            # Create a metadata prefix for the file content
            # We prepend the metadata to the file content as directly
            # adding as tags may cause issues with length
            metadata_prefix = self._create_metadata_prefix(
                file_path=file_path,
                repository=content.repository.name,
                repo_url=content.html_url,
                file_type=file_type,
                parent_class=parent_class,
                parent_function=parent_function,
                variable=variable,
                variable_member=variable_member,
                variable_parameter=variable_parameter,
                variable_builtin=variable_builtin,
                function_method=function_method,
                function_method_call=function_method_call,
                type_data=type_data,
                type_builtin=type_builtin,
                type_definition=type_definition,
                module=module,
                namespace=namespace,
                constant=constant,
                constant_macro=constant_macro,
                constant_builtin=constant_builtin,
                keyword=keyword,
                keyword_conditional=keyword_conditional,
                keyword_repeat=keyword_repeat,
                keyword_return=keyword_return,
                keyword_operator=keyword_operator,
                keyword_import=keyword_import,
                keyword_modifier=keyword_modifier,
                keyword_directive=keyword_directive,
                keyword_exception=keyword_exception,
                attribute=attribute,
                attribute_builtin=attribute_builtin,
                label=label,
                operator=operator,
                property_data=property_data,
                constructor=constructor,
            )

            # Prepend the metadata to the file content
            enhanced_content = metadata_prefix + "\n\n" + file_content

            # Chunk the text
            chunks = []
            try:
                # Split the text into chunks
                text_chunks = splitter.split_text(enhanced_content)

                # Create CodeChunk objects
                for i, chunk_text in enumerate(text_chunks):
                    # Create a unique chunk ID
                    chunk_id = f"{content.repository.full_name}:{file_path}:{i}"

                    # Extract line numbers for this chunk
                    start_line = (
                        file_content.count(
                            "\n", 0, file_content.find(chunk_text.strip()[:50])
                        )
                        + 1
                    )
                    end_line = start_line + chunk_text.count("\n")

                    # Create chunk with chunk-specific called functions
                    # Create a chunk with all extracted metadata
                    chunk = CodeChunk(
                        text=chunk_text,
                        file_path=file_path,
                        chunk_id=chunk_id,
                        repository=content.repository.name,
                        repo_url=content.html_url,
                        file_type=file_type,
                        parent_class=parent_class,
                        parent_function=parent_function,
                        start_line=start_line,
                        end_line=end_line,
                        updated_at=content.last_modified_datetime
                        or datetime.now(timezone.utc),
                    )
                    chunks.append(chunk)

            except Exception as e:
                logger.error(f"Error chunking file {file_path}: {e}")
                # Fall back to creating a single chunk with the entire file
                chunk_id = f"{content.html_url}:0"

                chunk = CodeChunk(
                    text=file_content,
                    file_path=file_path,
                    chunk_id=chunk_id,
                    repository=content.repository.name,
                    repo_url=content.repository.html_url,
                    file_type=file_type,
                    parent_class=parent_class,
                    parent_function=parent_function,
                    start_line=1,
                    end_line=file_content.count("\n") + 1,
                    updated_at=content.last_modified_datetime
                    or datetime.now(timezone.utc),
                )
                chunks.append(chunk)

            return chunks

        except Exception as e:
            logger.warning(f"Error processing file {file_path}: {e}")
            return []  # Return empty list instead of failing

    def _create_metadata_prefix(self, **kwargs: Any) -> str:
        """
        Create a metadata prefix with file information similar to Repomix.

        Args:
            Various metadata fields extracted from the file

        Returns:
            Formatted string with metadata information
        """
        file_path = kwargs.get("file_path", "")
        repository = kwargs.get("repository", "")
        repo_url = kwargs.get("repo_url", "")
        file_type = kwargs.get("file_type", "")
        parent_class = kwargs.get("parent_class", "")
        parent_function = kwargs.get("parent_function", "")

        # Extract all metadata fields
        variable = kwargs.get("variable", [])
        variable_member = kwargs.get("variable_member", [])
        variable_parameter = kwargs.get("variable_parameter", [])
        variable_builtin = kwargs.get("variable_builtin", [])

        function_method = kwargs.get("function_method", [])
        function_method_call = kwargs.get("function_method_call", [])

        type_data = kwargs.get("type_data", [])
        type_builtin = kwargs.get("type_builtin", [])
        type_definition = kwargs.get("type_definition", [])

        module = kwargs.get("module", [])
        namespace = kwargs.get("namespace", [])

        constant = kwargs.get("constant", [])
        constant_macro = kwargs.get("constant_macro", [])
        constant_builtin = kwargs.get("constant_builtin", [])

        keyword = kwargs.get("keyword", [])
        keyword_conditional = kwargs.get("keyword_conditional", [])
        keyword_repeat = kwargs.get("keyword_repeat", [])
        keyword_return = kwargs.get("keyword_return", [])
        keyword_operator = kwargs.get("keyword_operator", [])
        keyword_import = kwargs.get("keyword_import", [])
        keyword_modifier = kwargs.get("keyword_modifier", [])
        keyword_directive = kwargs.get("keyword_directive", [])
        keyword_exception = kwargs.get("keyword_exception", [])

        attribute = kwargs.get("attribute", [])
        attribute_builtin = kwargs.get("attribute_builtin", [])

        label = kwargs.get("label", [])
        operator = kwargs.get("operator", [])
        property_data = kwargs.get("property_data", [])
        constructor = kwargs.get("constructor", [])

        # Build the prefix
        prefix = [
            f"# Code File: {file_path}",
            f"Repository: {repository}",
            f"URL: {repo_url}",
            f"File Type: {file_type}",
        ]

        # Add classes and functions
        if parent_class:
            prefix.append(f"Primary Class: {parent_class}")
        if parent_function:
            prefix.append(f"Primary Function: {parent_function}")

        # Add key metadata sections
        if type_definition:
            prefix.append("\n## Type Definitions")
            prefix.append(", ".join(type_definition))

        if function_method:
            prefix.append("\n## Functions/Methods")
            prefix.append(", ".join(function_method))

        if function_method_call:
            prefix.append("\n## Function Calls")
            prefix.append(", ".join(function_method_call))

        if variable:
            prefix.append("\n## Variables")
            prefix.append(", ".join(variable))

        if module:
            prefix.append("\n## Modules/Imports")
            prefix.append(", ".join(module))

        # Add additional metadata sections
        if variable_member:
            prefix.append("\n## Member Variables")
            prefix.append(", ".join(variable_member))

        if variable_parameter:
            prefix.append("\n## Parameters")
            prefix.append(", ".join(variable_parameter))

        if variable_builtin:
            prefix.append("\n## Built-in Variables")
            prefix.append(", ".join(variable_builtin))

        if type_data:
            prefix.append("\n## Types")
            prefix.append(", ".join(type_data))

        if type_builtin:
            prefix.append("\n## Built-in Types")
            prefix.append(", ".join(type_builtin))

        if namespace:
            prefix.append("\n## Namespaces")
            prefix.append(", ".join(namespace))

        if constant:
            prefix.append("\n## Constants")
            prefix.append(", ".join(constant))

        if constant_macro:
            prefix.append("\n## Macros")
            prefix.append(", ".join(constant_macro))

        if constant_builtin:
            prefix.append("\n## Built-in Constants")
            prefix.append(", ".join(constant_builtin))

        if keyword:
            prefix.append("\n## Keywords")
            prefix.append(", ".join(keyword))

        if keyword_conditional:
            prefix.append("\n## Conditional Keywords")
            prefix.append(", ".join(keyword_conditional))

        if keyword_repeat:
            prefix.append("\n## Loop Keywords")
            prefix.append(", ".join(keyword_repeat))

        if keyword_return:
            prefix.append("\n## Return Keywords")
            prefix.append(", ".join(keyword_return))

        if keyword_operator:
            prefix.append("\n## Operator Keywords")
            prefix.append(", ".join(keyword_operator))

        if keyword_import:
            prefix.append("\n## Import Keywords")
            prefix.append(", ".join(keyword_import))

        if keyword_modifier:
            prefix.append("\n## Modifier Keywords")
            prefix.append(", ".join(keyword_modifier))

        if keyword_directive:
            prefix.append("\n## Directive Keywords")
            prefix.append(", ".join(keyword_directive))

        if keyword_exception:
            prefix.append("\n## Exception Keywords")
            prefix.append(", ".join(keyword_exception))

        if attribute:
            prefix.append("\n## Attributes/Decorators")
            prefix.append(", ".join(attribute))

        if attribute_builtin:
            prefix.append("\n## Built-in Attributes")
            prefix.append(", ".join(attribute_builtin))

        if label:
            prefix.append("\n## Labels")
            prefix.append(", ".join(label))

        if operator:
            prefix.append("\n## Operators")
            prefix.append(", ".join(operator))

        if property_data:
            prefix.append("\n## Properties")
            prefix.append(", ".join(property_data))

        if constructor:
            prefix.append("\n## Constructors")
            prefix.append(", ".join(constructor))

        # Add code summary section
        prefix.append("\n## Code Summary")
        prefix.append(
            "This chunk contains source code from the file. The metadata above provides a summary of key elements in the file."
        )
        prefix.append("---")

        return "\n".join(prefix)


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
        tree_sitter_chunker = TreeSitterChunker(query_dir="./queries")
        self.code_chunker = RecursiveCodeChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            tree_sitter_chunker=tree_sitter_chunker,
        )

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

        # Check for specific files to ignore
        filename = os.path.basename(file_path)
        ignored_filenames = ["package-lock.json"]
        if filename in ignored_filenames:
            return False

        # Check for generated, build output, and other binary files
        ignored_patterns = [
            # Build output directories often found at file level
            "bin/",
            "obj/",
            "dist/",
            "out/",
            # VS and .NET specific files
            ".csproj.user",
            ".suo",
            ".vssscc",
            ".vspscc",
            ".vs/",
            ".vscode/",
            "*.lock.json",
            "launchSettings.json",
            # Build artifacts
            "*.min.js",
            "*.min.css",
            "*.bundle.js",
            # Generated TypeScript files
            ".d.ts",
            "*.js.map",
            "*.d.ts.map",
            # PowerShell build files
            "*.ps1xml",
            "*.psd1",
            "*.psm1.signature",
            # Test files that might contain large mocks
            "**/TestResults/**",
            "**/coverage/**",
            "*.coverage",
            # NuGet packages
            "packages.config",
            "project.assets.json",
            # MSBuild files
            "*.nuget.targets",
            "*.nuget.props",
        ]

        for pattern in ignored_patterns:
            if pattern.startswith("*") and pattern[1:] in file_path:
                return False
            elif pattern.endswith("/") and pattern[:-1] in file_path.split(os.path.sep):
                return False
            elif pattern in file_path:
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

    def index_chunks_batch(self, chunks: list[CodeChunk]) -> list[Document]:
        """
        Index a batch of code chunks into Onyx.

        Args:
            chunks: list of CodeChunk objects to index

        Returns:
            Tuple of (chunks indexed, failures)
        """
        if not chunks:
            return []

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

            current_path = (
                checkpoint.directory_stack.pop()
            )  # Get the current directory path
            contents = repo.get_contents(
                current_path
            )  # Fetch contents of the directory

            # Fix for handling both ContentFile and list[ContentFile] return types
            if not isinstance(contents, list):
                # If contents is a single file, convert it to a list
                contents = [contents]

            checkpoint.curr_page += 1

            logger.info(f"Stack size: {len(checkpoint.directory_stack)}")
            logger.info(f"Current directory stack: {checkpoint.directory_stack}")
            logger.info(f"Processing directory: {current_path}")
            logger.info(f"Found {len(contents)} items in {current_path}")

            for content in contents:
                # Skip files updated before the start date
                if start is not None and content.last_modified_datetime < start:
                    continue
                # Skip files updated after the end date
                if end is not None and content.last_modified_datetime > end:
                    continue

                if content.type == "dir":
                    # Add the directory's path to the stack
                    checkpoint.directory_stack.append(content.path)
                elif self.should_process_file(content.path) is False:
                    logger.info(f"Skipping file {content.path} due to exclusion rules.")
                    continue
                else:
                    try:
                        doc_batch.extend(self.process_content_into_documents(content))
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

            # Whether we have files to yield or not, return the checkpoint
            # if we still have directories to process
            yield from doc_batch
            if len(checkpoint.directory_stack) > 0:
                # Return checkpoint if we have more directories to process
                # regardless of whether current directory had files
                return checkpoint

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
