import asyncio
from typing import Dict, List, Literal, Optional, TypeAlias, Union

from asyncer import asyncify
from fastmcp import FastMCP
from pydantic import BaseModel, Field

from ..dataverse import Dataverse
from .collection import (
    get_collection_metadata,
    get_graph_summary,
    list_content,
    query_sparql,
)
from .datacite import DataCite
from .dataset import (
    create_dataset,
    edit_dataset,
    edit_dataset_fields,
    get_dataset,
    list_files,
)
from .dataverse import get_metrics
from .file import read_file, read_tabular, tabular_schema
from .middleware import DataverseMiddleware
from .search import search
from .vocabulary import DEFAULT_SOURCES, VocabularySource, search_vocabulary

DataverseType: TypeAlias = Union[Dataverse, Dict[str, Dataverse]]


class MCPConfiguration(BaseModel):
    """
    Configuration class for controlling which tools are available at different levels of the Dataverse MCP server.

    This class defines which operations are enabled for different components of the Dataverse system,
    allowing fine-grained control over the available functionality.

    Attributes:
        dataverse: List of tools available at the dataverse level. Currently supports "metrics".
        collection: List of tools available at the collection level. Supports "read", "inspect", "write".
        dataset: List of tools available at the dataset level. Supports "read", "create", "edit".
        file: List of tools available at the file level. Supports "read", "metadata", "edit".
    """

    dataverse: List[Literal["metrics"]] = Field(
        default_factory=lambda: ["metrics"],
        description="Controls the tools that are available for the dataverse level.",
    )
    collection: List[Literal["read", "inspect", "write", "graph"]] = Field(
        default_factory=lambda: ["read", "graph"],
        description="Controls the tools that are available for the collection level.",
    )
    dataset: List[Literal["read", "create", "edit"]] = Field(
        default_factory=lambda: ["read"],
        description="Controls the tools that are available for the dataset level.",
    )
    file: List[Literal["read", "metadata", "edit"]] = Field(
        default_factory=lambda: ["read", "metadata"],
        description="Controls the tools that are available for the file level.",
    )
    vocabulary_sources: List[VocabularySource] = Field(
        default_factory=lambda: list(DEFAULT_SOURCES),
        description="Controlled-vocabulary services the Search_Vocabulary tool queries. "
        "Defaults to Wikidata + the EBI OLS; a deployment can add its own OLS-compatible endpoints.",
    )


class DataverseMCP:
    """
    Main class for creating and configuring a Dataverse MCP (Model Context Protocol) server.

    This class provides a high-level interface for setting up MCP tools that interact with
    a Dataverse instance. It handles the registration of tools based on the provided configuration
    and manages the middleware for Dataverse connections.

    Attributes:
        dataverse: The Dataverse instance to interact with.
        config: Configuration object controlling which tools are available.
    """

    def __init__(
        self,
        dataverse: Dataverse,
        config: Optional[MCPConfiguration] = None,
    ):
        """
        Initialize the DataverseMCP instance.

        Args:
            dataverse: The Dataverse instance to use for all operations.
            config: Optional configuration object. If not provided, uses default configuration
                   with basic read-only access enabled.
        """
        self.dataverse = dataverse
        self.dataverse.verbose = 0

        self.config = config or MCPConfiguration()

    @property
    def base_url(self) -> str:
        """
        Get the base URL of the dataverse.
        """
        return self.dataverse.base_url

    @property
    def base_url_instructions(self) -> str:
        """
        Get the base URL instructions for the dataverse.
        """
        return f"This MCP server is connected to the dataverse at {self.base_url}. Provide the `base_url` parameter if you want to use a different dataverse."

    def to_mcp(self, mcp: FastMCP):
        """
        Configure the provided FastMCP instance with Dataverse tools and middleware.

        This method adds the DataverseMiddleware and registers all enabled tools
        based on the configuration settings.

        Args:
            mcp: The FastMCP instance to configure with Dataverse tools.
        """
        mcp.add_middleware(
            DataverseMiddleware(
                dataverse=self.dataverse,
                vocabulary_sources=self.config.vocabulary_sources,
            )
        )
        self._add_dataverse_tools(mcp)
        self._add_dataset_tools(mcp)
        self._add_file_tools(mcp)
        self._add_collection_tools(mcp)
        self._add_search_tools(mcp)
        self._set_instructions(mcp)

    def _set_instructions(self, mcp: FastMCP):
        """Set model-facing server instructions describing the enabled workflow.

        Instructions are sent in the MCP handshake and are the always-in-context
        channel for guiding the model on how the tools relate. Built from the
        config so writing guidance only appears when those tools are enabled.
        """
        lines = [
            f"This server works with datasets on the Dataverse at {self.base_url}.",
            "Search and read datasets, collections, and files with the read tools.",
        ]

        can_create = "create" in self.config.dataset
        can_edit = "edit" in self.config.dataset
        if can_create or can_edit:
            verbs = []
            if can_create:
                verbs.append("create draft datasets (Create_Dataset)")
            if can_edit:
                verbs.append(
                    "enrich or modify existing drafts (Edit_Dataset_Metadata for common "
                    "citation fields, Edit_Dataset_Fields for any other field or block)"
                )
            lines.append("You can also " + " and ".join(verbs) + ".")
            lines.append(
                "A dataset is authored incrementally: create it from the core citation "
                "metadata (title, description, subject, one author and contact), then "
                "enrich it — more authors, keywords, funding, coverage — by editing."
                if can_create and can_edit
                else ""
            )
            lines.append(
                "When editing a multi-value field (authors, contacts, subjects), pass the "
                "COMPLETE desired list — it replaces the existing values. To add one item, "
                "read the current values first (Get_Dataset_Metadata), then send the full list."
            )
            lines.append(
                "All writes produce or update a DRAFT. Never publish; publishing is a "
                "separate step the user performs themselves."
            )

        mcp.instructions = "\n".join(line for line in lines if line)

    def _tool(self, mcp: FastMCP, fn, *, enabled: bool = True, **kwargs):
        """Register a tool, honoring a config gate.

        FastMCP v3 dropped the ``enabled=`` kwarg on ``mcp.tool``; we register
        then disable to preserve the same config-gated behaviour.
        """
        mcp.tool(fn, **kwargs)
        if not enabled:
            mcp.disable(names={kwargs["name"]}, components={"tool"})

    def _add_dataverse_tools(self, mcp: FastMCP):
        """
        Add dataverse-level tools to the MCP instance.

        Currently supports:
        - get_metrics: Retrieves dataverse metrics in TOON format

        Args:
            mcp: The FastMCP instance to add tools to.
        """
        if "metrics" in self.config.dataverse:
            self._tool(mcp, 
                get_metrics,
                name="Dataverse_Metrics",
                description=f"Get metrics from the dataverse in total and for the past 7 days. This tool returns the metrics of the dataverse in a TOON format. {self.base_url_instructions}",
            )

    def _add_dataset_tools(self, mcp: FastMCP):
        """
        Add dataset-level tools to the MCP instance.

        Currently supports:
        - get_dataset: Retrieves dataset information with optional metadata blocks
        - list_files: Lists files within a dataset

        Args:
            mcp: The FastMCP instance to add tools to.
        """
        available_metadatablocks = self._all_metadatablocks()

        self._tool(mcp, 
            get_dataset,
            name="Get_Dataset_Metadata",
            enabled="read" in self.config.dataset,
            description=f"""
            Get a dataset from the dataverse. This tool returns the dataset in a TOON format. 
            When `full` is True, you can specify the metadata blocks to return. 
            The available metadata blocks are: {", ".join(available_metadatablocks)}. 
            If you do not specify the metadata blocks, the function will return all metadata blocks.
            To save tokens and compute resources, you should first request the dataset without the `full` flag to get the available metadata blocks and then request the dataset with the `full` flag and the metadata blocks you want to see.
            
            {self.base_url_instructions}
            """.strip(),
        )

        self._tool(mcp, 
            list_files,
            name="List_Files_in_Dataset",
            enabled="read" in self.config.dataset,
            description=f"List the files in a dataset. This tool returns the files in a TOON format. {self.base_url_instructions}",
        )

        self._tool(mcp, 
            create_dataset,
            name="Create_Dataset",
            enabled="create" in self.config.dataset,
            description=f"""
            Create a new DRAFT dataset and add it to a collection. Provide the core citation metadata:
            title, description, authors, contacts, and subjects (from the controlled vocabulary).
            Start here for a new dataset, then enrich it (more authors, keywords, funding, coverage)
            with Edit_Dataset_Metadata. This does NOT publish — publishing is a separate, explicit step.
            Requires the server to be connected with an authenticated (API-token) Dataverse instance.
            Returns the new dataset's persistent identifier and URL.

            {self.base_url_instructions}
            """.strip(),
        )

        self._tool(
            mcp,
            edit_dataset,
            name="Edit_Dataset_Metadata",
            enabled="edit" in self.config.dataset,
            description=f"""
            Edit citation metadata on an existing DRAFT dataset (upsert). Change any of
            title, description, authors, contacts, subjects — only the fields you pass are replaced.
            Multi-value fields (authors/contacts/subjects) are replaced wholesale: to ADD one,
            first read current values with Get_Dataset_Metadata and pass the full merged list.
            Does NOT publish. Requires an authenticated (API-token) connection.

            {self.base_url_instructions}
            """.strip(),
        )

        self._tool(
            mcp,
            edit_dataset_fields,
            name="Edit_Dataset_Fields",
            enabled="edit" in self.config.dataset,
            description=f"""
            Edit ANY metadata field on a DRAFT dataset (generic upsert), for fields the friendly
            Edit_Dataset_Metadata tool doesn't expose (keywords, publications, funding, coverage,
            or any field in blocks like geospatial/EngMeta). Pass a mapping of field name -> value
            following the field's schema; each field is replaced wholesale. Does NOT publish.

            {self.base_url_instructions}
            """.strip(),
        )

    def _add_file_tools(self, mcp: FastMCP):
        """
        Add file-level tools to the MCP instance.

        Currently supports:
        - read_tabular: Reads tabular files with optional summarization and row limiting
        - read_file: Reads raw file content
        - tabular_schema: Retrieves schema information for tabular files

        Args:
            mcp: The FastMCP instance to add tools to.
        """
        self._tool(mcp, 
            read_tabular,
            name="Read_Tabular_File",
            enabled="read" in self.config.file,
            description=f"""
            Read a tabular file from a dataset. This tool returns the file in a TOON format.
            When `summarize` is True, the function will return the description of the file using the `describe` method of the pandas DataFrame.
            When `n_rows` is specified, the function will return the first `n_rows` rows of the file.
            When `n_rows` is not specified, the function will return the entire file. Please note that this is capped to 1000 rows.
            
            {self.base_url_instructions}
            """.strip(),
        )

        self._tool(mcp, 
            read_file,
            name="Read_File_Content",
            enabled="read" in self.config.file,
            description=f"""
            Read then content of a file from a dataset. This tool returns the raw content of the file.
            
            {self.base_url_instructions}
            """.strip(),
        )

        self._tool(mcp, 
            tabular_schema,
            name="Get_Tabular_File_Schema",
            enabled="read" in self.config.file,
            description=f"Get the schema of a tabular file. This tool returns the schema of the file in a TOON format. {self.base_url_instructions}",
        )

    def _add_collection_tools(self, mcp: FastMCP):
        """
        Add collection-level tools to the MCP instance.

        Currently no collection tools are implemented.

        Args:
            mcp: The FastMCP instance to add tools to.
        """

        self._tool(mcp, 
            get_collection_metadata,
            name="Get_Collection_Metadata",
            enabled="read" in self.config.collection,
            description=f"Get the metadata of a collection. This tool returns the metadata of the collection in a TOON format. {self.base_url_instructions}",
        )

        self._tool(mcp, 
            list_content,
            name="List_Content_of_Collection",
            enabled="read" in self.config.collection,
            description=f"""
            List the content of a collection. This tool returns the content in a TOON format.
            You can filter the content by type. The available types are: dataset, collection.
            
            {self.base_url_instructions}
            """.strip(),
        )

        self._tool(mcp, 
            get_graph_summary,
            name="Knowledge_Graph_Summary",
            enabled="graph" in self.config.collection,
            description=f"""
            Get the summary of the RDF graph of a collection. This tool returns all classes and predicates 
            found in the collection's knowledge graph, including their full URIs and short names. 
            Useful for exploring the semantic structure of a collection before writing SPARQL queries.
            The graph can be generated in different formats: "croissant" or "OAI_ORE".
            Returns the summary in a TOON format.
            
            {self.base_url_instructions}
            """.strip(),
        )

        self._tool(mcp, 
            query_sparql,
            name="Query_Knowledge_Graph",
            enabled="graph" in self.config.collection,
            description=f"""
            Execute a SPARQL query against a collection's RDF graph. This tool allows you to query the 
            semantic metadata of a collection using SPARQL syntax. The graph can be generated in different 
            formats: "croissant" or "OAI_ORE". Query results are returned as a list of dictionaries where 
            each dictionary represents a result row with variable names as keys. Use get_graph_summary first 
            to explore available classes and predicates before writing queries. Returns results in a TOON format.
            
            {self.base_url_instructions}
            """.strip(),
        )

    def _add_search_tools(self, mcp: FastMCP):
        """
        Add search tools to the MCP instance.

        Currently supports:
        - search_datasets_and_collections: Searches for datasets and collections in the dataverse

        Args:
            mcp: The FastMCP instance to add tools to.
        """

        self._tool(mcp, 
            search,
            name="Search_Dataverse",
            description="Search for datasets and collections in the dataverse. This tool returns the search results in a TOON format.",
        )

        self._tool(mcp,
            DataCite.search,
            name="Search_DataCite",
            description="Search for datasets in DataCite. This tool returns the search results in a TOON format.",
        )

        configured = ", ".join(s.name for s in self.config.vocabulary_sources)
        self._tool(mcp,
            search_vocabulary,
            name="Search_Vocabulary",
            description=f"""
            Look up a term in controlled vocabularies to link a keyword or topic classification.
            Configured sources: {configured}. By default searches all of them; pass `source` with a
            source name to narrow. Returns candidate terms with their vocabulary short name and term
            URI — use these to add a PROPERLY LINKED keyword via a metadata-edit tool (keywordValue +
            keywordVocabulary + keywordTermURI), rather than a bare free-text keyword. Present options
            and let the user pick. Returns results in a TOON format.
            """.strip(),
        )

    def _all_metadatablocks(self) -> List[str]:
        """
        Get all metadata blocks from the dataverse.
        """
        if isinstance(self.dataverse, Dataverse):
            return list(self.dataverse.metadatablocks.keys())
        else:
            return asyncio.run(self._async_metadatablocks())

    async def _async_metadatablocks(self) -> List[str]:
        """
        Get all metadata blocks from the dataverse asynchronously.
        """
        assert isinstance(self.dataverse, dict), (
            "Dataverse must be a dictionary for multi-dataverse setup."
        )

        tasks = [
            asyncify(dv.native_api.get_metadatablocks)(full=True)
            for dv in self.dataverse.values()
        ]
        results = await asyncio.gather(*tasks)
        blocks = set()
        for result in results:
            blocks.update(result.keys())
        return list(blocks)
