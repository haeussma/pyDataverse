from typing import (
    Annotated,
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Union,
    cast,
)

import pandas as pd
from fastmcp import Context
from fastmcp.dependencies import CurrentContext
from toon_format import encode
from typing_extensions import TypedDict

from ..dataverse import Author, Contact, Subject
from .utils import ensure_dataverse


class FileMeta(TypedDict):
    """
    Metadata information for a file in a dataset.

    Attributes:
        path: The file path within the dataset
        description: Optional description of the file
        mime_type: The MIME type of the file
        restricted: Whether the file has access restrictions
    """

    path: str
    description: Optional[str]
    mime_type: Optional[str]
    restricted: Optional[bool]


async def get_dataset(
    identifier: Annotated[
        Union[str, int],
        "The identifier of the dataset, which can be a persistent identifier or a dataverse identifier (int).",
    ],
    full: Annotated[
        bool,
        "Whether to return the full dataset metadata and files. This function is token-heavy, so you should only use it if you need to access the full dataset metadata and files.",
    ],
    base_url: Annotated[
        Optional[str],
        "The base URL of the dataverse to use. If not specified, the function will use the dataverse this MCP server is connected to by default.",
    ] = None,
    ctx: Context = CurrentContext(),
):
    """
    Retrieve dataset information from Dataverse.

    This function fetches dataset metadata from a Dataverse instance. It can return
    either a summary view with basic information or a full view with complete metadata
    including all metadata blocks.

    Args:
        identifier: The dataset identifier (persistent ID like DOI or numeric ID)
        full: If True, returns complete metadata; if False, returns summary info
        metadatablocks: Specific metadata blocks to include (only when full=True)
        ctx: The MCP context containing the Dataverse connection

    Returns:
        Encoded dataset metadata as a string

    Raises:
        RuntimeError: If no Dataverse connection is available in context
    """
    dataverse = await ensure_dataverse(ctx, base_url=base_url)
    dataset = dataverse.fetch_dataset(identifier=identifier)

    if full:
        dataset.metadata_blocks = {
            name: metadata_block
            for name, metadata_block in dataset.metadata_blocks.items()
            if not metadata_block.is_empty
        }

        serialized = dataset.model_dump()
    else:
        serialized = dataset.model_dump(
            include={
                "identifier",
                "persistent_identifier",
                "version",
                "title",
                "description",
                "authors",
                "subjects",
                "url",
            }
        )

        # We supply the available metadata blocks to the client so they can re-fetch
        # the dataset with the metadata blocks they want to see
        serialized["available_metadatablocks"] = ", ".join(
            [
                name
                for name, metadata_block in dataset.metadata_blocks.items()
                if not metadata_block.is_empty
            ]
        )

    return encode(serialized)


async def list_files(
    identifier: Annotated[
        Union[str, int],
        "The identifier of the dataset, which can be a persistent identifier or a dataverse identifier (int).",
    ],
    only_tabular: Annotated[
        bool,
        "Whether to return only tabular files. If True, the function will return only tabular files. If False, the function will return all files.",
    ],
    filter_mime_types: Annotated[
        Optional[Sequence[str]],
        "The MIME types to filter the files by. If not specified, the function will return all files.",
    ],
    base_url: Annotated[
        Optional[str],
        "The base URL of the dataverse to use. If not specified, the function will use the dataverse this MCP server is connected to by default.",
    ] = None,
    ctx: Context = CurrentContext(),
):
    """
    List all files in a dataset with optional MIME type filtering.

    This function retrieves information about all files in a dataset, including
    their paths, descriptions, MIME types, and access restrictions. Files can
    be filtered by MIME type if desired.

    Args:
        identifier: The dataset identifier (persistent ID like DOI or numeric ID)
        filter_mime_types: Optional list of MIME types to filter by
        ctx: The MCP context containing the Dataverse connection

    Returns:
        Encoded list of file metadata as a string

    Raises:
        RuntimeError: If no Dataverse connection is available in context
    """
    dataverse = await ensure_dataverse(ctx, base_url=base_url)
    dataset = dataverse.fetch_dataset(identifier=identifier)

    df = dataset.files.df
    if only_tabular:
        df = cast(pd.DataFrame, df[df["is_tabular"]])

    if filter_mime_types:
        df = cast(
            pd.DataFrame,
            df[df["mime_type"].isin(filter_mime_types)],
        )

    return encode(df.to_dict(orient="records"))


async def create_dataset(
    title: Annotated[str, "The title of the dataset."],
    description: Annotated[str, "The description / abstract of the dataset."],
    authors: Annotated[
        List[Author],
        "Authors of the dataset. Each is an object with a required `name` and optional `affiliation`, `identifier_scheme` (e.g. ORCID), and `identifier`.",
    ],
    contacts: Annotated[
        List[Contact],
        "Contact persons for the dataset. Each is an object with a required `name` and `email`, and an optional `affiliation`.",
    ],
    subjects: Annotated[
        List[Subject],
        "Subjects categorising the dataset. Must come from the Dataverse controlled vocabulary, e.g. 'Chemistry', 'Physics', 'Medicine, Health and Life Sciences', 'Other'.",
    ],
    collection: Annotated[
        str,
        "Alias of the collection to create the dataset in (e.g. 'root' or a sub-collection alias).",
    ],
    ctx: Context = CurrentContext(),
):
    """
    Create a new draft dataset and add it to a collection.

    This creates a DRAFT dataset in the given collection with the core citation
    metadata (title, description, authors, contacts, subjects). It does NOT publish
    the dataset — publishing remains a separate, explicit step. Writing requires the
    MCP server to be connected with an authenticated (API-token) Dataverse instance;
    the `base_url` override is intentionally not available here.

    Args:
        title: The dataset title.
        description: The dataset description / abstract.
        authors: Dataset authors.
        contacts: Dataset contact persons.
        subjects: Controlled-vocabulary subjects.
        collection: Alias of the target collection.
        ctx: The MCP context containing the Dataverse connection.

    Returns:
        Encoded object with the new dataset's `persistent_id` and `url`.
    """
    dataverse = await ensure_dataverse(ctx)
    dataset = dataverse.create_dataset(
        title=title,
        description=description,
        authors=authors,
        contacts=contacts,
        subjects=subjects,
        upload_to_collection=True,
        collection=collection,
    )

    return encode(
        {
            "persistent_id": dataset.persistent_identifier,
            "url": dataset.url,
        }
    )


async def edit_dataset(
    identifier: Annotated[
        str,
        "Persistent id of the DRAFT dataset to edit, e.g. 'doi:10.5072/FK2/ABC123'.",
    ],
    title: Annotated[Optional[str], "New title, if changing it."] = None,
    description: Annotated[
        Optional[str], "New description / abstract, if changing it."
    ] = None,
    authors: Annotated[
        Optional[List[Author]],
        "The COMPLETE author list to set — it REPLACES the existing authors, so include current authors you want to keep plus any new ones. Each: {name, affiliation?, identifier_scheme?, identifier?}.",
    ] = None,
    contacts: Annotated[
        Optional[List[Contact]],
        "The COMPLETE contact list to set (replaces existing). Each: {name?, email, affiliation?}.",
    ] = None,
    subjects: Annotated[
        Optional[List[Subject]],
        "The COMPLETE controlled-vocabulary subject list to set (replaces existing).",
    ] = None,
    keywords: Annotated[
        Optional[List[str]],
        "The COMPLETE keyword list to set (replaces existing). Free-text terms, e.g. ['data management', 'RDM'].",
    ] = None,
    ctx: Context = CurrentContext(),
):
    """
    Edit citation metadata on an existing DRAFT dataset (upsert).

    Only the fields you provide are changed; each is replaced in place. Multi-value
    fields (authors, contacts, subjects) are replaced wholesale — to ADD one, first
    read the current values (Get_Dataset_Metadata) and pass the full merged list.
    Does NOT publish. Requires an authenticated (API-token) connection.

    Returns:
        Encoded object with the dataset's `persistent_id`.
    """
    dataverse = await ensure_dataverse(ctx)
    resp = dataverse.edit_dataset_citation(
        identifier,
        title=title,
        description=description,
        authors=authors,
        contacts=contacts,
        subjects=subjects,
        keywords=keywords,
    )
    return encode({"persistent_id": resp.dataset_persistent_id})


async def edit_dataset_fields(
    identifier: Annotated[
        str,
        "Persistent id of the DRAFT dataset to edit, e.g. 'doi:10.5072/FK2/ABC123'.",
    ],
    fields: Annotated[
        Dict[str, Any],
        (
            "Mapping of field name -> value for the given metadata block. The value shape "
            "follows the field's schema: a scalar for a single primitive/controlled-vocabulary "
            "field (e.g. {'title': 'My title'}); a list of scalars for a multiple one "
            "(e.g. {'keyword': ['RDM', 'mcp'], 'subject': ['Chemistry']}); {child: value} for a "
            "single compound; and [{child: value}, ...] for a multiple compound "
            "(e.g. {'author': [{'authorName': 'Doe, Jane', 'authorAffiliation': 'Uni X'}]}). "
            "Each field is REPLACED wholesale — pass the COMPLETE value (read current values "
            "with Get_Dataset_Metadata first if adding to a list). If unsure of field/child "
            "names, they are the schema names shown by Get_Dataset_Metadata."
        ),
    ],
    block: Annotated[
        str,
        "Metadata block name. 'citation' covers the common fields; others e.g. 'geospatial', 'EngMeta'.",
    ] = "citation",
    ctx: Context = CurrentContext(),
):
    """
    Edit ANY metadata field on a DRAFT dataset (generic upsert).

    Builds the native Dataverse call from the live block schema, so any field in any
    block works without special-casing. Use this for fields the friendly
    Edit_Dataset_Metadata tool doesn't expose. Each field is replaced wholesale;
    does NOT publish. Requires an authenticated (API-token) connection.

    Returns:
        Encoded object with the dataset's `persistent_id`.
    """
    dataverse = await ensure_dataverse(ctx)
    resp = dataverse.edit_dataset_fields(identifier, fields, block=block)
    return encode({"persistent_id": resp.dataset_persistent_id})
