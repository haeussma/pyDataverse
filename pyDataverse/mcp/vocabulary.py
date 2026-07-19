"""Controlled-vocabulary term lookup for enriching keyword / topicClassification fields.

The standard Dataverse citation block lets keywords (and topic classifications) be
linked to a controlled vocabulary: a term value, a vocabulary short name, and a term
URI (keywordValue / keywordVocabulary / keywordTermURI / keywordVocabularyURI). This
resolves a plain term to candidates with URIs by querying one or more term services.

Two service types are supported generically:
  - "wikidata": the Wikidata entity search API
  - "ols":      any OLS4-compatible terminology service (identified by its base_url)

The set of services is configured by the deployment (see MCPConfiguration.vocabulary_sources),
so pyDataverse ships a neutral default (Wikidata + the EBI Ontology Lookup Service) and any
app can add its own OLS-compatible endpoints without changing this code.
"""

from typing import Annotated, List, Literal, Optional

import httpx
from fastmcp import Context
from fastmcp.dependencies import CurrentContext
from pydantic import BaseModel
from toon_format import encode


class VocabularySource(BaseModel):
    """A controlled-vocabulary service to query for terms."""

    name: str  # short label shown on results, e.g. "Wikidata", "TIB"
    type: Literal["wikidata", "ols"]
    base_url: Optional[str] = None  # required for type "ols"


DEFAULT_SOURCES: List[VocabularySource] = [
    VocabularySource(name="Wikidata", type="wikidata"),
    VocabularySource(name="EBI OLS", type="ols", base_url="https://www.ebi.ac.uk/ols4"),
]


async def _search_wikidata(
    client: httpx.AsyncClient, src: VocabularySource, query: str, limit: int
) -> List[dict]:
    r = await client.get(
        "https://www.wikidata.org/w/api.php",
        params={
            "action": "wbsearchentities",
            "search": query,
            "language": "en",
            "format": "json",
            "limit": limit,
        },
    )
    r.raise_for_status()
    out = []
    for x in r.json().get("search", []):
        if not x.get("concepturi"):
            continue
        out.append(
            {
                "term": x.get("label"),
                "vocabulary": src.name,
                "term_uri": x["concepturi"],
                "vocabulary_uri": "https://www.wikidata.org",
                "description": x.get("description", ""),
                "source": src.name,
            }
        )
    return out


async def _search_ols(
    client: httpx.AsyncClient, src: VocabularySource, query: str, limit: int
) -> List[dict]:
    r = await client.get(
        f"{src.base_url.rstrip('/')}/api/search", params={"q": query, "rows": limit}
    )
    r.raise_for_status()
    out = []
    for x in r.json().get("response", {}).get("docs", []):
        if not x.get("iri"):
            continue
        desc = x.get("description")
        if isinstance(desc, list):
            desc = desc[0] if desc else ""
        out.append(
            {
                "term": x.get("label"),
                "vocabulary": x.get("ontology_name") or src.name,
                "term_uri": x["iri"],
                "vocabulary_uri": None,
                "description": desc or "",
                "source": src.name,
            }
        )
    return out


async def _run_source(
    client: httpx.AsyncClient, src: VocabularySource, query: str, limit: int
) -> List[dict]:
    if src.type == "wikidata":
        return await _search_wikidata(client, src, query, limit)
    if src.type == "ols" and src.base_url:
        return await _search_ols(client, src, query, limit)
    return []


async def _configured_sources(ctx: Context) -> List[VocabularySource]:
    try:
        sources = await ctx.get_state("vocabulary_sources")
    except Exception:
        sources = None
    return sources or DEFAULT_SOURCES


async def search_vocabulary(
    query: Annotated[
        str,
        "The term to look up, e.g. 'research data management' or 'molecular dynamics simulation'.",
    ],
    source: Annotated[
        str,
        "Which configured source to query: 'all' (default) searches every configured vocabulary; or the name of one source (as shown in the 'source' field of previous results) to narrow.",
    ] = "all",
    limit: Annotated[int, "Max results per source."] = 5,
    ctx: Context = CurrentContext(),
):
    """
    Search controlled vocabularies for a term and its URI.

    Fans out to all vocabularies configured for this server (or one named source), and
    returns candidate terms with the vocabulary short name and term URI needed to link a
    Dataverse keyword or topic classification. Present the options and let the user pick,
    then set the field (e.g. via a metadata-edit tool) as a compound value, e.g.:
        {"keyword": [{"keywordValue": "Research Data Management",
                       "keywordVocabulary": "Wikidata",
                       "keywordTermURI": "http://www.wikidata.org/entity/Q30089794"}]}

    Returns:
        TOON-encoded list of {term, vocabulary, term_uri, vocabulary_uri, description, source}.
    """
    sources = await _configured_sources(ctx)
    if source != "all":
        sources = [s for s in sources if s.name.lower() == source.lower()]
        if not sources:
            raise ValueError(f"Unknown vocabulary source '{source}'.")

    # Wikimedia APIs require a descriptive User-Agent or they return 403.
    headers = {"User-Agent": "pyDataverse-MCP (controlled-vocabulary term lookup)"}
    async with httpx.AsyncClient(
        timeout=20, follow_redirects=True, headers=headers
    ) as client:
        results: List[dict] = []
        for src in sources:
            results += await _run_source(client, src, query, limit)
    return encode(results)
