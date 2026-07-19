from typing import List, Optional

from cachetools import TTLCache
from fastmcp.server.middleware import Middleware, MiddlewareContext

from ..dataverse import Dataverse
from .vocabulary import VocabularySource

TTL_TIME = 30 * 60  # 30 minutes
MAX_SIZE = 40


class DataverseMiddleware(Middleware):
    dataverse: Dataverse

    def __init__(
        self,
        dataverse: Dataverse,
        vocabulary_sources: Optional[List[VocabularySource]] = None,
    ):
        self.dataverse = dataverse
        self.vocabulary_sources = vocabulary_sources or []

    async def on_request(self, context: MiddlewareContext, call_next):
        """
        Handles a request with Dataverse instance lifecycle management.

        Sets the Dataverse instance in the FastMCP context state, calls the next
        request handler, and then ensures cleanup by clearing the state.

        Args:
            context (MiddlewareContext): The current middleware/request context.
            call_next (callable): The next handler to call.

        Returns:
            The result of `call_next(context)`.
        """
        assert context.fastmcp_context is not None, "FastMCP context is not set"
        # v3 validates state as serializable by default; these hold live objects
        # (a Dataverse client and a TTLCache), so store them request-scoped.
        await context.fastmcp_context.set_state(
            "dataverse", self.dataverse, serializable=False
        )
        await context.fastmcp_context.set_state(
            "dataverses",
            TTLCache(maxsize=MAX_SIZE, ttl=TTL_TIME),
            serializable=False,
        )
        await context.fastmcp_context.set_state(
            "vocabulary_sources", self.vocabulary_sources, serializable=False
        )

        try:
            return await call_next(context)
        finally:
            await context.fastmcp_context.set_state("dataverse", None)
