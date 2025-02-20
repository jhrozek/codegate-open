from itertools import chain
from typing import Any
from codegate.pipeline.mcp.manager import Manager as McpManager
from codegate.pipeline.base import PipelineContext, PipelineResult, PipelineStep

import structlog

logger = structlog.get_logger("codegate")

class AddMcpTools(PipelineStep):
    @property
    def name(self):
        return "add_mcp_tools"

    async def process(
        self, request: Any, context: PipelineContext
    ) -> PipelineResult:
        logger.debug(f"Processing request: {request}")
        if not context.mcp_manager:
            return PipelineResult(request=request, context=context)

        tools = await context.mcp_manager.list_tools()
        flat_list = list(chain.from_iterable(tools.values()))
        request.add_mcp_tools(flat_list)
        logger.debug(f"Returning request: {request}")
        return PipelineResult(request=request, context=context)