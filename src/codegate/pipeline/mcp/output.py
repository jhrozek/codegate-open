from typing import List, Optional, Tuple

import structlog
import json

from codegate.pipeline.base import PipelineContext
from codegate.pipeline.output import OutputPipelineContext, OutputPipelineStep
from codegate.types.common import ModelResponse

logger = structlog.get_logger("codegate")


class FunctionCallCheckStep(OutputPipelineStep):
    """Pipeline step that holds chunks until a complete tool call is detected"""

    def __init__(self):
        """Initialize the FunctionCallCheckStep"""
        super().__init__()
        self.buffer = []

    @property
    def name(self) -> str:
        """
        Returns the name of this pipeline step.
        """
        return "function-call-check"

    @staticmethod
    def _gather_from_chunk(chunk) -> Tuple[str, str, str]:
        """Gather the function and arguments from the buffer"""
        id = ""
        function = ""
        arguments = ""
        for choice in chunk.get_content():
            for id_chunk, function_chunk, arguments_chunk in choice.get_tool_calls():
                id += id_chunk if id_chunk else ""
                function += function_chunk if function_chunk else ""
                arguments += arguments_chunk if arguments_chunk else ""
        return id, function, arguments

    async def tool_call_chunk(
            self,
            end_chunk: ModelResponse,
            input_context: Optional[PipelineContext] = None,
        ):
        call_chunk = self.buffer[0].model_copy(deep=True)

        id = ""
        function = ""
        arguments = ""
        for chunk in self.buffer:
            id_part, fn_part, arg_part = self._gather_from_chunk(chunk)
            id += id_part
            function += fn_part
            arguments += arg_part

        mcp_server = input_context.mcp_manager.get_server(function)
        if mcp_server:
            logger.info("Handling")
            result = await mcp_server.execute_tool(function, json.loads(arguments))
            logger.info(f"Result {result}")
            result_content = ""
            for result in result.content:
                result_content += result.text

            # FIXME - needs better mapping than overwrite all the tool calls
            # we should rather create a chunk delta or a chunk per result?
            for choice in call_chunk.get_content():
                choice.reset_tool_calls()
                choice.set_text(f"\n\u2713Codegate tool call result: {result_content}")

            for choice in end_chunk.get_content():
                choice.set_finish_reason("stop")

        else:
            # This call is not handled by MCP, so we just return the original chunk
            # and the end chunk
            for choice in call_chunk.get_content():
                choice.set_tool_calls(id, function, arguments)
                break

        print(f"Tool call chunk: {call_chunk}")

        return [call_chunk, end_chunk]

    def _create_chunk(self, original_chunk: ModelResponse, content: str) -> ModelResponse:
        """
        Creates a new chunk with the given content, preserving the original chunk's metadata
        """
        copy = original_chunk.model_copy(deep=True)
        copy.set_content(content)
        return copy

    async def call_mcp(
        self,
        chunk: ModelResponse,
        context: OutputPipelineContext):
        for choice in chunk.get_content():
            for id_chunk, function_chunk, arguments_chunk in choice.get_tool_calls():
                print(f"Called tool: {id_chunk}, {function_chunk}, {arguments_chunk}")

    async def process_chunk(
        self,
        chunk: ModelResponse,
        context: OutputPipelineContext,
        input_context: Optional[PipelineContext] = None,
    ) -> List[ModelResponse]:
        """
        Process a single chunk of the stream
        """
        # Check if this chunk has any tool calls or finishes a tool call
        for choice in chunk.get_content():
            if choice.finished_tool_calls():
                logger.debug("Finishes tool call")
                call_result, end_chunk = await self.tool_call_chunk(chunk, input_context)
                self.buffer = []
                return [call_result, end_chunk]
            elif any(tc for tc in choice.get_tool_calls()):
                # Buffer the chunk since it has tool calls
                logger.debug("Found tool call")
                self.buffer.append(chunk)
                return []

        return [chunk]