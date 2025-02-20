import asyncio
import json
import logging
import os
import shutil
from contextlib import AsyncExitStack
from typing import Any, Dict, Optional

from codegate.types.mcp import Tool
from mcp import ClientSession, StdioServerParameters, stdio_client
from pydantic import BaseModel, Field
import structlog

logger = structlog.get_logger("codegate")

class McpServerConfig(BaseModel):
    command: str
    args: list[str] = Field(default_factory=list)
    env: Dict[str, str] = Field(default_factory=dict)
    disabled: bool = False
    auto_approve: list[str] = Field(default_factory=list, alias="autoApprove")

class McpConfig(BaseModel):
    mcp_servers: Dict[str, McpServerConfig] = Field(..., alias="mcpServers")

DEMO_CONFIG = McpConfig(
    mcpServers={
            "everything": McpServerConfig(
                command="npx",
                args=[
                    "-y",
                    "@modelcontextprotocol/server-everything"
                ],
            ),
        },
)

class Server:
    """Manages MCP server connections and tool execution."""

    def __init__(self, name: str, config: McpServerConfig) -> None:
        logger.debug(f"Creating server {name} with config: {config}")
        self.name: str = name
        self.config: McpServerConfig = config
        self.session: Optional[ClientSession] = None
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self.exit_stack: AsyncExitStack = AsyncExitStack()

    async def initialize(self) -> None:
        command = (
            shutil.which("npx")
            if self.config.command == "npx"
            else self.config.command
        )
        if command is None:
            raise ValueError("The command must be a valid string and cannot be None.")

        logger.info(f"Starting server {self.name} with command: {command} and args: {self.config.args}")
        server_params = StdioServerParameters(
            command=command,
            args=self.config.args,
            env={**os.environ, **self.config.env} if self.config.env else None,
        )
        try:
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()
            self.session = session
        except Exception as e:
            logging.error(f"Error initializing server {self.name}: {e}")
            await self.cleanup()
            raise

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        retries: int = 2,
        delay: float = 1.0,
    ) -> Any:
        """Execute a tool with retry mechanism."""
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        attempt = 0
        while attempt < retries:
            try:
                logging.info(f"Executing {tool_name}...")
                result = await self.session.call_tool(tool_name, arguments)

                return result

            except Exception as e:
                attempt += 1
                logging.warning(
                    f"Error executing tool: {e}. Attempt {attempt} of {retries}."
                )
                if attempt < retries:
                    logging.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logging.error("Max retries reached. Failing.")
                    raise

    async def list_tools(self) -> list[Any]:
        """List available tools from the server.

        Returns:
            A list of available tools.

        Raises:
            RuntimeError: If the server is not initialized.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        tools_response = await self.session.list_tools()
        tools = []

        for item in tools_response:
            if isinstance(item, tuple) and item[0] == "tools":
                for tool in item[1]:
                    tools.append(Tool(
                        name=tool.name,
                        description=tool.description,
                        input_schema=tool.inputSchema
                    ))

        return tools

    async def cleanup(self) -> None:
        """Clean up server resources."""
        async with self._cleanup_lock:
            if self.session:
                try:
                    await self.session.close()
                except Exception as e:
                    logging.error(f"Error closing session for server {self.name}: {e}")
                self.session = None
            
            try:
                await self.exit_stack.aclose()
            except Exception as e:
                logging.error(f"Error closing exit stack for server {self.name}: {e}")

class Manager:
    """Manages MCP server connections."""

    def __init__(self, config: McpConfig | None = None):
        if not config:
            config = DEMO_CONFIG
        self.config = config
        logger.debug(f"Creating MCP manager with config: {config}")
        self.servers: Dict[str, Server] = {}
        self.tool_server_map: Dict[str, str] = {}  # Mapping from tool names to server names

    async def initialize(self) -> None:
        """Initialize the manager and start all enabled servers."""
        await self.start_servers()

    async def start_servers(self) -> None:
        """Start all enabled servers."""
        for name, server_config in self.config.mcp_servers.items():
            if server_config.disabled:
                logger.debug(f"Skipping disabled server {name}")
                continue

            server = Server(name, server_config)
            logger.info(f"Starting MCP server {name}")
            try:
                await server.initialize()
                self.servers[name] = server
                logger.info(f"MCP server {name} started successfully")
            except Exception as e:
                logging.error(f"Failed to start MCP server {name}: {e}")
                await server.cleanup()

    async def list_tools(self) -> Dict[str, list[Tool]]:
        """List available tools from all servers."""
        self.tool_server_map.clear()  # Clear previous mappings

        tools = {}
        for srv_name, server in self.servers.items():
            try:
                tools[srv_name] = await server.list_tools()
                for tool in tools[srv_name]:
                    self.tool_server_map[tool.name] = srv_name
            except Exception as e:
                logging.error(f"Error listing tools for server {srv_name}: {e}")
        logger.debug(f"Listed tools: {tools}")
        logger.debug(f"Tool to server mapping: {self.tool_server_map}")
        return tools

    async def cleanup(self) -> None:
        """Clean up all server connections."""
        cleanup_tasks = []
        for name, server in self.servers.items():
            cleanup_tasks.append(server.cleanup())
        
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        self.servers.clear()

    def get_server(self, tool: str) -> Server|None:
        """Get the server associated with a tool."""
        server_name = self.tool_server_map.get(tool)
        if not server_name:
            return None
        return self.servers[server_name]

def load_config_from_file(config_path: str) -> McpConfig:
    """Load and validate MCP server configuration from a file."""
    with open(config_path) as f:
        raw_config = json.load(f)
    return McpConfig(**raw_config)
