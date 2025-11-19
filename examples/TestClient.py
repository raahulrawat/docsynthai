# mcp_stdio_client.py
import asyncio
import base64
import json
import logging
from typing import Any, Dict, List, Optional

# Keep your original imports (adjust package names if your MCP lib differs)
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger("DocumentClassifierMCPClient")
logger.setLevel(logging.INFO)


class DocumentClassifierMCPClient:
    """
    Async MCP stdio client for the Document Classifier MCP server.

    Usage (minimal):
        client = DocumentClassifierMCPClient()
        await client.connect("server.py")
        await client.setup_classifier(api_key="...")
        await client.disconnect()
    """

    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.available_tools: List[Dict[str, Any]] = []
        self._stdio_context = None
        self._read = None
        self._write = None

    # --------------------------
    # Connection / lifecycle
    # --------------------------
    async def connect(
        self,
        server_script_path: str,
        python_cmd: str = "python",
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Launch the server subprocess (stdio transport) and perform MCP initialization.

        Args:
            server_script_path: Path to server.py (or module) to run
            python_cmd: The Python interpreter to run (e.g. "python3" in some systems)
            args: Additional args passed after server_script_path
            env: Optional environment overrides passed to subprocess
        """
        if args is None:
            args = [server_script_path]

        server_params = StdioServerParameters(command=python_cmd, args=args, env=env)

        # Start stdio client (manages server subprocess)
        self._stdio_context = stdio_client(server_params)
        try:
            # Enter the stdio client context and get read/write transports
            self._read, self._write = await self._stdio_context.__aenter__()

            # Create MCP session and initialize handshake
            self.session = ClientSession(self._read, self._write)
            await self.session.__aenter__()

            # Perform MCP initialize/handshake
            await self.session.initialize()

            # Discover tools
            await self.refresh_tools()
            logger.info("Connected to MCP server; %d tools available.", len(self.available_tools))
        except Exception:
            # On any failure, ensure cleanup of partial state
            logger.exception("Failed to connect to MCP server; cleaning up resources.")
            await self._cleanup_on_error()
            raise

    async def _cleanup_on_error(self):
        # best effort cleanup
        try:
            if self.session:
                await self.session.__aexit__(None, None, None)
        except Exception:
            logger.debug("Session __aexit__ failed during cleanup", exc_info=True)
        try:
            if self._stdio_context:
                await self._stdio_context.__aexit__(None, None, None)
        except Exception:
            logger.debug("StdIO context __aexit__ failed during cleanup", exc_info=True)
        finally:
            self.session = None
            self._stdio_context = None
            self._read = self._write = None

    async def disconnect(self) -> None:
        """Gracefully disconnect and terminate the server subprocess."""
        if self.session:
            try:
                await self.session.__aexit__(None, None, None)
            except Exception:
                logger.debug("Error while closing MCP session", exc_info=True)
            self.session = None

        if self._stdio_context:
            try:
                await self._stdio_context.__aexit__(None, None, None)
            except Exception:
                logger.debug("Error while terminating stdio client", exc_info=True)
            self._stdio_context = None

        self._read = self._write = None
        logger.info("Disconnected from MCP server")

    # --------------------------
    # Tool discovery / helpers
    # --------------------------
    async def refresh_tools(self) -> None:
        """Query the server for available tools and cache them locally."""
        if not self.session:
            raise RuntimeError("Not connected to MCP server")

        tools_response = await self.session.list_tools()
        # defensive: tools_response may be different shapes depending on MCP implementation
        tools = getattr(tools_response, "tools", None) or tools_response
        self.available_tools = []

        try:
            for t in tools:
                self.available_tools.append({
                    "name": getattr(t, "name", t.get("name") if isinstance(t, dict) else str(t)),
                    "description": getattr(t, "description", t.get("description") if isinstance(t, dict) else ""),
                    "input_schema": getattr(t, "inputSchema", t.get("inputSchema") if isinstance(t, dict) else None)
                })
        except Exception:
            logger.exception("Failed to parse tools_response; storing raw response")
            # fallback: store raw response so user can debug
            self.available_tools = [{"raw": tools_response}]

    def list_tools(self) -> List[Dict[str, Any]]:
        return self.available_tools

    # --------------------------
    # Core tool invocation
    # --------------------------
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call a named tool on the MCP server and return parsed output.

        If the tool returns structured content, attempt to parse JSON from the
        first text content item. Otherwise returns raw content object.
        """
        if not self.session:
            raise RuntimeError("Not connected to MCP server. Call connect() first.")

        try:
            result = await self.session.call_tool(tool_name, arguments)
        except Exception as e:
            logger.exception("Tool call failed for %s", tool_name)
            raise

        # The result object shape depends on MCP implementation.
        # Common case: result.content is a list of ContentItems with .text
        try:
            content_items = getattr(result, "content", None)
            if content_items and len(content_items) > 0:
                text = getattr(content_items[0], "text", None) or (content_items[0].get("text") if isinstance(content_items[0], dict) else None)
                if text:
                    # Try JSON parse, else return raw text
                    try:
                        return json.loads(text)
                    except json.JSONDecodeError:
                        return text
            # If no content or non-text content, return the raw result object
            return result
        except Exception:
            logger.exception("Unexpected result shape from tool call; returning raw result")
            return result

    # --------------------------
    # Convenience wrappers for your tools
    # --------------------------
    async def setup_classifier(self, api_key: str, model: str = "gemini-1.5-flash") -> Dict[str, Any]:
        return await self.call_tool("setup_classifier", {"api_key": api_key, "model": model})

    async def create_rule(self, document_class: str, conditions: List[str], description: str = "") -> Dict[str, Any]:
        return await self.call_tool("create_rule", {"document_class": document_class, "conditions": conditions, "description": description})

    async def get_all_rules(self) -> Dict[str, Any]:
        return await self.call_tool("get_all_rules", {})

    async def delete_rule(self, document_class: str) -> Dict[str, Any]:
        return await self.call_tool("delete_rule", {"document_class": document_class})

    async def classify_document(self, document_base64: str, media_type: str = "image/jpeg", use_rules: bool = True) -> Dict[str, Any]:
        return await self.call_tool("classify_document", {"document_base64": document_base64, "media_type": media_type, "use_rules": use_rules})

    async def classify_from_file(self, file_path: str, use_rules: bool = True) -> Dict[str, Any]:
        with open(file_path, "rb") as f:
            file_bytes = f.read()
        doc_base64 = base64.b64encode(file_bytes).decode("utf-8")
        # Basic detection of media type
        mt = "image/jpeg"
        if file_path.lower().endswith(".png"):
            mt = "image/png"
        elif file_path.lower().endswith((".jpg", ".jpeg")):
            mt = "image/jpeg"
        return await self.classify_document(doc_base64, media_type=mt, use_rules=use_rules)


# --------------------------
# Demo helpers
# --------------------------
async def _prompt_async(prompt: str) -> str:
    """Run blocking input() in a thread so event loop is not blocked."""
    return await asyncio.to_thread(input, prompt)


async def demonstrate_mcp_workflow():
    client = DocumentClassifierMCPClient()
    try:
        await client.connect("server.py")  # or use ["server.py", "--some-arg"]
        tools = client.list_tools()
        print(f"Found {len(tools)} tools")
        for t in tools:
            print(" -", t.get("name"))

        # ask user for API key without blocking event loop
        api_key = await _prompt_async("Enter your Google API key (or press Enter to skip): ")
        if api_key:
            setup = await client.setup_classifier(api_key=api_key)
            print("setup:", setup)

        # ask for file path
        file_path = await _prompt_async("Enter path to document image to classify (or press Enter to skip): ")
        if file_path:
            result = await client.classify_from_file(file_path, use_rules=True)
            print("result:", result)
    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(demonstrate_mcp_workflow())
