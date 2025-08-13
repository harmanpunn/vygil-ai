"""
MCP Client for Vygil Activity Tracking

Implements thread-safe MCP client with JSON-RPC 2.0 over stdio transport.
Provides tools for screen capture, OCR, and activity logging.
"""

import asyncio
import json
import subprocess
import threading
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class MCPClient:
    """True MCP client implementing JSON-RPC 2.0 over stdio"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.mcp_config = config.get('mcp_server', {})
        self.command = self.mcp_config.get('command', 'node')
        self.args = self.mcp_config.get('args', ['../mcp-server/dist/vygil-mcp-server.js'])
        self.timeout = self.mcp_config.get('timeout', 30)
        self.is_initialized = False
        self.request_id = 0
        self.process = None
        self._id_lock = threading.Lock()
        self._process_lock = asyncio.Lock()  # Add process-level async lock
        
    def _next_request_id(self) -> int:
        """Generate unique request ID - thread safe"""
        with self._id_lock:
            self.request_id += 1
            return self.request_id
        
    async def _send_mcp_request(self, method: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Send JSON-RPC 2.0 request to MCP server via stdio - thread-safe"""
        
        # Acquire process lock to prevent concurrent access
        async with self._process_lock:
            request_id = self._next_request_id()
            
            # Construct JSON-RPC 2.0 request
            rpc_request = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method
            }
            
            if params is not None:
                rpc_request["params"] = params
                
            try:
                # Start MCP server process if not already running
                if self.process is None or self.process.returncode is not None:
                    logger.debug(f"Starting MCP server: {self.command} {' '.join(self.args)}")
                    self.process = await asyncio.create_subprocess_exec(
                        self.command, *self.args,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    
                    # Give the server a moment to start up
                    await asyncio.sleep(0.5)
                    
                    # Read any startup messages from stderr
                    try:
                        startup_msg = await asyncio.wait_for(
                            self.process.stderr.readline(),
                            timeout=2.0
                        )
                        if startup_msg:
                            logger.debug(f"MCP server startup: {startup_msg.decode().strip()}")
                    except asyncio.TimeoutError:
                        pass  # No startup message, continue
                
                # Send request
                request_json = json.dumps(rpc_request) + '\n'
                logger.debug(f"Sending MCP request: {method} -> {request_json.strip()}")
                
                self.process.stdin.write(request_json.encode())
                await self.process.stdin.drain()
                
                # Read response with detailed logging
                logger.debug(f"Waiting for MCP response (timeout: {self.timeout}s)")
                response_line = await asyncio.wait_for(
                    self.process.stdout.readline(),
                    timeout=self.timeout
                )
                
                logger.debug(f"MCP response received: {response_line.decode().strip() if response_line else 'No response'}")
                
                if not response_line:
                    raise Exception("No response from MCP server")
                
                result = json.loads(response_line.decode().strip())
                
                # Validate JSON-RPC response
                if result.get("jsonrpc") != "2.0":
                    logger.error(f"Invalid JSON-RPC version: {result}")
                    return {"success": False, "error": "Invalid JSON-RPC version"}
                
                # Handle error responses first
                if "error" in result:
                    error = result["error"]
                    logger.error(f"MCP server error: {error}")
                    return {"success": False, "error": error.get("message", "Unknown error")}
                
                # Validate request ID - log mismatch but don't crash
                response_id = result.get("id")
                if response_id != request_id:
                    logger.error(f"MCP response ID mismatch: expected {request_id}, got {response_id}. This may indicate a concurrency issue.")
                    return {"success": False, "error": f"Response ID mismatch - expected {request_id}, got {response_id}"}
                
                # Return successful result
                return {"success": True, "data": result.get("result", {})}
                        
            except asyncio.TimeoutError:
                logger.error(f"MCP request timeout after {self.timeout}s")
                return {"success": False, "error": "Request timeout"}
            except Exception as e:
                logger.error(f"MCP request failed: {e}")
                return {"success": False, "error": str(e)}
    
    async def initialize(self) -> bool:
        """Initialize MCP connection"""
        if self.is_initialized:
            return True
            
        logger.info("Initializing MCP connection...")
        
        response = await self._send_mcp_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "Vygil Activity Agent",
                "version": "1.0.0"
            }
        })
        
        if response.get("success"):
            self.is_initialized = True
            logger.info("✅ MCP connection initialized successfully")
            return True
        else:
            logger.error(f"❌ MCP initialization failed: {response.get('error')}")
            return False
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools from MCP server"""
        if not self.is_initialized:
            await self.initialize()
            
        response = await self._send_mcp_request("tools/list")
        
        if response.get("success"):
            tools = response.get("data", {}).get("tools", [])
            logger.info(f"📋 Found {len(tools)} available tools")
            return tools
        else:
            logger.error(f"Failed to list tools: {response.get('error')}")
            return []
    
    async def call_tool(self, tool_name: str, arguments: Dict = None) -> Dict[str, Any]:
        """Call MCP server tool using JSON-RPC 2.0"""
        if arguments is None:
            arguments = {}
            
        # Ensure MCP connection is initialized
        if not self.is_initialized:
            init_success = await self.initialize()
            if not init_success:
                return {"success": False, "error": "Failed to initialize MCP connection"}
        
        logger.debug(f"Calling MCP tool: {tool_name} with args: {arguments}")
        
        # Send tools/call request
        response = await self._send_mcp_request("tools/call", {
            "name": tool_name,
            "arguments": arguments
        })
        
        if response.get("success"):
            # Extract result from MCP response format
            data = response.get("data", {})
            content = data.get("content", [])
            
            if content and len(content) > 0:
                # Parse the text content (which contains JSON)
                try:
                    result_text = content[0].get("text", "{}")
                    parsed_result = json.loads(result_text)
                    
                    # Return in expected format
                    return {
                        "success": parsed_result.get("success", True),
                        **parsed_result
                    }
                except json.JSONDecodeError:
                    # Fallback if parsing fails
                    return {
                        "success": True,
                        "result": result_text
                    }
            else:
                return {"success": True, "data": data}
        else:
            error_msg = response.get("error", "Unknown error")
            logger.error(f"MCP tool call failed: {error_msg}")
            return {"success": False, "error": error_msg}

    async def cleanup(self):
        """Clean up MCP client resources"""
        if self.process:
            try:
                self.process.terminate()
                await self.process.wait()
                logger.debug("MCP server process terminated")
            except Exception as e:
                logger.error(f"Error terminating MCP process: {e}")
            finally:
                self.process = None
                self.is_initialized = False