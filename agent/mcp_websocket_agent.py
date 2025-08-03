# mcp_websocket_agent.py
import asyncio
import base64
import io
import json
import logging
import time
import uuid
from PIL import Image, ImageGrab
import socketio
from typing import Dict, Any, Optional, List

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('MCPAgent')

class MCPWebSocketAgent:
    """
    Agent that connects to MCP server using Socket.IO (WebSockets)
    instead of REST API calls.
    """
    
    def __init__(self, agent_id: Optional[str] = None, server_url: str = "http://localhost:3000"):
        """
        Initialize the MCP WebSocket Agent
        """
        self.agent_id = agent_id or f"agent-{uuid.uuid4()}"
        self.server_url = server_url
        self.sio = socketio.AsyncClient()
        self.connected = False
        self.pending_requests = {}
        
        # Set up event handlers
        self._setup_event_handlers()
    
    def _setup_event_handlers(self):
        """Set up Socket.IO event handlers"""
        self.sio.on("connect", self._on_connect)
        self.sio.on("disconnect", self._on_disconnect)
        self.sio.on("agent-registered", self._on_agent_registered)
        self.sio.on("agent-ocr-result", self._on_ocr_result)
        self.sio.on("agent-screen-result", self._on_screen_result)
        self.sio.on("error", self._on_error)
    
    async def connect(self):
        """Connect to the MCP server"""
        if not self.connected:
            try:
                logger.info(f"Connecting to MCP server at {self.server_url}")
                await self.sio.connect(self.server_url)
                await self.register()
                return True
            except Exception as e:
                logger.error(f"Failed to connect to Agentic MCP server: {e}")
    
    async def disconnect(self):
        """Disconnect from the MCP server"""
        if self.connected:
            await self.sio.disconnect()
    
    async def register(self):
        """Register the agent with the MCP server"""
        logger.info(f"Registering agent {self.agent_id}")
        await self.sio.emit("agent-register", {
            "agentId": self.agent_id,
            "capabilities": ["ocr"]
        })
    
    # Socket.IO event handlers
    async def _on_connect(self):
        """Handle socket.io connect event"""
        self.connected = True
        logger.info("Connected to MCP server")
    
    async def _on_disconnect(self):
        """Handle socket.io disconnect event"""
        self.connected = False
        logger.info("Disconnected from MCP server")
    
    async def _on_agent_registered(self, data):
        """Handle agent-registered event"""
        logger.info(f"Agent registered with server: {data}")
    
    async def _on_ocr_result(self, data):
        """Handle OCR result from server"""
        request_id = data.get('requestId')
        if request_id in self.pending_requests:
            future = self.pending_requests.pop(request_id)
            future.set_result(data)
        logger.info(f"Received OCR result: {len(data['text'])} chars, confidence: {data.get('confidence', 0):.2f}")
    
    async def _on_screen_result(self, data):
        """Handle screen capture result from server"""
        request_id = data.get('requestId')
        if request_id in self.pending_requests:
            future = self.pending_requests.pop(request_id)
            future.set_result(data)
        logger.info(f"Received screen capture: {len(data['imageData'])} bytes")
    
    async def _on_error(self, data):
        """Handle error from server"""
        request_id = data.get('requestId')
        if request_id in self.pending_requests:
            future = self.pending_requests.pop(request_id)
            future.set_exception(Exception(data.get('message', 'Unknown error')))
        logger.error(f"Error from MCP server: {data.get('message')}")
    
    # Public API methods
    async def request_ocr(self, image_data: str = None) -> Dict[str, Any]:
        """
        Request OCR processing for an image
        
        Args:
            image_data: Base64 encoded image data or None to capture screenshot
            
        Returns:
            Dict with OCR results
        """
        await self._ensure_connected()
        
        # Generate a unique request ID
        request_id = str(uuid.uuid4())
        future = asyncio.Future()
        self.pending_requests[request_id] = future
        
        # If no image data provided, take a screenshot
        if not image_data:
            image_data = self._capture_screenshot()
        
        # Send OCR request to server
        await self.sio.emit("agent-request-ocr", {
            "agentId": self.agent_id,
            "requestId": request_id,
            "imageData": image_data
        })
        
        # Wait for result with timeout
        try:
            result = await asyncio.wait_for(future, timeout=30.0)
            return {
                "success": True,
                "text": result.get("text", ""),
                "confidence": result.get("confidence", 0),
                "processing_time_ms": result.get("processingTimeMs", 0)
            }
        except asyncio.TimeoutError:
            self.pending_requests.pop(request_id, None)
            logger.error("OCR request timed out")
            return {
                "success": False,
                "error": "Request timed out"
            }
        except Exception as e:
            logger.error(f"OCR request failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def request_screen_capture(self) -> Dict[str, Any]:
        """
        Request screen capture from the MCP server
        
        Returns:
            Dict with screen capture results
        """
        await self._ensure_connected()
        
        # Generate a unique request ID
        request_id = str(uuid.uuid4())
        future = asyncio.Future()
        self.pending_requests[request_id] = future
        
        # Send screen capture request to server
        await self.sio.emit("agent-request-screen", {
            "agentId": self.agent_id,
            "requestId": request_id
        })
        
        # Wait for result with timeout
        try:
            result = await asyncio.wait_for(future, timeout=30.0)
            return {
                "success": True,
                "image_data": result.get("imageData", ""),
                "dimensions": result.get("dimensions", {"width": 0, "height": 0})
            }
        except asyncio.TimeoutError:
            self.pending_requests.pop(request_id, None)
            logger.error("Screen capture request timed out")
            return {
                "success": False,
                "error": "Request timed out"
            }
        except Exception as e:
            logger.error(f"Screen capture request failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def call_tool(self, tool_name: str, arguments: Dict = None) -> Dict[str, Any]:
        """
        Call MCP server tool using WebSockets - matches the original MCPClient interface
        """
        if arguments is None:
            arguments = {}
            
        try:
            await self._ensure_connected()
            
            if tool_name == "screen_capture":
                # For screen capture, we capture locally and return the image
                image_data = self._capture_screenshot()
                return {
                    "success": True,
                    "image_base64": image_data
                }
                
            elif tool_name == "extract_text":
                # Get image data from arguments
                image_data = arguments.get("image_data")
                if not image_data:
                    return {"success": False, "error": "Image data required"}
                
                # Request OCR using WebSockets
                result = await self.request_ocr(image_data)
                return {
                    "success": result["success"],
                    "text": result.get("text", ""),
                    "confidence": result.get("confidence", 0)
                }
            else:
                return {"success": False, "error": f"Unknown tool: {tool_name}"}
                
        except Exception as e:
            logger.error(f"MCP tool call failed for {tool_name}: {e}")
            return {"success": False, "error": str(e)}
    
    # Helper methods
    async def _ensure_connected(self):
        """Ensure agent is connected to the server"""
        if not self.connected:
            await self.connect()
            if not self.connected:
                raise ConnectionError("Failed to connect to MCP server")
    
    def _capture_screenshot(self) -> str:
        """Capture a screenshot and return as base64 data URL"""
        try:
            # Capture screenshot using PIL
            screenshot = ImageGrab.grab()
            buffer = io.BytesIO()
            screenshot.save(buffer, format="JPEG", quality=85)
            buffer.seek(0)
            
            # Convert to base64
            b64_bytes = base64.b64encode(buffer.read())
            b64_str = b64_bytes.decode('utf-8')
            
            # Return as data URL
            return f"data:image/jpeg;base64,{b64_str}"
        except Exception as e:
            logger.error(f"Screenshot capture failed: {e}")
            raise

# Example usage
async def main():
    agent = MCPWebSocketAgent()
    try:
        await agent.connect()
        
        # Take a screenshot and request OCR
        result = await agent.request_ocr()
        if result["success"]:
            print(f"OCR Text: {result['text'][:100]}...")
        else:
            print(f"OCR failed: {result.get('error')}")
        
        # Wait a bit then disconnect
        await asyncio.sleep(1)
    finally:
        await agent.disconnect()

if __name__ == "__main__":
    asyncio.run(main())