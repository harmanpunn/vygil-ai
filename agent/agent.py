#!/usr/bin/env python3
"""
Vygil Activity Tracking Agent - MVP Implementation

Simple AI agent that monitors screen activity using MCP server tools.
Follows the MVP flow: screen capture → OCR → LLM analysis → activity logging
"""

import uuid
import asyncio
import logging
import json
import time
import os
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import yaml
from pathlib import Path

# Third-party imports
from dotenv import load_dotenv
import socketio

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("vygil-agent")


class ConfigLoader:
    """Load and validate agent configuration from YAML"""
    
    @staticmethod
    def load_config(config_path: str = "config/activity-tracking-agent.yaml") -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                logger.info(f"Configuration loaded from {config_path}")
                return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            raise


# Simple memory management functions
def get_memory_file(agent_id: str) -> Path:
    """Get memory file path for agent"""
    memory_dir = Path("memory")
    memory_dir.mkdir(exist_ok=True)
    return memory_dir / f"{agent_id}_memory.txt"

def get_memory(agent_id: str = "vygil-activity-tracker") -> str:
    """Load agent memory from file"""
    try:
        memory_file = get_memory_file(agent_id)
        if memory_file.exists():
            return memory_file.read_text(encoding='utf-8').strip()
        return ""
    except Exception as e:
        logger.error(f"Failed to read memory: {e}")
        return ""

def update_memory(content: str, agent_id: str = "vygil-activity-tracker"):
    """Update agent memory file"""
    try:
        memory_file = get_memory_file(agent_id)
        memory_file.write_text(content, encoding='utf-8')
        logger.debug(f"Memory updated for {agent_id}")
    except Exception as e:
        logger.error(f"Failed to update memory: {e}")

def get_current_time() -> str:
    """Get current time formatted for memory entries"""
    return datetime.now().strftime('%I:%M %p')

def inject_memory_context(prompt: str, agent_id: str) -> str:
    """Replace $MEMORY placeholder with actual memory content"""
    memory_content = get_memory(agent_id)
    if not memory_content:
        memory_content = "No previous activities recorded."
    return prompt.replace('$MEMORY', memory_content)

def execute_autonomous_code(code: str, activity_result: str, agent_id: str):
    """Execute autonomous code with access to memory functions"""
    if not code.strip():
        return
    
    try:
        # Create execution context with utility functions
        context = {
            'get_memory': lambda: get_memory(agent_id),
            'update_memory': lambda content: update_memory(content, agent_id), 
            'get_current_time': get_current_time,
            'activity_result': activity_result,
            'agent_id': agent_id
        }
        
        # Execute the code
        exec(code, context)
        logger.debug(f"Autonomous code executed for {agent_id}")
        
    except Exception as e:
        logger.error(f"Autonomous code execution failed: {e}")


class LLMProcessor:
    """Handle LLM communication for activity classification"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm_config = config.get('llm', {})
        self.system_prompt = config.get('instructions', {}).get('system_prompt', '')
        self.clients = {}
        self._setup_clients()
    
    def _setup_clients(self):
        """Initialize LLM clients based on configuration"""
        primary_provider = self.llm_config.get('provider', 'openai')
        
        # Setup primary provider
        if primary_provider == 'openai' and os.getenv('OPENAI_API_KEY'):
            try:
                import openai
                self.clients['openai'] = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
                logger.info("OpenAI client initialized successfully")
            except ImportError:
                logger.warning("OpenAI package not available")
        
        elif primary_provider == 'anthropic' and os.getenv('ANTHROPIC_API_KEY'):
            try:
                import anthropic
                self.clients['anthropic'] = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
                logger.info("Anthropic client initialized")
            except ImportError:
                logger.warning("Anthropic package not available")
        
        # Setup fallback providers
        fallback_providers = self.llm_config.get('fallback_providers', [])
        for fallback in fallback_providers:
            provider = fallback.get('provider')
            if provider == 'anthropic' and provider not in self.clients and os.getenv('ANTHROPIC_API_KEY'):
                try:
                    import anthropic
                    self.clients['anthropic'] = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
                    logger.info("Anthropic fallback client initialized")
                except ImportError:
                    logger.warning("Anthropic package not available for fallback")
    
    async def classify_activity(self, screen_text: str, agent_id: str) -> Tuple[str, float]:
        """
        Classify user activity based on screen text
        Returns: (activity_description, confidence_score)
        """
        if not screen_text or len(screen_text.strip()) < 10:
            return "ACTIVITY: Insufficient screen content", 0.2
        
        # Get system prompt from config and inject memory context
        system_prompt = self.config.get('instructions', {}).get('system_prompt', '')
        if not system_prompt:
            logger.warning("No system prompt found in config")
            return "ACTIVITY: Configuration error", 0.0
        
        # Inject memory context into prompt
        system_prompt = inject_memory_context(system_prompt, agent_id)
        
        # Truncate text to avoid token limits
        truncated_text = screen_text[:2000]
        if len(screen_text) > 2000:
            truncated_text += "..."
        
        user_prompt = f"""<Screen Content>
{truncated_text}
</Screen Content>"""

        # Try primary provider first
        primary_provider = self.llm_config.get('provider', 'openai')
        if primary_provider in self.clients:
            try:
                response = await self._query_llm(primary_provider, user_prompt, system_prompt)
                if response:
                    confidence = self._calculate_confidence(screen_text, response)
                    return self._format_response(response), confidence
            except Exception as e:
                logger.warning(f"Primary LLM provider {primary_provider} failed: {e}")
        
        # Try fallback providers
        for fallback in self.llm_config.get('fallback_providers', []):
            provider = fallback.get('provider')
            if provider in self.clients:
                try:
                    response = await self._query_llm(provider, user_prompt, system_prompt)
                    if response:
                        confidence = self._calculate_confidence(screen_text, response)
                        return self._format_response(response), confidence
                except Exception as e:
                    logger.warning(f"Fallback LLM provider {provider} failed: {e}")
        
        # All providers failed
        logger.error("All LLM providers failed")
        return "ACTIVITY: LLM analysis failed", 0.0
    
    async def _query_llm(self, provider: str, user_prompt: str, system_prompt: str = "") -> Optional[str]:
        """Query specific LLM provider"""
        try:
            if provider == 'openai':
                return await self._query_openai(user_prompt, system_prompt)
            elif provider == 'anthropic':
                return await self._query_anthropic(user_prompt, system_prompt)
            else:
                logger.warning(f"Unknown LLM provider: {provider}")
                return None
        except Exception as e:
            logger.error(f"Error querying {provider}: {e}")
            return None
    
    async def _query_openai(self, user_prompt: str, system_prompt: str = "") -> str:
        """Query OpenAI API"""
        client = self.clients['openai']
        model = self.llm_config.get('model', 'gpt-4o-mini')
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=model,
            messages=messages,
            max_tokens=self.llm_config.get('max_tokens', 50),
            temperature=self.llm_config.get('temperature', 0.1),
            timeout=self.llm_config.get('timeout', 10)
        )
        return response.choices[0].message.content.strip()
    
    async def _query_anthropic(self, user_prompt: str, system_prompt: str = "") -> str:
        """Query Anthropic Claude API"""
        client = self.clients['anthropic']
        model = self.llm_config.get('model', 'claude-3-haiku-20240307')
        
        # Build request parameters
        params = {
            'model': model,
            'max_tokens': self.llm_config.get('max_tokens', 50),
            'messages': [{"role": "user", "content": user_prompt}],
            'timeout': self.llm_config.get('timeout', 10)
        }
        
        # Add system prompt if provided
        if system_prompt:
            params['system'] = system_prompt
        
        response = await asyncio.to_thread(client.messages.create, **params)
        return response.content[0].text.strip()
    
    def _calculate_confidence(self, screen_text: str, response: str) -> float:
        """Calculate confidence score based on text length and response quality"""
        base_confidence = 0.5
        
        # Higher confidence for longer screen text
        if len(screen_text) > 100:
            base_confidence += 0.2
        if len(screen_text) > 500:
            base_confidence += 0.1
        
        # Higher confidence if response follows format
        if response.startswith("ACTIVITY:"):
            base_confidence += 0.2
        
        return min(base_confidence, 1.0)
    
    def _format_response(self, response: str) -> str:
        """Ensure response follows ACTIVITY: format"""
        if not response.startswith("ACTIVITY:"):
            return f"ACTIVITY: {response}"
        return response


class MCPClient:
    """MCP client for communicating with MCP server tools using WebSockets"""
    
    def __init__(self, config, agent_id: str = None):
        """Initialize MCP client"""
        # Handle both dict config and direct server_url
        if isinstance(config, dict):
            self.server_url = config.get("mcp_server_url", "http://localhost:3000")
        else:
            # If config is a string, treat it as server_url
            self.server_url = config
            
        self.agent_id = agent_id or f"agent-{uuid.uuid4()}"
        self.sio = socketio.AsyncClient()
        self.connected = False
        self.pending_requests = {}
        
        # Set up event handlers
        self.sio.on("connect", self._on_connect)
        self.sio.on("disconnect", self._on_disconnect)
        self.sio.on("agent-registered", self._on_agent_registered)
        self.sio.on("agent-ocr-result", self._on_ocr_result)
        self.sio.on("agent-screen-result", self._on_screen_result)
        self.sio.on("error", self._on_error)
        
        logger.info(f"MCP client initialized with ID {self.agent_id}")
    
    async def connect(self):
        """Connect to MCP server"""
        if not self.connected:
            try:
                await self.sio.connect(self.server_url)
                await self.sio.emit("agent-register", {
                    "agentId": self.agent_id,
                    "capabilities": ["ocr"]
                })
                return True
            except Exception as e:
                logger.error(f"Failed to connect to MCP server: {e}")
                return False
        return True
    
    async def disconnect(self):
        """Disconnect from MCP server"""
        if self.connected:
            await self.sio.disconnect()
    
    async def call_tool(self, tool_name: str, arguments: Dict = None) -> Dict[str, Any]:
        """
        Call MCP server tool using WebSockets
        """
        if arguments is None:
            arguments = {}
            
        try:
            # Make sure we're connected
            if not self.connected:
                await self.connect()
                if not self.connected:
                    return {"success": False, "error": "Failed to connect to MCP server"}
            
            # Generate a unique request ID
            request_id = str(uuid.uuid4())
            future = asyncio.Future()
            self.pending_requests[request_id] = future
            
            if tool_name == "screen_capture":
                # Request screenshot using WebSockets
                await self.sio.emit("agent-request-screen", {
                    "agentId": self.agent_id,
                    "requestId": request_id
                })
                
                # Wait for result
                try:
                    result = await asyncio.wait_for(future, timeout=30.0)
                    return {
                        "success": True,
                        "image_base64": result.get("imageData", "")
                    }
                except Exception as e:
                    return {"success": False, "error": str(e)}
                
            elif tool_name == "extract_text":
                # Get image data from arguments
                image_data = arguments.get("image_data")
                if not image_data:
                    return {"success": False, "error": "Image data required"}
                
                # Request OCR using WebSockets
                await self.sio.emit("agent-request-ocr", {
                    "agentId": self.agent_id,
                    "requestId": request_id,
                    "imageData": image_data
                })
                
                # Wait for result
                try:
                    result = await asyncio.wait_for(future, timeout=30.0)
                    return {
                        "success": True,
                        "text": result.get("text", ""),
                        "confidence": result.get("confidence", 0)
                    }
                except Exception as e:
                    return {"success": False, "error": str(e)}
            else:
                return {"success": False, "error": f"Unknown tool: {tool_name}"}
                
        except Exception as e:
            logger.error(f"MCP tool call failed for {tool_name}: {e}")
            return {"success": False, "error": str(e)}
    
    # Socket.IO event handlers
    async def _on_connect(self):
        self.connected = True
        logger.info("Connected to MCP server")
    
    async def _on_disconnect(self):
        self.connected = False
        logger.info("Disconnected from MCP server")
    
    async def _on_agent_registered(self, data):
        logger.info(f"Agent registered with server: {data}")
    
    async def _on_ocr_result(self, data):
        request_id = data.get('requestId')
        if request_id in self.pending_requests:
            future = self.pending_requests.pop(request_id)
            future.set_result(data)
        logger.info(f"Received OCR result: {len(data['text'])} chars")
    
    async def _on_screen_result(self, data):
        request_id = data.get('requestId')
        if request_id in self.pending_requests:
            future = self.pending_requests.pop(request_id)
            future.set_result(data)
        logger.info(f"Received screen capture: {len(data['imageData'])} bytes")
    
    async def _on_error(self, data):
        request_id = data.get('requestId')
        if request_id in self.pending_requests:
            future = self.pending_requests.pop(request_id)
            future.set_exception(Exception(data.get('message', 'Unknown error')))
        logger.error(f"Error from MCP server: {data.get('message')}")


class ActivityTrackingAgent:
    """Main activity tracking agent implementing the MVP flow"""
    
    def __init__(self, config_path: str = "config/activity-tracking-agent.yaml"):
        self.config = ConfigLoader.load_config(config_path)
        self.agent_config = self.config.get('agent', {})
        self.agent_id = self.agent_config.get('id', 'vygil-activity-tracker')
        
        # Initialize components
        self.llm_processor = LLMProcessor(self.config)
        
        # Initialize MCP client with WebSocket connection
        self.mcp_client = MCPClient(
            config=self.config,  # Pass the full config dict
            agent_id=self.agent_id
        )
        
        # Agent state
        self.running = False
        self.loop_interval = self.agent_config.get('loop_interval', 60)
        self.max_retries = self.agent_config.get('max_retries', 3)
        self.consecutive_failures = 0
        
        # Statistics
        self.total_iterations = 0
        self.successful_iterations = 0
        self.start_time = None
        
        logger.info(f"Agent initialized: {self.agent_config.get('name', 'Vygil Agent')}")
        logger.info(f"Loop interval: {self.loop_interval} seconds")
    
    async def start_monitoring(self):
        """Start the 60-second activity monitoring loop"""
        if self.running:
            logger.warning("Agent already running")
            return
        
        logger.info("🚀 Starting Vygil Activity Tracking Agent...")
        
        # Check API keys
        if not self._check_api_keys():
            logger.error("No LLM API keys found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY")
            return
        
        # Request screen permission (placeholder for MVP)
        self._request_screen_permission()
        
        self.running = True
        self.start_time = datetime.now()
        
        try:
            # Connect to MCP server
            connected = await self.mcp_client.connect()
            if not connected:
                logger.error("Failed to connect to MCP server, falling back to mock mode")
            
            while self.running:
                await self._execute_monitoring_cycle()
                
                # Check for too many consecutive failures
                max_failures = self.config.get('error_handling', {}).get('max_consecutive_failures', 5)
                if self.consecutive_failures >= max_failures:
                    logger.error(f"Stopping agent after {self.consecutive_failures} consecutive failures")
                    break
                
                # Wait for next cycle
                await asyncio.sleep(self.loop_interval)
                
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        except Exception as e:
            logger.error(f"Agent loop error: {e}")
        finally:
            await self.stop_monitoring()
    
    async def stop_monitoring(self):
        """Stop the activity monitoring"""
        logger.info("🛑 Stopping activity tracking agent...")
        self.running = False
        
        # Print final statistics
        stats = self.get_statistics()
        logger.info(f"Final statistics: {stats}")
    
    async def _execute_monitoring_cycle(self):
        """Execute one complete monitoring cycle following MVP flow"""
        self.total_iterations += 1
        cycle_start = time.time()
        
        try:
            logger.debug(f"Starting monitoring cycle {self.total_iterations}")
            
            # Step 1: Capture screen (via MCP server)
            screen_result = await self.mcp_client.call_tool("screen_capture")
            if not screen_result.get("success"):
                raise Exception(f"Screen capture failed: {screen_result.get('error')}")
            
            # Step 2: Extract text from screen (OCR via MCP server)
            ocr_result = await self.mcp_client.call_tool("extract_text", {
                "image_data": screen_result.get("image_base64")
            })
            if not ocr_result.get("success"):
                raise Exception(f"OCR extraction failed: {ocr_result.get('error')}")
            
            screen_text = ocr_result.get("text", "")
            
            # Step 3: Classify activity using LLM (with memory context)
            activity_description, confidence = await self.llm_processor.classify_activity(screen_text, self.agent_id)
            
            # Step 4: Execute autonomous code (agentic behavior)
            autonomous_code = self.config.get('code', '')
            if autonomous_code:
                execute_autonomous_code(autonomous_code, activity_description, self.agent_id)
            
            # Step 5: Log activity (via MCP server)
            log_result = await self.mcp_client.call_tool("log_activity", {
                "description": activity_description,
                "confidence": confidence,
                "screen_text_length": len(screen_text),
                "processing_time": time.time() - cycle_start
            })
            
            if log_result.get("success"):
                self.successful_iterations += 1
                self.consecutive_failures = 0
                
                # Log successful cycle
                cycle_time = time.time() - cycle_start
                logger.info(f"✅ [{datetime.now().strftime('%H:%M:%S')}] {activity_description} "
                          f"(confidence: {confidence:.2f}, time: {cycle_time:.1f}s)")
            else:
                raise Exception(f"Activity logging failed: {log_result.get('error')}")
                
        except Exception as e:
            self.consecutive_failures += 1
            logger.error(f"❌ Monitoring cycle {self.total_iterations} failed: {e}")
            
            # Wait before retry
            retry_delay = self.config.get('error_handling', {}).get('retry_delay', 2)
            await asyncio.sleep(retry_delay)
    
    def _check_api_keys(self) -> bool:
        """Check if required API keys are available"""
        return bool(os.getenv('OPENAI_API_KEY') or os.getenv('ANTHROPIC_API_KEY'))
    
    def _request_screen_permission(self):
        """Request screen recording permission (placeholder for MVP)"""
        consent_msg = self.config.get('privacy', {}).get('user_consent_message', 
                                                        'Screen recording permission required')
        logger.info(f"📺 {consent_msg}")
        logger.info("Screen permission granted (MVP mode)")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get agent performance statistics"""
        runtime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        success_rate = (self.successful_iterations / self.total_iterations * 100) if self.total_iterations > 0 else 0
        
        return {
            "agent_name": self.agent_config.get('name', 'Vygil Agent'),
            "runtime_seconds": round(runtime, 1),
            "total_iterations": self.total_iterations,
            "successful_iterations": self.successful_iterations,
            "success_rate_percent": round(success_rate, 1),
            "consecutive_failures": self.consecutive_failures,
            "is_running": self.running
        }


async def main():
    """Main entry point for the activity tracking agent"""
    logger.info("🎯 Vygil Activity Tracking Agent MVP")
    
    try:
        # Create and start agent
        agent = ActivityTrackingAgent()
        await agent.start_monitoring()
        
    except Exception as e:
        logger.error(f"Failed to start agent: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)