#!/usr/bin/env python3
"""
Vygil Activity Tracking Agent - MVP Implementation

Simple AI agent that monitors screen activity using MCP server tools.
Follows the MVP flow: screen capture → OCR → LLM analysis → activity logging
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, Optional, Tuple, Any, List
import yaml
from pathlib import Path

# Third-party imports
from dotenv import load_dotenv

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
        logger.info(f"OpenAI response: {response.choices[0].message.content.strip()}")
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
        
    def _next_request_id(self) -> int:
        """Generate unique request ID"""
        self.request_id += 1
        return self.request_id
        
    async def _send_mcp_request(self, method: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Send JSON-RPC 2.0 request to MCP server via stdio"""
        import subprocess
        import json
        
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
            if result.get("jsonrpc") != "2.0" or result.get("id") != request_id:
                logger.error(f"Invalid JSON-RPC response: {result}")
                return {"success": False, "error": "Invalid JSON-RPC response"}
            
            # Check for errors
            if "error" in result:
                error = result["error"]
                logger.error(f"MCP server error: {error}")
                return {"success": False, "error": error.get("message", "Unknown error")}
            
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
                    import json
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


class ActivityTrackingAgent:
    """Main activity tracking agent implementing the MVP flow"""
    
    def __init__(self, config_path: str = "config/activity-tracking-agent.yaml"):
        self.config = ConfigLoader.load_config(config_path)
        self.agent_config = self.config.get('agent', {})
        self.agent_id = self.agent_config.get('id', 'vygil-activity-tracker')
        
        # Initialize components
        self.llm_processor = LLMProcessor(self.config)
        self.mcp_client = MCPClient(self.config)
        
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


class FocusAssistantAgent(ActivityTrackingAgent):
    """Focus Assistant Agent - extends ActivityTrackingAgent with focus-specific features"""
    
    def __init__(self, config_path: str):
        super().__init__(config_path)
        self.focus_metrics = []
        self.distraction_count = 0
        self.productivity_score = 0.0
        
    async def _process_activity_result(self, result: str) -> Dict[str, Any]:
        """Process focus-specific results"""
        try:
            import json
            import re
            
            # Extract JSON from "ACTIVITY: {JSON}" format
            json_str = result
            if result.startswith("ACTIVITY:"):
                # Remove "ACTIVITY:" prefix and extract JSON
                json_str = result[9:].strip()
            
            # Try to find JSON in the string if it's not pure JSON
            if not json_str.startswith('{'):
                json_match = re.search(r'\{.*\}', result, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    raise json.JSONDecodeError("No JSON found", result, 0)
            
            # Parse JSON
            focus_data = json.loads(json_str)
            
            # Update focus metrics
            self.productivity_score = focus_data.get('productivity_score', 0.0)
            
            if focus_data.get('category') == 'distraction':
                self.distraction_count += 1
            else:
                self.distraction_count = max(0, self.distraction_count - 1)
            
            # Store focus metrics
            self.focus_metrics.append({
                'timestamp': time.time(),
                'focus_level': focus_data.get('focus_level'),
                'productivity_score': self.productivity_score,
                'category': focus_data.get('category'),
                'suggestion': focus_data.get('suggestion')
            })
            
            # Keep only last 50 entries
            if len(self.focus_metrics) > 50:
                self.focus_metrics = self.focus_metrics[-50:]
            
            logger.info(f"📊 Focus metrics updated: productivity={self.productivity_score:.2f}, category={focus_data.get('category')}")
            return focus_data
            
        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning(f"Failed to parse focus JSON from: {result[:100]}... Error: {e}")
            # Fallback to regular activity processing
            return await super()._process_activity_result(result)
    
    def get_focus_summary(self) -> Dict[str, Any]:
        """Get current focus session summary"""
        if not self.focus_metrics:
            return {
                'productivity_score': 0.0,
                'focus_sessions': 0,
                'distractions': 0,
                'total_focus_time': 0,
                'status': 'no_data'
            }
        
        recent_metrics = self.focus_metrics[-10:]  # Last 10 entries
        avg_productivity = sum(m['productivity_score'] for m in recent_metrics) / len(recent_metrics)
        
        focus_levels = [m['focus_level'] for m in recent_metrics]
        dominant_focus = max(set(focus_levels), key=focus_levels.count)
        
        # Calculate focus sessions (consecutive periods of medium/high focus)
        focus_sessions = 0
        in_focus_session = False
        for metric in self.focus_metrics:
            if metric.get('focus_level') in ['medium', 'high']:
                if not in_focus_session:
                    focus_sessions += 1
                    in_focus_session = True
            else:
                in_focus_session = False
        
        # Calculate total focus time (approximate, assuming 60 second intervals)
        focus_entries = [m for m in self.focus_metrics if m.get('focus_level') in ['medium', 'high']]
        total_focus_time = len(focus_entries) * 60  # 60 seconds per entry
        
        return {
            'productivity_score': avg_productivity,
            'focus_sessions': focus_sessions,
            'distractions': self.distraction_count,
            'total_focus_time': total_focus_time,
            'dominant_focus_level': dominant_focus,
            'total_sessions': len(self.focus_metrics),
            'current_suggestion': recent_metrics[-1].get('suggestion', '') if recent_metrics else ''
        }


# Missing in agent.py - needed for Focus Assistant
def store_focus_metrics(metrics: Dict[str, Any], agent_id: str = "vygil-focus-assistant"):
    """Store focus metrics to a JSON file for persistence"""
    try:
        metrics_file = Path(__file__).parent / "memory" / f"{agent_id}_focus_metrics.json"
        metrics_file.parent.mkdir(exist_ok=True)
        
        # Load existing metrics
        existing_metrics = []
        if metrics_file.exists():
            try:
                with open(metrics_file, 'r') as f:
                    existing_metrics = json.load(f)
            except (json.JSONDecodeError, IOError):
                existing_metrics = []
        
        # Add new metrics
        existing_metrics.append({
            'timestamp': time.time(),
            **metrics
        })
        
        # Keep only last 100 entries
        if len(existing_metrics) > 100:
            existing_metrics = existing_metrics[-100:]
        
        # Save back to file
        with open(metrics_file, 'w') as f:
            json.dump(existing_metrics, f, indent=2)
            
        logger.debug(f"Stored focus metrics for {agent_id}")
        
    except Exception as e:
        logger.error(f"Failed to store focus metrics: {e}")


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)