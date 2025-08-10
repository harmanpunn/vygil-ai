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
                
        elif primary_provider == 'gemini' and os.getenv('GOOGLE_API_KEY'):
            try:
                import google.generativeai as genai
                genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
                self.clients['gemini'] = genai
                logger.info("Google Gemini client initialized")
            except ImportError:
                logger.warning("Google Generative AI package not available")
        
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
                    
            elif provider == 'gemini' and provider not in self.clients and os.getenv('GOOGLE_API_KEY'):
                try:
                    import google.generativeai as genai
                    genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
                    self.clients['gemini'] = genai
                    logger.info("Google Gemini fallback client initialized")
                except ImportError:
                    logger.warning("Google Generative AI package not available for fallback")
                    
            elif provider == 'openai' and provider not in self.clients and os.getenv('OPENAI_API_KEY'):
                try:
                    import openai
                    self.clients['openai'] = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
                    logger.info("OpenAI fallback client initialized")
                except ImportError:
                    logger.warning("OpenAI package not available for fallback")
    
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
                    return self._format_response(response, agent_id), confidence
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
                        return self._format_response(response, agent_id), confidence
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
            elif provider == 'gemini':
                return await self._query_gemini(user_prompt, system_prompt)
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
        
        response_content = response.choices[0].message.content.strip()
        
        # For focus agents, log only the activity instead of full JSON
        if response_content.startswith('{') and 'focus_level' in response_content:
            try:
                import json
                focus_data = json.loads(response_content)
                activity = focus_data.get('activity', 'Unknown activity')
                logger.info(f"OpenAI response: ACTIVITY: {activity}")
            except:
                logger.info(f"OpenAI response: {response_content}")
        else:
            logger.info(f"OpenAI response: {response_content}")
        
        return response_content
    
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
    
    async def _query_gemini(self, user_prompt: str, system_prompt: str = "") -> str:
        """Query Google Gemini API"""
        genai = self.clients['gemini']
        model_name = self.llm_config.get('model', 'gemini-1.5-flash')
        
        # Combine system prompt and user prompt for Gemini
        if system_prompt:
            combined_prompt = f"{system_prompt}\n\n{user_prompt}"
        else:
            combined_prompt = user_prompt
        
        # Configure generation parameters
        generation_config = {
            'temperature': self.llm_config.get('temperature', 0.1),
            'max_output_tokens': self.llm_config.get('max_tokens', 50),
        }
        
        # Initialize the model
        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config
        )
        
        # Generate response
        response = await asyncio.to_thread(
            model.generate_content,
            combined_prompt
        )
        
        result_text = response.text.strip()
        logger.info(f"Gemini response: {result_text}")
        return result_text
    
    def _calculate_confidence(self, screen_text: str, response: str) -> float:
        """Calculate confidence score based on text length and response quality"""
        base_confidence = 0.3
        
        # Text length indicators (more gradual scoring)
        text_len = len(screen_text.strip())
        if text_len < 20:
            base_confidence += 0.1  # Very little text, low confidence
        elif text_len < 100:
            base_confidence += 0.2  # Some text available
        elif text_len < 500:
            base_confidence += 0.3  # Good amount of text
        else:
            base_confidence += 0.4  # Lots of context
        
        # Response quality indicators
        if response.startswith("ACTIVITY:"):
            base_confidence += 0.1  # Formatted correctly
        
        # Response specificity (longer, more specific responses get higher confidence)
        activity_text = response.replace("ACTIVITY:", "").strip()
        if len(activity_text) > 20:
            base_confidence += 0.1  # Detailed description
        elif len(activity_text) < 10:
            base_confidence -= 0.1  # Very generic description
        
        # Add some randomness for realistic variation (±0.05)
        import random
        variation = random.uniform(-0.05, 0.05)
        
        return max(0.1, min(0.95, base_confidence + variation))
    
    def _format_response(self, response: str, agent_id: str = None) -> str:
        """Format response based on agent type"""
        # For Focus Assistant, extract human-readable message from JSON
        if agent_id and 'focus-assistant' in agent_id:
            try:
                import json
                import re
                
                # Extract JSON from response if it's wrapped
                json_str = response
                if response.startswith("ACTIVITY:"):
                    json_str = response[9:].strip()
                
                # Find JSON in the string
                if not json_str.startswith('{'):
                    json_match = re.search(r'\{.*\}', response, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                
                # Parse and extract meaningful message
                focus_data = json.loads(json_str)
                activity = focus_data.get('activity', 'Unknown activity')
                
                # For logs, keep it concise like activity tracker - only show the activity
                # The suggestion and other data are used internally for focus metrics
                return f"ACTIVITY: {activity}"
                    
            except (json.JSONDecodeError, AttributeError):
                # Fallback to original response if JSON parsing fails
                pass
        
        # For any agent with focus_features, extract activity from JSON
        # This handles the new ConfigurableAgent with focus_features
        if response.startswith('{') and response.endswith('}'):
            try:
                import json
                focus_data = json.loads(response)
                if 'activity' in focus_data:
                    activity = focus_data.get('activity', 'Unknown activity')
                    return f"ACTIVITY: {activity}"
            except (json.JSONDecodeError, ValueError):
                # If JSON parsing fails, continue to default formatting
                pass
        
        # Default formatting for regular activity tracker
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


class SensorManager:
    """Manages different sensor inputs (screen, audio, camera, etc.)"""
    
    def __init__(self, sensors_config: Dict[str, Any]):
        self.sensors_config = sensors_config
        self.available_sensors = self._discover_sensors()
        logger.info(f"SensorManager initialized with {len(self.available_sensors)} sensors")
    
    def _discover_sensors(self) -> Dict[str, Dict]:
        """Discover available sensors based on configuration"""
        sensors = {}
        
        # Screen sensor (always available via MCP)
        if self.sensors_config.get('screen', {}).get('enabled', True):
            sensors['screen'] = {
                'type': 'visual',
                'tools': ['screen_capture', 'extract_text'],
                'description': 'Captures and analyzes screen content'
            }
        
        # Audio sensor (future implementation)
        if self.sensors_config.get('audio', {}).get('enabled', False):
            sensors['audio'] = {
                'type': 'auditory', 
                'tools': ['capture_audio', 'transcribe_audio'],
                'description': 'Captures and transcribes audio input'
            }
        
        # Camera sensor (future implementation)
        if self.sensors_config.get('camera', {}).get('enabled', False):
            sensors['camera'] = {
                'type': 'visual',
                'tools': ['capture_image', 'analyze_visual'],
                'description': 'Captures and analyzes camera input'
            }
        
        # Microphone sensor (future implementation)
        if self.sensors_config.get('microphone', {}).get('enabled', False):
            sensors['microphone'] = {
                'type': 'auditory',
                'tools': ['capture_voice', 'voice_analysis'], 
                'description': 'Captures and analyzes voice input'
            }
        
        return sensors
    
    def get_sensor_capabilities(self, sensor_name: str) -> Dict[str, Any]:
        """Get capabilities of a specific sensor"""
        return self.available_sensors.get(sensor_name, {})
    
    def get_sensors_by_type(self, sensor_type: str) -> List[str]:
        """Get all sensors of a specific type (visual, auditory, etc.)"""
        return [name for name, config in self.available_sensors.items() 
                if config.get('type') == sensor_type]


class ConfigurableAgent:
    """Pure YAML-driven agent that can use any sensors and tools"""
    
    def __init__(self, config_path: str):
        self.config = ConfigLoader.load_config(config_path)
        self.agent_config = self.config.get('agent', {})
        self.agent_id = self.agent_config.get('id', 'configurable-agent')
        
        # Initialize sensor manager
        sensors_config = self.config.get('sensors', {})
        self.sensor_manager = SensorManager(sensors_config)
        
        # Initialize LLM and MCP components
        self.llm_processor = LLMProcessor(self.config)
        self.mcp_client = MCPClient(self.config)
        
        # Agent state
        self.running = False
        self.loop_interval = self.agent_config.get('loop_interval', 20)
        self.max_retries = self.agent_config.get('max_retries', 3)
        self.consecutive_failures = 0
        
        # Statistics and custom data storage
        self.total_iterations = 0
        self.successful_iterations = 0
        self.start_time = None
        self.custom_data = {}  # For agent-specific data like focus metrics
        
        logger.info(f"ConfigurableAgent initialized: {self.agent_config.get('name', 'Unnamed Agent')}")
        logger.info(f"Available sensors: {list(self.sensor_manager.available_sensors.keys())}")
        logger.info(f"Loop interval: {self.loop_interval} seconds")
    
    async def process_input(self, input_data: Dict[str, Any], timestamp: str) -> Dict[str, Any]:
        """
        Universal sensor-aware processing method.
        Agent autonomously decides which sensors to use and how to process data.
        
        Args:
            input_data: {
                "screen": {"image": "base64_data", "metadata": {}},
                "audio": {"data": "base64_audio", "metadata": {}},
                "camera": {"image": "base64_data", "metadata": {}},
                # ... future sensors
            }
            timestamp: ISO timestamp
            
        Returns: {
            "result": str,  # Agent's analysis result
            "confidence": float,
            "processing_time": float,
            "sensors_used": List[str],
            "agent_reasoning": str,
            "custom_data": Dict  # Agent-specific data (focus metrics, etc.)
        }
        """
        start_time = time.time()
        
        try:
            logger.info(f"🤖 Agent {self.agent_id} autonomously processing multi-sensor input...")
            
            # STEP 1: Agent analyzes available sensors and input data
            available_sensors = list(self.sensor_manager.available_sensors.keys())
            present_sensors = [sensor for sensor in available_sensors if sensor in input_data]
            
            logger.info(f"🔍 Available sensors: {available_sensors}")
            logger.info(f"📡 Present input sensors: {present_sensors}")
            
            # STEP 2: Agent decides processing strategy based on configuration
            agent_prompt = self._build_sensor_decision_prompt(present_sensors, input_data)
            
            decision_response = await self.llm_processor._query_llm(
                self.llm_processor.llm_config.get('provider', 'openai'),
                agent_prompt
            )
            
            # Parse agent's decision
            try:
                import json
                decision = json.loads(decision_response)
                chosen_sensors = decision.get('chosen_sensors', present_sensors[:1])
                processing_plan = decision.get('processing_plan', 'Extract and analyze data')
                reasoning = decision.get('reasoning', 'Default sensor selection')
                logger.info(f"🎯 Agent decision: sensors={chosen_sensors}, plan={processing_plan}")
            except:
                chosen_sensors = present_sensors[:1]  # Fallback to first available
                processing_plan = 'Extract and analyze available data'
                reasoning = 'Fallback sensor selection'
                logger.warning("Agent decision parsing failed, using fallback")
            
            # STEP 3: Execute sensor-specific processing
            processed_data = {}
            for sensor in chosen_sensors:
                if sensor in input_data:
                    sensor_result = await self._process_sensor_data(sensor, input_data[sensor])
                    processed_data[sensor] = sensor_result
            
            # STEP 4: Agent analyzes combined sensor data using its configuration
            combined_data = self._combine_sensor_data(processed_data)
            
            # For focus agents, get raw LLM response first before formatting
            if self.config.get('focus_features'):
                # Get raw JSON response for focus processing
                system_prompt = self.config.get('instructions', {}).get('system_prompt', '')
                system_prompt_with_memory = inject_memory_context(system_prompt, self.agent_id)
                
                raw_response = await self.llm_processor._query_llm(
                    self.llm_processor.llm_config.get('provider', 'openai'),
                    f"""<Screen Content>\n{combined_data[:2000]}\n</Screen Content>""",
                    system_prompt_with_memory
                )
                
                if raw_response:
                    # Use raw response for autonomous code (contains JSON)
                    result = raw_response
                    confidence = self.llm_processor._calculate_confidence(combined_data, raw_response)
                    # Log clean activity instead of raw JSON
                    try:
                        import json
                        focus_data = json.loads(raw_response)
                        activity = focus_data.get('activity', 'Unknown activity')
                        logger.debug(f"🎯 Focus agent processing: {activity}")
                    except:
                        logger.debug(f"🎯 Focus agent raw response: {raw_response[:100]}...")
                else:
                    # Fallback to formatted response
                    result, confidence = await self.llm_processor.classify_activity(combined_data, self.agent_id)
            else:
                # Regular agent processing
                result, confidence = await self.llm_processor.classify_activity(combined_data, self.agent_id)
            
            # STEP 5: Execute autonomous code with appropriate result format
            autonomous_code = self.config.get('code', '')
            if autonomous_code:
                execute_autonomous_code(autonomous_code, result, self.agent_id)
                logger.debug("✅ Agent executed autonomous code")
            
            # STEP 6: Agent-specific data processing (from YAML config)
            custom_data = await self._process_agent_specific_data(result, processed_data)
            
            # Format result for API response (after autonomous processing)
            if self.config.get('focus_features') and result and result.startswith('{'):
                # For focus agents, format JSON response for UI display
                try:
                    import json
                    focus_data = json.loads(result)
                    formatted_result = f"ACTIVITY: {focus_data.get('activity', 'Focus analysis')}"
                    result = formatted_result
                    logger.info(f"🎯 Focus agent activity: {focus_data.get('activity', 'Focus analysis')}")
                except:
                    result = f"ACTIVITY: {result}"  # Fallback formatting
            
            processing_time = time.time() - start_time
            
            return {
                "result": result,
                "confidence": confidence,
                "processing_time": processing_time,
                "sensors_used": chosen_sensors,
                "agent_reasoning": reasoning,
                "custom_data": custom_data,
                "timestamp": timestamp
            }
            
        except Exception as e:
            logger.error(f"❌ Agent processing failed: {e}")
            return {
                "result": "RESULT: Processing failed",
                "confidence": 0.0,
                "processing_time": time.time() - start_time,
                "sensors_used": [],
                "agent_reasoning": f"Error: {str(e)}",
                "custom_data": {},
                "timestamp": timestamp
            }
    
    def _build_sensor_decision_prompt(self, present_sensors: List[str], input_data: Dict) -> str:
        """Build agent's autonomous sensor decision prompt"""
        sensor_descriptions = []
        for sensor in present_sensors:
            sensor_info = self.sensor_manager.get_sensor_capabilities(sensor)
            data_size = len(str(input_data.get(sensor, {})))
            sensor_descriptions.append(f"- {sensor}: {sensor_info.get('description', 'No description')} (data: {data_size} bytes)")
        
        return f"""
You are {self.agent_config.get('name', 'an AI agent')}.

Your goal: {self.agent_config.get('description', 'Process and analyze user activity')}

Available sensors with current data:
{chr(10).join(sensor_descriptions)}

Available MCP tools:
{chr(10).join(f'- {tool}: {desc.get("description", "")}' for tool, desc in self.config.get("mcp_server", {}).get("tools", {}).items())}

Decide your processing strategy. Respond with JSON:
{{
    "chosen_sensors": ["list", "of", "sensors"],
    "processing_plan": "detailed plan for analysis",
    "reasoning": "why you chose this approach"
}}
"""
    
    async def _process_sensor_data(self, sensor: str, sensor_data: Dict) -> str:
        """Process data from a specific sensor"""
        sensor_capabilities = self.sensor_manager.get_sensor_capabilities(sensor)
        
        if sensor == 'screen':
            # Use MCP tools for screen processing
            if 'image' in sensor_data:
                ocr_result = await self.mcp_client.call_tool("extract_text", {
                    "imageData": sensor_data['image']
                })
                if ocr_result.get("success"):
                    return ocr_result.get("text", "")
                else:
                    return "Screen text extraction failed"
            return "No screen image data"
        
        elif sensor == 'audio':
            # Future: audio processing
            return "Audio processing not implemented yet"
        
        elif sensor == 'camera':
            # Future: camera processing  
            return "Camera processing not implemented yet"
        
        elif sensor == 'microphone':
            # Future: microphone processing
            return "Microphone processing not implemented yet"
        
        else:
            return f"Unknown sensor: {sensor}"
    
    def _combine_sensor_data(self, processed_data: Dict[str, str]) -> str:
        """Combine data from multiple sensors into unified text"""
        combined = []
        for sensor, data in processed_data.items():
            combined.append(f"[{sensor.upper()}]: {data}")
        return "\n".join(combined)
    
    async def _process_agent_specific_data(self, result: str, processed_data: Dict) -> Dict:
        """Process agent-specific data based on YAML configuration"""
        custom_data = {}
        
        # Check if agent has focus-specific features (from YAML)
        if self.config.get('focus_features'):
            # Try to extract focus data from result - check if result contains JSON
            try:
                import json
                import re
                
                # The result might be formatted as "ACTIVITY: {JSON}" or just "{JSON}"
                json_text = result
                
                # Remove "ACTIVITY:" prefix if present
                if result.startswith("ACTIVITY:"):
                    json_text = result[9:].strip()
                
                # Look for JSON in the text
                json_match = re.search(r'\{.*\}', json_text, re.DOTALL)
                if json_match:
                    focus_data = json.loads(json_match.group(0))
                    
                    # Update agent's custom data (for get_focus_summary)
                    if 'focus_metrics' not in self.custom_data:
                        self.custom_data['focus_metrics'] = []
                    
                    self.custom_data['focus_metrics'].append({
                        'timestamp': time.time(),
                        'focus_level': focus_data.get('focus_level', 'medium'),
                        'productivity_score': focus_data.get('productivity_score', 0.5),
                        'category': focus_data.get('category', 'neutral'),
                        'suggestion': focus_data.get('suggestion', '')
                    })
                    
                    # Keep only last 50 entries
                    if len(self.custom_data['focus_metrics']) > 50:
                        self.custom_data['focus_metrics'] = self.custom_data['focus_metrics'][-50:]
                    
                    custom_data = focus_data
                    logger.info(f"📊 Focus data processed: {focus_data.get('focus_level')} focus, {focus_data.get('productivity_score', 0.0):.2f} productivity")
                else:
                    logger.warning(f"No JSON found in result: {result[:100]}...")
                    
            except Exception as e:
                logger.warning(f"Failed to parse focus data from: {result[:100]}... Error: {e}")
                # Fallback for non-JSON results
                custom_data = {"type": "activity_only", "data": result}
        
        return custom_data
    
    def get_focus_summary(self) -> Dict[str, Any]:
        """Get current focus session summary from custom_data"""
        focus_metrics = self.custom_data.get('focus_metrics', [])
        
        if not focus_metrics:
            return {
                'productivity_score': 0.0,
                'focus_sessions': 0,
                'distractions': 0,
                'total_focus_time': 0,
                'status': 'no_data'
            }
        
        recent_metrics = focus_metrics[-10:]  # Last 10 entries
        avg_productivity = sum(m.get('productivity_score', 0.0) for m in recent_metrics) / len(recent_metrics)
        
        focus_levels = [m.get('focus_level', 'medium') for m in recent_metrics]
        dominant_focus = max(set(focus_levels), key=focus_levels.count) if focus_levels else 'medium'
        
        # Calculate focus sessions (consecutive periods of medium/high focus)
        focus_sessions = 0
        in_focus_session = False
        for metric in focus_metrics:
            if metric.get('focus_level') in ['medium', 'high']:
                if not in_focus_session:
                    focus_sessions += 1
                    in_focus_session = True
            else:
                in_focus_session = False
        
        # Calculate total focus time (approximate)
        focus_entries = [m for m in focus_metrics if m.get('focus_level') in ['medium', 'high']]
        total_focus_time = len(focus_entries) * 60  # 60 seconds per entry
        
        # Count distractions
        distractions = len([m for m in focus_metrics if m.get('category') == 'distraction'])
        
        return {
            'productivity_score': avg_productivity,
            'focus_sessions': focus_sessions,
            'distractions': distractions,
            'total_focus_time': total_focus_time,
            'dominant_focus_level': dominant_focus,
            'total_sessions': len(focus_metrics),
            'current_suggestion': recent_metrics[-1].get('suggestion', '') if recent_metrics else ''
        }
    
    # Legacy method for backward compatibility with API
    async def process_image(self, image_data: str, timestamp: str) -> Dict[str, Any]:
        """Legacy method - delegates to process_input"""
        input_data = {"screen": {"image": image_data}}
        result = await self.process_input(input_data, timestamp)
        
        # Convert to legacy format
        return {
            "activity": result["result"],
            "confidence": result["confidence"],
            "processing_time": result["processing_time"],
            "agent_reasoning": result["agent_reasoning"],
            "timestamp": timestamp
        }
    
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


# Type alias for backward compatibility
ActivityTrackingAgent = ConfigurableAgent

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