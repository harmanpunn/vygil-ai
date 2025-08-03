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
from typing import Dict, Optional, Tuple, Any
import yaml

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
    
    async def classify_activity(self, screen_text: str) -> Tuple[str, float]:
        """
        Classify user activity based on screen text
        Returns: (activity_description, confidence_score)
        """
        if not screen_text or len(screen_text.strip()) < 10:
            return "ACTIVITY: Insufficient screen content", 0.2
        
        # Truncate text to avoid token limits
        truncated_text = screen_text[:2000]
        if len(screen_text) > 2000:
            truncated_text += "..."
        
        user_prompt = f"""Analyze this screen content and identify the user's activity:

<Screen Content>
{truncated_text}
</Screen Content>

What is the user primarily doing? Follow the format rules exactly."""

        # Try primary provider first
        primary_provider = self.llm_config.get('provider', 'openai')
        if primary_provider in self.clients:
            try:
                response = await self._query_llm(primary_provider, user_prompt)
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
                    response = await self._query_llm(provider, user_prompt)
                    if response:
                        confidence = self._calculate_confidence(screen_text, response)
                        return self._format_response(response), confidence
                except Exception as e:
                    logger.warning(f"Fallback LLM provider {provider} failed: {e}")
        
        # All providers failed
        logger.error("All LLM providers failed")
        return "ACTIVITY: LLM analysis failed", 0.0
    
    async def _query_llm(self, provider: str, user_prompt: str) -> Optional[str]:
        """Query specific LLM provider"""
        try:
            if provider == 'openai':
                return await self._query_openai(user_prompt)
            elif provider == 'anthropic':
                return await self._query_anthropic(user_prompt)
            else:
                logger.warning(f"Unknown LLM provider: {provider}")
                return None
        except Exception as e:
            logger.error(f"Error querying {provider}: {e}")
            return None
    
    async def _query_openai(self, user_prompt: str) -> str:
        """Query OpenAI API"""
        client = self.clients['openai']
        model = self.llm_config.get('model', 'gpt-4o-mini')
        
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=self.llm_config.get('max_tokens', 50),
            temperature=self.llm_config.get('temperature', 0.1),
            timeout=self.llm_config.get('timeout', 10)
        )
        return response.choices[0].message.content.strip()
    
    async def _query_anthropic(self, user_prompt: str) -> str:
        """Query Anthropic Claude API"""
        client = self.clients['anthropic']
        model = self.llm_config.get('model', 'claude-3-haiku-20240307')
        
        response = await asyncio.to_thread(
            client.messages.create,
            model=model,
            max_tokens=self.llm_config.get('max_tokens', 50),
            system=self.system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            timeout=self.llm_config.get('timeout', 10)
        )
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
    """MCP client for communicating with MCP server tools"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.mcp_config = config.get('mcp_server', {})
        self.host = self.mcp_config.get('host', 'localhost')
        self.port = self.mcp_config.get('port', 3000)
        self.timeout = self.mcp_config.get('timeout', 5)
        
    async def call_tool(self, tool_name: str, arguments: Dict = None) -> Dict[str, Any]:
        """
        Call MCP server tool
        For MVP: simplified HTTP client implementation
        """
        if arguments is None:
            arguments = {}
            
        try:
            # For MVP: Mock implementation until MCP server is ready
            # In production, this would make actual HTTP/stdio calls to MCP server
            
            if tool_name == "screen_capture":
                return await self._mock_screen_capture()
            elif tool_name == "extract_text":
                return await self._mock_extract_text(arguments)
            elif tool_name == "log_activity":
                return await self._mock_log_activity(arguments)
            elif tool_name == "get_recent_activities":
                return await self._mock_get_recent_activities(arguments)
            else:
                return {"success": False, "error": f"Unknown tool: {tool_name}"}
                
        except Exception as e:
            logger.error(f"MCP tool call failed for {tool_name}: {e}")
            return {"success": False, "error": str(e)}
    
    async def _mock_screen_capture(self) -> Dict[str, Any]:
        """Mock screen capture for MVP testing"""
        logger.debug("Mock screen capture called")
        return {
            "success": True,
            "image_base64": "mock_base64_image_data",
            "timestamp": datetime.now().isoformat()
        }
    
    async def _mock_extract_text(self, arguments: Dict) -> Dict[str, Any]:
        """Mock OCR text extraction for MVP testing"""
        logger.debug("Mock OCR extraction called")
        
        # Simulate different screen content for testing
        mock_texts = [
            "VS Code - Python development environment with terminal open",
            "Chrome browser - reading technical documentation on GitHub",
            "Slack conversation about project updates and team coordination",
            "Excel spreadsheet with quarterly financial data and charts",
            "Zoom video call - team standup meeting in progress",
            "Terminal window running git commands and code compilation",
            "Email client composing message to client about project status"
        ]
        
        import random
        mock_text = random.choice(mock_texts)
        
        return {
            "success": True,
            "text": mock_text,
            "confidence": 0.85,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _mock_log_activity(self, arguments: Dict) -> Dict[str, Any]:
        """Mock activity logging for MVP testing"""
        description = arguments.get('description', 'Unknown activity')
        confidence = arguments.get('confidence', 0.0)
        
        logger.info(f"Activity logged: {description} (confidence: {confidence:.2f})")
        
        return {
            "success": True,
            "logged_at": datetime.now().isoformat(),
            "activity_id": f"activity_{int(time.time())}"
        }
    
    async def _mock_get_recent_activities(self, arguments: Dict) -> Dict[str, Any]:
        """Mock recent activities retrieval for MVP testing"""
        limit = arguments.get('limit', 5)
        
        mock_activities = [
            {"timestamp": "2024-01-01T10:30:00", "description": "ACTIVITY: Coding in Python"},
            {"timestamp": "2024-01-01T10:25:00", "description": "ACTIVITY: Reading documentation"},
            {"timestamp": "2024-01-01T10:20:00", "description": "ACTIVITY: Email communication"},
        ]
        
        return {
            "success": True,
            "activities": mock_activities[:limit],
            "total_count": len(mock_activities)
        }


class ActivityTrackingAgent:
    """Main activity tracking agent implementing the MVP flow"""
    
    def __init__(self, config_path: str = "config/activity-tracking-agent.yaml"):
        self.config = ConfigLoader.load_config(config_path)
        self.agent_config = self.config.get('agent', {})
        
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
            
            # Step 3: Classify activity using LLM
            activity_description, confidence = await self.llm_processor.classify_activity(screen_text)
            
            # Step 4: Log activity (via MCP server)
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