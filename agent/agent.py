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

# Local imports
from mcp_client import MCPClient

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
    
    async def classify_activity(self, screen_text: str, agent_id: str, return_raw: bool = False) -> Tuple[str, float] | Tuple[str, float, str]:
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

        raw_response_text: Optional[str] = None

        # Try primary provider first
        primary_provider = self.llm_config.get('provider', 'openai')
        if primary_provider in self.clients:
            try:
                response = await self._query_llm(primary_provider, user_prompt, system_prompt)
                if response:
                    raw_response_text = response
                    confidence = self._calculate_confidence(screen_text, response)
                    formatted = self._format_response(response, agent_id)
                    if return_raw:
                        return formatted, confidence, response
                    return formatted, confidence
            except Exception as e:
                logger.warning(f"Primary LLM provider {primary_provider} failed: {e}")
        
        # Try fallback providers
        for fallback in self.llm_config.get('fallback_providers', []):
            provider = fallback.get('provider')
            if provider in self.clients:
                try:
                    response = await self._query_llm(provider, user_prompt, system_prompt)
                    if response:
                        raw_response_text = response
                        confidence = self._calculate_confidence(screen_text, response)
                        formatted = self._format_response(response, agent_id)
                        if return_raw:
                            return formatted, confidence, response
                        return formatted, confidence
                except Exception as e:
                    logger.warning(f"Fallback LLM provider {provider} failed: {e}")
        
        # All providers failed
        logger.error("All LLM providers failed")
        if return_raw:
            return "ACTIVITY: LLM analysis failed", 0.0, (raw_response_text or "")
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
        
        # Default formatting for regular activity tracker
        if not response.startswith("ACTIVITY:"):
            return f"ACTIVITY: {response}"
        return response


class ActivityTrackingAgent:
    """Deprecated: retained for import compatibility. Use GenericAgent instead."""
    def __init__(self, config_path: str = "config/activity-tracking-agent.yaml"):
        self.config = ConfigLoader.load_config(config_path)
        self.agent_config = self.config.get('agent', {})
        self.agent_id = self.agent_config.get('id', 'vygil-activity-tracker')
        self.llm_processor = LLMProcessor(self.config)
        self.mcp_client = MCPClient(self.config)
        self.running = False
        self.loop_interval = self.agent_config.get('loop_interval', 20)
        self.max_retries = self.agent_config.get('max_retries', 3)
        self.consecutive_failures = 0
        self.total_iterations = 0
        self.successful_iterations = 0
        self.start_time = None
        logger.info("ActivityTrackingAgent is deprecated. Switch to GenericAgent.")


async def main():
    """Main entry point for the activity tracking agent"""
    logger.info("ActivityTrackingAgent is deprecated. Use API with GenericAgent.")
    return 0


class FocusAssistantAgent(ActivityTrackingAgent):
    """Deprecated wrapper to keep import compatibility. Use GenericAgent."""
    def __init__(self, config_path: str):
        super().__init__(config_path)


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