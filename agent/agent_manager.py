#!/usr/bin/env python3
"""
Agent Manager - Handles multiple agent configurations and selection
"""

import os
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

from agent import ConfigLoader
from generic_agent import GenericAgent

logger = logging.getLogger("agent-manager")

class AgentManager:
    """Manages multiple agents and their configurations"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(__file__).parent / config_dir
        self.agents: Dict[str, Any] = {}
        self.current_agent: Optional[Any] = None
        self.current_agent_id: Optional[str] = None
        
        # Discover available agents
        self.available_agents = self._discover_agents()
        
        # Auto-select default agent if none selected
        if self.available_agents and not self.current_agent_id:
            # Prefer focus assistant, fallback to activity tracker
            default_agent = "vygil-focus-assistant" if "vygil-focus-assistant" in self.available_agents else list(self.available_agents.keys())[0]
            logger.info(f"Auto-selecting default agent: {default_agent}")
            # Use sync version for initialization
            self._select_agent_sync(default_agent)
        
    def _discover_agents(self) -> Dict[str, Dict[str, Any]]:
        """Discover all available agent configurations"""
        agents = {}
        
        logger.info(f"Looking for agent configs in: {self.config_dir}")
        
        # Look for YAML config files
        for config_file in self.config_dir.glob("*-agent.yaml"):
            try:
                config = ConfigLoader.load_config(str(config_file))
                agent_info = config.get('agent', {})
                
                agent_id = agent_info.get('id', config_file.stem)
                agents[agent_id] = {
                    'name': agent_info.get('name', agent_id),
                    'description': agent_info.get('description', 'No description'),
                    'version': agent_info.get('version', '1.0.0'),
                    'config_file': str(config_file),
                    'config': config,
                    'loop_interval': agent_info.get('loop_interval', 20),
                    'features': self._extract_features(config)
                }
                
                logger.info(f"Discovered agent: {agents[agent_id]['name']} - {agents[agent_id]['description']}")
                logger.debug(f"Agent info loaded: {agents[agent_id]}")
                
            except Exception as e:
                logger.error(f"Failed to load agent config {config_file}: {e}")
                
        return agents
    
    def _extract_features(self, config: Dict) -> List[str]:
        """Extract key features from agent config"""
        features = []
        
        # Core features all agents have
        if config.get('memory') is not None:  # Memory section exists
            features.append('Memory Persistence')
        
        if '$MEMORY' in config.get('instructions', {}).get('system_prompt', ''):
            features.append('Context Injection')
            
        if config.get('code'):  # Autonomous code execution
            features.append('Autonomous Execution')
            
        if config.get('mcp_server'):  # MCP integration
            features.append('MCP Tools')
            
        if len(config.get('llm', {}).get('fallback_providers', [])) > 0:
            features.append('Multi-LLM Support')
        
        # Agent-specific features
        if 'focus_features' in config:
            features.extend(['Focus Tracking', 'Productivity Analysis', 'Distraction Alerts'])
        
        if config.get('monitoring', {}).get('confidence_scoring'):
            features.append('Confidence Scoring')
            
        if config.get('monitoring', {}).get('adaptive_prompts'):
            features.append('Adaptive Learning')
            
        return features
    
    def get_available_agents(self) -> Dict[str, Dict[str, Any]]:
        """Get list of all available agents"""
        return self.available_agents
    
    async def select_agent(self, agent_id: str) -> bool:
        """Select and initialize an agent"""
        if agent_id not in self.available_agents:
            logger.error(f"Agent {agent_id} not found")
            return False
            
        try:
            # Stop current agent if running
            if self.current_agent:
                logger.info(f"Stopping current agent: {self.current_agent_id}")
                if hasattr(self.current_agent, 'stop_monitoring'):
                    await self.current_agent.stop_monitoring()
            
            # Create new agent instance (GenericAgent)
            agent_info = self.available_agents[agent_id]
            config_file = agent_info['config_file']
            self.current_agent = GenericAgent.from_yaml(config_file)
            
            self.current_agent_id = agent_id
            logger.info(f"Selected agent: {agent_info['name']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to select agent {agent_id}: {e}")
            return False
    
    def _select_agent_sync(self, agent_id: str) -> bool:
        """Select and initialize an agent synchronously (for initialization)"""
        if agent_id not in self.available_agents:
            logger.error(f"Agent {agent_id} not found")
            return False
            
        try:
            # Create new agent instance (GenericAgent)
            agent_info = self.available_agents[agent_id]
            config_file = agent_info['config_file']
            self.current_agent = GenericAgent.from_yaml(config_file)
            
            self.current_agent_id = agent_id
            logger.info(f"Selected agent: {agent_info['name']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to select agent {agent_id}: {e}")
            return False
    
    def get_current_agent(self) -> Optional[Any]:
        """Get the currently active agent"""
        return self.current_agent
    
    def get_current_agent_info(self) -> Optional[Dict[str, Any]]:
        """Get info about the currently active agent"""
        if self.current_agent_id:
            agent_info = self.available_agents.get(self.current_agent_id)
            if agent_info:
                # Add the ID to the agent info
                agent_info_with_id = agent_info.copy()
                agent_info_with_id['id'] = self.current_agent_id
                return agent_info_with_id
        return None

