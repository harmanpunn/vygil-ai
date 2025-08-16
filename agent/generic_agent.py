#!/usr/bin/env python3
"""
GenericAgent — YAML-driven agent executor

Loads agent configuration from YAML and executes a simple pipeline:
  1) tool_selection (LLM chooses a tool)
  2) execute_tools (invoke MCP tools)
  3) analyze_results (LLM classifies)

This avoids agent-specific subclasses and keeps onboarding YAML-only.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

# Local imports are resolved by api/main.py which adds the agent dir to sys.path
from agent import ConfigLoader, LLMProcessor  # type: ignore
from mcp_client import MCPClient  # type: ignore


logger = logging.getLogger("generic-agent")


class GenericAgent:
    """Generic, YAML-configured agent executor."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agent_config = config.get("agent", {})
        self.agent_id = self.agent_config.get("id", "generic-agent")
        self.sensors: List[str] = self.agent_config.get("sensors", [])
        self.pipeline_steps: List[str] = config.get("pipeline", [])
        self.prompts: Dict[str, str] = config.get("prompts", {})
        self.llm_processor = LLMProcessor(config)

    @classmethod
    def from_yaml(cls, path: str) -> "GenericAgent":
        config = ConfigLoader.load_config(path)
        return cls(config)

    def available_tool_names(self) -> List[str]:
        tools = self.config.get("mcp_server", {}).get("tools", {})
        return list(tools.keys())

    async def select_tool(self, context: Dict[str, Any]) -> str:
        """Select a tool using an LLM prompt defined in YAML; fallback to extract_text."""
        prompt_template = self.prompts.get("tool_selection")
        if not prompt_template:
            return "extract_text" if "extract_text" in self.available_tool_names() else (self.available_tool_names()[0] if self.available_tool_names() else "")

        available_tools = self.available_tool_names()
        tool_descriptions = []
        for name in available_tools:
            desc = self.config.get("mcp_server", {}).get("tools", {}).get(name, {}).get("description", "")
            tool_descriptions.append(f"- {name}: {desc}")

        # Basic context for prompt injection without interfering with JSON braces
        screen_present = bool(context.get("screen", {}).get("imageData"))
        filled_prompt = prompt_template
        filled_prompt = filled_prompt.replace("{available_tools}", "\n".join(tool_descriptions))
        filled_prompt = filled_prompt.replace("{has_screen}", str(screen_present).lower())

        try:
            # Use internal LLM call to preserve minimal changes
            response_text = await self.llm_processor._query_llm(
                self.llm_processor.llm_config.get("provider", "openai"),
                user_prompt=filled_prompt,
                system_prompt=""
            )
            if not response_text:
                return available_tools[0] if available_tools else ""

            text = response_text.strip()
            if not text.startswith("{"):
                start = text.find("{")
                end = text.rfind("}") + 1
                if start != -1 and end > start:
                    text = text[start:end]

            decision = json.loads(text)
            chosen = decision.get("tool")
            if chosen in available_tools:
                return chosen
            return available_tools[0] if available_tools else ""
        except Exception as e:
            logger.warning(f"Tool selection failed, using fallback. Error: {e}")
            return available_tools[0] if available_tools else ""

    async def execute_tool(self, tool_name: str, mcp_client: MCPClient, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the selected tool with minimal argument mapping."""
        arguments: Dict[str, Any] = {}

        # Minimal mapping for common tools
        if tool_name == "extract_text":
            image_data = context.get("screen", {}).get("imageData")
            if image_data:
                arguments = {"imageData": image_data}

        result = await mcp_client.call_tool(tool_name, arguments)
        return result

    async def analyze_results(self, intermediate: Dict[str, Any]) -> Tuple[str, float, str]:
        """Analyze tool results with LLM, returning (activity, confidence, raw)."""
        screen_text = intermediate.get("text") or intermediate.get("result") or ""
        # Ask for raw to allow YAML code to store richer details
        try:
            result = await self.llm_processor.classify_activity(screen_text, self.agent_id, return_raw=True)  # type: ignore[arg-type]
            if isinstance(result, tuple) and len(result) == 3:
                activity, confidence, raw = result
                return activity, confidence, (raw or "")
        except TypeError:
            # Older signature fallback
            pass
        activity, confidence = await self.llm_processor.classify_activity(screen_text, self.agent_id)
        return activity, confidence, ""

    async def plan_and_act(self, inputs: Dict[str, Any]) -> Tuple[str, float, str]:
        """Run the YAML-declared pipeline with provided inputs (sensors)."""
        # Build context from sensors
        context: Dict[str, Any] = {}
        if "screen" in self.sensors and inputs.get("screen"):
            image_base64 = inputs["screen"].get("image")
            if image_base64 and not image_base64.startswith("data:"):
                image_base64 = f"data:image/png;base64,{image_base64}"
            context["screen"] = {"imageData": image_base64}

        mcp_client = MCPClient(self.config)
        try:
            await mcp_client.initialize()

            chosen_tool = None
            intermediate: Dict[str, Any] = {}

            for step in self.pipeline_steps:
                if step == "tool_selection":
                    chosen_tool = await self.select_tool(context)
                elif step == "execute_tools":
                    if chosen_tool:
                        tool_result = await self.execute_tool(chosen_tool, mcp_client, context)
                        # Normalize a few fields into intermediate
                        if tool_result.get("success"):
                            if "text" in tool_result:
                                intermediate["text"] = tool_result.get("text")
                            elif "result" in tool_result:
                                intermediate["result"] = tool_result.get("result")
                        else:
                            logger.warning(f"Selected tool '{chosen_tool}' failed: {tool_result.get('error')}")
                    else:
                        logger.warning("No tool selected before execute_tools step")
                elif step == "analyze_results":
                    activity, confidence, raw = await self.analyze_results(intermediate)
                    return activity, confidence, raw

            # If pipeline didn't conclude with analyze_results, do a best-effort analysis
            activity, confidence, raw = await self.analyze_results(intermediate)
            return activity, confidence, raw
        finally:
            await mcp_client.cleanup()

    async def get_focus_summary(self) -> Dict[str, Any]:
        """Focus summary via MCP if tools exist; graceful fallback otherwise."""
        mcp_client = MCPClient(self.config)
        try:
            await mcp_client.initialize()
            tools = await mcp_client.list_tools()
            tool_names = {t.get("name") for t in tools}
            if "get_focus_history" in tool_names:
                history = await mcp_client.call_tool("get_focus_history", {"limit": 10})
                return {
                    "message": "Focus summary retrieved",
                    "summary": history.get("history") if history.get("success") else [],
                    "agent": self.agent_config.get("name", self.agent_id)
                }
            return {
                "message": "Focus summary not available",
                "agent": self.agent_config.get("name", self.agent_id)
            }
        finally:
            await mcp_client.cleanup()


