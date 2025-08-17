#!/usr/bin/env python3
"""
Vygil FastAPI Backend - Activity Processing API

Converts the agent.py logic into REST API endpoints for frontend integration.
Handles image processing, MCP communication, and activity logging.
"""

import asyncio
import base64
import json
import logging
import os
import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add agent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'agent'))

from agent import ConfigLoader, LLMProcessor, MCPClient, execute_autonomous_code, get_memory
from agent_manager import AgentManager  # type: ignore
from generic_agent import GenericAgent  # type: ignore

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("vygil-api")

# Pydantic models for API requests/responses
class ActivityRequest(BaseModel):
    image: str  # base64 encoded image
    timestamp: str

class ActivityResponse(BaseModel):
    activity: str
    confidence: float
    timestamp: str
    processing_time: float

class ActivityLog(BaseModel):
    id: str
    activity: str
    confidence: float
    timestamp: str

class ScreenshotRequest(BaseModel):
    filename: str
    imageData: str  # base64 encoded image

class ScreenshotResponse(BaseModel):
    success: bool
    filepath: str
    message: str

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    agent_ready: bool

class AgentInfo(BaseModel):
    id: str
    name: str
    description: str
    features: List[str]

class AgentListResponse(BaseModel):
    agents: List[AgentInfo]

class AgentSelectionRequest(BaseModel):
    agent_id: str

# Global components
app = FastAPI(
    title="Vygil Activity Tracking API",
    description="AI-powered activity monitoring backend",
    version="1.0.0"
)

# CORS configuration for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:3000"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
config = None
llm_processor = None
agent_manager = None
activity_logs: List[ActivityLog] = []

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    global config, llm_processor, agent_manager
    
    try:
        # Initialize Agent Manager first
        agent_manager = AgentManager()
        
        # Load configuration
        config_path = os.path.join(os.path.dirname(__file__), '..', 'agent', 'config', 'activity-tracking-agent.yaml')
        config = ConfigLoader.load_config(config_path)
        
        # Initialize LLM processor
        llm_processor = LLMProcessor(config)
        
        logger.info("Vygil API server started successfully")
        logger.info(f"Discovered {len(agent_manager.get_available_agents())} agents")
        
    except Exception as e:
        logger.error(f"Failed to initialize server: {e}")
        raise

@app.get("/health", response_model=HealthResponse)
@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        agent_ready=llm_processor is not None and agent_manager is not None
    )

@app.post("/api/process-activity", response_model=ActivityResponse)
async def process_activity(request: ActivityRequest):
    """Process screen capture and return activity classification"""
    start_time = asyncio.get_event_loop().time()
    
    try:
        if not llm_processor:
            raise HTTPException(status_code=503, detail="Server not ready")

        # Resolve current agent YAML and create GenericAgent
        current_agent_info = agent_manager.get_current_agent_info()
        if current_agent_info:
            agent_yaml_path = current_agent_info.get('config_file')
            generic_agent = GenericAgent.from_yaml(agent_yaml_path)
            logger.info(f"Using agent from YAML: {current_agent_info['name']} ({current_agent_info['id']})")
        else:
            default_yaml = os.path.join(os.path.dirname(__file__), '..', 'agent', 'config', 'activity-tracking-agent.yaml')
            generic_agent = GenericAgent.from_yaml(default_yaml)
            logger.warning("No current agent selected, using default YAML")

        # Prepare inputs based on YAML-declared sensors
        inputs: Dict[str, Any] = {}
        if 'screen' in generic_agent.sensors:
            inputs['screen'] = {"image": request.image}

        # Execute pipeline
        activity, confidence, raw = await generic_agent.plan_and_act(inputs)
        
        logger.info(f"🎯 Final result: activity='{activity}', confidence={confidence}")
        
        # Execute autonomous code if defined in YAML
        autonomous_code = generic_agent.config.get('code', '')
        agent_id = generic_agent.agent_id
        if autonomous_code:
            # Prefer compact JSON payload if raw contains a JSON object
            activity_for_memory = activity
            if raw:
                try:
                    import re as _re
                    _m = _re.search(r"\{.*\}", raw, flags=_re.DOTALL)
                    if _m:
                        _obj = json.loads(_m.group(0))
                        # Normalize fields
                        fl = str(_obj.get('focus_level', 'medium')).lower()
                        if fl not in ['low', 'medium', 'high']:
                            fl = 'medium'
                        try:
                            ps = float(_obj.get('productivity_score', 0.5))
                        except Exception:
                            ps = 0.5
                        if ps < 0.0:
                            ps = 0.0
                        if ps > 1.0:
                            ps = 1.0
                        cat = str(_obj.get('category', 'neutral')).lower()
                        cat_map = {'productive': 'PRODUCTIVE', 'neutral': 'NEUTRAL', 'distraction': 'DISTRACTION'}
                        cat_norm = cat_map.get(cat, 'NEUTRAL')
                        suggestion = str(_obj.get('suggestion', ''))
                        activity_short = str(_obj.get('activity', ''))
                        _clean = {
                            'focus_level': fl,
                            'productivity_score': round(ps, 2),
                            'category': cat_norm,
                            'suggestion': suggestion,
                            'activity': activity_short,
                        }
                        activity_for_memory = json.dumps(_clean, separators=(',', ':'))
                except Exception:
                    pass
            execute_autonomous_code(autonomous_code, activity_for_memory, agent_id)
        
        # Calculate processing time
        processing_time = asyncio.get_event_loop().time() - start_time
        
        # Log activity
        activity_log = ActivityLog(
            id=str(len(activity_logs) + 1),
            activity=activity,
            confidence=confidence,
            timestamp=request.timestamp
        )
        activity_logs.append(activity_log)
        
        # Keep only last 100 activities
        if len(activity_logs) > 100:
            activity_logs.pop(0)
        
        logger.info(f"Processed activity: {activity} (confidence: {confidence:.2f})")
        
        return ActivityResponse(
            activity=activity,
            confidence=confidence,
            timestamp=request.timestamp,
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Activity processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/api/activities", response_model=List[ActivityLog])
async def get_activities(limit: int = 50):
    """Get recent activity logs"""
    return activity_logs[-limit:] if activity_logs else []

@app.delete("/api/activities")
async def clear_activities():
    """Clear all activity logs"""
    global activity_logs
    activity_logs.clear()
    return {"message": "Activity logs cleared"}

@app.post("/api/save-screenshot", response_model=ScreenshotResponse)
async def save_screenshot(request: ScreenshotRequest):
    """Save screenshot to local directory for reference"""
    try:
        # Get screenshots directory from environment variable
        screenshots_path = os.getenv("SCREENSHOTS_DIR", "./screenshots")
        screenshots_dir = Path(screenshots_path).resolve()
        screenshots_dir.mkdir(exist_ok=True)
        
        # Decode base64 image
        image_data = base64.b64decode(request.imageData)
        
        # Save to file
        file_path = screenshots_dir / request.filename
        with open(file_path, 'wb') as f:
            f.write(image_data)
        
        logger.info(f"💾 Screenshot saved: {file_path}")
        
        return ScreenshotResponse(
            success=True,
            filepath=str(file_path),
            message=f"Screenshot saved successfully: {request.filename}"
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to save screenshot: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save screenshot: {str(e)}")

@app.get("/api/stats")
async def get_stats():
    """Get activity statistics"""
    if not activity_logs:
        return {
            "total_activities": 0,
            "average_confidence": 0.0,
            "session_start": None
        }
    
    total = len(activity_logs)
    avg_confidence = sum(log.confidence for log in activity_logs) / total
    session_start = activity_logs[0].timestamp if activity_logs else None
    
    return {
        "total_activities": total,
        "average_confidence": avg_confidence,
        "session_start": session_start
    }

@app.get("/api/agents", response_model=AgentListResponse)
async def get_available_agents():
    """List all discovered agents"""
    if not agent_manager:
        raise HTTPException(status_code=503, detail="Agent manager not initialized")
    
    available_agents = agent_manager.get_available_agents()
    agents_list = []
    
    for agent_id, agent_info in available_agents.items():
        agents_list.append(AgentInfo(
            id=agent_id,
            name=agent_info.get('name', agent_id),
            description=agent_info.get('description', 'No description'),
            features=agent_info.get('features', [])
        ))
    
    return AgentListResponse(agents=agents_list)

@app.post("/api/agents/select") 
async def select_agent(request: AgentSelectionRequest):
    """Switch active agent"""
    if not agent_manager:
        raise HTTPException(status_code=503, detail="Agent manager not initialized")
    
    agent_id = request.agent_id
    available_agents = agent_manager.get_available_agents()
    
    if agent_id not in available_agents:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    
    # Switch to the selected agent
    success = await agent_manager.select_agent(agent_id)
    if not success:
        raise HTTPException(status_code=500, detail=f"Failed to switch to agent '{agent_id}'")
    
    logger.info(f"Active agent switched to: {agent_id}")
    
    return {
        "message": f"Active agent switched to: {available_agents[agent_id]['name']}",
        "agent_id": agent_id,
        "agent_name": available_agents[agent_id]['name']
    }

@app.get("/api/focus/summary") 
async def get_focus_summary():
    """Return focus session summary for Focus Assistant using persisted memory."""
    if not agent_manager:
        raise HTTPException(status_code=503, detail="Agent manager not initialized")

    current_agent_info = agent_manager.get_current_agent_info()
    if not current_agent_info:
        return {"message": "No agent selected"}

    agent_id = current_agent_info.get('id', 'vygil-focus-assistant')
    agent_name = current_agent_info.get('name', 'Focus Assistant')

    # Only summarize for Focus Assistant
    if 'focus-assistant' not in agent_id:
        return {"message": "Focus assistant not active", "active_agent": agent_name}

    # Load loop interval from YAML for time estimates
    try:
        config_file = current_agent_info.get('config_file')
        cfg = ConfigLoader.load_config(config_file) if config_file else {}
        loop_interval = int(cfg.get('agent', {}).get('loop_interval', 60))
    except Exception:
        loop_interval = 60

    # Read persisted focus memory and compute summary
    memory_text = get_memory(agent_id)
    if not memory_text:
        return {"message": "No focus data yet", "agent": agent_name}

    lines = [ln.strip() for ln in memory_text.split('\n') if ln.strip()]

    # Parse lines (compact, one line per event):
    # [12:34 PM] Focus: high | Score: 0.85 | PRODUCTIVE | Coding
    import re
    focus_levels = []
    categories = []
    scores = []
    activities = []

    # Example line:
    # [01:05 AM] Focus: medium | Score: 0.50 | NEUTRAL | ACTIVITY: Coding and debugging
    pattern = re.compile(r"Focus:\s*(high|medium|low)\s*\|\s*Score:\s*([0-9]+\.?[0-9]*)\s*\|\s*([A-Z]+)\s*\|\s*(.*)$")

    for ln in lines:
        m = pattern.search(ln)
        if not m:
            continue
        level = m.group(1)
        try:
            score = float(m.group(2))
        except ValueError:
            score = 0.0
        category = m.group(3)
        activity_text = m.group(4).strip()

        focus_levels.append(level)
        scores.append(score)
        categories.append(category)
        activities.append(activity_text)

    if not scores:
        return {"message": "No focus data yet", "agent": agent_name}

    # Compute metrics with exponential moving average (EMA) to react faster but smooth noise
    if scores:
        alpha = 0.3  # smoothing factor
        ema = scores[0]
        for s in scores[1:]:
            ema = alpha * s + (1 - alpha) * ema
        avg_productivity = ema
    else:
        avg_productivity = 0.0

    # Focus sessions: detect consecutive runs of medium/high (true sessions)
    focus_sessions = 0
    in_session = False
    for lvl in focus_levels:
        if lvl in ['medium', 'high']:
            if not in_session:
                focus_sessions += 1
                in_session = True
        else:
            in_session = False

    # Distractions: count DISTRACTION category, with heuristic fallback by activity text
    distraction_keywords = ['instagram', 'facebook', 'twitter', 'tiktok', 'reddit', 'youtube', 'netflix']
    distractions = 0
    for cat, act in zip(categories, activities):
        if cat == 'DISTRACTION':
            distractions += 1
            continue
        act_lower = act.lower()
        if any(k in act_lower for k in distraction_keywords):
            distractions += 1

    # Total focus time approximation
    focus_entries = sum(1 for lvl in focus_levels if lvl in ['medium', 'high'])
    total_focus_time = focus_entries * loop_interval

    # Simple AI insight based on productivity
    if avg_productivity >= 0.75:
        suggestion = "Great focus! Keep it up."
    elif avg_productivity >= 0.5:
        suggestion = "Solid progress. Consider a short stretch break."
    elif avg_productivity >= 0.3:
        suggestion = "Attention drifting. Close distractions and refocus."
    else:
        suggestion = "Low focus detected. Take a 5-min reset, then resume."

    summary = {
        "productivity_score": avg_productivity,
        "focus_sessions": focus_sessions,
        "distractions": distractions,
        "total_focus_time": total_focus_time,
        "current_suggestion": suggestion
    }

    return {
        "message": "Focus summary retrieved",
        "summary": summary,
        "agent": agent_name
    }

# Serve static frontend files (for production)
if os.path.exists("../frontend/dist"):
    app.mount("/", StaticFiles(directory="../frontend/dist", html=True), name="static")

def main():
    """Run the FastAPI server"""
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()