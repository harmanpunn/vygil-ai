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
from typing import Dict, List, Optional

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

from agent import ConfigLoader, LLMProcessor, MCPClient, execute_autonomous_code
from agent_manager import AgentManager

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
mcp_client = None
agent_manager = None
activity_logs: List[ActivityLog] = []

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    global config, llm_processor, mcp_client, agent_manager
    
    try:
        # Initialize Agent Manager first
        agent_manager = AgentManager()
        
        # Load configuration
        config_path = os.path.join(os.path.dirname(__file__), '..', 'agent', 'config', 'activity-tracking-agent.yaml')
        config = ConfigLoader.load_config(config_path)
        
        # Initialize LLM processor
        llm_processor = LLMProcessor(config)
        
        # Initialize MCP client
        mcp_client = MCPClient(config)
        
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
        agent_ready=llm_processor is not None and mcp_client is not None and agent_manager is not None
    )

@app.post("/api/process-activity", response_model=ActivityResponse)
async def process_activity(request: ActivityRequest):
    """Process screen capture and return activity classification - Truly Agentic Approach"""
    try:
        if not agent_manager:
            raise HTTPException(status_code=503, detail="Agent manager not ready")
        
        # Validate image data
        try:
            image_data = base64.b64decode(request.image)
            logger.info(f"🖼️ Received image data: {len(image_data)} bytes")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 image: {e}")
        
        # Get currently selected agent
        current_agent = agent_manager.get_current_agent()
        current_agent_info = agent_manager.get_current_agent_info()
        
        if not current_agent:
            raise HTTPException(status_code=503, detail="No agent selected")
        
        # Prepare image data for agent
        image_base64 = request.image if request.image.startswith('data:') else f"data:image/png;base64,{request.image}"
        
        logger.info(f"🤖 Delegating to agent: {current_agent_info['name']} ({current_agent_info['id']})")
        
        # 🚀 TRULY AGENTIC: Let the selected agent decide everything autonomously
        result = await current_agent.process_image(image_base64, request.timestamp)
        
        # Log activity for API tracking
        activity_log = ActivityLog(
            id=str(len(activity_logs) + 1),
            activity=result["activity"],
            confidence=result["confidence"],
            timestamp=result["timestamp"]
        )
        activity_logs.append(activity_log)
        
        # Keep only last 100 activities
        if len(activity_logs) > 100:
            activity_logs.pop(0)
        
        logger.info(f"✅ Agent completed: {result['activity']} (confidence: {result['confidence']:.2f}, time: {result['processing_time']:.1f}s)")
        
        return ActivityResponse(
            activity=result["activity"],
            confidence=result["confidence"],
            timestamp=result["timestamp"],
            processing_time=result["processing_time"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Activity processing failed: {e}")
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
    """Focus-specific metrics - delegated to current agent"""
    if not agent_manager:
        raise HTTPException(status_code=503, detail="Agent manager not initialized")
    
    current_agent = agent_manager.get_current_agent()
    current_agent_info = agent_manager.get_current_agent_info()
    
    if not current_agent_info:
        raise HTTPException(status_code=503, detail="No agent selected")
    
    # Let any agent provide its metrics (focus or otherwise)
    if hasattr(current_agent, 'get_focus_summary'):
        focus_summary = current_agent.get_focus_summary()
        return {
            "message": "Agent metrics retrieved",
            "summary": focus_summary,
            "agent": current_agent_info.get('name'),
            "agent_type": current_agent_info.get('id')
        }
    else:
        return {
            "message": "Agent does not provide focus metrics",
            "active_agent": current_agent_info.get('name'),
            "agent_type": current_agent_info.get('id')
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