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

# Add agent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'agent'))

from agent import ConfigLoader, LLMProcessor, MCPClient

# Configure logging
logging.basicConfig(level=logging.INFO)
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
activity_logs: List[ActivityLog] = []

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    global config, llm_processor, mcp_client
    
    try:
        # Load configuration
        config_path = os.path.join(os.path.dirname(__file__), '..', 'agent', 'config', 'activity-tracking-agent.yaml')
        config = ConfigLoader.load_config(config_path)
        
        # Initialize LLM processor
        llm_processor = LLMProcessor(config)
        
        # Initialize MCP client
        mcp_client = MCPClient(config)
        
        logger.info("Vygil API server started successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize server: {e}")
        raise

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        agent_ready=llm_processor is not None and mcp_client is not None
    )

@app.post("/api/process-activity", response_model=ActivityResponse)
async def process_activity(request: ActivityRequest):
    """Process screen capture and return activity classification"""
    start_time = asyncio.get_event_loop().time()
    
    try:
        if not llm_processor or not mcp_client:
            raise HTTPException(status_code=503, detail="Server not ready")
        
        # Decode base64 image (for future OCR processing)
        try:
            image_data = base64.b64decode(request.image)
            logger.info(f"Received image data: {len(image_data)} bytes")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 image: {e}")
        
        # For MVP, simulate OCR text extraction with variety
        # In production, this would use MCP server's OCR tool
        import random
        mock_ocr_texts = [
            "VS Code editor with Python file open, showing function definitions and import statements",
            "Chrome browser showing Stack Overflow page with coding questions and answers",
            "Terminal window with git commands and file listings visible",
            "Slack application with team chat messages and notifications",
            "Gmail inbox with emails from colleagues and project updates",
            "Notion page with meeting notes and task lists",
            "YouTube video player showing programming tutorial",
            "Figma design interface with UI mockups and components",
            "Discord chat with gaming channel and voice call active",
            "Microsoft Word document with project documentation"
        ]
        mock_ocr_text = random.choice(mock_ocr_texts)
        logger.info(f"🔤 Using mock OCR text: {mock_ocr_text}")
        
        # Process with LLM
        logger.info("🤖 Calling LLM processor...")
        activity, confidence = await llm_processor.classify_activity(mock_ocr_text)
        logger.info(f"🎯 LLM returned: activity='{activity}', confidence={confidence}")
        
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
        # Create screenshots directory if it doesn't exist
        screenshots_dir = Path("/Users/harman/Home/Projects/vygil-ai/screenshots")
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