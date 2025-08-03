#!/usr/bin/env python3
"""
Test script for MCP server integration
Tests the full workflow: API -> MCP REST server -> OCR -> LLM
"""

import asyncio
import base64
import json
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def create_test_image_base64():
    """Create a small test image with text for OCR testing"""
    # This is a tiny PNG with "TEST" text (you can replace with actual image)
    # For now, we'll use a minimal PNG that might not have readable text
    test_png_base64 = """
iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==
""".strip()
    return f"data:image/png;base64,{test_png_base64}"

async def test_mcp_api_health():
    """Test MCP REST API health endpoint"""
    print("🏥 Testing MCP REST API health...")
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:3001/api/health")
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ MCP API healthy: {result}")
                return True
            else:
                print(f"❌ MCP API health check failed: {response.status_code}")
                return False
    except Exception as e:
        print(f"❌ MCP API connection failed: {e}")
        return False

async def test_mcp_ocr():
    """Test MCP REST API OCR endpoint"""
    print("🔤 Testing MCP OCR endpoint...")
    
    try:
        test_image = create_test_image_base64()
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "http://localhost:3001/api/ocr",
                json={"imageData": test_image}
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    print(f"✅ OCR successful:")
                    print(f"   - Text: '{result.get('text', '')}'")
                    print(f"   - Confidence: {result.get('confidence', 0):.2f}")
                    print(f"   - Processing time: {result.get('processingTime', 0)}ms")
                    return result.get("text", "")
                else:
                    print(f"❌ OCR failed: {result.get('error')}")
                    return None
            else:
                print(f"❌ OCR request failed: {response.status_code} - {response.text}")
                return None
                
    except Exception as e:
        print(f"❌ OCR test failed: {e}")
        return None

async def test_vygil_api():
    """Test Vygil API process-activity endpoint"""
    print("🎯 Testing Vygil API process-activity endpoint...")
    
    try:
        test_image = create_test_image_base64()
        # Remove data URL prefix for API
        image_b64 = test_image.split(',')[1] if ',' in test_image else test_image
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "http://localhost:8000/api/process-activity",
                json={
                    "image": image_b64,
                    "timestamp": "2024-01-01T10:30:00Z"
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Activity processing successful:")
                print(f"   - Activity: {result.get('activity')}")
                print(f"   - Confidence: {result.get('confidence', 0):.2f}")
                print(f"   - Processing time: {result.get('processing_time', 0):.2f}s")
                return True
            else:
                print(f"❌ Activity processing failed: {response.status_code} - {response.text}")
                return False
                
    except Exception as e:
        print(f"❌ Vygil API test failed: {e}")
        return False

async def test_full_workflow():
    """Test the complete workflow"""
    print("🚀 Testing complete MCP integration workflow...\n")
    
    # Test 1: MCP API Health
    mcp_healthy = await test_mcp_api_health()
    print()
    
    # Test 2: MCP OCR
    if mcp_healthy:
        ocr_text = await test_mcp_ocr()
        print()
    else:
        print("⚠️ Skipping OCR test - MCP server not healthy")
        ocr_text = None
        print()
    
    # Test 3: Vygil API (will use MCP or fallback)
    vygil_success = await test_vygil_api()
    print()
    
    # Summary
    print("📊 Test Summary:")
    print(f"   - MCP REST API Health: {'✅' if mcp_healthy else '❌'}")
    print(f"   - MCP OCR Processing: {'✅' if ocr_text is not None else '❌'}")
    print(f"   - Vygil API Integration: {'✅' if vygil_success else '❌'}")
    
    if mcp_healthy and vygil_success:
        print("\n🎉 Full integration working! The workflow is:")
        print("   Frontend → Vygil API → MCP REST API → OCR → LLM → Response")
    elif vygil_success:
        print("\n⚠️ Vygil API working with fallback (MCP server not available)")
        print("   Frontend → Vygil API → Fallback OCR → LLM → Response")
    else:
        print("\n❌ Integration not working - check server status")

def print_startup_instructions():
    """Print instructions for starting servers"""
    print("🚀 MCP Integration Test\n")
    print("Before running this test, make sure the following servers are running:\n")
    print("1. MCP REST API Server (port 3001):")
    print("   cd mcp-server")
    print("   npm run dev:api\n")
    print("2. Vygil API Server (port 8000):")
    print("   cd api")
    print("   python main.py\n")
    print("3. Optional - MCP Socket.IO Server (port 3000):")
    print("   cd mcp-server")
    print("   npm run dev\n")
    print("-" * 60)

async def main():
    """Run all tests"""
    print_startup_instructions()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("\nUsage:")
        print("  python test_mcp_integration.py          # Run full workflow test")
        print("  python test_mcp_integration.py --help   # Show this help")
        return
    
    await test_full_workflow()

if __name__ == "__main__":
    asyncio.run(main())