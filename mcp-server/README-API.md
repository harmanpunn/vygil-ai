# MCP REST API Server

A simple REST API server for OCR processing, designed to work alongside the main Socket.IO MCP server.

## Overview

This API server provides HTTP endpoints for the Vygil agent integration:
- **Port**: 3001 (different from Socket.IO server on 3000)
- **Purpose**: Simple request/response OCR processing
- **Complementary**: Works alongside the original Socket.IO server for future streaming features

## Quick Start

### 1. Install Dependencies
```bash
cd mcp-server
npm install
```

### 2. Run API Server Only
```bash
# Development mode
npm run dev:api

# Production mode
npm run build
npm run start:api
```

### 3. Run Both Servers (Socket.IO + REST API)
```bash
npm run dev:both
```

## API Endpoints

### Health Check
```http
GET /api/health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "MCP REST API Server", 
  "version": "1.0.0",
  "timestamp": "2024-01-01T10:30:00.000Z"
}
```

### OCR Processing
```http
POST /api/ocr
Content-Type: application/json

{
  "imageData": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
}
```

**Response:**
```json
{
  "success": true,
  "text": "Extracted text from the image",
  "confidence": 0.89,
  "processingTime": 1250,
  "metadata": {
    "textLength": 42,
    "wordCount": 7,
    "timestamp": "2024-01-01T10:30:00.000Z"
  }
}
```

**Error Response:**
```json
{
  "success": false,
  "error": "OCR processing failed",
  "details": "Invalid image format",
  "processingTime": 150
}
```

## Integration with Vygil API

The Vygil API (`api/main.py`) calls this OCR endpoint:

```python
# api/main.py - process_activity function
async with httpx.AsyncClient() as client:
    response = await client.post("http://localhost:3001/api/ocr", json={
        "imageData": base64_image
    })
    result = response.json()
    extracted_text = result.get("text", "")
```

## Testing

### Test OCR Endpoint
```bash
curl -X POST http://localhost:3001/api/ocr \
  -H "Content-Type: application/json" \
  -d '{"imageData": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="}'
```

### Test Health Check
```bash
curl http://localhost:3001/api/health
```

## Architecture

```
┌─────────────────┐    ┌─────────────────┐
│   Socket.IO     │    │   REST API      │
│   Server        │    │   Server        │
│   (port 3000)   │    │   (port 3001)   │
│                 │    │                 │
│ • Screen Share  │    │ • OCR Endpoint  │
│ • Real-time     │    │ • Health Check  │
│ • Future Video  │    │ • Agent API     │
└─────────────────┘    └─────────────────┘
        │                       │
        │                       │
        ▼                       ▼
┌─────────────────────────────────────────┐
│          Shared Dependencies            │
│     • Tesseract.js • Express.js         │
└─────────────────────────────────────────┘
```

## Configuration

Environment variables:
```bash
API_PORT=3001              # REST API server port
PORT=3000                  # Socket.IO server port (original)
```

## Development

- **Original server**: `src/server.ts` (Socket.IO, streaming, sessions)
- **New API server**: `src/api-server.ts` (REST endpoints, OCR processing)
- **Shared config**: `src/config.ts`

Both servers can run simultaneously for different use cases:
- Use Socket.IO for real-time streaming features
- Use REST API for simple OCR requests from agents

## Future Enhancements

- Add authentication/API keys
- Rate limiting for OCR requests  
- Batch processing for multiple images
- Additional image processing options
- Integration with cloud OCR services