import express from 'express';
import cors from 'cors';
import { Request, Response } from 'express';
import Tesseract from 'tesseract.js';
import { config } from './config';

/**
 * MCP REST API Server - Simple HTTP endpoints for OCR processing
 * 
 * This server provides REST API endpoints for the Vygil agent integration,
 * while keeping the original Socket.IO server (server.ts) for future streaming features.
 * 
 * Endpoints:
 * - POST /api/ocr - Process image and return OCR text
 * - GET /api/health - Health check
 */

// Initialize Express app
const app = express();

// Middleware
app.use(cors());
app.use(express.json({ limit: '10mb' })); // Increase limit for base64 images

// Health check endpoint
app.get('/api/health', (req: Request, res: Response) => {
  res.json({
    status: 'healthy',
    service: 'MCP REST API Server',
    version: '1.0.0',
    timestamp: new Date().toISOString()
  });
});

// OCR processing endpoint
app.post('/api/ocr', async (req: Request, res: Response) => {
  const startTime = Date.now();
  
  try {
    const { imageData } = req.body;
    
    // Validate input
    if (!imageData) {
      return res.status(400).json({ 
        success: false, 
        error: 'Image data is required in request body' 
      });
    }
    
    console.log(`📸 OCR request received: ${imageData.length} characters of image data`);
    
    // Process the image data for OCR
    let base64Data: string;
    
    // Handle both raw base64 and data URL formats
    if (imageData.includes(',')) {
      base64Data = imageData.split(',')[1]; // Remove data URL prefix
    } else {
      base64Data = imageData;
    }
    
    if (!base64Data) {
      return res.status(400).json({ 
        success: false, 
        error: 'Invalid image data format. Expected base64 string or data URL.' 
      });
    }
    
    // Convert base64 to buffer
    let imageBuffer: Buffer;
    try {
      imageBuffer = Buffer.from(base64Data, 'base64');
      console.log(`🖼️ Image buffer created: ${imageBuffer.length} bytes`);
    } catch (error) {
      return res.status(400).json({
        success: false,
        error: 'Failed to decode base64 image data'
      });
    }
    
    // Perform OCR using Tesseract.js
    console.log('🔤 Starting OCR processing...');
    const result = await Tesseract.recognize(
      imageBuffer,
      'eng',
      { 
        logger: (m) => {
          if (m.status === 'recognizing text') {
            console.log(`OCR Progress: ${Math.round(m.progress * 100)}%`);
          }
        }
      }
    );
    
    const extractedText = result.data.text.trim();
    const confidence = result.data.confidence / 100; // Convert to 0-1 scale
    const processingTime = Date.now() - startTime;
    
    console.log(`✅ OCR completed in ${processingTime}ms:`);
    console.log(`   - Text length: ${extractedText.length} characters`);
    console.log(`   - Confidence: ${(confidence * 100).toFixed(1)}%`);
    console.log(`   - Preview: "${extractedText.substring(0, 100)}${extractedText.length > 100 ? '...' : ''}"`);  
    
    // Return success response
    res.json({
      success: true,
      text: extractedText,
      confidence: confidence,
      processingTime: processingTime,
      metadata: {
        textLength: extractedText.length,
        wordCount: extractedText.split(/\s+/).filter(word => word.length > 0).length,
        timestamp: new Date().toISOString()
      }
    });
    
  } catch (error) {
    const processingTime = Date.now() - startTime;
    console.error('❌ OCR processing failed:', error);
    
    res.status(500).json({ 
      success: false, 
      error: 'OCR processing failed',
      details: error instanceof Error ? error.message : 'Unknown error',
      processingTime: processingTime
    });
  }
});

// Error handling middleware
app.use((error: Error, req: Request, res: Response, next: any) => {
  console.error('Unhandled error:', error);
  res.status(500).json({
    success: false,
    error: 'Internal server error',
    details: error.message
  });
});

// 404 handler
app.use((req: Request, res: Response) => {
  res.status(404).json({
    success: false,
    error: 'Endpoint not found',
    availableEndpoints: [
      'GET /api/health',
      'POST /api/ocr'
    ]
  });
});

// Start the server
const PORT = process.env.API_PORT || 3001; // Different port from Socket.IO server
const server = app.listen(PORT, () => {
  console.log(`🚀 MCP REST API Server running on port ${PORT}`);
  console.log(`📍 Health check: http://localhost:${PORT}/api/health`);
  console.log(`🔤 OCR endpoint: http://localhost:${PORT}/api/ocr`);
});

// Graceful shutdown handling
process.on('SIGTERM', () => {
  console.log('SIGTERM received, shutting down gracefully...');
  server.close(() => {
    console.log('Server closed');
    process.exit(0);
  });
});

process.on('SIGINT', () => {
  console.log('SIGINT received, shutting down gracefully...');
  server.close(() => {
    console.log('Server closed');
    process.exit(0);
  });
});

export { app, server };