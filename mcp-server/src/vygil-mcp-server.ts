#!/usr/bin/env node

/**
 * Vygil MCP Server - Simple and Compliant
 * 
 * A minimal MCP server using the official @modelcontextprotocol/sdk
 * that provides tools for screen capture, OCR, and activity logging.
 * 
 * Usage: node dist/vygil-mcp-server.js
 */

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import Tesseract from 'tesseract.js';

// Create MCP server
const server = new McpServer({
  name: "Vygil Activity Tracker",
  version: "1.0.0",
});

// Tool 1: Screen Capture (mock implementation for MVP)
server.tool(
  "screen_capture",
  {
    // No parameters needed
  },
  async () => {
    return {
      content: [
        {
          type: "text",
          text: JSON.stringify({
            success: true,
            imageData: "mock_base64_image_data",
            timestamp: new Date().toISOString(),
            metadata: {
              width: 1920,
              height: 1080,
              format: "png"
            }
          })
        }
      ]
    };
  }
);

// Tool 2: Extract Text (OCR)
server.tool(
  "extract_text",
  {
    imageData: z.string().describe("Base64 encoded image data or data URL")
  },
  async ({ imageData }) => {
    try {
      const startTime = Date.now();
      
      // Process the image data for OCR
      let base64Data: string;
      
      // Handle both raw base64 and data URL formats
      if (imageData.includes(',')) {
        const parts = imageData.split(',');
        base64Data = parts[1] || '';
      } else {
        base64Data = imageData;
      }
      
      if (!base64Data) {
        throw new Error('Invalid image data format. Expected base64 string or data URL.');
      }
      
      // Convert base64 to buffer
      const imageBuffer = Buffer.from(base64Data, 'base64');
      
      // Perform OCR using Tesseract.js
      const result = await Tesseract.recognize(imageBuffer, 'eng', {
        logger: () => {} // Silent logging
      });
      
      const extractedText = result.data.text.trim();
      const confidence = result.data.confidence / 100;
      const processingTime = Date.now() - startTime;
      
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify({
              success: true,
              text: extractedText,
              confidence: confidence,
              processingTime: processingTime,
              timestamp: new Date().toISOString(),
              metadata: {
                textLength: extractedText.length,
                wordCount: extractedText.split(/\s+/).filter(word => word.length > 0).length
              }
            })
          }
        ]
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify({
              success: false,
              error: error instanceof Error ? error.message : 'OCR processing failed'
            })
          }
        ]
      };
    }
  }
);

// Tool 3: Log Activity
server.tool(
  "log_activity",
  {
    description: z.string().describe("Activity description"),
    confidence: z.number().optional().describe("Confidence score between 0 and 1"),
    screen_text_length: z.number().optional().describe("Length of the processed screen text"),
    processing_time: z.number().optional().describe("Processing time in seconds")
  },
  async ({ description, confidence = 0, screen_text_length = 0, processing_time = 0 }) => {
    const logEntry = {
      timestamp: new Date().toISOString(),
      description,
      confidence,
      screen_text_length,
      processing_time,
      id: `activity_${Date.now()}`
    };
    
    // Log to console (this will appear in the MCP client's logs)
    console.error(`📝 Activity logged: ${description} (confidence: ${confidence})`);
    
    return {
      content: [
        {
          type: "text",
          text: JSON.stringify({
            success: true,
            logged_at: logEntry.timestamp,
            activity_id: logEntry.id,
            entry: logEntry
          })
        }
      ]
    };
  }
);

// Start the server
async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  
  // Server is now running and listening on stdin/stdout
  console.error("🚀 Vygil MCP Server started successfully!");
  console.error("📡 Ready to receive MCP requests via stdio");
}

// Handle errors
process.on('uncaughtException', (error) => {
  console.error('❌ Uncaught exception:', error);
  process.exit(1);
});

process.on('unhandledRejection', (reason, promise) => {
  console.error('❌ Unhandled rejection at:', promise, 'reason:', reason);
  process.exit(1);
});

// Start server
main().catch((error) => {
  console.error('❌ Failed to start server:', error);
  process.exit(1);
});