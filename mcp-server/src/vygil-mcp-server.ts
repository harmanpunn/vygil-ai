#!/usr/bin/env node

/**
 * Vygil MCP Server - Phase 4: Production Ready
 * 
 * Features:
 * - Dual transport support (stdio + Streamable HTTP)
 * - Enhanced health checks with system metrics
 * - Graceful shutdown handling
 * - Environment-based configuration
 * - Improved logging with levels
 * - Request metrics and monitoring
 * 
 * Usage: 
 *   node dist/vygil-mcp-server.js          # stdio (default)
 *   node dist/vygil-mcp-server.js stdio    # stdio explicitly  
 *   node dist/vygil-mcp-server.js http     # Streamable HTTP
 */

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import { z } from "zod";
import Tesseract from 'tesseract.js';
import express, { Request, Response, NextFunction } from 'express';
import { createServer } from 'http';
import { randomUUID } from 'crypto';

// Environment configuration
const config = {
  // Server settings
  port: parseInt(process.env.PORT || '3000'),
  transport: process.argv[2] || process.env.MCP_TRANSPORT || 'stdio',
  logLevel: process.env.LOG_LEVEL || 'info',
  
  // Security settings
  allowedOrigins: (process.env.ALLOWED_ORIGINS || 'http://localhost:3000,http://localhost:8080,http://127.0.0.1:3000').split(','),
  enableDnsRebinding: process.env.ENABLE_DNS_REBINDING_PROTECTION !== 'false',
  
  // Feature flags
  enableMetrics: process.env.ENABLE_METRICS !== 'false',
  enableHealthCheck: process.env.ENABLE_HEALTH_CHECK !== 'false',
  
  // Timeouts and limits
  shutdownTimeout: parseInt(process.env.SHUTDOWN_TIMEOUT || '10000'),
  requestTimeout: parseInt(process.env.REQUEST_TIMEOUT || '30000'),
};

// Logging utility
const logger = {
  debug: (msg: string, ...args: any[]) => {
    if (['debug'].includes(config.logLevel)) {
      console.error(`🔍 [DEBUG] ${msg}`, ...args);
    }
  },
  info: (msg: string, ...args: any[]) => {
    if (['debug', 'info'].includes(config.logLevel)) {
      console.error(`ℹ️  [INFO] ${msg}`, ...args);
    }
  },
  warn: (msg: string, ...args: any[]) => {
    if (['debug', 'info', 'warn'].includes(config.logLevel)) {
      console.error(`⚠️  [WARN] ${msg}`, ...args);
    }
  },
  error: (msg: string, ...args: any[]) => {
    console.error(`❌ [ERROR] ${msg}`, ...args);
  }
};

// Metrics tracking
class MetricsCollector {
  private metrics = {
    requests: 0,
    errors: 0,
    screenCaptures: 0,
    ocrRequests: 0,
    activityLogs: 0,
    uptime: Date.now(),
    lastRequest: null as Date | null
  };

  increment(metric: keyof typeof this.metrics) {
    if (typeof this.metrics[metric] === 'number') {
      (this.metrics[metric] as number)++;
    }
  }

  updateLastRequest() {
    this.metrics.lastRequest = new Date();
  }

  getMetrics() {
    return {
      ...this.metrics,
      uptimeSeconds: Math.floor((Date.now() - this.metrics.uptime) / 1000),
      memory: process.memoryUsage(),
      cpu: process.cpuUsage()
    };
  }
}

const metrics = new MetricsCollector();

// Create MCP server
const server = new McpServer({
  name: "Vygil Activity Tracker",
  version: "1.0.0",
});

// Tool 1: Screen Capture (using modern registerTool API)
server.registerTool(
  "screen_capture",
  {
    description: "Capture screenshot of the current screen",
    inputSchema: {}
  },
  async () => {
    const startTime = Date.now();
    metrics.increment('requests');
    metrics.increment('screenCaptures');
    metrics.updateLastRequest();
    
    try {
      logger.debug('Processing screen capture request');
      
      // Mock implementation for MVP - replace with real screen capture
      const mockImageData = "mock_base64_image_data";
      
      const processingTime = Date.now() - startTime;
      logger.info(`Screen capture completed in ${processingTime}ms`);
      
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify({
              success: true,
              imageData: mockImageData,
              timestamp: new Date().toISOString(),
              processingTime,
              metadata: {
                width: 1920,
                height: 1080,
                format: "png"
              }
            })
          }
        ]
      };
    } catch (error) {
      metrics.increment('errors');
      logger.error("Screen capture failed:", error);
      
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify({
              success: false,
              error: error instanceof Error ? error.message : 'Screen capture failed',
              timestamp: new Date().toISOString()
            })
          }
        ]
      };
    }
  }
);

// Tool 2: OCR Text Extraction (using modern registerTool API)
server.registerTool(
  "extract_text",
  {
    description: "Extract text from image using OCR technology",
    inputSchema: {
      imageData: z.string().describe("Base64 encoded image data or data URL")
    }
  },
  async ({ imageData }) => {
    const startTime = Date.now();
    metrics.increment('requests');
    metrics.increment('ocrRequests');
    metrics.updateLastRequest();
    
    try {
      logger.debug('Processing OCR request');
      
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
      logger.debug(`Processing ${imageBuffer.length} bytes of image data`);
      
      // Perform OCR using Tesseract.js
      const result = await Tesseract.recognize(imageBuffer, 'eng', {
        logger: () => {} // Silent logging
      });
      
      const extractedText = result.data.text.trim();
      const confidence = result.data.confidence / 100;
      const processingTime = Date.now() - startTime;
      
      logger.info(`OCR completed in ${processingTime}ms, extracted ${extractedText.length} characters`);
      
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
      metrics.increment('errors');
      logger.error("OCR processing failed:", error);
      
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify({
              success: false,
              error: error instanceof Error ? error.message : 'OCR processing failed',
              timestamp: new Date().toISOString()
            })
          }
        ]
      };
    }
  }
);

// Tool 3: Activity Logging (using modern registerTool API)
server.registerTool(
  "log_activity",
  {
    description: "Log user activity with confidence metrics and metadata",
    inputSchema: {
      description: z.string().describe("Activity description"),
      confidence: z.number().min(0).max(1).optional().describe("Confidence score between 0 and 1"),
      screen_text_length: z.number().optional().describe("Length of the processed screen text"),
      processing_time: z.number().optional().describe("Processing time in seconds")
    }
  },
  async ({ description, confidence = 0, screen_text_length = 0, processing_time = 0 }) => {
    metrics.increment('requests');
    metrics.increment('activityLogs');
    metrics.updateLastRequest();
    
    const logEntry = {
      timestamp: new Date().toISOString(),
      description,
      confidence,
      screen_text_length,
      processing_time,
      id: `activity_${Date.now()}`
    };
    
    logger.info(`Activity logged: ${description} (confidence: ${confidence})`);
    
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

// Graceful shutdown handler
class GracefulShutdown {
  private isShuttingDown = false;
  private httpServer: any = null;

  setHttpServer(server: any) {
    this.httpServer = server;
  }

  async shutdown(signal: string) {
    if (this.isShuttingDown) {
      logger.warn('Shutdown already in progress, forcing exit');
      process.exit(1);
    }

    this.isShuttingDown = true;
    logger.info(`Received ${signal}, starting graceful shutdown...`);

    const shutdownTimer = setTimeout(() => {
      logger.error('Shutdown timeout exceeded, forcing exit');
      process.exit(1);
    }, config.shutdownTimeout);

    try {
      // Close HTTP server if running
      if (this.httpServer) {
        await new Promise<void>((resolve) => {
          this.httpServer.close((err: any) => {
            if (err) {
              logger.error('Error closing HTTP server:', err);
            } else {
              logger.info('HTTP server closed');
            }
            resolve();
          });
        });
      }

      // Additional cleanup can be added here
      logger.info('Graceful shutdown completed');
      clearTimeout(shutdownTimer);
      process.exit(0);
    } catch (error) {
      logger.error('Error during shutdown:', error);
      clearTimeout(shutdownTimer);
      process.exit(1);
    }
  }
}

const gracefulShutdown = new GracefulShutdown();

async function startStdioServer() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  logger.info("Vygil MCP Server started with STDIO transport");
  logger.info("Ready for Claude Desktop connection");
}

async function startHTTPServer() {
  const app = express();
  app.use(express.json({ limit: '10mb' }));
  
  // Request timeout middleware
  app.use((req: Request, res: Response, next: NextFunction) => {
    res.setTimeout(config.requestTimeout, () => {
      logger.warn(`Request timeout for ${req.method} ${req.path}`);
      res.status(408).json({ error: 'Request timeout' });
    });
    next();
  });

  // Enhanced health check endpoint
  if (config.enableHealthCheck) {
    app.get('/health', (_req: Request, res: Response) => {
      const healthData = {
        status: 'healthy',
        timestamp: new Date().toISOString(),
        transport: 'streamable-http',
        version: '1.0.0',
        uptime: Math.floor(process.uptime()),
        ...(config.enableMetrics ? { metrics: metrics.getMetrics() } : {})
      };
      
      res.json(healthData);
    });

    // Detailed health check
    app.get('/health/detailed', (_req: Request, res: Response) => {
      res.json({
        status: 'healthy',
        timestamp: new Date().toISOString(),
        config: {
          transport: config.transport,
          logLevel: config.logLevel,
          enableMetrics: config.enableMetrics,
          enableDnsRebinding: config.enableDnsRebinding
        },
        metrics: metrics.getMetrics(),
        system: {
          nodeVersion: process.version,
          platform: process.platform,
          arch: process.arch,
          pid: process.pid
        }
      });
    });
  }

  // Metrics endpoint
  if (config.enableMetrics) {
    app.get('/metrics', (_req: Request, res: Response) => {
      res.json(metrics.getMetrics());
    });
  }

  // MCP endpoint with proper security (2025 standards)
  app.all('/mcp', async (req: Request, res: Response) => {
    try {
      const transport = new StreamableHTTPServerTransport({
        sessionIdGenerator: () => randomUUID(),
        // Required: Origin validation for security (DNS rebinding protection)
        allowedOrigins: config.allowedOrigins,
        // Enable DNS rebinding protection
        enableDnsRebindingProtection: config.enableDnsRebinding,
      });
      
      await server.connect(transport);
      await transport.handleRequest(req, res, req.body);
    } catch (error) {
      metrics.increment('errors');
      logger.error('HTTP transport error:', error);
      res.status(500).json({ 
        error: 'Internal server error',
        message: error instanceof Error ? error.message : 'Unknown error',
        timestamp: new Date().toISOString()
      });
    }
  });

  // Error handling middleware
  app.use((err: any, _req: any, res: any, _next: any) => {
    metrics.increment('errors');
    logger.error('Express error:', err);
    res.status(500).json({
      error: 'Internal server error',
      timestamp: new Date().toISOString()
    });
  });

  // 404 handler
  app.use('*', (req: Request, res: Response) => {
    res.status(404).json({
      error: 'Not found',
      path: req.originalUrl,
      timestamp: new Date().toISOString()
    });
  });

  const httpServer = createServer(app);
  gracefulShutdown.setHttpServer(httpServer);

  return new Promise<void>((resolve) => {
    httpServer.listen(config.port, () => {
      logger.info(`Vygil MCP Server started with Streamable HTTP transport`);
      logger.info(`Ready at http://localhost:${config.port}/mcp`);
      if (config.enableHealthCheck) {
        logger.info(`Health check: http://localhost:${config.port}/health`);
      }
      if (config.enableMetrics) {
        logger.info(`Metrics: http://localhost:${config.port}/metrics`);
      }
      resolve();
    });
  });
}

// Start appropriate server based on transport type
async function main() {
  logger.info(`Starting Vygil MCP Server with ${config.transport} transport...`);
  logger.debug('Configuration:', config);
  
  switch (config.transport) {
    case 'stdio':
      await startStdioServer();
      break;
    case 'http':
    case 'streamable-http':
      await startHTTPServer();
      break;
    default:
      logger.error(`Unknown transport type: ${config.transport}`);
      logger.info('Available options: stdio, http');
      process.exit(1);
  }
}

// Enhanced error handling
process.on('uncaughtException', (error) => {
  logger.error('Uncaught exception:', error);
  gracefulShutdown.shutdown('UNCAUGHT_EXCEPTION');
});

process.on('unhandledRejection', (reason, promise) => {
  logger.error('Unhandled rejection at:', promise, 'reason:', reason);
  gracefulShutdown.shutdown('UNHANDLED_REJECTION');
});

// Graceful shutdown signals
process.on('SIGTERM', () => gracefulShutdown.shutdown('SIGTERM'));
process.on('SIGINT', () => gracefulShutdown.shutdown('SIGINT'));

// Start server
main().catch((error) => {
  logger.error('Failed to start server:', error);
  process.exit(1);
});