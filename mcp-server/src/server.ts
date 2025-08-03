import express from 'express';
import { createServer } from 'http';
import { Server } from 'socket.io';
import cors from 'cors';
import { v4 as uuidv4 } from 'uuid';
import { Request, Response } from 'express';
import { config } from './config';
import { MCPSession, MCPScreenChunk, MCPMessageType, MCPAgentRegistration, MCPAgentOCRRequest, MCPAgentScreenRequest } from './types/protocol';
import Tesseract from 'tesseract.js';

// Initialize app and server
const app = express();
const server = createServer(app);

// Enable CORS for all routes
app.use(cors({
  origin: "*",
  methods: ["GET", "POST", "OPTIONS"],
  credentials: true
}));

app.use(express.json());

// Store active sessions
const activeSessions: Record<string, MCPSession> = {};

// Socket.IO connection handling
const io = new Server(server, {
  cors: {
    origin: "*",
    methods: ["GET", "POST"],
    credentials: true
  },
  transports: ['websocket', 'polling'],
  allowEIO3: true
});

io.on('connection', (socket) => {
  console.log(`✅ New client connected: ${socket.id}`);

  // Handle screen sharing session creation
  socket.on('create-session', () => {
    const sessionId = uuidv4();
    activeSessions[sessionId] = {
      id: sessionId,
      host: socket.id,
      viewers: [],
      startTime: Date.now()
    };
    
    socket.join(sessionId);
    console.log(`New session created: ${sessionId} by host: ${socket.id}`);
    socket.emit('session-created', { sessionId });
  });

  // Handle joining a session
  socket.on('join-session', (data: { sessionId: string }) => {
    const { sessionId } = data;
    const session = activeSessions[sessionId];
    
    if (!session) {
      socket.emit('error', { message: 'Session not found' });
      return;
    }
    
    session.viewers.push(socket.id);
    socket.join(sessionId);
    console.log(`Viewer ${socket.id} joined session ${sessionId}`);
    
    socket.emit('session-joined', { sessionId });
    io.to(session.host).emit('viewer-joined', { viewerId: socket.id });
  });

  // Handle screen data from client and perform OCR
  socket.on('screen-data', async (data: { sessionId: string, chunk: MCPScreenChunk }) => {
    const { sessionId, chunk } = data;
    const session = activeSessions[sessionId];
    
    if (!session) {
      socket.emit('error', { message: 'Session not found' });
      return;
    }
    
    try {
      // Forward screen data to viewers
      socket.to(sessionId).emit('screen-data', { chunk });
      
      // Perform OCR on the received image
      if (chunk.data && typeof chunk.data === 'string') {
        // Split to remove data URL prefix if present (e.g., "data:image/jpeg;base64,")
        const parts = chunk.data.split(',');
        const imageData = parts.length > 1 ? parts[1] : parts[0];
        
        // Make sure imageData is not undefined before creating buffer
        if (imageData) {
          // Convert base64 to buffer for Tesseract
          const imageBuffer = Buffer.from(imageData, 'base64');
          
          // Perform OCR
          const result = await Tesseract.recognize(
            imageBuffer,
            'eng',
            { logger: m => console.log(m) }
          );
          
          const extractedText = result.data.text;
          console.log(`OCR completed for session ${sessionId}, text length: ${extractedText.length} chars`);
          
          // Send OCR result back to the client
          socket.emit('ocr-result', {
            sessionId,
            timestamp: Date.now(),
            text: extractedText
          });
        } else {
          console.error('Image data is empty or invalid');
          socket.emit('error', { message: 'Invalid image data' });
        }
      }
    } catch (error) {
      console.error('Error processing screen data for OCR:', error);
      socket.emit('error', { message: 'Failed to process OCR' });
    }
  });

  // Handle request for OCR on specific image
  socket.on('request-ocr', async (data: { sessionId: string, imageData: string }) => {
    const { sessionId, imageData } = data;
    const session = activeSessions[sessionId];
    
    if (!session) {
      socket.emit('error', { message: 'Session not found' });
      return;
    }
    
    try {
      // Process the image data for OCR
      const base64Data = imageData.split(',')[1]; // Remove data URL prefix if present
      if (!base64Data) {
        socket.emit('error', { message: 'Invalid image data' });
        return;
      }
      const imageBuffer = Buffer.from(base64Data, 'base64');
      
      // Perform OCR
      const result = await Tesseract.recognize(
        imageBuffer,
        'eng',
        { logger: m => console.log(m) }
      );
      
      const extractedText = result.data.text;
      
      // Send OCR result back to the client
      socket.emit('ocr-result', {
        sessionId,
        timestamp: Date.now(),
        text: extractedText
      });
      
      console.log(`On-demand OCR completed for session ${sessionId}`);
    } catch (error) {
      console.error('OCR error:', error);
      socket.emit('error', { message: 'Failed to perform OCR' });
    }
  });

  // Handle agent registration
  socket.on(MCPMessageType.AGENT_REGISTER, (data: MCPAgentRegistration) => {
    const { agentId } = data;
    console.log(`Agent registered: ${agentId}`);
    
    socket.emit(MCPMessageType.AGENT_REGISTERED, { 
      agentId, 
      timestamp: Date.now(),
      status: 'registered',
      serverCapabilities: ['ocr', 'screen-sharing']
    });
  });

  // Handle OCR requests from agents
  socket.on(MCPMessageType.AGENT_REQUEST_OCR, async (data: MCPAgentOCRRequest) => {
    const { agentId, imageData, options = {} } = data;
    const startTime = Date.now();
    
    try {
      console.log(`Processing OCR request from agent ${agentId}`);
      
      // Extract base64 image data
      const parts = imageData.split(',');
      const base64Data = parts.length > 1 ? parts[1] : parts[0];
      
      if (!base64Data) {
        socket.emit(MCPMessageType.ERROR, { 
          message: 'Invalid image data',
          agentId
        });
        return;
      }
      
      // Convert base64 to buffer for Tesseract
      const imageBuffer = Buffer.from(base64Data, 'base64');
      
      // Perform OCR
      const result = await Tesseract.recognize(
        imageBuffer,
        options.language || 'eng',
        { 
          logger: m => {
            if (m.status === 'recognizing text') {
              console.log(`OCR Progress for ${agentId}: ${Math.round(m.progress * 100)}%`);
            }
          }
        }
      );
      
      const extractedText = result.data.text;
      const processingTime = Date.now() - startTime;
      
      console.log(`OCR completed for agent ${agentId}, text length: ${extractedText.length} chars, took ${processingTime}ms`);
      
      // Send OCR result back to the agent
      socket.emit(MCPMessageType.AGENT_OCR_RESULT, {
        agentId,
        text: extractedText,
        confidence: result.data.confidence / 100, // Convert to 0-1 scale
        timestamp: Date.now(),
        processingTimeMs: processingTime
      });
      
    } catch (error) {
      console.error(`OCR error for agent ${agentId}:`, error);
      socket.emit(MCPMessageType.ERROR, { 
        message: 'Failed to perform OCR',
        agentId,
        error: error instanceof Error ? error.message : 'Unknown error'
      });
    }
  });

  // Handle screen capture requests from agents
  socket.on(MCPMessageType.AGENT_REQUEST_SCREEN, (data: MCPAgentScreenRequest) => {
    const { agentId } = data;
    
    console.log(`Agent ${agentId} requested screen capture`);
    
    // We need to notify a client to capture the screen
    const sessionIds = Object.keys(activeSessions);
    if (sessionIds.length > 0) {
      // Use a non-null assertion or type assertion since we've checked length > 0
      const sessionId = sessionIds[0] as string; // This tells TypeScript that we know it's a string
      
      const session = activeSessions[sessionId];
      
      if (session) {
        // Send request to session host
        io.to(session.host).emit('request-screenshot-for-agent', { 
          agentId,
          sessionId
        });
      } else {
        socket.emit(MCPMessageType.ERROR, {
          message: 'Session not found',
          agentId
        });
      }
    } else {
      socket.emit(MCPMessageType.ERROR, {
        message: 'No active screen sharing sessions available',
        agentId
      });
    }
  });

  // Handle screenshot data from client (to be forwarded to agent)
  socket.on('screenshot-for-agent', (data: { agentId: string, imageData: string, dimensions: { width: number, height: number } }) => {
    const { agentId, imageData, dimensions } = data;
    
    // Find the socket for the agent
    // In a real implementation, you would store agent socket IDs
    // For this example, we'll just forward it to the requesting socket
    socket.emit(MCPMessageType.AGENT_SCREEN_RESULT, {
      agentId,
      imageData,
      timestamp: Date.now(),
      dimensions
    });
  });

  // Handle disconnection
  socket.on('disconnect', () => {
    console.log(`❌ Client disconnected: ${socket.id}`);
    
    // Clean up sessions if host disconnects
    for (const sessionId in activeSessions) {
      const session = activeSessions[sessionId];
      
      if (session && session.host === socket.id) {
        // Notify all viewers
        io.to(sessionId).emit('session-ended', { 
          message: 'Host has disconnected' 
        });
        delete activeSessions[sessionId];
        console.log(`Session ${sessionId} terminated because host disconnected`);
      } else if (session && session.viewers.includes(socket.id)) {
        // Remove viewer from session
        session.viewers = session.viewers.filter(id => id !== socket.id);
        io.to(session.host).emit('viewer-left', { viewerId: socket.id });
      }
    }
  });
});

// API routes
app.get('/api/sessions', (req: Request, res: Response) => {
  res.json({ sessions: Object.keys(activeSessions) });
});

app.get('/', (req, res) => {
  res.json({
    message: 'MCP WebSocket Server',
    status: 'running',
    timestamp: new Date().toISOString()
  });
});

app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    service: 'MCP WebSocket Server',
    timestamp: new Date().toISOString()
  });
});

// Start the server
const PORT = config.server.port;
server.listen(PORT, () => {
  console.log(`🚀 MCP Server running on port ${PORT}`);
  console.log(`📡 WebSocket server ready for connections`);
});

// Export for testing or other uses
export { app, server, io };
export { config as serverConfig };