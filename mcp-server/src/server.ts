import express from 'express';
import http from 'http';
import { Server, Socket } from 'socket.io';
import cors from 'cors';
import { v4 as uuidv4 } from 'uuid';
import { Request, Response } from 'express';
import { config } from './config';
import { MCPSession, MCPScreenChunk, MCPMessageType } from './types/protocol';
import Tesseract from 'tesseract.js';

// Initialize app and server
const app = express();
const server = http.createServer(app);
const io = new Server(server, {
  cors: config.server.cors
});

// Use middleware
app.use(cors());
app.use(express.json());

// Store active sessions
const activeSessions: Record<string, MCPSession> = {};

// Socket.IO connection handling
io.on('connection', (socket: Socket) => {
  console.log(`New connection: ${socket.id}`);

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

  // Handle disconnection
  socket.on('disconnect', () => {
    console.log(`Disconnected: ${socket.id}`);
    
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

// Start the server
const PORT = config.server.port;
server.listen(PORT, () => {
  console.log(`MCP Server running on port ${PORT}`);
});

// Export for testing or other uses
export { app, server, io };
export { config as serverConfig };