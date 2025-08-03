export interface MCPSession {
  id: string;
  host: string;
  viewers: string[];
  startTime?: number;
  captureInterval?: NodeJS.Timeout;
}

export interface MCPScreenChunk {
  type: 'full' | 'delta';
  data: string; // Base64 encoded image data
  timestamp: number;
  dimensions?: {
    width: number;
    height: number;
  };
}

export enum MCPMessageType {
  SESSION_CREATE = 'create-session',
  SESSION_JOIN = 'join-session',
  SESSION_CREATED = 'session-created',
  SESSION_JOINED = 'session-joined',
  VIEWER_JOINED = 'viewer-joined',
  VIEWER_LEFT = 'viewer-left',
  SESSION_ENDED = 'session-ended',
  SCREEN_DATA = 'screen-data',
  REQUEST_SCREENSHOT = 'request-screenshot',
  SCREENSHOT_DATA = 'screenshot-data',
  START_SCREEN_CAPTURE = 'start-screen-capture',
  SCREEN_CAPTURE_STARTED = 'screen-capture-started',
  STOP_SCREEN_CAPTURE = 'stop-screen-capture',
  SCREEN_CAPTURE_STOPPED = 'screen-capture-stopped',
  ERROR = 'error',
  
  // New agent-specific types
  AGENT_REGISTER = 'agent-register',
  AGENT_REGISTERED = 'agent-registered',
  AGENT_REQUEST_OCR = 'agent-request-ocr',
  AGENT_OCR_RESULT = 'agent-ocr-result',
  AGENT_REQUEST_SCREEN = 'agent-request-screen',
  AGENT_SCREEN_RESULT = 'agent-screen-result'
}

export interface MCPMessage {
  type: MCPMessageType;
  payload: any;
}

// New agent-specific interfaces
export interface MCPAgentRegistration {
  agentId: string;
  capabilities?: string[];
}

export interface MCPAgentOCRRequest {
  agentId: string;
  imageData: string;  // Base64 encoded image
  options?: {
    language?: string;
  };
}

export interface MCPAgentOCRResult {
  agentId: string;
  text: string;
  confidence?: number;
  timestamp: number;
  processingTimeMs: number;
}

export interface MCPAgentScreenRequest {
  agentId: string;
}

export interface MCPAgentScreenResult {
  agentId: string;
  imageData: string;
  timestamp: number;
  dimensions: {
    width: number;
    height: number;
  };
}