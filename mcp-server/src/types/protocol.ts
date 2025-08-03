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
  ERROR = 'error'
}

export interface MCPMessage {
  type: MCPMessageType;
  payload: any;
}