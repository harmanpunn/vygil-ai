export const config = {
  server: {
    port: process.env.PORT || 3000,
    cors: {
      origin: process.env.CORS_ORIGIN || '*',
      methods: ['GET', 'POST']
    }
  },
  sessions: {
    timeoutMs: 3600000, // 1 hour in milliseconds
    cleanupIntervalMs: 300000 // Check for expired sessions every 5 minutes
  },
  stream: {
    maxChunkSize: 1024 * 1024, // 1MB
    maxWidth: 1920, // Max width for screenshots
    defaultQuality: 80, // Default JPEG quality (1-100)
    maxFps: 10 // Maximum frames per second
  }
};