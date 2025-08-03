# MCP Server: Modal Context Protocol for Screen Sharing and OCR

A powerful Node.js server for screen sharing with real-time OCR capabilities. This server facilitates remote screen viewing and text extraction using the Modal Context Protocol.

## Features

- **Live Screen Sharing**: Create sessions to share screens with multiple viewers
- **Real-time OCR**: Extract text from shared screens using Tesseract.js
- **WebSocket Communication**: Low-latency data transmission via Socket.IO
- **Session Management**: Simple session creation and joining
- **RESTful API**: Endpoints for session discovery and management

## Prerequisites

- Node.js (v16+)
- npm (v7+)

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/vygil-ai.git
cd vygil-ai/mcp-server
```

2. **Install dependencies**

```bash
npm install
```

3. **Configure the server** (optional)

Edit `src/config.ts` to customize server settings:

```typescript
// Example configuration
export const config = {
  server: {
    port: process.env.PORT || 3000,
    cors: {
      origin: process.env.CORS_ORIGIN || '*',
      methods: ['GET', 'POST']
    }
  },
  // ... other settings
};
```

## Running the Server

### Development Mode

Run the server with hot-reloading for development:

```bash
npm run dev
```

The server will restart automatically when you make changes to the source files.

### Production Mode

Build and run the server for production:

```bash
npm run build
npm start
```

## Testing the Server

### Using the Test HTML Client

1. Create a file named `test-client.html` with the HTML test client code provided below
2. Open the HTML file in a browser
3. Click "Connect" to connect to the server
4. Click "Create Session" to create a new screen sharing session
5. Use the session ID to connect from another browser tab or device

```html
<!DOCTYPE html>
<html>
<head>
    <title>MCP Server Test</title>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <!-- HTML client code from previous message -->
</head>
<body>
    <!-- Client UI -->
</body>
</html>
```

### Using API Endpoints

Test the REST API with curl:

```bash
# Get list of active sessions
curl http://localhost:3000/api/sessions
```

## Server Events

| Event | Description |
|-------|-------------|
| `create-session` | Create a new screen sharing session |
| `join-session` | Join an existing session as a viewer |
| `screen-data` | Send screen data (client → server) |
| `ocr-result` | Receive OCR text results (server → client) |
| `request-ocr` | Request OCR on a specific image |

## Troubleshooting

### Common Issues

- **Connection Refused**: Make sure the server is running and the port is accessible
- **CORS Errors**: Check the CORS configuration in `config.ts`
- **OCR Not Working**: Ensure Tesseract.js language data is properly loaded

### Debug Mode

For more detailed logs, set the environment variable:

```bash
DEBUG=mcp-server:* npm run dev
```

## Project Structure

```
mcp-server/
├── src/
│   ├── server.ts             # Main server file
│   ├── config.ts             # Server configuration
│   ├── types/
│   │   └── protocol.ts       # MCP protocol definitions
│   └── utils/                # Utility functions
├── dist/                     # Compiled JavaScript files
├── package.json              # Project dependencies
└── tsconfig.json             # TypeScript configuration
```

## License

This project is licensed under the ISC License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add some amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request


---

For technical support or questions, please open an issue in the GitHub repository.