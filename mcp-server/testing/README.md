# MCP Server Testing

This directory contains comprehensive tests for the Vygil MCP HTTP server.

## Test Files

### 🌐 Browser Test Client
**File:** `test-http-mcp.html`

A comprehensive HTML test client with interactive UI for manual testing:

- **Server Health & Info** - Check server status and capabilities
- **MCP Tools Testing** - Test all available tools (screen capture, OCR, activity logging)
- **Multi-User Testing** - Test concurrent user sessions
- **Stress Testing** - Performance testing with configurable concurrent requests
- **Session Management** - Test user ID and session ID functionality

**Usage:**
1. Start the MCP HTTP server: `npm run dev:http`
2. Open `test-http-mcp.html` in your browser
3. Configure server URL and user credentials
4. Run individual tests or comprehensive test suites

### 🤖 Automated Test Script
**File:** `test-mcp-tools.js`

Node.js script for automated testing and CI/CD integration:

**Features:**
- Automated testing of all MCP tools
- Multi-user concurrent request testing
- Performance benchmarking
- Error handling validation
- Colored console output with detailed results
- Exit codes for CI integration (0 = success, 1 = failure)

**Usage:**
```bash
# Run all automated tests
npm test

# Run with custom server URL
npm test -- --server http://localhost:3002

# Or with environment variable
MCP_SERVER_URL=http://production-server.com npm test

# Show help
npm test -- --help
```

## Test Commands

```bash
# Install dependencies (if not already done)
npm install

# Start the MCP HTTP server for testing
npm run dev:http

# Run automated tests
npm test

# Open browser test instructions
npm run test:browser

# Run both automated and show browser test info
npm run test:all
```

## Test Coverage

### ✅ Functional Tests
- [x] Server health check
- [x] Server information endpoint
- [x] Screen capture tool
- [x] OCR/text extraction tool
- [x] Activity logging tool
- [x] Multi-user session management
- [x] Error handling and validation

### ✅ Performance Tests
- [x] Concurrent request handling
- [x] Response time measurement
- [x] Session isolation verification
- [x] Stress testing with configurable load

### ✅ Integration Tests
- [x] HTTP Streamable transport protocol
- [x] JSON-RPC 2.0 compliance
- [x] CORS functionality
- [x] Session timeout handling

## Expected Test Results

**Healthy Server Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-XX-XXTXX:XX:XX.XXXZ",
  "activeSessions": 0,
  "version": "2.0.0",
  "transport": "http-streamable"
}
```

**Successful Tool Call:**
```json
{
  "jsonrpc": "2.0",
  "id": 12345,
  "result": {
    "success": true,
    "message": "MCP request processed",
    "session": {
      "userId": "test-user",
      "requestCount": 1
    }
  }
}
```

## Troubleshooting

### Common Issues

**❌ Connection Refused**
- Ensure MCP server is running: `npm run dev:http`
- Check server URL in test configuration
- Verify port 3001 is available

**❌ CORS Errors (Browser Tests)**
- Ensure ALLOWED_ORIGINS includes your test domain
- Check browser console for specific CORS messages

**❌ Test Failures**
- Check server logs for detailed error messages
- Verify environment variables are set correctly
- Ensure all dependencies are installed

### Debug Mode

Enable detailed logging by setting environment variable:
```bash
LOG_LEVEL=debug npm run dev:http
```

## CI/CD Integration

The automated test script returns appropriate exit codes:
- **Exit 0**: All tests passed
- **Exit 1**: Some tests failed

Example GitHub Actions workflow:
```yaml
- name: Install dependencies
  run: npm install
  working-directory: mcp-server

- name: Start MCP server
  run: npm run dev:http &
  working-directory: mcp-server

- name: Wait for server
  run: sleep 5

- name: Run MCP tests
  run: npm test
  working-directory: mcp-server
```

## Contributing

When adding new MCP tools or functionality:

1. Update `test-http-mcp.html` with new test sections
2. Add corresponding test functions in `test-mcp-tools.js`
3. Update this README with new test coverage
4. Ensure tests handle both success and error cases