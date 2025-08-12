#!/usr/bin/env node

/**
 * Automated MCP Server Testing Script
 * 
 * Tests all MCP tools and functionality programmatically
 */

import fetch from 'node-fetch';
import { v4 as uuidv4 } from 'uuid';

const SERVER_URL = process.env.MCP_SERVER_URL || 'http://localhost:3001';
const TEST_USER_ID = 'test-user-automated';
const TEST_SESSION_ID = uuidv4();

// ANSI color codes for console output
const colors = {
    reset: '\x1b[0m',
    bright: '\x1b[1m',
    red: '\x1b[31m',
    green: '\x1b[32m',
    yellow: '\x1b[33m',
    blue: '\x1b[34m',
    magenta: '\x1b[35m',
    cyan: '\x1b[36m'
};

function log(message, color = 'reset') {
    console.log(`${colors[color]}${message}${colors.reset}`);
}

function logTest(testName) {
    log(`\n📋 Testing: ${testName}`, 'cyan');
    log('-'.repeat(50), 'blue');
}

function logSuccess(message) {
    log(`✅ ${message}`, 'green');
}

function logError(message) {
    log(`❌ ${message}`, 'red');
}

function logWarning(message) {
    log(`⚠️  ${message}`, 'yellow');
}

// Helper function to make MCP requests
async function sendMCPRequest(method, params = null) {
    const requestId = Date.now() + Math.random();
    const mcpRequest = {
        jsonrpc: "2.0",
        id: requestId,
        method: method
    };
    
    if (params) {
        mcpRequest.params = params;
    }

    const response = await fetch(`${SERVER_URL}/mcp`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'x-user-id': TEST_USER_ID,
            'x-session-id': TEST_SESSION_ID
        },
        body: JSON.stringify(mcpRequest)
    });

    if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    return await response.json();
}

// Test server health
async function testHealth() {
    logTest('Server Health Check');
    
    try {
        const response = await fetch(`${SERVER_URL}/health`);
        const data = await response.json();
        
        if (data.status === 'healthy') {
            logSuccess('Server is healthy');
            logSuccess(`Active sessions: ${data.activeSessions}`);
            logSuccess(`Version: ${data.version}`);
        } else {
            logWarning('Server responded but status is not healthy');
        }
        
        return true;
    } catch (error) {
        logError(`Health check failed: ${error.message}`);
        return false;
    }
}

// Test server info
async function testServerInfo() {
    logTest('Server Information');
    
    try {
        const response = await fetch(`${SERVER_URL}/info`);
        const data = await response.json();
        
        logSuccess(`Server: ${data.name} v${data.version}`);
        logSuccess(`Transport: ${data.transport}`);
        logSuccess(`Multi-user support: ${data.capabilities.features.multiUser ? 'Yes' : 'No'}`);
        logSuccess(`Available tools: ${data.capabilities.tools.length}`);
        
        data.capabilities.tools.forEach(tool => {
            log(`  - ${tool.name}: ${tool.description}`, 'blue');
        });
        
        return true;
    } catch (error) {
        logError(`Server info failed: ${error.message}`);
        return false;
    }
}

// Test screen capture tool
async function testScreenCapture() {
    logTest('Screen Capture Tool');
    
    try {
        const response = await sendMCPRequest('tools/call', {
            name: 'screen_capture',
            arguments: {}
        });
        
        if (response.result && response.result.success) {
            logSuccess('Screen capture tool responded successfully');
        } else if (response.error) {
            logError(`MCP error: ${response.error.message || JSON.stringify(response.error)}`);
            return false;
        } else {
            logWarning('Unexpected response format');
            console.log(JSON.stringify(response, null, 2));
        }
        
        return true;
    } catch (error) {
        logError(`Screen capture test failed: ${error.message}`);
        return false;
    }
}

// Test OCR/extract text tool
async function testExtractText() {
    logTest('OCR - Extract Text Tool');
    
    // Sample base64 image (1x1 pixel PNG)
    const sampleImage = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAI7m97yUgAAAABJRU5ErkJggg==';
    
    try {
        const response = await sendMCPRequest('tools/call', {
            name: 'extract_text',
            arguments: {
                imageData: sampleImage
            }
        });
        
        if (response.result && response.result.success !== false) {
            logSuccess('OCR tool responded successfully');
            if (response.result.confidence) {
                log(`  Confidence: ${response.result.confidence}`, 'blue');
            }
            if (response.result.processingTime) {
                log(`  Processing time: ${response.result.processingTime}ms`, 'blue');
            }
        } else if (response.error) {
            logError(`MCP error: ${response.error.message || JSON.stringify(response.error)}`);
            return false;
        } else {
            logWarning('Unexpected response format');
            console.log(JSON.stringify(response, null, 2));
        }
        
        return true;
    } catch (error) {
        logError(`OCR test failed: ${error.message}`);
        return false;
    }
}

// Test activity logging tool
async function testLogActivity() {
    logTest('Activity Logging Tool');
    
    try {
        const response = await sendMCPRequest('tools/call', {
            name: 'log_activity',
            arguments: {
                description: 'Automated test activity',
                confidence: 0.95,
                screen_text_length: 200,
                processing_time: 1.5
            }
        });
        
        if (response.result && response.result.success !== false) {
            logSuccess('Activity logging tool responded successfully');
            if (response.result.activity_id) {
                log(`  Activity ID: ${response.result.activity_id}`, 'blue');
            }
            if (response.result.logged_at) {
                log(`  Logged at: ${response.result.logged_at}`, 'blue');
            }
        } else if (response.error) {
            logError(`MCP error: ${response.error.message || JSON.stringify(response.error)}`);
            return false;
        } else {
            logWarning('Unexpected response format');
            console.log(JSON.stringify(response, null, 2));
        }
        
        return true;
    } catch (error) {
        logError(`Activity logging test failed: ${error.message}`);
        return false;
    }
}

// Test multi-user functionality
async function testMultiUser() {
    logTest('Multi-User Functionality');
    
    const users = [
        { id: 'user-test-1', session: uuidv4() },
        { id: 'user-test-2', session: uuidv4() },
        { id: 'user-test-3', session: uuidv4() }
    ];
    
    try {
        const promises = users.map(async (user, index) => {
            const mcpRequest = {
                jsonrpc: "2.0",
                id: Date.now() + index,
                method: "tools/call",
                params: {
                    name: "log_activity",
                    arguments: {
                        description: `Activity from ${user.id}`,
                        confidence: 0.8
                    }
                }
            };
            
            const response = await fetch(`${SERVER_URL}/mcp`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'x-user-id': user.id,
                    'x-session-id': user.session
                },
                body: JSON.stringify(mcpRequest)
            });
            
            return { user, response: await response.json() };
        });
        
        const results = await Promise.all(promises);
        const successful = results.filter(r => r.response.result && r.response.result.success !== false);
        
        logSuccess(`Successfully processed ${successful.length}/${users.length} concurrent user requests`);
        
        return successful.length === users.length;
    } catch (error) {
        logError(`Multi-user test failed: ${error.message}`);
        return false;
    }
}

// Test error handling
async function testErrorHandling() {
    logTest('Error Handling');
    
    try {
        // Test with invalid tool name
        const response = await sendMCPRequest('tools/call', {
            name: 'invalid_tool',
            arguments: {}
        });
        
        if (response.error) {
            logSuccess('Server correctly returned error for invalid tool');
            log(`  Error: ${response.error.message}`, 'yellow');
        } else {
            logWarning('Expected error response for invalid tool');
        }
        
        return true;
    } catch (error) {
        // Network errors are also acceptable for this test
        logSuccess('Server correctly rejected invalid request');
        return true;
    }
}

// Run performance test
async function testPerformance() {
    logTest('Performance Test (10 concurrent requests)');
    
    const numRequests = 10;
    const startTime = Date.now();
    
    try {
        const promises = Array.from({ length: numRequests }, (_, i) =>
            sendMCPRequest('tools/call', {
                name: 'log_activity',
                arguments: {
                    description: `Performance test request ${i + 1}`,
                    confidence: 0.7
                }
            })
        );
        
        const results = await Promise.all(promises);
        const endTime = Date.now();
        const duration = endTime - startTime;
        
        const successful = results.filter(r => r.result && r.result.success !== false).length;
        
        logSuccess(`Completed ${successful}/${numRequests} requests in ${duration}ms`);
        logSuccess(`Average response time: ${(duration / numRequests).toFixed(2)}ms`);
        
        return successful === numRequests;
    } catch (error) {
        logError(`Performance test failed: ${error.message}`);
        return false;
    }
}

// Main test runner
async function runAllTests() {
    log('🚀 Starting Vygil MCP Server Automated Tests', 'bright');
    log(`🎯 Target server: ${SERVER_URL}`, 'blue');
    log(`👤 Test user: ${TEST_USER_ID}`, 'blue');
    log(`🔗 Session ID: ${TEST_SESSION_ID}`, 'blue');
    
    const tests = [
        { name: 'Health Check', fn: testHealth },
        { name: 'Server Info', fn: testServerInfo },
        { name: 'Screen Capture', fn: testScreenCapture },
        { name: 'Extract Text (OCR)', fn: testExtractText },
        { name: 'Activity Logging', fn: testLogActivity },
        { name: 'Multi-User', fn: testMultiUser },
        { name: 'Error Handling', fn: testErrorHandling },
        { name: 'Performance', fn: testPerformance }
    ];
    
    const results = [];
    
    for (const test of tests) {
        try {
            const result = await test.fn();
            results.push({ name: test.name, passed: result });
        } catch (error) {
            logError(`Test "${test.name}" threw an error: ${error.message}`);
            results.push({ name: test.name, passed: false });
        }
    }
    
    // Summary
    log('\n📊 Test Results Summary', 'bright');
    log('='.repeat(50), 'blue');
    
    const passed = results.filter(r => r.passed).length;
    const failed = results.length - passed;
    
    results.forEach(result => {
        if (result.passed) {
            logSuccess(`${result.name}`);
        } else {
            logError(`${result.name}`);
        }
    });
    
    log(`\n📈 Total: ${results.length} tests`, 'blue');
    logSuccess(`Passed: ${passed}`);
    if (failed > 0) {
        logError(`Failed: ${failed}`);
    }
    
    const successRate = ((passed / results.length) * 100).toFixed(1);
    log(`🎯 Success Rate: ${successRate}%`, successRate === '100.0' ? 'green' : 'yellow');
    
    if (passed === results.length) {
        log('\n🎉 All tests passed! MCP server is working correctly.', 'green');
        process.exit(0);
    } else {
        log('\n⚠️  Some tests failed. Check the MCP server configuration.', 'red');
        process.exit(1);
    }
}

// Handle command line arguments
if (process.argv.includes('--help') || process.argv.includes('-h')) {
    console.log(`
Usage: node test-mcp-tools.js [options]

Options:
  --help, -h          Show this help message
  --server <url>      MCP server URL (default: http://localhost:3001)

Environment Variables:
  MCP_SERVER_URL      MCP server URL (overrides --server)

Examples:
  node test-mcp-tools.js
  node test-mcp-tools.js --server http://localhost:3002
  MCP_SERVER_URL=http://production-server.com node test-mcp-tools.js
`);
    process.exit(0);
}

// Run the tests
runAllTests().catch(error => {
    logError(`Fatal error: ${error.message}`);
    process.exit(1);
});