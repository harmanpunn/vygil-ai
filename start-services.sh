#!/bin/bash

# Get port from environment variable or default to 5173
PORT=${PORT:-5173}

# Function to start a service in the background
start_service() {
    local name=$1
    local command=$2
    echo "Starting $name..."
    $command &
    local pid=$!
    echo "$name started with PID $pid"
    echo $pid > "/tmp/${name}.pid"
}

# Function to stop a service
stop_service() {
    local name=$1
    local pid_file="/tmp/${name}.pid"
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        echo "Stopping $name (PID: $pid)..."
        kill $pid 2>/dev/null || true
        rm -f "$pid_file"
    fi
}

# Function to cleanup on exit
cleanup() {
    echo "Shutting down services..."
    stop_service "frontend"
    stop_service "api"
    exit 0
}

# Set trap for cleanup
trap cleanup SIGTERM SIGINT

# Activate Python virtual environment
source .venv/bin/activate

# Start Frontend with explicit PORT environment variable
echo "Starting Frontend on port $PORT..."
cd frontend
# Export PORT so Vite can see it
export PORT=$PORT
echo "Frontend will use PORT: $PORT"
npm run dev &
FRONTEND_PID=$!
echo $FRONTEND_PID > /tmp/frontend.pid
cd ..

# Start API
echo "Starting API..."
cd api
python main.py &
API_PID=$!
echo $API_PID > /tmp/api.pid
cd ..

echo "All services started!"
echo "Frontend PID: $FRONTEND_PID (Port: $PORT)"
echo "API PID: $API_PID"

# Wait for all background processes
wait
