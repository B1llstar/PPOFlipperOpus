#!/bin/bash

# PPO Flipper Dashboard Startup Script

echo "Starting PPO Flipper Dashboard..."

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Start backend server in background
echo "Starting backend server on http://localhost:8000..."
cd "$PROJECT_ROOT"
python dashboard/backend/server.py &
BACKEND_PID=$!

# Wait for backend to start
sleep 2

# Start frontend dev server
echo "Starting frontend on http://localhost:5173..."
cd "$SCRIPT_DIR/frontend"
npm run dev &
FRONTEND_PID=$!

echo ""
echo "Dashboard is running!"
echo "  - Backend API: http://localhost:8000"
echo "  - Frontend:    http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop both servers..."

# Handle cleanup on exit
cleanup() {
    echo ""
    echo "Stopping servers..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit 0
}

trap cleanup INT TERM

# Wait for both processes
wait
