#!/bin/bash

# PPO Flipper Dashboard Startup Script
# Starts both the backend API server and Vue frontend

echo "=============================================="
echo "PPO Flipper Dashboard"
echo "=============================================="

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$DIR/.." && pwd )"

# Activate virtual environment
source "$PROJECT_ROOT/venv/bin/activate"

# Start backend in background
echo "Starting backend API server on http://localhost:5001..."
cd "$DIR/backend"
python server.py &
BACKEND_PID=$!

# Wait for backend to start
sleep 2

# Start frontend
echo "Starting Vue frontend on http://localhost:5173..."
cd "$DIR"
npm run dev &
FRONTEND_PID=$!

# Wait for frontend to start
sleep 3

echo ""
echo "=============================================="
echo "Dashboard is running!"
echo "  Frontend: http://localhost:5173"
echo "  Backend:  http://localhost:5001"
echo "=============================================="
echo ""
echo "Press Ctrl+C to stop both servers"

# Handle cleanup on exit
cleanup() {
    echo ""
    echo "Shutting down..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM

# Wait for processes
wait
