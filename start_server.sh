#!/bin/bash

# FramePack-FastAPI Server Startup Script for uv

echo "Starting FramePack-FastAPI server with uv..."

# Use full path to uv
UV_PATH="$HOME/.local/bin/uv"

# Check if uv is available
if [ ! -f "$UV_PATH" ]; then
    echo "Error: uv is not installed at $UV_PATH"
    echo "Please install uv first: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    "$UV_PATH" venv --python 3.10
fi

# Install basic dependencies if not already installed
echo "Installing/checking dependencies..."
"$UV_PATH" pip install fastapi uvicorn python-multipart pydantic watchdog

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Start the server
echo "Starting FastAPI server..."
echo "Server will be available at: http://localhost:8080"
echo "API docs will be available at: http://localhost:8080/docs"
echo ""

# Use uv run to execute in the virtual environment
"$UV_PATH" run uvicorn api.api:app --host 0.0.0.0 --port 8080 --reload