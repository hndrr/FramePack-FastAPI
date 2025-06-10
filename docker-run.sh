#!/bin/bash

# Docker run script for FramePack-FastAPI

set -e

echo "🐳 FramePack-FastAPI Docker Run Script"
echo "======================================"

# Parse command line arguments
DETACHED=false
PORT="8000"

while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--detached)
            DETACHED=true
            shift
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -d, --detached Run in detached mode"
            echo "  -p, --port    Port to expose (default: 8000)"
            echo "  -h, --help    Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set compose file
COMPOSE_FILE="docker-compose.yml"
echo "🚀 Starting FramePack-FastAPI..."
echo "⚠️  This requires NVIDIA GPU with 24GB+ VRAM!"

echo "📋 Run Configuration:"
echo "   Compose file: $COMPOSE_FILE"
echo "   Port: $PORT"
echo "   Detached: $DETACHED"
echo ""

# Create required directories
echo "📁 Creating required directories..."
mkdir -p outputs/images temp_queue_images loras hf_download

# Update port in compose file if different from 8000
if [ "$PORT" != "8000" ]; then
    echo "🔧 Updating port configuration..."
    sed -i.bak "s/8000:8000/$PORT:8000/g" "$COMPOSE_FILE"
fi

# Run docker-compose
if [ "$DETACHED" = true ]; then
    echo "🔄 Starting containers in detached mode..."
    docker-compose -f "$COMPOSE_FILE" up -d
else
    echo "🔄 Starting containers..."
    docker-compose -f "$COMPOSE_FILE" up
fi

# Restore original compose file if port was changed
if [ "$PORT" != "8000" ] && [ -f "$COMPOSE_FILE.bak" ]; then
    mv "$COMPOSE_FILE.bak" "$COMPOSE_FILE"
fi

if [ "$DETACHED" = true ]; then
    echo ""
    echo "🎉 Containers started successfully!"
    echo ""
    echo "API is available at: http://localhost:$PORT"
    echo "API docs: http://localhost:$PORT/docs"
    echo ""
    echo "To view logs:"
    echo "   docker-compose -f $COMPOSE_FILE logs -f"
    echo ""
    echo "To stop:"
    echo "   docker-compose -f $COMPOSE_FILE down"
fi