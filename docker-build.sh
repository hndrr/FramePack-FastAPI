#!/bin/bash

# Docker build script for FramePack-FastAPI

set -e

echo "🐳 FramePack-FastAPI Docker Build Script"
echo "========================================"

# Parse command line arguments
PUSH=false
TAG="latest"

while [[ $# -gt 0 ]]; do
    case $1 in
        --push)
            PUSH=true
            shift
            ;;
        --tag)
            TAG="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --push    Push to registry after build"
            echo "  --tag     Docker image tag (default: latest)"
            echo "  -h, --help Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set image name and dockerfile
IMAGE_NAME="framepack-fastapi"
DOCKERFILE="Dockerfile"
echo "🚀 Building FramePack-FastAPI..."
echo "⚠️  This requires NVIDIA GPU for inference!"

FULL_IMAGE_NAME="$IMAGE_NAME:$TAG"

echo "📋 Build Configuration:"
echo "   Type: $BUILD_TYPE"
echo "   Image: $FULL_IMAGE_NAME"
echo "   Dockerfile: $DOCKERFILE"
echo "   Push: $PUSH"
echo ""

# Build the image
echo "🔨 Building Docker image..."
docker build -f "$DOCKERFILE" -t "$FULL_IMAGE_NAME" .

if [ $? -eq 0 ]; then
    echo "✅ Build completed successfully!"
    echo "   Image: $FULL_IMAGE_NAME"
else
    echo "❌ Build failed!"
    exit 1
fi

# Push if requested
if [ "$PUSH" = true ]; then
    echo "📤 Pushing image to registry..."
    docker push "$FULL_IMAGE_NAME"
    
    if [ $? -eq 0 ]; then
        echo "✅ Push completed successfully!"
    else
        echo "❌ Push failed!"
        exit 1
    fi
fi

echo ""
echo "🎉 All done!"
echo ""
echo "To run the container:"
if [ "$BUILD_TYPE" = "gpu" ]; then
    echo "   docker run --gpus all -p 8000:8000 $FULL_IMAGE_NAME"
    echo ""
    echo "Or use docker-compose:"
    echo "   docker-compose --profile gpu up"
    echo "   # or"
    echo "   docker-compose -f docker-compose.gpu.yml up"
    echo ""
    echo "⚠️  Requires NVIDIA GPU with 24GB+ VRAM for full functionality"
else
    echo "   docker run -p 8000:8000 $FULL_IMAGE_NAME"
    echo ""
    echo "Or use docker-compose:"
    echo "   docker-compose up"
    echo ""
    echo "ℹ️  This is development server only - API will return 503 for inference requests"
fi

echo ""
echo "API will be available at: http://localhost:8000"
echo "API docs: http://localhost:8000/docs"