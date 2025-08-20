#!/bin/bash

echo "==================================="
echo "  Datarus Pipeline Setup"
echo "==================================="

# Check if Python 3 exists
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not found"
    exit 1
fi

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed"
    exit 1
fi

# Install dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Setup configuration
if [ ! -f config/config.yaml ]; then
    echo "Creating configuration file..."
    cp config/config.yaml.example config/config.yaml
fi

# Setup environment
if [ ! -f .env ]; then
    echo "Creating environment file..."
    cp .env.example .env
fi

# Pull Docker images
echo "Pulling Docker images..."
docker pull jupyter/tensorflow-notebook

mkdir -p notebooks

echo ""
echo "✓ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit config/config.yaml with your settings"
echo "2. Set REMOTE_SERVER_IP in .env (default: 127.0.0.1)"
echo "3. Run: python src/main.py --challenge challenges/datascience_challenge_0bdebbb9.json"
