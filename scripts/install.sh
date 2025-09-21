#!/bin/bash

# BingX Trading Bot - Installation Script
set -e

echo "🔧 Installing BingX Trading Bot..."

# Check if Python 3.9+ is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3.9+ is required but not installed"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [[ $(echo "$PYTHON_VERSION < 3.9" | bc -l) -eq 1 ]]; then
    echo "❌ Python 3.9+ is required. Current version: $PYTHON_VERSION"
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🚀 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📚 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs backups data/models data/cache data/historical

# Copy environment template
if [ ! -f .env ]; then
    echo "📋 Copying environment template..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your actual configuration"
fi

# Initialize database
echo "🗄️ Initializing database..."
python -c "
from src.data.database import init_db
import asyncio
asyncio.run(init_db())
"

echo "✅ Installation completed successfully!"
echo "📝 Next steps:"
echo "   1. Edit .env file with your API keys and settings"
echo "   2. Run: ./scripts/start.sh to start the bot"
echo "   3. Run: ./scripts/monitor.sh to check status"