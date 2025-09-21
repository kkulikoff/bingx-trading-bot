#!/bin/bash

# BingX Trading Bot - Startup Script
set -e

echo "🚀 Starting BingX Trading Bot..."

# Check if .env exists
if [ ! -f .env ]; then
    echo "❌ .env file not found. Please run ./scripts/install.sh first"
    exit 1
fi

# Load environment variables
export $(grep -v '^#' .env | xargs)

# Check if using Docker
if [ "$USE_DOCKER" = "true" ]; then
    echo "🐳 Starting with Docker Compose..."
    docker-compose up -d
    echo "✅ Bot started in Docker container"
    exit 0
fi

# Regular startup
echo "🔍 Checking if bot is already running..."
if [ -f bot.pid ] && ps -p $(cat bot.pid) > /dev/null; then
    echo "⚠️ Bot is already running with PID $(cat bot.pid)"
    echo "ℹ️ Use ./scripts/stop.sh to stop it first"
    exit 1
fi

# Activate virtual environment
if [ -d "venv" ]; then
    echo "🚀 Activating virtual environment..."
    source venv/bin/activate
fi

# Start the bot
echo "🤖 Starting trading bot..."
nohup python -m src.main > logs/bot_console.log 2>&1 &

# Save PID
echo $! > bot.pid

echo "✅ Bot started successfully with PID $(cat bot.pid)"
echo "📋 Logs are being written to logs/bot_console.log"
echo "🌐 Web interface will be available at http://${HOST:-localhost}:${PORT:-5000}"