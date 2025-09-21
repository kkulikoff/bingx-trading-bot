#!/bin/bash

# BingX Trading Bot - Stop Script

echo "🛑 Stopping BingX Trading Bot..."

# Check if using Docker
if [ "$USE_DOCKER" = "true" ] && [ -f "docker-compose.yml" ]; then
    echo "🐳 Stopping Docker containers..."
    docker-compose down
    echo "✅ Docker containers stopped"
    exit 0
fi

# Regular stop
if [ ! -f bot.pid ]; then
    echo "❌ No PID file found. Is the bot running?"
    exit 1
fi

PID=$(cat bot.pid)

if ps -p $PID > /dev/null; then
    echo "⏳ Stopping bot process $PID..."
    kill -TERM $PID
    
    # Wait for process to exit
    for i in {1..10}; do
        if ! ps -p $PID > /dev/null; then
            break
        fi
        sleep 1
    done
    
    if ps -p $PID > /dev/null; then
        echo "❌ Process did not stop gracefully, forcing kill..."
        kill -KILL $PID
    fi
    
    rm -f bot.pid
    echo "✅ Bot stopped successfully"
else
    echo "⚠️ Process $PID not found, removing PID file"
    rm -f bot.pid
fi