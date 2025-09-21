#!/bin/bash

# BingX Trading Bot - Monitoring Script

echo "📊 BingX Trading Bot Status Monitor"
echo "==================================="

# Check if using Docker
if [ "$USE_DOCKER" = "true" ] && [ -f "docker-compose.yml" ]; then
    echo "🐳 Docker Containers Status:"
    docker-compose ps
    echo ""
    
    echo "📊 Docker Logs (last 10 lines):"
    docker-compose logs --tail=10
    echo ""
    
    # Check web interface
    if curl -s -f http://localhost:${PORT:-5000}/api/health > /dev/null; then
        echo "✅ Web interface is responding"
    else
        echo "❌ Web interface is not responding"
    fi
    
    exit 0
fi

# Regular monitoring
if [ ! -f bot.pid ]; then
    echo "❌ Bot is not running (no PID file)"
    exit 1
fi

PID=$(cat bot.pid)

if ps -p $PID > /dev/null; then
    echo "✅ Bot is running with PID $PID"
    echo ""
    
    # Check memory usage
    MEMORY_USAGE=$(ps -o rss= -p $PID | awk '{printf "%.2f MB", $1/1024}')
    echo "📈 Memory Usage: $MEMORY_USAGE"
    
    # Check CPU usage
    CPU_USAGE=$(ps -o %cpu= -p $PID)
    echo "📊 CPU Usage: ${CPU_USAGE:-0}%"
    
    # Check uptime
    START_TIME=$(ps -o lstart= -p $PID)
    echo "⏰ Started: $START_TIME"
    
    # Check web interface
    if curl -s -f http://localhost:${PORT:-5000}/api/health > /dev/null; then
        echo "✅ Web interface is responding"
        
        # Get basic stats
        echo ""
        echo "🤖 Bot Statistics:"
        curl -s http://localhost:${PORT:-5000}/api/performance | \
            python3 -c "
import json, sys
data = json.load(sys.stdin)
print(f'  Total Trades: {data.get(\"total_trades\", 0)}')
print(f'  Winning Trades: {data.get(\"winning_trades\", 0)}')
print(f'  Win Rate: {data.get(\"win_rate\", 0):.2f}%')
print(f'  Total PnL: {data.get(\"total_pnl\", 0):.2f} USDT')
"
    else
        echo "❌ Web interface is not responding"
    fi
    
    # Show recent logs
    echo ""
    echo "📋 Recent Logs:"
    tail -n 10 logs/bot_console.log 2>/dev/null || echo "No log file found"
    
else
    echo "❌ Bot is not running (process $PID not found)"
    rm -f bot.pid
fi

echo ""
echo "==================================="