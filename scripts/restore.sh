#!/bin/bash

# BingX Trading Bot - Restore Script
set -e

echo "🔄 Restoring BingX Trading Bot from backup..."

if [ $# -eq 0 ]; then
    echo "❌ Please specify backup file to restore"
    echo "Usage: $0 <backup_file.tar.gz>"
    ls -la backups/backup_*.tar.gz 2>/dev/null || echo "No backups found"
    exit 1
fi

BACKUP_FILE=$1

if [ ! -f "$BACKUP_FILE" ]; then
    echo "❌ Backup file $BACKUP_FILE not found"
    exit 1
fi

# Stop bot if running
if [ -f bot.pid ] && ps -p $(cat bot.pid) > /dev/null; then
    echo "⏳ Stopping bot before restore..."
    ./scripts/stop.sh
fi

# Create temporary directory
TEMP_DIR=$(mktemp -d)

echo "📂 Extracting backup..."
tar -xzf $BACKUP_FILE -C $TEMP_DIR

echo "🔄 Restoring files..."

# Restore database
if [ -f "$TEMP_DIR/data/trading_bot.db" ]; then
    echo "💾 Restoring database..."
    mkdir -p data
    cp $TEMP_DIR/data/trading_bot.db data/
fi

# Restore models
if [ -d "$TEMP_DIR/data/models" ]; then
    echo "🧠 Restoring ML models..."
    mkdir -p data/models
    cp -r $TEMP_DIR/data/models/* data/models/
fi

# Restore configuration
if [ -f "$TEMP_DIR/.env" ]; then
    echo "⚙️ Restoring configuration..."
    cp $TEMP_DIR/.env .env.restored
    echo "⚠️ Configuration restored to .env.restored - please review before using"
fi

# Cleanup
rm -rf $TEMP_DIR

echo "✅ Restore completed successfully from $BACKUP_FILE"
echo "🚀 You can now start the bot with: ./scripts/start.sh"