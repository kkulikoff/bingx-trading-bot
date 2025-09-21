#!/bin/bash

# BingX Trading Bot - Backup Script
set -e

echo "📦 Creating backup of BingX Trading Bot..."

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="backups/backup_$TIMESTAMP"
BACKUP_FILE="backups/backup_$TIMESTAMP.tar.gz"

# Create backup directory
mkdir -p $BACKUP_DIR

echo "🔍 Collecting data for backup..."

# Backup database
if [ -f "data/trading_bot.db" ]; then
    echo "💾 Backing up database..."
    cp data/trading_bot.db $BACKUP_DIR/
fi

# Backup configuration
echo "⚙️ Backing up configuration..."
cp .env $BACKUP_DIR/ 2>/dev/null || echo "⚠️ No .env file found"

# Backup models
echo "🧠 Backing up ML models..."
if [ -d "data/models" ]; then
    mkdir -p $BACKUP_DIR/models
    cp -r data/models/* $BACKUP_DIR/models/ 2>/dev/null || echo "⚠️ No models found"
fi

# Backup logs (last 7 days)
echo "📝 Backing up recent logs..."
find logs/ -name "*.log" -mtime -7 -exec cp --parents {} $BACKUP_DIR/ \; 2>/dev/null || echo "⚠️ No logs found"

# Create archive
echo "🗜️ Creating backup archive..."
tar -czf $BACKUP_FILE -C $BACKUP_DIR .

# Cleanup
rm -rf $BACKUP_DIR

# Remove old backups (keep last 30)
echo "🧹 Cleaning up old backups..."
ls -tp backups/backup_*.tar.gz | grep -v '/$' | tail -n +31 | xargs -I {} rm -- {}

echo "✅ Backup completed successfully: $BACKUP_FILE"
echo "📊 Backup size: $(du -h $BACKUP_FILE | cut -f1)"