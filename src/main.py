#!/usr/bin/env python3
"""
Main entry point for BingX Trading Bot.
"""

import asyncio
import signal
import sys
import logging
from pathlib import Path
from typing import Optional

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Correct imports without src prefix
from utils.logger import setup_logging
from bot.core import AdvancedBingXBot
from data.cache import init_cache, close_cache
from data.database import init_db
from utils.prometheus_metrics import start_prometheus_server

# Global variables
bot: Optional[AdvancedBingXBot] = None
logger = logging.getLogger(__name__)

def handle_exception(exc_type, exc_value, exc_traceback):
    """Global exception handler for uncaught exceptions"""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    
    logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    asyncio.run(shutdown())

# Set global exception handler
sys.excepthook = handle_exception

async def startup():
    """Initialize the trading bot application"""
    global bot
    
    try:
        # Setup logging
        setup_logging()
        logger.info("Starting BingX Trading Bot")
        
        # Initialize cache
        init_cache()
        logger.info("Cache initialized")
        
        # Initialize database
        await init_db()
        logger.info("Database initialized")
        
        # Create and initialize bot
        bot = AdvancedBingXBot()
        await bot.initialize()
        logger.info("Trading bot initialized")
        
        # Start Prometheus metrics server
        start_prometheus_server(9090)
        logger.info("Prometheus metrics server started on port 9090")
        
        logger.info("BingX Trading Bot started successfully")
        return True
        
    except Exception as e:
        logger.critical(f"Failed to start application: {e}", exc_info=True)
        await shutdown()
        return False

async def run():
    """Main application loop"""
    global bot
    
    if not bot:
        logger.error("Bot not initialized")
        return
    
    try:
        # Main loop
        while getattr(bot, 'is_running', False):
            try:
                # Run one iteration of the bot
                await bot.run_iteration()
                
                # Short sleep to prevent CPU overload
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                await asyncio.sleep(5)
                
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.critical(f"Fatal error in main loop: {e}", exc_info=True)
    finally:
        await shutdown()

async def shutdown():
    """Gracefully shutdown the application"""
    global bot
    
    logger.info("Shutting down BingX Trading Bot")
    
    try:
        # Shutdown bot
        if bot:
            await bot.shutdown()
            logger.info("Trading bot shutdown completed")
        
        # Close cache
        close_cache()
        logger.info("Cache closed")
        
        logger.info("Cleanup completed")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}", exc_info=True)
    finally:
        logger.info("BingX Trading Bot shutdown complete")
        sys.exit(0)

def handle_signal(signum, frame):
    """Handle OS signals for graceful shutdown"""
    logger.info(f"Received signal {signum}, initiating shutdown")
    asyncio.create_task(shutdown())

async def main():
    """Main application entry point"""
    # Setup signal handlers
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    
    # Initialize application
    success = await startup()
    if not success:
        logger.critical("Failed to initialize application")
        sys.exit(1)
    
    # Run main application loop
    await run()

if __name__ == "__main__":
    # Run the application
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
    except Exception as e:
        logger.critical(f"Application failed: {e}", exc_info=True)
        sys.exit(1)