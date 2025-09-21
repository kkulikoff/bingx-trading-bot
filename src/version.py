"""
BingX Trading Bot package.
Advanced algorithmic trading system for BingX exchange.
"""

__version__ = "1.0.0"
__author__ = "BingX Trading Bot Team"
__email__ = "support@bingx-trading-bot.com"

# Import key components for easier access
from src.bot.core import AdvancedBingXBot
from src.web.app import create_app

__all__ = ['AdvancedBingXBot', 'create_app']