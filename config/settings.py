"""
Настройки приложения торгового бота.
Все настройки загружаются из переменных окружения.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# BingX API Configuration
BINGX_API_KEY = os.getenv('BINGX_API_KEY', '')
BINGX_SECRET_KEY = os.getenv('BINGX_SECRET_KEY', '')

# Application Settings
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
DEBUG = os.getenv('DEBUG', 'false').lower() == 'true'
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
HOST = os.getenv('HOST', '0.0.0.0')
PORT = int(os.getenv('PORT', '5000'))

# Database Configuration
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///data/trading_bot.db')

# ML Model Settings
ML_MODEL_PATH = os.getenv('ML_MODEL_PATH', 'data/models/trading_model.pkl')
ML_SCALER_PATH = os.getenv('ML_SCALER_PATH', 'data/models/scaler.pkl')

# Risk Management
INITIAL_BALANCE = float(os.getenv('INITIAL_BALANCE', '10000'))
RISK_PER_TRADE = float(os.getenv('RISK_PER_TRADE', '0.02'))
MAX_POSITION_SIZE = float(os.getenv('MAX_POSITION_SIZE', '0.1'))

# Trading Symbols
SYMBOLS = [s.strip() for s in os.getenv('SYMBOLS', 'BTC-USDT,ETH-USDT,SOL-USDT,XRP-USDT').split(',')]
TIMEFRAMES = [t.strip() for t in os.getenv('TIMEFRAMES', '15m,1h,4h').split(',')]

# Backup Configuration
BACKUP_SCHEDULE = os.getenv('BACKUP_SCHEDULE', 'daily')
MAX_BACKUPS = int(os.getenv('MAX_BACKUPS', '30'))

# Notification Settings
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')
ALERT_THRESHOLD = int(os.getenv('ALERT_THRESHOLD', '5'))

# External APIs
NEWS_API_KEY = os.getenv('NEWS_API_KEY', '')
ALPHA_VANTAGE_KEY = os.getenv('ALPHA_VANTAGE_KEY', '')

# Security
ENCRYPTION_KEY = os.getenv('ENCRYPTION_KEY', '')
API_RATE_LIMIT = int(os.getenv('API_RATE_LIMIT', '120'))
REQUEST_TIMEOUT = int(os.getenv('REQUEST_TIMEOUT', '30'))

# Валидация обязательных параметров
if not BINGX_API_KEY or not BINGX_SECRET_KEY:
    raise ValueError("BingX API ключи обязательны для работы бота")