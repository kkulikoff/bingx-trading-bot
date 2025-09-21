"""
Pytest configuration and fixtures for BingX Trading Bot tests.
"""
import pytest
import asyncio
import aiohttp
from unittest.mock import AsyncMock, MagicMock
import sys
import os

# Add source directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.bot.api_client import BingXAPI
from src.bot.risk_manager import RiskManager
from src.bot.signal_generator import SignalGenerator
from src.bot.trading_engine import TradingEngine
from src.bot.ml_model import MLModel
from src.web.app import create_app
from src.data.database import init_db, get_engine

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def mock_api_response():
    """Fixture for mock API responses."""
    def _create_response(data=None, status=200):
        response = MagicMock()
        response.status = status
        response.json = AsyncMock(return_value=data or {})
        return response
    return _create_response

@pytest.fixture
async def mock_bingx_api():
    """Fixture for mock BingX API client."""
    api = BingXAPI()
    api.session = AsyncMock()
    
    # Mock API methods
    api.get_account_balance = AsyncMock(return_value={
        'balances': [{'asset': 'USDT', 'free': '10000', 'locked': '0'}]
    })
    
    api.get_klines = AsyncMock(return_value={
        'data': [
            [1640995200000, '50000', '51000', '49000', '50500', '1000'],
            [1640998800000, '50500', '51500', '50000', '51000', '1200']
        ]
    })
    
    api.place_order = AsyncMock(return_value={
        'orderId': '123456',
        'symbol': 'BTC-USDT',
        'status': 'FILLED'
    })
    
    return api

@pytest.fixture
def sample_ohlc_data():
    """Fixture for sample OHLC data."""
    return {
        'timestamp': [1640995200000, 1640998800000],
        'open': [50000.0, 50500.0],
        'high': [51000.0, 51500.0],
        'low': [49000.0, 50000.0],
        'close': [50500.0, 51000.0],
        'volume': [1000.0, 1200.0]
    }

@pytest.fixture
def sample_dataframe(sample_ohlc_data):
    """Fixture for sample DataFrame."""
    import pandas as pd
    return pd.DataFrame(sample_ohlc_data)

@pytest.fixture
def risk_manager():
    """Fixture for RiskManager instance."""
    return RiskManager(initial_balance=10000, risk_per_trade=0.02)

@pytest.fixture
def signal_generator():
    """Fixture for SignalGenerator instance."""
    return SignalGenerator()

@pytest.fixture
async def trading_engine(mock_bingx_api, risk_manager):
    """Fixture for TradingEngine instance."""
    return TradingEngine(mock_bingx_api, risk_manager)

@pytest.fixture
def ml_model():
    """Fixture for MLModel instance."""
    model = MLModel()
    model.is_trained = True
    model.predict = MagicMock(return_value=0.75)
    return model

@pytest.fixture
def test_app():
    """Fixture for Flask test application."""
    app = create_app()
    app.config['TESTING'] = True
    return app

@pytest.fixture
def test_client(test_app):
    """Fixture for Flask test client."""
    return test_app.test_client()

@pytest.fixture(autouse=True)
async def setup_database():
    """Fixture to set up test database."""
    await init_db(testing=True)
    yield
    # Clean up after tests
    engine = get_engine()
    await engine.dispose()

@pytest.fixture
def sample_signal():
    """Fixture for sample trade signal."""
    return {
        'symbol': 'BTC-USDT',
        'direction': 'LONG',
        'price': 51000.0,
        'confidence': 0.85,
        'stop_loss': 50000.0,
        'take_profit': 53000.0,
        'risk_reward': 2.0,
        'timestamp': '2023-01-01T12:00:00'
    }