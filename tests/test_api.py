"""
Tests for BingX API client.
"""
import pytest
from unittest.mock import patch, AsyncMock
import aiohttp
from src.bot.api_client import BingXAPI

@pytest.mark.asyncio
async def test_api_initialization():
    """Test API client initialization."""
    api = BingXAPI()
    assert api.api_key is not None
    assert api.secret_key is not None
    assert api.base_url == "https://open-api.bingx.com"

@pytest.mark.asyncio
async def test_generate_signature():
    """Test signature generation."""
    api = BingXAPI()
    params = {'symbol': 'BTC-USDT', 'timestamp': 1640995200000}
    signature = api._generate_signature(params)
    
    assert signature is not None
    assert isinstance(signature, str)
    assert len(signature) == 64  # SHA256 hash length

@pytest.mark.asyncio
async def test_rate_limiting(mock_bingx_api):
    """Test API rate limiting."""
    import time
    
    start_time = time.time()
    for _ in range(5):
        await mock_bingx_api._rate_limit()
    
    # Should have taken at least 0.5 seconds (5 requests * 0.1s delay)
    assert time.time() - start_time >= 0.4

@pytest.mark.asyncio
async def test_get_account_balance(mock_bingx_api):
    """Test getting account balance."""
    balance = await mock_bingx_api.get_account_balance()
    
    assert balance is not None
    assert 'balances' in balance
    assert balance['balances'][0]['asset'] == 'USDT'

@pytest.mark.asyncio
async def test_get_klines(mock_bingx_api):
    """Test getting klines data."""
    klines = await mock_bingx_api.get_klines('BTC-USDT', '15m')
    
    assert klines is not None
    assert 'data' in klines
    assert len(klines['data']) == 2

@pytest.mark.asyncio
async def test_place_order(mock_bingx_api):
    """Test placing an order."""
    order_result = await mock_bingx_api.place_order(
        'BTC-USDT', 'BUY', 0.01
    )
    
    assert order_result is not None
    assert 'orderId' in order_result
    assert order_result['symbol'] == 'BTC-USDT'

@pytest.mark.asyncio
async def test_api_error_handling():
    """Test API error handling."""
    api = BingXAPI()
    api.session = AsyncMock()
    
    # Mock failed request
    response = AsyncMock()
    response.status = 400
    response.json = AsyncMock(return_value={'code': 1001, 'msg': 'Invalid symbol'})
    api.session.get.return_value.__aenter__.return_value = response
    
    result = await api._make_request('GET', '/test', {})
    assert result is None

@pytest.mark.asyncio
async def test_request_timeout():
    """Test request timeout handling."""
    api = BingXAPI()
    api.session = AsyncMock()
    
    # Mock timeout
    api.session.get.side_effect = aiohttp.ClientTimeoutError()
    
    result = await api._make_request('GET', '/test', {})
    assert result is None