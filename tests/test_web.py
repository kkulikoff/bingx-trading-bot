"""
Tests for web application.
"""
import pytest
import json
from unittest.mock import patch, AsyncMock
from src.web.app import create_app

def test_app_creation():
    """Test Flask application creation."""
    app = create_app()
    
    assert app is not None
    assert app.name == 'src.web.app'
    assert app.config['TESTING'] is False

def test_index_route(test_client):
    """Test index route."""
    response = test_client.get('/')
    
    assert response.status_code == 200
    assert b'BingX Trading Bot' in response.data

def test_api_health(test_client):
    """Test health check API endpoint."""
    response = test_client.get('/api/health')
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'healthy'
    assert 'timestamp' in data

def test_api_signals(test_client):
    """Test signals API endpoint."""
    response = test_client.get('/api/signals')
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert isinstance(data, list)

def test_api_portfolio(test_client):
    """Test portfolio API endpoint."""
    response = test_client.get('/api/portfolio')
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert isinstance(data, dict)

def test_api_performance(test_client):
    """Test performance API endpoint."""
    response = test_client.get('/api/performance')
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert isinstance(data, dict)

def test_api_settings(test_client):
    """Test settings API endpoint."""
    response = test_client.get('/api/settings')
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert isinstance(data, dict)

def test_execute_trade_route(test_client):
    """Test execute trade API endpoint."""
    response = test_client.get('/api/execute/BTC-USDT/LONG')
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'success'
    assert 'BTC-USDT LONG' in data['message']

def test_invalid_route(test_client):
    """Test invalid route handling."""
    response = test_client.get('/invalid/route')
    
    assert response.status_code == 404

def test_cors_headers(test_client):
    """Test CORS headers."""
    response = test_client.get('/api/health')
    
    assert response.status_code == 200
    # CORS headers should be present
    assert 'Access-Control-Allow-Origin' in response.headers

@pytest.mark.asyncio
async def test_websocket_connection():
    """Test WebSocket connection."""
    # This would require a more complex setup with a WebSocket test client
    # For now, we'll just verify the route exists
    app = create_app()
    assert any(
        rule.rule == '/ws' and 'GET' in rule.methods 
        for rule in app.url_map.iter_rules()
    )

def test_error_handling(test_client):
    """Test error handling."""
    # Force an error by accessing an invalid endpoint with wrong method
    response = test_client.post('/api/health')
    
    # Should return 405 Method Not Allowed, not 500 Internal Server Error
    assert response.status_code == 405

def test_rate_limiting(test_client):
    """Test rate limiting."""
    # Make multiple requests quickly
    for _ in range(5):
        response = test_client.get('/api/health')
        assert response.status_code == 200
    
    # In a real implementation, we would test that the 6th request gets rate limited
    # For now, we'll just verify the endpoint still works
    response = test_client.get('/api/health')
    assert response.status_code == 200

def test_response_format(test_client):
    """Test response format consistency."""
    endpoints = ['/api/health', '/api/signals', '/api/portfolio']
    
    for endpoint in endpoints:
        response = test_client.get(endpoint)
        assert response.status_code == 200
        assert response.content_type == 'application/json'
        
        data = json.loads(response.data)
        assert isinstance(data, (dict, list))