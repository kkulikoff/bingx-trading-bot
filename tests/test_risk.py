"""
Tests for RiskManager class.
"""
import pytest
import numpy as np
from src.bot.risk_manager import RiskManager

def test_risk_manager_initialization():
    """Test RiskManager initialization."""
    rm = RiskManager(initial_balance=10000, risk_per_trade=0.02)
    
    assert rm.balance == 10000
    assert rm.risk_per_trade == 0.02
    assert rm.max_position_size == 0.1
    assert len(rm.positions) == 0
    assert len(rm.equity_curve) == 0

def test_calculate_position_size(risk_manager):
    """Test position size calculation."""
    # Test LONG position
    position_size = risk_manager.calculate_position_size(
        50000,  # entry_price
        49000,  # stop_loss_price
        'BTC-USDT'
    )
    
    # Risk amount = 10000 * 0.02 = 200 USDT
    # Price risk = (50000 - 49000) / 50000 = 0.02 (2%)
    # Position size = 200 / (0.02 * 50000) = 0.2 BTC
    expected_size = 200 / (0.02 * 50000)
    assert position_size == pytest.approx(expected_size, 0.01)
    
    # Test with zero risk (should return 0)
    zero_risk_manager = RiskManager(initial_balance=0, risk_per_trade=0.02)
    position_size = zero_risk_manager.calculate_position_size(
        50000, 49000, 'BTC-USDT'
    )
    assert position_size == 0

def test_update_balance(risk_manager):
    """Test balance update."""
    initial_balance = risk_manager.balance
    pnl = 500
    
    risk_manager.update_balance(pnl)
    
    assert risk_manager.balance == initial_balance + pnl
    assert len(risk_manager.equity_curve) == 1
    assert risk_manager.equity_curve[0]['pnl'] == pnl

def test_add_trade(risk_manager):
    """Test adding trade to history."""
    trade_data = {
        'symbol': 'BTC-USDT',
        'direction': 'LONG',
        'entry_price': 50000,
        'quantity': 0.1,
        'pnl': 500
    }
    
    risk_manager.add_trade(trade_data)
    
    assert len(risk_manager.trade_history) == 1
    assert risk_manager.trade_history[0]['symbol'] == 'BTC-USDT'

def test_get_portfolio_status(risk_manager):
    """Test getting portfolio status."""
    status = risk_manager.get_portfolio_status()
    
    assert 'balance' in status
    assert 'risk_per_trade' in status
    assert 'current_positions' in status
    assert 'equity_curve' in status
    assert 'total_trades' in status

def test_calculate_portfolio_metrics(risk_manager):
    """Test portfolio metrics calculation."""
    # Add some trades
    risk_manager.add_trade({'symbol': 'BTC-USDT', 'direction': 'LONG', 'pnl': 500})
    risk_manager.add_trade({'symbol': 'ETH-USDT', 'direction': 'SHORT', 'pnl': -200})
    risk_manager.add_trade({'symbol': 'SOL-USDT', 'direction': 'LONG', 'pnl': 300})
    
    metrics = risk_manager.calculate_portfolio_metrics()
    
    assert 'total_pnl' in metrics
    assert 'win_rate' in metrics
    assert 'avg_profit' in metrics
    assert 'avg_loss' in metrics
    assert 'profit_factor' in metrics
    assert 'max_drawdown' in metrics
    
    assert metrics['total_pnl'] == 600  # 500 - 200 + 300
    assert metrics['win_rate'] == pytest.approx(66.67, 0.01)  # 2 wins out of 3

def test_validate_trade(risk_manager):
    """Test trade validation."""
    # Valid trade
    is_valid = risk_manager.validate_trade('BTC-USDT', 0.1, 50000)
    assert is_valid is True
    
    # Invalid trade (insufficient funds)
    is_valid = risk_manager.validate_trade('BTC-USDT', 10.0, 50000)
    assert is_valid is False
    
    # Add a position and test duplicate validation
    risk_manager.positions['BTC-USDT'] = {'quantity': 0.1}
    is_valid = risk_manager.validate_trade('BTC-USDT', 0.1, 50000)
    assert is_valid is False

def test_max_drawdown_calculation(risk_manager):
    """Test maximum drawdown calculation."""
    # Simulate equity curve with a drawdown
    risk_manager.equity_curve = [
        {'balance': 10000, 'pnl': 0},
        {'balance': 11000, 'pnl': 1000},  # peak
        {'balance': 10500, 'pnl': 500},
        {'balance': 9500, 'pnl': -500},   # trough
        {'balance': 10000, 'pnl': 0}
    ]
    
    metrics = risk_manager.calculate_portfolio_metrics()
    
    # Drawdown from 11000 to 9500 = (11000-9500)/11000 ≈ 13.64%
    assert metrics['max_drawdown'] == pytest.approx(13.64, 0.01)