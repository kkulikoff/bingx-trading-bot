"""
Tests for SignalGenerator class.
"""
import pytest
import pandas as pd
import numpy as np
from src.bot.signal_generator import SignalGenerator

def test_signal_generator_initialization():
    """Test SignalGenerator initialization."""
    sg = SignalGenerator()
    
    assert sg.indicators is not None
    assert len(sg.indicators) > 0
    assert 'rsi' in sg.indicators
    assert 'macd' in sg.indicators

def test_calculate_indicators(signal_generator, sample_dataframe):
    """Test indicator calculation."""
    indicators = signal_generator.calculate_indicators(sample_dataframe)
    
    assert indicators is not None
    assert isinstance(indicators, dict)
    assert 'rsi' in indicators
    assert 'macd' in indicators
    assert 'bollinger_bands' in indicators
    assert 'atr' in indicators

def test_calculate_rsi(signal_generator, sample_dataframe):
    """Test RSI calculation."""
    rsi = signal_generator._calculate_rsi(sample_dataframe['close'])
    
    assert rsi is not None
    assert isinstance(rsi, pd.Series)
    assert len(rsi) == len(sample_dataframe)
    assert all(0 <= val <= 100 for val in rsi.dropna())

def test_calculate_macd(signal_generator, sample_dataframe):
    """Test MACD calculation."""
    macd, signal_line = signal_generator._calculate_macd(sample_dataframe['close'])
    
    assert macd is not None
    assert signal_line is not None
    assert isinstance(macd, pd.Series)
    assert isinstance(signal_line, pd.Series)
    assert len(macd) == len(sample_dataframe)

def test_calculate_bollinger_bands(signal_generator, sample_dataframe):
    """Test Bollinger Bands calculation."""
    upper, middle, lower = signal_generator._calculate_bollinger_bands(sample_dataframe['close'])
    
    assert upper is not None
    assert middle is not None
    assert lower is not None
    assert isinstance(upper, pd.Series)
    assert isinstance(middle, pd.Series)
    assert isinstance(lower, pd.Series)
    assert all(upper >= middle)
    assert all(middle >= lower)

def test_calculate_atr(signal_generator, sample_dataframe):
    """Test ATR calculation."""
    atr = signal_generator._calculate_atr(sample_dataframe)
    
    assert atr is not None
    assert isinstance(atr, pd.Series)
    assert len(atr) == len(sample_dataframe)
    assert all(val >= 0 for val in atr.dropna())

@pytest.mark.asyncio
async def test_generate_signal(signal_generator, sample_dataframe, ml_model):
    """Test signal generation."""
    signal = await signal_generator.generate_signal(
        sample_dataframe, 'BTC-USDT', '15m', ml_model
    )
    
    # Signal might be None if conditions aren't met
    if signal is not None:
        assert 'symbol' in signal
        assert 'direction' in signal
        assert 'price' in signal
        assert 'confidence' in signal
        assert 'reasons' in signal
        assert signal['symbol'] == 'BTC-USDT'
        assert signal['direction'] in ['LONG', 'SHORT']
        assert 0 <= signal['confidence'] <= 1

def test_generate_from_indicators(signal_generator, sample_dataframe):
    """Test signal generation from indicators."""
    indicators = signal_generator.calculate_indicators(sample_dataframe)
    signal = signal_generator._generate_from_indicators(
        indicators, sample_dataframe, 'BTC-USDT', '15m'
    )
    
    # Signal might be None if conditions aren't met
    if signal is not None:
        assert 'direction' in signal
        assert 'price' in signal
        assert 'confidence' in signal
        assert 'reasons' in signal

def test_rsi_based_signals(signal_generator):
    """Test RSI-based signal generation."""
    # Create test data with oversold condition
    oversold_data = pd.DataFrame({
        'close': [50000, 49000, 48000, 47000, 46000] * 10
    })
    
    indicators = signal_generator.calculate_indicators(oversold_data)
    signal = signal_generator._generate_from_indicators(
        indicators, oversold_data, 'BTC-USDT', '15m'
    )
    
    if signal is not None and signal['direction'] == 'LONG':
        assert any('RSI' in reason for reason in signal['reasons'])

def test_macd_based_signals(signal_generator):
    """Test MACD-based signal generation."""
    # Create test data with MACD crossover
    trend_data = pd.DataFrame({
        'close': np.linspace(50000, 60000, 50)  # Upward trend
    })
    
    indicators = signal_generator.calculate_indicators(trend_data)
    signal = signal_generator._generate_from_indicators(
        indicators, trend_data, 'BTC-USDT', '15m'
    )
    
    if signal is not None and signal['direction'] == 'LONG':
        assert any('MACD' in reason for reason in signal['reasons'])

def test_volatility_expansion_signals(signal_generator):
    """Test volatility expansion signals."""
    # Create test data with high volatility
    volatile_data = pd.DataFrame({
        'open': [50000, 51000, 49000, 52000, 48000],
        'high': [51000, 52000, 50000, 53000, 49000],
        'low': [49000, 50000, 48000, 51000, 47000],
        'close': [50500, 49500, 51500, 48500, 52500]
    })
    
    indicators = signal_generator.calculate_indicators(volatile_data)
    signal = signal_generator._generate_from_indicators(
        indicators, volatile_data, 'BTC-USDT', '15m'
    )
    
    if signal is not None:
        assert any('ATR' in reason or 'volatility' in reason for reason in signal['reasons'])