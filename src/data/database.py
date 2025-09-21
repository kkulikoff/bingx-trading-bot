"""
Database module for BingX Trading Bot.
Handles all database operations with async support.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

import aiosqlite
import sqlalchemy
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean

from config.settings import DATABASE_URL

logger = logging.getLogger(__name__)

# SQLAlchemy base class
Base = declarative_base()

class Signal(Base):
    __tablename__ = 'signals'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)
    direction = Column(String(10), nullable=False)
    price = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    risk_reward = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    reasons = Column(Text)
    executed = Column(Boolean, default=False)

class Trade(Base):
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True)
    signal_id = Column(Integer)
    order_id = Column(String(50))
    symbol = Column(String(20), nullable=False)
    direction = Column(String(10), nullable=False)
    quantity = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float)
    pnl = Column(Float)
    status = Column(String(20), default='OPEN')
    opened_at = Column(DateTime, default=datetime.utcnow)
    closed_at = Column(DateTime)
    stop_loss = Column(Float)
    take_profit = Column(Float)

class Portfolio(Base):
    __tablename__ = 'portfolio'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    total_balance = Column(Float, nullable=False)
    available_balance = Column(Float, nullable=False)
    equity = Column(Float, nullable=False)

# Database engine and session
engine = None
AsyncSessionLocal = None

async def init_db():
    """Initialize database connection and create tables"""
    global engine, AsyncSessionLocal
    
    try:
        # For SQLite, use aiosqlite for async support
        if DATABASE_URL.startswith('sqlite'):
            # Create async SQLite engine
            engine = create_async_engine(
                DATABASE_URL.replace('sqlite://', 'sqlite+aiosqlite://'),
                echo=False,
                future=True
            )
        else:
            # For other databases (PostgreSQL)
            engine = create_async_engine(DATABASE_URL, echo=False, future=True)
        
        # Create session factory
        AsyncSessionLocal = sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )
        
        # Create tables
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        logger.info("Database initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        return False

async def get_db_session():
    """Get async database session"""
    if not AsyncSessionLocal:
        await init_db()
    
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            raise e
        finally:
            await session.close()

async def save_signal(signal_data: Dict[str, Any]) -> int:
    """Save signal to database"""
    try:
        async with AsyncSessionLocal() as session:
            signal = Signal(
                symbol=signal_data.get('symbol'),
                direction=signal_data.get('direction'),
                price=signal_data.get('price'),
                confidence=signal_data.get('confidence', 0.0),
                stop_loss=signal_data.get('stop_loss'),
                take_profit=signal_data.get('take_profit'),
                risk_reward=signal_data.get('risk_reward'),
                reasons=str(signal_data.get('reasons', [])),
                executed=signal_data.get('executed', False)
            )
            session.add(signal)
            await session.commit()
            await session.refresh(signal)
            return signal.id
            
    except Exception as e:
        logger.error(f"Error saving signal: {e}")
        return -1

async def save_trade(trade_data: Dict[str, Any]) -> int:
    """Save trade to database"""
    try:
        async with AsyncSessionLocal() as session:
            trade = Trade(
                signal_id=trade_data.get('signal_id'),
                order_id=trade_data.get('order_id'),
                symbol=trade_data.get('symbol'),
                direction=trade_data.get('direction'),
                quantity=trade_data.get('quantity'),
                entry_price=trade_data.get('entry_price'),
                exit_price=trade_data.get('exit_price'),
                pnl=trade_data.get('pnl'),
                status=trade_data.get('status', 'OPEN'),
                opened_at=trade_data.get('opened_at', datetime.utcnow()),
                closed_at=trade_data.get('closed_at'),
                stop_loss=trade_data.get('stop_loss'),
                take_profit=trade_data.get('take_profit')
            )
            session.add(trade)
            await session.commit()
            await session.refresh(trade)
            return trade.id
            
    except Exception as e:
        logger.error(f"Error saving trade: {e}")
        return -1

async def update_portfolio(balance_data: Dict[str, Any]) -> bool:
    """Update portfolio balance"""
    try:
        async with AsyncSessionLocal() as session:
            portfolio = Portfolio(
                total_balance=balance_data.get('total_balance', 0.0),
                available_balance=balance_data.get('available_balance', 0.0),
                equity=balance_data.get('equity', 0.0)
            )
            session.add(portfolio)
            await session.commit()
            return True
            
    except Exception as e:
        logger.error(f"Error updating portfolio: {e}")
        return False

async def get_recent_signals(limit: int = 50) -> List[Dict[str, Any]]:
    """Get recent signals from database"""
    try:
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                sqlalchemy.select(Signal)
                .order_by(Signal.timestamp.desc())
                .limit(limit)
            )
            signals = result.scalars().all()
            return [{
                'id': s.id,
                'symbol': s.symbol,
                'direction': s.direction,
                'price': s.price,
                'confidence': s.confidence,
                'timestamp': s.timestamp,
                'executed': s.executed
            } for s in signals]
            
    except Exception as e:
        logger.error(f"Error getting recent signals: {e}")
        return []

async def get_open_trades() -> List[Dict[str, Any]]:
    """Get all open trades"""
    try:
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                sqlalchemy.select(Trade)
                .where(Trade.status == 'OPEN')
                .order_by(Trade.opened_at.desc())
            )
            trades = result.scalars().all()
            return [{
                'id': t.id,
                'symbol': t.symbol,
                'direction': t.direction,
                'quantity': t.quantity,
                'entry_price': t.entry_price,
                'status': t.status,
                'opened_at': t.opened_at
            } for t in trades]
            
    except Exception as e:
        logger.error(f"Error getting open trades: {e}")
        return []

async def close_db():
    """Close database connection"""
    if engine:
        await engine.dispose()
        logger.info("Database connection closed")