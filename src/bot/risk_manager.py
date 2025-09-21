"""
Модуль управления рисками торгового бота.
Обеспечивает расчет размера позиций и управление капиталом.
"""
import numpy as np
from typing import Dict, List, Optional, Any
import logging

from src.config.settings import INITIAL_BALANCE, RISK_PER_TRADE, MAX_POSITION_SIZE
from src.utils.logger import setup_logging

logger = setup_logging(__name__)

class RiskManager:
    """Класс для управления рисками и расчета размера позиции"""
    
    def __init__(self, initial_balance: float = None, risk_per_trade: float = None):
        self.balance = initial_balance or INITIAL_BALANCE
        self.risk_per_trade = risk_per_trade or RISK_PER_TRADE
        self.max_position_size = MAX_POSITION_SIZE
        self.positions = {}
        self.equity_curve = []
        self.trade_history = []

    def calculate_position_size(self, entry_price: float, stop_loss_price: float, symbol: str) -> float:
        """
        Расчет размера позиции на основе волатильности и баланса.
        """
        try:
            # Расчет риска в процентах от цены
            price_risk_pct = abs(entry_price - stop_loss_price) / entry_price
            
            # Расчет суммы риска на сделку
            risk_amount = self.balance * self.risk_per_trade
            
            # Расчет размера позиции
            position_size = risk_amount / (price_risk_pct * entry_price)
            
            # Ограничение максимального размера позиции
            max_allowed = self.balance * self.max_position_size / entry_price
            position_size = min(position_size, max_allowed)
            
            # Округление в зависимости от символа
            if "BTC" in symbol:
                position_size = round(position_size, 6)
            elif "ETH" in symbol:
                position_size = round(position_size, 5)
            else:
                position_size = round(position_size, 2)
            
            logger.info(f"Рассчитан размер позиции для {symbol}: {position_size}, риск: {risk_amount:.2f} USDT")
            return max(0, position_size)
            
        except Exception as e:
            logger.error(f"Ошибка расчета размера позиции: {e}")
            return 0

    def update_balance(self, pnl: float):
        """
        Обновление баланса после сделки.
        """
        self.balance += pnl
        self.equity_curve.append({
            'timestamp': np.datetime64('now'),
            'balance': self.balance,
            'pnl': pnl
        })
        logger.info(f"Баланс обновлен: {self.balance:.2f} USDT, PnL: {pnl:.2f} USDT")

    def add_trade(self, trade_data: Dict[str, Any]):
        """
        Добавление сделки в историю.
        """
        self.trade_history.append(trade_data)
        logger.info(f"Добавлена сделка в историю: {trade_data['symbol']} {trade_data['direction']}")

    def get_portfolio_status(self) -> Dict[str, Any]:
        """
        Получение текущего статуса портфеля.
        """
        return {
            'balance': self.balance,
            'risk_per_trade': self.risk_per_trade,
            'current_positions': len(self.positions),
            'equity_curve': self.equity_curve[-100:] if self.equity_curve else [],
            'total_trades': len(self.trade_history),
            'winning_trades': len([t for t in self.trade_history if t.get('pnl', 0) > 0]),
            'losing_trades': len([t for t in self.trade_history if t.get('pnl', 0) <= 0])
        }

    def calculate_portfolio_metrics(self) -> Dict[str, Any]:
        """
        Расчет метрик портфеля.
        """
        if not self.trade_history:
            return {}
        
        # Расчет общей прибыли
        total_pnl = sum(trade.get('pnl', 0) for trade in self.trade_history)
        
        # Расчет процента прибыльных сделок
        winning_trades = len([t for t in self.trade_history if t.get('pnl', 0) > 0])
        win_rate = winning_trades / len(self.trade_history) * 100
        
        # Расчет средней прибыли и убытка
        profit_trades = [t for t in self.trade_history if t.get('pnl', 0) > 0]
        loss_trades = [t for t in self.trade_history if t.get('pnl', 0) <= 0]
        
        avg_profit = np.mean([t.get('pnl', 0) for t in profit_trades]) if profit_trades else 0
        avg_loss = np.mean([t.get('pnl', 0) for t in loss_trades]) if loss_trades else 0
        
        # Расчет коэффициента прибыльности
        profit_factor = abs(avg_profit / avg_loss) if avg_loss != 0 else float('inf')
        
        # Расчет максимальной просадки
        balances = [point['balance'] for point in self.equity_curve] if self.equity_curve else [self.balance]
        peak = balances[0]
        max_drawdown = 0
        
        for balance in balances:
            if balance > peak:
                peak = balance
            drawdown = (peak - balance) / peak * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        return {
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown
        }

    def validate_trade(self, symbol: str, position_size: float, price: float) -> bool:
        """
        Проверка возможности совершения сделки.
        """
        # Проверка достаточности средств
        trade_cost = position_size * price
        if trade_cost > self.balance * self.max_position_size:
            logger.warning(f"Недостаточно средств для сделки {symbol}. Нужно: {trade_cost:.2f}, доступно: {self.balance * self.max_position_size:.2f}")
            return False
        
        # Проверка на существующую позицию
        if symbol in self.positions:
            logger.warning(f"Позиция по {symbol} уже открыта")
            return False
        
        return True