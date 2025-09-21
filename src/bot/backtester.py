"""
Модуль бэктестинга торговых стратегий.
Обеспечивает тестирование стратегий на исторических данных с расчетом ключевых метрик.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable
import logging
from datetime import datetime
import json
import matplotlib.pyplot as plt
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class Backtester:
    """Класс для тестирования торговых стратегий на исторических данных"""
    
    def __init__(self, initial_balance: float = 10000):
        """
        Инициализация бэктестера.
        
        Args:
            initial_balance: Начальный баланс для тестирования
        """
        self.initial_balance = initial_balance
        self.results = {}
        self.trades = []
        self.equity_curve = []
        
    def run_backtest(self, 
                    historical_data: pd.DataFrame, 
                    strategy_function: Callable,
                    **strategy_params) -> Dict[str, Any]:
        """
        Запуск бэктеста на исторических данных.
        
        Args:
            historical_data: DataFrame с историческими данными
            strategy_function: Функция стратегии для тестирования
            **strategy_params: Параметры стратегии
            
        Returns:
            Dict: Результаты бэктеста
        """
        try:
            logger.info("Запуск бэктеста торговой стратегии")
            
            # Инициализация переменных
            balance = self.initial_balance
            position = None
            self.trades = []
            self.equity_curve = []
            
            # Проходим по всем историческим данным
            for i in range(100, len(historical_data)):
                current_data = historical_data.iloc[:i]
                current_candle = historical_data.iloc[i]
                current_time = current_candle.name
                current_price = current_candle['close']
                
                # Получаем сигнал от стратегии
                signal = strategy_function(current_data, **strategy_params)
                
                # Если есть открытая позиция, проверяем是否需要 закрыть
                if position:
                    # Проверка стоп-лосса и тейк-профита
                    if self._check_exit_conditions(position, current_price, current_time):
                        # Закрываем позицию
                        pnl = self._calculate_pnl(position, current_price)
                        balance += pnl
                        
                        # Сохраняем информацию о сделке
                        trade = {
                            'symbol': position['symbol'],
                            'direction': position['direction'],
                            'entry_price': position['entry_price'],
                            'exit_price': current_price,
                            'quantity': position['quantity'],
                            'pnl': pnl,
                            'entry_time': position['entry_time'],
                            'exit_time': current_time,
                            'duration': (current_time - position['entry_time']).total_seconds() / 60
                        }
                        self.trades.append(trade)
                        position = None
                
                # Если есть сигнал на открытие и нет открытых позиций
                if signal and signal.get('direction') and not position:
                    # Расчет размера позиции на основе риска
                    risk_amount = balance * strategy_params.get('risk_per_trade', 0.02)
                    price_risk_pct = abs(signal['price'] - signal.get('stop_loss', 0)) / signal['price']
                    
                    if price_risk_pct > 0:
                        position_size = risk_amount / (price_risk_pct * signal['price'])
                        
                        # Открываем новую позицию
                        position = {
                            'symbol': signal.get('symbol', 'BTC-USDT'),
                            'direction': signal['direction'],
                            'entry_price': signal['price'],
                            'entry_time': current_time,
                            'quantity': position_size,
                            'stop_loss': signal.get('stop_loss'),
                            'take_profit': signal.get('take_profit')
                        }
                        
                        # Вычитаем стоимость позиции из баланса
                        balance -= position_size * signal['price']
            
                # Записываем текущий баланс в кривую equity
                self.equity_curve.append({
                    'timestamp': current_time,
                    'balance': balance,
                    'price': current_price
                })
            
            # Расчет метрик производительности
            self._calculate_performance_metrics()
            
            logger.info(f"Бэктест завершен. Итоговый баланс: {balance:.2f} USDT")
            return self.results
            
        except Exception as e:
            logger.error(f"Ошибка выполнения бэктеста: {e}")
            raise
    
    def _check_exit_conditions(self, position: Dict[str, Any], current_price: float, current_time: datetime) -> bool:
        """
        Проверка условий для выхода из позиции.
        
        Args:
            position: Информация о позиции
            current_price: Текущая цена
            current_time: Текущее время
            
        Returns:
            bool: True если нужно выйти из позиции
        """
        # Проверка стоп-лосса и тейк-профита
        if position['direction'] == 'LONG':
            if current_price <= position['stop_loss']:
                logger.debug(f"Сработал стоп-лосс: {current_price} <= {position['stop_loss']}")
                return True
            if current_price >= position['take_profit']:
                logger.debug(f"Сработал тейк-профит: {current_price} >= {position['take_profit']}")
                return True
        elif position['direction'] == 'SHORT':
            if current_price >= position['stop_loss']:
                logger.debug(f"Сработал стоп-лосс: {current_price} >= {position['stop_loss']}")
                return True
            if current_price <= position['take_profit']:
                logger.debug(f"Сработал тейк-профит: {current_price} <= {position['take_profit']}")
                return True
        
        # Дополнительные условия выхода (например, по времени)
        # position_duration = (current_time - position['entry_time']).total_seconds() / 3600
        # if position_duration > 24:  # Закрыть через 24 часа
        #     return True
        
        return False
    
    def _calculate_pnl(self, position: Dict[str, Any], exit_price: float) -> float:
        """
        Расчет прибыли/убытка по позиции.
        
        Args:
            position: Информация о позиции
            exit_price: Цена выхода
            
        Returns:
            float: Прибыль/убыток
        """
        if position['direction'] == 'LONG':
            return (exit_price - position['entry_price']) * position['quantity']
        else:  # SHORT
            return (position['entry_price'] - exit_price) * position['quantity']
    
    def _calculate_performance_metrics(self):
        """Расчет метрик производительности стратегии."""
        if not self.trades or not self.equity_curve:
            self.results = {
                'error': 'Нет данных для расчета метрик'
            }
            return
        
        # Базовые метрики
        final_balance = self.equity_curve[-1]['balance']
        total_return = (final_balance - self.initial_balance) / self.initial_balance * 100
        
        # Метрики сделок
        winning_trades = [t for t in self.trades if t['pnl'] > 0]
        losing_trades = [t for t in self.trades if t['pnl'] <= 0]
        
        win_rate = len(winning_trades) / len(self.trades) * 100 if self.trades else 0
        
        avg_profit = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        
        profit_factor = abs(sum(t['pnl'] for t in winning_trades) / 
                           sum(t['pnl'] for t in losing_trades)) if losing_trades else float('inf')
        
        # Расчет максимальной просадки
        balances = [point['balance'] for point in self.equity_curve]
        drawdowns = self._calculate_drawdowns(balances)
        max_drawdown = min(drawdowns) if drawdowns else 0
        
        # Расчет коэффициента Шарпа
        returns = self._calculate_returns(balances)
        sharpe_ratio = self._calculate_sharpe_ratio(returns) if returns else 0
        
        # Сохранение результатов
        self.results = {
            'initial_balance': self.initial_balance,
            'final_balance': final_balance,
            'total_return': total_return,
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'largest_profit': max([t['pnl'] for t in winning_trades]) if winning_trades else 0,
            'largest_loss': min([t['pnl'] for t in losing_trades]) if losing_trades else 0,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'trades': self.trades,
            'equity_curve': self.equity_curve
        }
    
    def _calculate_drawdowns(self, balances: List[float]) -> List[float]:
        """
        Расчет просадок на основе кривой баланса.
        
        Args:
            balances: Список значений баланса
            
        Returns:
            List: Список просадок в процентах
        """
        drawdowns = []
        peak = balances[0]
        
        for balance in balances:
            if balance > peak:
                peak = balance
            drawdown = (balance - peak) / peak * 100
            drawdowns.append(drawdown)
        
        return drawdowns
    
    def _calculate_returns(self, balances: List[float]) -> List[float]:
        """
        Расчет ежедневной доходности.
        
        Args:
            balances: Список значений баланса
            
        Returns:
            List: Список доходностей в процентах
        """
        returns = []
        for i in range(1, len(balances)):
            daily_return = (balances[i] - balances[i-1]) / balances[i-1] * 100
            returns.append(daily_return)
        return returns
    
    def _calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.0) -> float:
        """
        Расчет коэффициента Шарпа.
        
        Args:
            returns: Список доходностей
            risk_free_rate: Безрисковая ставка
            
        Returns:
            float: Коэффициент Шарпа
        """
        if not returns:
            return 0
        
        returns = np.array(returns)
        excess_returns = returns - risk_free_rate
        
        if np.std(excess_returns) == 0:
            return 0
        
        return np.mean(excess_returns) / np.std(excess_returns)
    
    def generate_report(self, report_path: str = None) -> str:
        """
        Генерация отчета о результатах бэктеста.
        
        Args:
            report_path: Путь для сохранения отчета
            
        Returns:
            str: Текст отчета
        """
        if not self.results:
            return "Нет данных для отчета"
        
        report = [
            "=" * 60,
            "ОТЧЕТ О БЭКТЕСТЕ ТОРГОВОЙ СТРАТЕГИИ",
            "=" * 60,
            f"Начальный баланс: {self.results['initial_balance']:.2f} USDT",
            f"Конечный баланс: {self.results['final_balance']:.2f} USDT",
            f"Общая доходность: {self.results['total_return']:.2f}%",
            "",
            "СТАТИСТИКА СДЕЛОК:",
            f"Всего сделок: {self.results['total_trades']}",
            f"Прибыльных сделок: {self.results['winning_trades']}",
            f"Убыточных сделок: {self.results['losing_trades']}",
            f"Процент прибыльных: {self.results['win_rate']:.2f}%",
            f"Средняя прибыль: {self.results['avg_profit']:.2f} USDT",
            f"Средний убыток: {self.results['avg_loss']:.2f} USDT",
            f"Максимальная прибыль: {self.results['largest_profit']:.2f} USDT",
            f"Максимальный убыток: {self.results['largest_loss']:.2f} USDT",
            f"Фактор прибыли: {self.results['profit_factor']:.2f}",
            "",
            "РИСК-МЕТРИКИ:",
            f"Максимальная просадка: {self.results['max_drawdown']:.2f}%",
            f"Коэффициент Шарпа: {self.results['sharpe_ratio']:.2f}",
            "=" * 60
        ]
        
        report_text = "\n".join(report)
        
        # Сохранение отчета в файл
        if report_path:
            try:
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(report_text)
                
                # Сохранение детальных данных в JSON
                data_path = report_path.replace('.txt', '_data.json')
                with open(data_path, 'w', encoding='utf-8') as f:
                    json.dump(self.results, f, indent=2, default=str)
                
                logger.info(f"Отчет сохранен: {report_path}")
            except Exception as e:
                logger.error(f"Ошибка сохранения отчета: {e}")
        
        return report_text
    
    def plot_equity_curve(self, save_path: str = None):
        """
        Построение графика кривой баланса.
        
        Args:
            save_path: Путь для сохранения графика
        """
        if not self.equity_curve:
            logger.warning("Нет данных для построения графика")
            return
        
        # Подготовка данных
        timestamps = [point['timestamp'] for point in self.equity_curve]
        balances = [point['balance'] for point in self.equity_curve]
        prices = [point['price'] for point in self.equity_curve] if 'price' in self.equity_curve[0] else None
        
        # Создание графика
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # График баланса
        ax1.plot(timestamps, balances, label='Баланс', color='blue')
        ax1.set_title('Кривая баланса')
        ax1.set_ylabel('Баланс (USDT)')
        ax1.grid(True)
        ax1.legend()
        
        # График цены (если есть)
        if prices:
            ax2.plot(timestamps, prices, label='Цена', color='green')
            ax2.set_title('Цена актива')
            ax2.set_ylabel('Цена (USDT)')
            ax2.set_xlabel('Время')
            ax2.grid(True)
            ax2.legend()
        
        plt.tight_layout()
        
        # Сохранение графика
        if save_path:
            try:
                plt.savefig(save_path)
                logger.info(f"График сохранен: {save_path}")
            except Exception as e:
                logger.error(f"Ошибка сохранения графика: {e}")
        
        plt.show()


# Пример использования класса Backtester
if __name__ == "__main__":
    # Создание тестовых данных
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    prices = np.random.normal(100, 10, len(dates)).cumsum() + 10000
    
    test_data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': prices + np.random.normal(5, 2, len(dates)),
        'low': prices - np.random.normal(5, 2, len(dates)),
        'close': prices,
        'volume': np.random.normal(1000, 100, len(dates))
    })
    test_data.set_index('timestamp', inplace=True)
    
    # Тестовая стратегия (покупка при падении цены на 2%)
    def test_strategy(data, **kwargs):
        if len(data) < 2:
            return None
        
        current_price = data['close'].iloc[-1]
        previous_price = data['close'].iloc[-2]
        
        if current_price < previous_price * 0.98:  # Падение на 2%
            return {
                'direction': 'LONG',
                'price': current_price,
                'stop_loss': current_price * 0.95,
                'take_profit': current_price * 1.06
            }
        return None
    
    # Запуск бэктеста
    backtester = Backtester(initial_balance=10000)
    results = backtester.run_backtest(
        test_data, 
        test_strategy, 
        risk_per_trade=0.02
    )
    
    # Генерация отчета
    report = backtester.generate_report('backtest_report.txt')
    print(report)
    
    # Построение графика
    backtester.plot_equity_curve('equity_curve.png')