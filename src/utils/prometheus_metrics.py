"""
Модуль для работы с Prometheus метриками торгового бота.
Обеспечивает сбор и экспорт метрик для мониторинга производительности.
"""

from prometheus_client import Counter, Gauge, Histogram, Summary, start_http_server
from prometheus_client.core import CollectorRegistry
import time
import logging
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class PrometheusMetrics:
    """Класс для управления метриками Prometheus"""
    
    def __init__(self, port: int = 9090):
        self.port = port
        self.registry = CollectorRegistry()
        
        # Инициализация метрик
        self._init_metrics()
        
    def _init_metrics(self):
        """Инициализация всех метрик"""
        
        # Метрики торговой активности
        self.trade_signals_total = Counter(
            'trade_signals_total',
            'Total number of trading signals generated',
            ['symbol', 'direction', 'timeframe'],
            registry=self.registry
        )
        
        self.trades_executed_total = Counter(
            'trades_executed_total',
            'Total number of trades executed',
            ['symbol', 'direction', 'status'],
            registry=self.registry
        )
        
        self.trade_profit_loss = Gauge(
            'trade_profit_loss',
            'Profit or loss from trades',
            ['symbol', 'direction'],
            registry=self.registry
        )
        
        # Метрики производительности
        self.signal_generation_time = Histogram(
            'signal_generation_time_seconds',
            'Time taken to generate trading signals',
            ['symbol', 'timeframe'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0],
            registry=self.registry
        )
        
        self.api_request_duration = Histogram(
            'api_request_duration_seconds',
            'Duration of API requests',
            ['endpoint', 'method', 'status'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            registry=self.registry
        )
        
        self.api_errors_total = Counter(
            'api_errors_total',
            'Total number of API errors',
            ['endpoint', 'method', 'error_type'],
            registry=self.registry
        )
        
        # Метрики состояния системы
        self.bot_balance = Gauge(
            'bot_balance',
            'Current balance of the trading bot',
            ['currency'],
            registry=self.registry
        )
        
        self.bot_equity = Gauge(
            'bot_equity',
            'Current equity of the trading bot',
            registry=self.registry
        )
        
        self.open_positions = Gauge(
            'open_positions',
            'Number of currently open positions',
            registry=self.registry
        )
        
        self.portfolio_value = Gauge(
            'portfolio_value',
            'Total portfolio value',
            ['currency'],
            registry=self.registry
        )
        
        # Метрики рисков
        self.risk_exposure = Gauge(
            'risk_exposure',
            'Current risk exposure percentage',
            ['symbol'],
            registry=self.registry
        )
        
        self.drawdown = Gauge(
            'drawdown_percentage',
            'Current drawdown percentage',
            registry=self.registry
        )
        
        # Метрики ML модели
        self.ml_prediction_confidence = Gauge(
            'ml_prediction_confidence',
            'Confidence of ML predictions',
            ['symbol', 'timeframe'],
            registry=self.registry
        )
        
        self.ml_model_accuracy = Gauge(
            'ml_model_accuracy',
            'Accuracy of ML model predictions',
            ['symbol'],
            registry=self.registry
        )
        
        # Метрики производительности системы
        self.memory_usage = Gauge(
            'memory_usage_bytes',
            'Memory usage of the application',
            registry=self.registry
        )
        
        self.cpu_usage = Gauge(
            'cpu_usage_percentage',
            'CPU usage of the application',
            registry=self.registry
        )
        
        self.uptime = Gauge(
            'uptime_seconds',
            'Application uptime in seconds',
            registry=self.registry
        )
        
        # Время запуска для расчета uptime
        self.start_time = datetime.now()
        
        logger.info("Prometheus метрики инициализированы")
    
    def start_metrics_server(self):
        """Запуск HTTP сервера для экспорта метрик"""
        try:
            start_http_server(self.port, registry=self.registry)
            logger.info(f"Prometheus метрики доступны на порту {self.port}")
            return True
        except Exception as e:
            logger.error(f"Ошибка запуска сервера метрик: {e}")
            return False
    
    def record_signal_generated(self, symbol: str, direction: str, timeframe: str):
        """Запись метрики о сгенерированном сигнале"""
        try:
            self.trade_signals_total.labels(
                symbol=symbol,
                direction=direction,
                timeframe=timeframe
            ).inc()
        except Exception as e:
            logger.error(f"Ошибка записи метрики сигнала: {e}")
    
    def record_trade_executed(self, symbol: str, direction: str, status: str, pnl: float = 0):
        """Запись метрики о выполненной сделке"""
        try:
            self.trades_executed_total.labels(
                symbol=symbol,
                direction=direction,
                status=status
            ).inc()
            
            if pnl != 0:
                self.trade_profit_loss.labels(
                    symbol=symbol,
                    direction=direction
                ).set(pnl)
        except Exception as e:
            logger.error(f"Ошибка записи метрики сделки: {e}")
    
    def record_api_request(self, endpoint: str, method: str, duration: float, status: str = "success"):
        """Запись метрики о запросе к API"""
        try:
            self.api_request_duration.labels(
                endpoint=endpoint,
                method=method,
                status=status
            ).observe(duration)
        except Exception as e:
            logger.error(f"Ошибка записи метрики API запроса: {e}")
    
    def record_api_error(self, endpoint: str, method: str, error_type: str):
        """Запись метрики об ошибке API"""
        try:
            self.api_errors_total.labels(
                endpoint=endpoint,
                method=method,
                error_type=error_type
            ).inc()
        except Exception as e:
            logger.error(f"Ошибка записи метрики ошибки API: {e}")
    
    def update_balance(self, balance: float, currency: str = "USDT"):
        """Обновление метрики баланса"""
        try:
            self.bot_balance.labels(currency=currency).set(balance)
        except Exception as e:
            logger.error(f"Ошибка обновления метрики баланса: {e}")
    
    def update_equity(self, equity: float):
        """Обновление метрики эквити"""
        try:
            self.bot_equity.set(equity)
        except Exception as e:
            logger.error(f"Ошибка обновления метрики эквити: {e}")
    
    def update_open_positions(self, count: int):
        """Обновление метрики открытых позиций"""
        try:
            self.open_positions.set(count)
        except Exception as e:
            logger.error(f"Ошибка обновления метрики открытых позиций: {e}")
    
    def update_portfolio_value(self, value: float, currency: str = "USDT"):
        """Обновление метрики стоимости портфеля"""
        try:
            self.portfolio_value.labels(currency=currency).set(value)
        except Exception as e:
            logger.error(f"Ошибка обновления метрики стоимости портфеля: {e}")
    
    def update_risk_exposure(self, symbol: str, exposure: float):
        """Обновление метрики рискового воздействия"""
        try:
            self.risk_exposure.labels(symbol=symbol).set(exposure)
        except Exception as e:
            logger.error(f"Ошибка обновления метрики рискового воздействия: {e}")
    
    def update_drawdown(self, drawdown: float):
        """Обновление метрики просадки"""
        try:
            self.drawdown.set(drawdown)
        except Exception as e:
            logger.error(f"Ошибка обновления метрики просадки: {e}")
    
    def update_ml_confidence(self, symbol: str, timeframe: str, confidence: float):
        """Обновление метрики уверенности ML модели"""
        try:
            self.ml_prediction_confidence.labels(
                symbol=symbol,
                timeframe=timeframe
            ).set(confidence)
        except Exception as e:
            logger.error(f"Ошибка обновления метрики уверенности ML: {e}")
    
    def update_ml_accuracy(self, symbol: str, accuracy: float):
        """Обновление метрики точности ML модели"""
        try:
            self.ml_model_accuracy.labels(symbol=symbol).set(accuracy)
        except Exception as e:
            logger.error(f"Ошибка обновления метрики точности ML: {e}")
    
    def update_system_metrics(self):
        """Обновление метрик системы"""
        try:
            # Обновление uptime
            uptime = (datetime.now() - self.start_time).total_seconds()
            self.uptime.set(uptime)
            
            # Обновление использования памяти (упрощенная версия)
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            self.memory_usage.set(memory_info.rss)
            
            # Обновление использования CPU (упрощенная версия)
            cpu_percent = process.cpu_percent(interval=0.1)
            self.cpu_usage.set(cpu_percent)
            
        except ImportError:
            logger.warning("psutil не установлен, системные метрики недоступны")
        except Exception as e:
            logger.error(f"Ошибка обновления системных метрик: {e}")
    
    def get_metrics_registry(self):
        """Получение реестра метрик"""
        return self.registry


# Глобальный экземпляр для использования во всем приложении
metrics = PrometheusMetrics()

# Функции для обратной совместимости
def start_prometheus_server(port: int = 9090):
    """Запуск сервера Prometheus (обертка для обратной совместимости)"""
    return metrics.start_metrics_server()

def record_signal_generated(symbol: str, direction: str, timeframe: str):
    """Запись метрики о сгенерированном сигнале (обертка для обратной совместимости)"""
    metrics.record_signal_generated(symbol, direction, timeframe)

def record_trade_executed(symbol: str, direction: str, status: str, pnl: float = 0):
    """Запись метрики о выполненной сделке (обертка для обратной совместимости)"""
    metrics.record_trade_executed(symbol, direction, status, pnl)

def record_api_request(endpoint: str, method: str, duration: float, status: str = "success"):
    """Запись метрики о запросе к API (обертка для обратной совместимости)"""
    metrics.record_api_request(endpoint, method, duration, status)

def record_api_error(endpoint: str, method: str, error_type: str):
    """Запись метрики об ошибке API (обертка для обратной совместимости)"""
    metrics.record_api_error(endpoint, method, error_type)