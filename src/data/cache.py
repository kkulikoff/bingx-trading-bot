"""
Модуль кэширования данных для торгового бота.
Обеспечивает временное хранение часто используемых данных для уменьшения нагрузки на API и ускорения работы системы.
"""

import time
import json
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import diskcache as dc
from pathlib import Path

from config.settings import ENVIRONMENT, DEBUG
from src.utils.logger import setup_logging

logger = setup_logging(__name__)

class CacheManager:
    """Класс для управления кэшированием данных"""
    
    def __init__(self, cache_dir: str = "data/cache", default_ttl: int = 3600):
        """
        Инициализация менеджера кэша
        
        Args:
            cache_dir: Директория для хранения кэша
            default_ttl: Время жизни кэша по умолчанию в секундах
        """
        self.cache_dir = Path(cache_dir)
        self.default_ttl = default_ttl
        
        # Создание директории кэша если не существует
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Инициализация кэша
        self.cache = dc.Cache(str(self.cache_dir))
        logger.info(f"Кэш инициализирован в директории: {self.cache_dir}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Получение значения из кэша
        
        Args:
            key: Ключ кэша
            default: Значение по умолчанию если ключ не найден
            
        Returns:
            Значение из кэша или default
        """
        try:
            value = self.cache.get(key, default)
            if value is not default:
                logger.debug(f"Кэш попадание для ключа: {key}")
            else:
                logger.debug(f"Кэш промах для ключа: {key}")
            return value
        except Exception as e:
            logger.error(f"Ошибка получения из кэша ключа {key}: {e}")
            return default
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Установка значения в кэш
        
        Args:
            key: Ключ кэша
            value: Значение для кэширования
            ttl: Время жизни в секундах (None - использовать по умолчанию)
            
        Returns:
            bool: True если успешно, False если ошибка
        """
        try:
            expire_time = ttl if ttl is not None else self.default_ttl
            self.cache.set(key, value, expire=expire_time)
            logger.debug(f"Значение установлено в кэш для ключа: {key}, TTL: {expire_time}с")
            return True
        except Exception as e:
            logger.error(f"Ошибка установки в кэш ключа {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """
        Удаление значения из кэша
        
        Args:
            key: Ключ кэша
            
        Returns:
            bool: True если успешно, False если ошибка
        """
        try:
            result = self.cache.delete(key)
            logger.debug(f"Значение удалено из кэша для ключа: {key}")
            return result
        except Exception as e:
            logger.error(f"Ошибка удаления из кэша ключа {key}: {e}")
            return False
    
    def clear(self) -> bool:
        """
        Очистка всего кэша
        
        Returns:
            bool: True если успешно, False если ошибка
        """
        try:
            self.cache.clear()
            logger.info("Кэш полностью очищен")
            return True
        except Exception as e:
            logger.error(f"Ошибка очистки кэша: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Получение статистики кэша
        
        Returns:
            Dict: Статистика кэша
        """
        try:
            stats = {
                'size': self.cache.volume(),
                'count': len(self.cache),
                'directory': str(self.cache_dir),
                'default_ttl': self.default_ttl
            }
            return stats
        except Exception as e:
            logger.error(f"Ошибка получения статистики кэша: {e}")
            return {}
    
    def close(self):
        """Корректное закрытие кэша"""
        try:
            self.cache.close()
            logger.info("Кэш корректно закрыт")
        except Exception as e:
            logger.error(f"Ошибка закрытия кэша: {e}")

# Глобальный экземпляр кэша
cache_manager = CacheManager()

# Специализированные методы для различных типов данных
class DataCache:
    """Класс со специализированными методами кэширования для различных типов данных"""
    
    @staticmethod
    def cache_historical_data(symbol: str, timeframe: str, data: List[Dict[str, Any]], ttl: int = 300) -> bool:
        """
        Кэширование исторических данных
        
        Args:
            symbol: Торговая пара
            timeframe: Таймфрейм
            data: Исторические данные
            ttl: Время жизни в секундах
            
        Returns:
            bool: True если успешно
        """
        key = f"historical_{symbol}_{timeframe}"
        return cache_manager.set(key, data, ttl)
    
    @staticmethod
    def get_historical_data(symbol: str, timeframe: str) -> Optional[List[Dict[str, Any]]]:
        """
        Получение исторических данных из кэша
        
        Args:
            symbol: Торговая пара
            timeframe: Таймфрейм
            
        Returns:
            Optional[List[Dict]]: Исторические данные или None
        """
        key = f"historical_{symbol}_{timeframe}"
        return cache_manager.get(key)
    
    @staticmethod
    def cache_indicators(symbol: str, timeframe: str, indicators: Dict[str, Any], ttl: int = 600) -> bool:
        """
        Кэширование рассчитанных индикаторов
        
        Args:
            symbol: Торговая пара
            timeframe: Таймфрейм
            indicators: Данные индикаторов
            ttl: Время жизни в секундах
            
        Returns:
            bool: True если успешно
        """
        key = f"indicators_{symbol}_{timeframe}"
        return cache_manager.set(key, indicators, ttl)
    
    @staticmethod
    def get_indicators(symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        """
        Получение индикаторов из кэша
        
        Args:
            symbol: Торговая пара
            timeframe: Таймфрейм
            
        Returns:
            Optional[Dict]: Данные индикаторов или None
        """
        key = f"indicators_{symbol}_{timeframe}"
        return cache_manager.get(key)
    
    @staticmethod
    def cache_signal(symbol: str, timeframe: str, signal: Dict[str, Any], ttl: int = 300) -> bool:
        """
        Кэширование торгового сигнала
        
        Args:
            symbol: Торговая пара
            timeframe: Таймфрейм
            signal: Торговый сигнал
            ttl: Время жизни в секундах
            
        Returns:
            bool: True если успешно
        """
        key = f"signal_{symbol}_{timeframe}"
        return cache_manager.set(key, signal, ttl)
    
    @staticmethod
    def get_signal(symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        """
        Получение торгового сигнала из кэша
        
        Args:
            symbol: Торговая пара
            timeframe: Таймфрейм
            
        Returns:
            Optional[Dict]: Торговый сигнал или None
        """
        key = f"signal_{symbol}_{timeframe}"
        return cache_manager.get(key)
    
    @staticmethod
    def cache_ml_prediction(symbol: str, timeframe: str, prediction: float, ttl: int = 1800) -> bool:
        """
        Кэширование ML прогноза
        
        Args:
            symbol: Торговая пара
            timeframe: Таймфрейм
            prediction: Прогноз ML модели
            ttl: Время жизни в секундах
            
        Returns:
            bool: True если успешно
        """
        key = f"ml_prediction_{symbol}_{timeframe}"
        return cache_manager.set(key, prediction, ttl)
    
    @staticmethod
    def get_ml_prediction(symbol: str, timeframe: str) -> Optional[float]:
        """
        Получение ML прогноза из кэша
        
        Args:
            symbol: Торговая пара
            timeframe: Таймфрейм
            
        Returns:
            Optional[float]: ML прогноз или None
        """
        key = f"ml_prediction_{symbol}_{timeframe}"
        return cache_manager.get(key)
    
    @staticmethod
    def cache_market_sentiment(sentiment_data: Dict[str, Any], ttl: int = 3600) -> bool:
        """
        Кэширование данных настроения рынка
        
        Args:
            sentiment_data: Данные настроения рынка
            ttl: Время жизни в секундах
            
        Returns:
            bool: True если успешно
        """
        key = "market_sentiment"
        return cache_manager.set(key, sentiment_data, ttl)
    
    @staticmethod
    def get_market_sentiment() -> Optional[Dict[str, Any]]:
        """
        Получение данных настроения рынка из кэша
        
        Returns:
            Optional[Dict]: Данные настроения рынка или None
        """
        key = "market_sentiment"
        return cache_manager.get(key)

# Декоратор для кэширования результатов функций
def cache_result(ttl: int = 300, key_prefix: str = "func"):
    """
    Декоратор для кэширования результатов функций
    
    Args:
        ttl: Время жизни кэша в секундах
        key_prefix: Префикс для ключа кэша
        
    Returns:
        Декоратор функции
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Создание ключа на основе функции и аргументов
            key_args = [str(arg) for arg in args]
            key_kwargs = [f"{k}={v}" for k, v in sorted(kwargs.items())]
            key = f"{key_prefix}_{func.__name__}_{'_'.join(key_args + key_kwargs)}"
            
            # Попытка получить результат из кэша
            cached_result = cache_manager.get(key)
            if cached_result is not None:
                logger.debug(f"Кэшированный результат найден для функции {func.__name__}")
                return cached_result
            
            # Выполнение функции и сохранение результата в кэш
            result = func(*args, **kwargs)
            cache_manager.set(key, result, ttl)
            logger.debug(f"Результат функции {func.__name__} сохранен в кэш")
            
            return result
        return wrapper
    return decorator

# Инициализация глобального экземпляра кэша при импорте модуля
def init_cache():
    """Инициализация кэша при запуске приложения"""
    global cache_manager
    cache_manager = CacheManager()
    logger.info("Кэш инициализирован")

# Корректное закрытие кэша при завершении приложения
def close_cache():
    """Корректное закрытие кэша"""
    global cache_manager
    cache_manager.close()
    logger.info("Кэш закрыт")

# Регистрация обработчиков для корректного завершения
import atexit
atexit.register(close_cache)