"""
Вспомогательные утилиты для торгового бота BingX.

Содержит общие функции для работы с данными, временем, валидации
и других вспомогательных операций.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List, Optional, Any, Union
import json
import re
from decimal import Decimal, ROUND_DOWN
import hashlib

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def safe_round(value: float, precision: int = 8) -> float:
    """
    Безопасное округление чисел с плавающей точкой.
    
    Args:
        value: Число для округления
        precision: Количество знаков после запятой
        
    Returns:
        float: Округленное число
    """
    if value is None:
        return 0.0
        
    if not isinstance(value, (int, float, Decimal)):
        try:
            value = float(value)
        except (ValueError, TypeError):
            return 0.0
            
    if precision <= 0:
        return round(value)
    
    # Используем Decimal для точного округления
    decimal_value = Decimal(str(value))
    rounded = decimal_value.quantize(
        Decimal('1.' + '0' * precision), 
        rounding=ROUND_DOWN
    )
    return float(rounded)

def normalize_symbol(symbol: str) -> str:
    """
    Нормализация названия торговой пары.
    
    Args:
        symbol: Торговая пара (например, 'btcusdt', 'BTC-USDT')
        
    Returns:
        str: Нормализованное название (например, 'BTC-USDT')
    """
    if not symbol:
        return ""
        
    # Удаляем все не-алфавитные символы
    clean_symbol = re.sub(r'[^a-zA-Z]', '', symbol).upper()
    
    # Определяем базовую и котируемую валюту
    if len(clean_symbol) >= 6:
        # Предполагаем, что базовая валюта состоит из 3-4 символов
        for i in range(3, 5):
            base = clean_symbol[:i]
            quote = clean_symbol[i:]
            
            # Проверяем распространенные котируемые валюты
            if quote in ['USDT', 'USD', 'BTC', 'ETH', 'BNB']:
                return f"{base}-{quote}"
    
    # Если не удалось определить, возвращаем в верхнем регистре с дефисом
    if '-' in symbol:
        parts = symbol.split('-')
        if len(parts) == 2:
            return f"{parts[0].upper()}-{parts[1].upper()}"
    
    return symbol.upper()

def parse_timeframe(timeframe: str) -> timedelta:
    """
    Парсинг таймфрейма в объект timedelta.
    
    Args:
        timeframe: Таймфрейм (например, '15m', '1h', '4h', '1d')
        
    Returns:
        timedelta: Соответствующий временной интервал
        
    Raises:
        ValueError: Если таймфрейм имеет неправильный формат
    """
    timeframe = timeframe.lower()
    
    if timeframe.endswith('m'):
        minutes = int(timeframe[:-1])
        return timedelta(minutes=minutes)
    elif timeframe.endswith('h'):
        hours = int(timeframe[:-1])
        return timedelta(hours=hours)
    elif timeframe.endswith('d'):
        days = int(timeframe[:-1])
        return timedelta(days=days)
    elif timeframe.endswith('w'):
        weeks = int(timeframe[:-1])
        return timedelta(weeks=weeks)
    else:
        raise ValueError(f"Неизвестный формат таймфрейма: {timeframe}")

def timeframe_to_minutes(timeframe: str) -> int:
    """
    Конвертация таймфрейма в количество минут.
    
    Args:
        timeframe: Таймфрейм (например, '15m', '1h', '4h')
        
    Returns:
        int: Количество минут
    """
    delta = parse_timeframe(timeframe)
    return int(delta.total_seconds() / 60)

def format_timestamp(timestamp: Union[datetime, int, float], 
                    fmt: str = '%Y-%m-%d %H:%M:%S') -> str:
    """
    Форматирование временной метки в строку.
    
    Args:
        timestamp: Временная метка (datetime, Unix timestamp)
        fmt: Формат строки
        
    Returns:
        str: Отформатированная строка времени
    """
    if isinstance(timestamp, (int, float)):
        if timestamp > 1e12:  # Миллисекунды
            timestamp = timestamp / 1000
        timestamp = datetime.fromtimestamp(timestamp)
    
    if isinstance(timestamp, datetime):
        return timestamp.strftime(fmt)
    
    return str(timestamp)

def parse_timestamp(timestamp_str: str, 
                   fmt: str = '%Y-%m-%d %H:%M:%S') -> datetime:
    """
    Парсинг строки времени в объект datetime.
    
    Args:
        timestamp_str: Строка времени
        fmt: Формат строки
        
    Returns:
        datetime: Объект datetime
    """
    return datetime.strptime(timestamp_str, fmt)

def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """
    Расчет процентного изменения между двумя значениями.
    
    Args:
        old_value: Старое значение
        new_value: Новое значение
        
    Returns:
        float: Процентное изменение
    """
    if old_value == 0:
        return 0.0
    
    return ((new_value - old_value) / old_value) * 100

def calculate_volatility(prices: List[float], 
                        period: int = 20) -> float:
    """
    Расчет волатильности на основе цен.
    
    Args:
        prices: Список цен
        period: Период для расчета
        
    Returns:
        float: Волатильность в процентах
    """
    if len(prices) < period:
        return 0.0
    
    recent_prices = prices[-period:]
    returns = np.diff(recent_prices) / recent_prices[:-1]
    volatility = np.std(returns) * 100
    
    return float(volatility)

def generate_hash(data: Any, length: int = 12) -> str:
    """
    Генерация хэша для данных.
    
    Args:
        data: Данные для хэширования
        length: Длина хэша (макс. 64)
        
    Returns:
        str: Хэш строка
    """
    if not data:
        return ""
    
    data_str = json.dumps(data, sort_keys=True, default=str)
    hash_object = hashlib.sha256(data_str.encode())
    hash_hex = hash_object.hexdigest()
    
    return hash_hex[:min(length, 64)]

def format_currency(value: float, currency: str = "USD") -> str:
    """
    Форматирование денежных значений.
    
    Args:
        value: Числовое значение
        currency: Валюта
        
    Returns:
        str: Отформатированная строка
    """
    if currency.upper() in ["USD", "USDT"]:
        return f"${safe_round(value, 2):,.2f}"
    elif currency.upper() == "BTC":
        return f"₿{safe_round(value, 8)}"
    else:
        return f"{safe_round(value, 2)} {currency.upper()}"

def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """
    Безопасная загрузка JSON из строки.
    
    Args:
        json_str: JSON строка
        default: Значение по умолчанию при ошибке
        
    Returns:
        Any: Распарсенный JSON или значение по умолчанию
    """
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return default

def deep_merge(dict1: Dict, dict2: Dict) -> Dict:
    """
    Рекурсивное слияние двух словарей.
    
    Args:
        dict1: Первый словарь
        dict2: Второй словарь
        
    Returns:
        Dict: Объединенный словарь
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if (key in result and isinstance(result[key], dict) 
            and isinstance(value, dict)):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result

def filter_dict(data: Dict, keys: List[str]) -> Dict:
    """
    Фильтрация словаря по списку ключей.
    
    Args:
        data: Исходный словарь
        keys: Список ключей для сохранения
        
    Returns:
        Dict: Отфильтрованный словарь
    """
    return {k: v for k, v in data.items() if k in keys}

def retry_on_exception(max_retries: int = 3, 
                      delay: float = 1.0, 
                      exceptions: tuple = (Exception,)):
    """
    Декоратор для повторного выполнения функции при возникновении исключений.
    
    Args:
        max_retries: Максимальное количество попыток
        delay: Задержка между попытками (в секундах)
        exceptions: Кортеж исключений для перехвата
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    retries += 1
                    if retries >= max_retries:
                        logger.error(f"Функция {func.__name__} завершилась с ошибкой после {max_retries} попыток: {e}")
                        raise
                    
                    logger.warning(f"Ошибка в функции {func.__name__}, попытка {retries}/{max_retries}: {e}")
                    time.sleep(delay * (2 ** (retries - 1)))  # Экспоненциальная задержка
        return wrapper
    return decorator

def timing_decorator(func):
    """
    Декоратор для измерения времени выполнения функции.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        
        logger.debug(f"Функция {func.__name__} выполнена за {execution_time:.4f} секунд")
        return result
    return wrapper

def validate_email(email: str) -> bool:
    """
    Валидация email адреса.
    
    Args:
        email: Email адрес
        
    Returns:
        bool: True если email валиден, иначе False
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def format_bytes(size: int) -> str:
    """
    Форматирование размера в байтах в читаемый вид.
    
    Args:
        size: Размер в байтах
        
    Returns:
        str: Отформатированная строка
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"

def is_market_hours(now: datetime = None) -> bool:
    """
    Проверка, работает ли рынок в текущее время.
    Криптовалютные рынки работают 24/7, но функция может быть адаптирована
    для традиционных рынков при необходимости.
    
    Args:
        now: Текущее время (по умолчанию сейчас)
        
    Returns:
        bool: True если рынок открыт, иначе False
    """
    # Для криптовалютных рынков всегда возвращаем True
    return True

def calculate_compound_interest(principal: float, 
                               rate: float, 
                               time_periods: int) -> float:
    """
    Расчет сложного процента.
    
    Args:
        principal: Основная сумма
        rate: Процентная ставка за период
        time_periods: Количество периодов
        
    Returns:
        float: Итоговая сумма
    """
    return principal * (1 + rate) ** time_periods

def create_dataframe_chunking(data: List[Dict], 
                             chunk_size: int = 1000) -> List[pd.DataFrame]:
    """
    Создание DataFrame с разбивкой на части для больших datasets.
    
    Args:
        data: Список словарей с данными
        chunk_size: Размер части
        
    Returns:
        List[pd.DataFrame]: Список DataFrame
    """
    chunks = []
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        chunks.append(pd.DataFrame(chunk))
    
    return chunks

def clean_numeric_data(value: Any, default: float = 0.0) -> float:
    """
    Очистка и преобразование числовых данных.
    
    Args:
        value: Значение для очистки
        default: Значение по умолчанию при ошибке
        
    Returns:
        float: Числовое значение
    """
    if isinstance(value, (int, float)):
        return float(value)
    
    if isinstance(value, str):
        # Удаляем нечисловые символы, кроме точки и минуса
        cleaned = re.sub(r'[^\d.-]', '', value)
        try:
            return float(cleaned)
        except ValueError:
            pass
    
    return default

def generate_id(prefix: str = "", length: int = 8) -> str:
    """
    Генерация уникального ID.
    
    Args:
        prefix: Префикс ID
        length: Длина случайной части
        
    Returns:
        str: Уникальный ID
    """
    random_part = hashlib.sha256(str(time.time()).encode()).hexdigest()[:length]
    return f"{prefix}{random_part}" if prefix else random_part

# Алиасы для обратной совместимости
round_float = safe_round
get_hash = generate_hash
format_time = format_timestamp