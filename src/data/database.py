"""
Модуль для работы с базой данных торгового бота.
Обеспечивает хранение и управление данными о сделках, сигналах и состоянии системы.
"""

import sqlite3
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import threading
from contextlib import contextmanager

from config.settings import DATABASE_URL
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class DatabaseManager:
    """Класс для управления базой данных торгового бота"""
    
    def __init__(self, db_url: str = None):
        """
        Инициализация менеджера базы данных.
        
        Args:
            db_url: URL базы данных (если None, используется значение из настроек)
        """
        self.db_url = db_url or DATABASE_URL
        self._local = threading.local()
        self.init_database()
        
        logger.info(f"Менеджер базы данных инициализирован: {self.db_url}")
    
    def get_connection(self):
        """Получение соединения с базой данных (с поддержкой threading)"""
        if not hasattr(self._local, 'connection'):
            self._local.connection = sqlite3.connect(self.db_url, check_same_thread=False)
            self._local.connection.row_factory = sqlite3.Row
        return self._local.connection
    
    def close_connection(self):
        """Закрытие соединения с базой данных"""
        if hasattr(self._local, 'connection'):
            self._local.connection.close()
            delattr(self._local, 'connection')
    
    @contextmanager
    def get_cursor(self):
        """
        Контекстный менеджер для работы с курсором базы данных.
        
        Yields:
            sqlite3.Cursor: Курсор базы данных
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            cursor.close()
    
    def init_database(self):
        """Инициализация базы данных и создание таблиц"""
        try:
            with self.get_cursor() as cursor:
                # Таблица сделок
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        direction TEXT NOT NULL,
                        entry_price REAL NOT NULL,
                        exit_price REAL,
                        quantity REAL NOT NULL,
                        stop_loss REAL,
                        take_profit REAL,
                        pnl REAL,
                        fee REAL DEFAULT 0,
                        status TEXT DEFAULT 'open',
                        order_id TEXT,
                        signal_id INTEGER,
                        entry_time DATETIME NOT NULL,
                        exit_time DATETIME,
                        duration INTEGER,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (signal_id) REFERENCES signals (id)
                    )
                """)
                
                # Таблица сигналов
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS signals (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        timeframe TEXT NOT NULL,
                        direction TEXT NOT NULL,
                        price REAL NOT NULL,
                        stop_loss REAL NOT NULL,
                        take_profit REAL NOT NULL,
                        confidence REAL,
                        risk_reward REAL,
                        position_size REAL,
                        reasons TEXT,
                        indicators TEXT,
                        ml_confidence REAL,
                        status TEXT DEFAULT 'generated',
                        executed BOOLEAN DEFAULT FALSE,
                        executed_price REAL,
                        generated_at DATETIME NOT NULL,
                        executed_at DATETIME,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Таблица состояния бота
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS bot_state (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        balance REAL NOT NULL,
                        equity REAL NOT NULL,
                        available_balance REAL NOT NULL,
                        total_trades INTEGER DEFAULT 0,
                        winning_trades INTEGER DEFAULT 0,
                        losing_trades INTEGER DEFAULT 0,
                        win_rate REAL DEFAULT 0,
                        total_pnl REAL DEFAULT 0,
                        daily_pnl REAL DEFAULT 0,
                        max_drawdown REAL DEFAULT 0,
                        sharpe_ratio REAL DEFAULT 0,
                        timestamp DATETIME NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Таблица позиций
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS positions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        direction TEXT NOT NULL,
                        entry_price REAL NOT NULL,
                        quantity REAL NOT NULL,
                        stop_loss REAL,
                        take_profit REAL,
                        current_price REAL,
                        unrealized_pnl REAL,
                        margin REAL DEFAULT 0,
                        leverage INTEGER DEFAULT 1,
                        status TEXT DEFAULT 'open',
                        order_id TEXT,
                        signal_id INTEGER,
                        opened_at DATETIME NOT NULL,
                        closed_at DATETIME,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (signal_id) REFERENCES signals (id)
                    )
                """)
                
                # Таблица ошибок
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS errors (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        module TEXT NOT NULL,
                        error_type TEXT NOT NULL,
                        message TEXT NOT NULL,
                        traceback TEXT,
                        severity TEXT DEFAULT 'error',
                        resolved BOOLEAN DEFAULT FALSE,
                        occurred_at DATETIME NOT NULL,
                        resolved_at DATETIME,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Таблица производительности
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME NOT NULL,
                        cpu_usage REAL,
                        memory_usage REAL,
                        disk_usage REAL,
                        api_response_time REAL,
                        active_connections INTEGER,
                        trade_execution_time REAL,
                        signal_processing_time REAL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Таблица настроек
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS settings (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        key TEXT UNIQUE NOT NULL,
                        value TEXT NOT NULL,
                        description TEXT,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Индексы для улучшения производительности
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_generated_at ON signals(generated_at)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_errors_occurred_at ON errors(occurred_at)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_performance_timestamp ON performance(timestamp)")
                
                # Вставка начальных настроек
                initial_settings = [
                    ('risk_per_trade', '0.02', 'Риск на сделку (2% от баланса)'),
                    ('max_position_size', '0.1', 'Максимальный размер позиции (10% от баланса)'),
                    ('default_leverage', '10', 'Кредитное плечо по умолчанию'),
                    ('stop_loss_multiplier', '2.0', 'Множитель для стоп-лосса (ATR * multiplier)'),
                    ('take_profit_multiplier', '4.0', 'Множитель для тейк-профита (ATR * multiplier)'),
                    ('trade_confirmation_threshold', '0.7', 'Порог уверенности для исполнения сделки'),
                    ('max_daily_trades', '20', 'Максимальное количество сделок в день'),
                    ('auto_restart_on_error', 'true', 'Автоматический перезапуск при ошибках')
                ]
                
                cursor.executemany("""
                    INSERT OR IGNORE INTO settings (key, value, description)
                    VALUES (?, ?, ?)
                """, initial_settings)
            
            logger.info("База данных инициализирована успешно")
            
        except Exception as e:
            logger.error(f"Ошибка инициализации базы данных: {e}")
            raise
    
    def add_trade(self, trade_data: Dict[str, Any]) -> int:
        """
        Добавление сделки в базу данных.
        
        Args:
            trade_data: Данные о сделке
            
        Returns:
            int: ID добавленной сделки
        """
        try:
            with self.get_cursor() as cursor:
                cursor.execute("""
                    INSERT INTO trades (
                        symbol, direction, entry_price, quantity, stop_loss, take_profit,
                        order_id, signal_id, entry_time
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade_data['symbol'],
                    trade_data['direction'],
                    trade_data['entry_price'],
                    trade_data['quantity'],
                    trade_data.get('stop_loss'),
                    trade_data.get('take_profit'),
                    trade_data.get('order_id'),
                    trade_data.get('signal_id'),
                    trade_data.get('entry_time', datetime.now())
                ))
                
                trade_id = cursor.lastrowid
                logger.info(f"Сделка добавлена в базу данных: ID {trade_id}")
                return trade_id
                
        except Exception as e:
            logger.error(f"Ошибка добавления сделки: {e}")
            raise
    
    def update_trade(self, trade_id: int, update_data: Dict[str, Any]) -> bool:
        """
        Обновление данных о сделке.
        
        Args:
            trade_id: ID сделки
            update_data: Данные для обновления
            
        Returns:
            bool: True если обновление успешно, False в противном случае
        """
        try:
            with self.get_cursor() as cursor:
                set_clause = ", ".join([f"{key} = ?" for key in update_data.keys()])
                values = list(update_data.values())
                values.append(trade_id)
                
                cursor.execute(f"""
                    UPDATE trades 
                    SET {set_clause}, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, values)
                
                logger.info(f"Сделка обновлена: ID {trade_id}")
                return cursor.rowcount > 0
                
        except Exception as e:
            logger.error(f"Ошибка обновления сделки {trade_id}: {e}")
            return False
    
    def close_trade(self, trade_id: int, exit_price: float, pnl: float, fee: float = 0) -> bool:
        """
        Закрытие сделки.
        
        Args:
            trade_id: ID сделки
            exit_price: Цена выхода
            pnl: Прибыль/убыток
            fee: Комиссия
            
        Returns:
            bool: True если закрытие успешно, False в противном случае
        """
        try:
            with self.get_cursor() as cursor:
                cursor.execute("""
                    UPDATE trades 
                    SET exit_price = ?, pnl = ?, fee = ?, status = 'closed',
                        exit_time = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP,
                        duration = CAST((julianday(CURRENT_TIMESTAMP) - julianday(entry_time)) * 24 * 60 * 60 AS INTEGER)
                    WHERE id = ?
                """, (exit_price, pnl, fee, trade_id))
                
                success = cursor.rowcount > 0
                if success:
                    logger.info(f"Сделка закрыта: ID {trade_id}, PnL: {pnl:.2f}")
                else:
                    logger.warning(f"Сделка не найдена при закрытии: ID {trade_id}")
                
                return success
                
        except Exception as e:
            logger.error(f"Ошибка закрытия сделки {trade_id}: {e}")
            return False
    
    def add_signal(self, signal_data: Dict[str, Any]) -> int:
        """
        Добавление сигнала в базу данных.
        
        Args:
            signal_data: Данные о сигнале
            
        Returns:
            int: ID добавленного сигнала
        """
        try:
            # Преобразование списков в JSON строки
            reasons = json.dumps(signal_data.get('reasons', [])) if signal_data.get('reasons') else None
            indicators = json.dumps(signal_data.get('indicators', {})) if signal_data.get('indicators') else None
            
            with self.get_cursor() as cursor:
                cursor.execute("""
                    INSERT INTO signals (
                        symbol, timeframe, direction, price, stop_loss, take_profit,
                        confidence, risk_reward, position_size, reasons, indicators,
                        ml_confidence, generated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    signal_data['symbol'],
                    signal_data['timeframe'],
                    signal_data['direction'],
                    signal_data['price'],
                    signal_data['stop_loss'],
                    signal_data['take_profit'],
                    signal_data.get('confidence'),
                    signal_data.get('risk_reward'),
                    signal_data.get('position_size'),
                    reasons,
                    indicators,
                    signal_data.get('ml_confidence'),
                    signal_data.get('generated_at', datetime.now())
                ))
                
                signal_id = cursor.lastrowid
                logger.info(f"Сигнал добавлен в базу данных: ID {signal_id}")
                return signal_id
                
        except Exception as e:
            logger.error(f"Ошибка добавления сигнала: {e}")
            raise
    
    def mark_signal_executed(self, signal_id: int, executed_price: float) -> bool:
        """
        Отметка сигнала как исполненного.
        
        Args:
            signal_id: ID сигнала
            executed_price: Цена исполнения
            
        Returns:
            bool: True если обновление успешно, False в противном случае
        """
        try:
            with self.get_cursor() as cursor:
                cursor.execute("""
                    UPDATE signals 
                    SET executed = TRUE, executed_price = ?, executed_at = CURRENT_TIMESTAMP,
                        status = 'executed'
                    WHERE id = ?
                """, (executed_price, signal_id))
                
                success = cursor.rowcount > 0
                if success:
                    logger.info(f"Сигнал отмечен как исполненный: ID {signal_id}")
                else:
                    logger.warning(f"Сигнал не найден: ID {signal_id}")
                
                return success
                
        except Exception as e:
            logger.error(f"Ошибка отметки сигнала как исполненного {signal_id}: {e}")
            return False
    
    def save_bot_state(self, state_data: Dict[str, Any]) -> int:
        """
        Сохранение состояния бота.
        
        Args:
            state_data: Данные о состоянии бота
            
        Returns:
            int: ID сохраненного состояния
        """
        try:
            with self.get_cursor() as cursor:
                cursor.execute("""
                    INSERT INTO bot_state (
                        balance, equity, available_balance, total_trades, winning_trades,
                        losing_trades, win_rate, total_pnl, daily_pnl, max_drawdown,
                        sharpe_ratio, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    state_data['balance'],
                    state_data['equity'],
                    state_data['available_balance'],
                    state_data.get('total_trades', 0),
                    state_data.get('winning_trades', 0),
                    state_data.get('losing_trades', 0),
                    state_data.get('win_rate', 0),
                    state_data.get('total_pnl', 0),
                    state_data.get('daily_pnl', 0),
                    state_data.get('max_drawdown', 0),
                    state_data.get('sharpe_ratio', 0),
                    state_data.get('timestamp', datetime.now())
                ))
                
                state_id = cursor.lastrowid
                logger.debug(f"Состояние бота сохранено: ID {state_id}")
                return state_id
                
        except Exception as e:
            logger.error(f"Ошибка сохранения состояния бота: {e}")
            raise
    
    def add_position(self, position_data: Dict[str, Any]) -> int:
        """
        Добавление позиции в базу данных.
        
        Args:
            position_data: Данные о позиции
            
        Returns:
            int: ID добавленной позиции
        """
        try:
            with self.get_cursor() as cursor:
                cursor.execute("""
                    INSERT INTO positions (
                        symbol, direction, entry_price, quantity, stop_loss, take_profit,
                        margin, leverage, order_id, signal_id, opened_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    position_data['symbol'],
                    position_data['direction'],
                    position_data['entry_price'],
                    position_data['quantity'],
                    position_data.get('stop_loss'),
                    position_data.get('take_profit'),
                    position_data.get('margin', 0),
                    position_data.get('leverage', 1),
                    position_data.get('order_id'),
                    position_data.get('signal_id'),
                    position_data.get('opened_at', datetime.now())
                ))
                
                position_id = cursor.lastrowid
                logger.info(f"Позиция добавлена в базу данных: ID {position_id}")
                return position_id
                
        except Exception as e:
            logger.error(f"Ошибка добавления позиции: {e}")
            raise
    
    def update_position(self, position_id: int, update_data: Dict[str, Any]) -> bool:
        """
        Обновление данных о позиции.
        
        Args:
            position_id: ID позиции
            update_data: Данные для обновления
            
        Returns:
            bool: True если обновление успешно, False в противном случае
        """
        try:
            with self.get_cursor() as cursor:
                set_clause = ", ".join([f"{key} = ?" for key in update_data.keys()])
                values = list(update_data.values())
                values.append(position_id)
                
                cursor.execute(f"""
                    UPDATE positions 
                    SET {set_clause}, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, values)
                
                logger.debug(f"Позиция обновлена: ID {position_id}")
                return cursor.rowcount > 0
                
        except Exception as e:
            logger.error(f"Ошибка обновления позиции {position_id}: {e}")
            return False
    
    def close_position(self, position_id: int, exit_price: float, pnl: float) -> bool:
        """
        Закрытие позиции.
        
        Args:
            position_id: ID позиции
            exit_price: Цена выхода
            pnl: Прибыль/убыток
            
        Returns:
            bool: True если закрытие успешно, False в противном случае
        """
        try:
            with self.get_cursor() as cursor:
                cursor.execute("""
                    UPDATE positions 
                    SET current_price = ?, unrealized_pnl = ?, status = 'closed',
                        closed_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (exit_price, pnl, position_id))
                
                success = cursor.rowcount > 0
                if success:
                    logger.info(f"Позиция закрыта: ID {position_id}, PnL: {pnl:.2f}")
                else:
                    logger.warning(f"Позиция не найдена при закрытии: ID {position_id}")
                
                return success
                
        except Exception as e:
            logger.error(f"Ошибка закрытия позиции {position_id}: {e}")
            return False
    
    def log_error(self, error_data: Dict[str, Any]) -> int:
        """
        Логирование ошибки в базу данных.
        
        Args:
            error_data: Данные об ошибке
            
        Returns:
            int: ID добавленной ошибки
        """
        try:
            with self.get_cursor() as cursor:
                cursor.execute("""
                    INSERT INTO errors (
                        module, error_type, message, traceback, severity, occurred_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    error_data['module'],
                    error_data['error_type'],
                    error_data['message'],
                    error_data.get('traceback'),
                    error_data.get('severity', 'error'),
                    error_data.get('occurred_at', datetime.now())
                ))
                
                error_id = cursor.lastrowid
                logger.debug(f"Ошибка записана в базу данных: ID {error_id}")
                return error_id
                
        except Exception as e:
            logger.error(f"Ошибка записи ошибки в базу данных: {e}")
            raise
    
    def save_performance_metrics(self, metrics_data: Dict[str, Any]) -> int:
        """
        Сохранение метрик производительности.
        
        Args:
            metrics_data: Данные о производительности
            
        Returns:
            int: ID добавленных метрик
        """
        try:
            with self.get_cursor() as cursor:
                cursor.execute("""
                    INSERT INTO performance (
                        timestamp, cpu_usage, memory_usage, disk_usage, api_response_time,
                        active_connections, trade_execution_time, signal_processing_time
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metrics_data.get('timestamp', datetime.now()),
                    metrics_data.get('cpu_usage'),
                    metrics_data.get('memory_usage'),
                    metrics_data.get('disk_usage'),
                    metrics_data.get('api_response_time'),
                    metrics_data.get('active_connections'),
                    metrics_data.get('trade_execution_time'),
                    metrics_data.get('signal_processing_time')
                ))
                
                metrics_id = cursor.lastrowid
                logger.debug(f"Метрики производительности сохранены: ID {metrics_id}")
                return metrics_id
                
        except Exception as e:
            logger.error(f"Ошибка сохранения метрик производительности: {e}")
            raise
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """
        Получение значения настройки.
        
        Args:
            key: Ключ настройки
            default: Значение по умолчанию
            
        Returns:
            Any: Значение настройки или значение по умолчанию
        """
        try:
            with self.get_cursor() as cursor:
                cursor.execute("SELECT value FROM settings WHERE key = ?", (key,))
                result = cursor.fetchone()
                
                if result:
                    # Попытка преобразования типов
                    value = result['value']
                    try:
                        if value.lower() in ('true', 'false'):
                            return value.lower() == 'true'
                        elif '.' in value:
                            return float(value)
                        else:
                            return int(value)
                    except (ValueError, AttributeError):
                        return value
                else:
                    return default
                    
        except Exception as e:
            logger.error(f"Ошибка получения настройки {key}: {e}")
            return default
    
    def set_setting(self, key: str, value: Any, description: str = None) -> bool:
        """
        Установка значения настройки.
        
        Args:
            key: Ключ настройки
            value: Значение настройки
            description: Описание настройки
            
        Returns:
            bool: True если установка успешна, False в противном случае
        """
        try:
            with self.get_cursor() as cursor:
                if description:
                    cursor.execute("""
                        INSERT OR REPLACE INTO settings (key, value, description, updated_at)
                        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                    """, (key, str(value), description))
                else:
                    cursor.execute("""
                        INSERT OR REPLACE INTO settings (key, value, updated_at)
                        VALUES (?, ?, CURRENT_TIMESTAMP)
                    """, (key, str(value)))
                
                logger.info(f"Настройка обновлена: {key} = {value}")
                return True
                
        except Exception as e:
            logger.error(f"Ошибка установки настройки {key}: {e}")
            return False
    
    def get_trades(self, limit: int = 100, offset: int = 0, **filters) -> List[Dict[str, Any]]:
        """
        Получение списка сделок с фильтрацией.
        
        Args:
            limit: Максимальное количество записей
            offset: Смещение
            **filters: Параметры фильтрации
            
        Returns:
            List[Dict]: Список сделок
        """
        try:
            with self.get_cursor() as cursor:
                where_clause = ""
                params = []
                
                if filters:
                    conditions = []
                    for key, value in filters.items():
                        if value is not None:
                            conditions.append(f"{key} = ?")
                            params.append(value)
                    
                    if conditions:
                        where_clause = "WHERE " + " AND ".join(conditions)
                
                cursor.execute(f"""
                    SELECT * FROM trades 
                    {where_clause}
                    ORDER BY entry_time DESC 
                    LIMIT ? OFFSET ?
                """, params + [limit, offset])
                
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Ошибка получения списка сделок: {e}")
            return []
    
    def get_signals(self, limit: int = 100, offset: int = 0, **filters) -> List[Dict[str, Any]]:
        """
        Получение списка сигналов с фильтрацией.
        
        Args:
            limit: Максимальное количество записей
            offset: Смещение
            **filters: Параметры фильтрации
            
        Returns:
            List[Dict]: Список сигналов
        """
        try:
            with self.get_cursor() as cursor:
                where_clause = ""
                params = []
                
                if filters:
                    conditions = []
                    for key, value in filters.items():
                        if value is not None:
                            conditions.append(f"{key} = ?")
                            params.append(value)
                    
                    if conditions:
                        where_clause = "WHERE " + " AND ".join(conditions)
                
                cursor.execute(f"""
                    SELECT * FROM signals 
                    {where_clause}
                    ORDER BY generated_at DESC 
                    LIMIT ? OFFSET ?
                """, params + [limit, offset])
                
                results = []
                for row in cursor.fetchall():
                    signal = dict(row)
                    # Преобразование JSON строк обратно в объекты
                    if signal.get('reasons'):
                        signal['reasons'] = json.loads(signal['reasons'])
                    if signal.get('indicators'):
                        signal['indicators'] = json.loads(signal['indicators'])
                    results.append(signal)
                
                return results
                
        except Exception as e:
            logger.error(f"Ошибка получения списка сигналов: {e}")
            return []
    
    def get_performance_stats(self, days: int = 7) -> Dict[str, Any]:
        """
        Получение статистики производительности за указанный период.
        
        Args:
            days: Количество дней для анализа
            
        Returns:
            Dict: Статистика производительности
        """
        try:
            with self.get_cursor() as cursor:
                # Статистика сделок
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_trades,
                        SUM(CASE WHEN status = 'closed' THEN 1 ELSE 0 END) as closed_trades,
                        SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                        SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
                        AVG(pnl) as avg_pnl,
                        SUM(pnl) as total_pnl,
                        MAX(pnl) as max_profit,
                        MIN(pnl) as max_loss
                    FROM trades 
                    WHERE entry_time >= datetime('now', ?)
                """, (f"-{days} days",))
                
                trade_stats = dict(cursor.fetchone())
                
                # Статистика сигналов
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_signals,
                        SUM(CASE WHEN executed = 1 THEN 1 ELSE 0 END) as executed_signals,
                        AVG(confidence) as avg_confidence,
                        AVG(risk_reward) as avg_risk_reward
                    FROM signals 
                    WHERE generated_at >= datetime('now', ?)
                """, (f"-{days} days",))
                
                signal_stats = dict(cursor.fetchone())
                
                # Объединение статистики
                stats = {
                    'period_days': days,
                    'trades': trade_stats,
                    'signals': signal_stats,
                    'timestamp': datetime.now()
                }
                
                return stats
                
        except Exception as e:
            logger.error(f"Ошибка получения статистики производительности: {e}")
            return {}
    
    def backup_database(self, backup_path: str = None) -> str:
        """
        Создание резервной копии базы данных.
        
        Args:
            backup_path: Путь для сохранения резервной копии
            
        Returns:
            str: Путь к созданной резервной копии
        """
        try:
            if backup_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = f"backups/database_backup_{timestamp}.db"
            
            # Создание директории для резервных копий, если не существует
            Path("backups").mkdir(exist_ok=True)
            
            # Копирование базы данных
            with self.get_connection() as source:
                with sqlite3.connect(backup_path) as target:
                    source.backup(target)
            
            logger.info(f"Создана резервная копия базы данных: {backup_path}")
            return backup_path
            
        except Exception as e:
            logger.error(f"Ошибка создания резервной копии базы данных: {e}")
            raise
    
    def optimize_database(self):
        """Оптимизация базы данных (VACUUM и перестроение индексов)"""
        try:
            with self.get_cursor() as cursor:
                # VACUUM для освобождения пространства и дефрагментации
                cursor.execute("VACUUM")
                
                # ANALYZE для обновления статистики
                cursor.execute("ANALYZE")
                
                logger.info("База данных оптимизирована")
                
        except Exception as e:
            logger.error(f"Ошибка оптимизации базы данных: {e}")
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """
        Очистка старых данных из базы данных.
        
        Args:
            days_to_keep: Количество дней для хранения данных
        """
        try:
            with self.get_cursor() as cursor:
                cutoff_date = datetime.now() - timedelta(days=days_to_keep)
                
                # Удаление старых записей
                tables_to_clean = ['trades', 'signals', 'bot_state', 'errors', 'performance']
                
                for table in tables_to_clean:
                    if table == 'trades':
                        date_field = 'entry_time'
                    elif table == 'signals':
                        date_field = 'generated_at'
                    elif table == 'bot_state':
                        date_field = 'timestamp'
                    elif table == 'errors':
                        date_field = 'occurred_at'
                    elif table == 'performance':
                        date_field = 'timestamp'
                    
                    cursor.execute(f"""
                        DELETE FROM {table} 
                        WHERE {date_field} < ?
                    """, (cutoff_date,))
                    
                    deleted_count = cursor.rowcount
                    logger.info(f"Удалено {deleted_count} записей из таблицы {table}")
                
                # Оптимизация после очистки
                self.optimize_database()
                
        except Exception as e:
            logger.error(f"Ошибка очистки старых данных: {e}")


# Глобальный экземпляр менеджера базы данных
_db_instance = None

def get_database() -> DatabaseManager:
    """
    Получение глобального экземпляра менеджера базы данных.
    
    Returns:
        DatabaseManager: Экземпляр менеджера базы данных
    """
    global _db_instance
    if _db_instance is None:
        _db_instance = DatabaseManager()
    return _db_instance

def init_database():
    """Инициализация базы данных при запуске приложения"""
    db = get_database()
    # Планирование регулярной очистки старых данных
    # (должно быть интегрировано с системой планирования задач)
    logger.info("База данных готова к использованию")


if __name__ == "__main__":
    # Пример использования
    db = DatabaseManager()
    
    # Добавление тестовой сделки
    trade_id = db.add_trade({
        'symbol': 'BTC-USDT',
        'direction': 'BUY',
        'entry_price': 50000.0,
        'quantity': 0.1,
        'stop_loss': 49000.0,
        'take_profit': 52000.0,
        'order_id': 'test_order_123'
    })
    
    # Закрытие сделки
    db.close_trade(trade_id, 51000.0, 1000.0)
    
    # Добавление тестового сигнала
    signal_id = db.add_signal({
        'symbol': 'BTC-USDT',
        'timeframe': '15m',
        'direction': 'LONG',
        'price': 50000.0,
        'stop_loss': 49000.0,
        'take_profit': 52000.0,
        'confidence': 0.8,
        'risk_reward': 2.0,
        'position_size': 0.1,
        'reasons': ['RSI oversold', 'MACD bullish crossover'],
        'indicators': {'rsi': 30.5, 'macd': 50.2}
    })
    
    # Получение списка сделок
    trades = db.get_trades(limit=10)
    print(f"Последние {len(trades)} сделок:")
    for trade in trades:
        print(f"  {trade['symbol']} {trade['direction']} @ {trade['entry_price']}")
    
    # Получение статистики
    stats = db.get_performance_stats(days=7)
    print(f"Статистика за 7 дней: {stats['trades']['total_trades']} сделок")
    
    # Создание резервной копии
    backup_path = db.backup_database()
    print(f"Резервная копия создана: {backup_path}")