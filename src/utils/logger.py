"""
Утилиты для настройки и инициализации системы логирования.
"""

import os
import logging
import logging.config
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import json
from pathlib import Path

def setup_logging(
    default_path='config/logging.conf',
    default_level=logging.INFO,
    env_key='LOG_CFG'
):
    """
    Настройка системы логирования из конфигурационного файла.
    
    Args:
        default_path: Путь к файлу конфигурации по умолчанию
        default_level: Уровень логирования по умолчанию
        env_key: Ключ переменной окружения с путем к конфигурации
    """
    # Создание директории для логов, если не существует
    Path("logs").mkdir(exist_ok=True)
    
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    
    if os.path.exists(path):
        try:
            logging.config.fileConfig(path)
            logging.info(f"Логирование настроено из файла: {path}")
        except Exception as e:
            logging.basicConfig(level=default_level)
            logging.error(f"Ошибка загрузки конфигурации логирования: {e}")
    else:
        logging.basicConfig(level=default_level)
        logging.warning(f"Файл конфигурации логирования не найден: {path}")
    
    # Дополнительная настройка для JSON логов
    setup_json_logging()

def setup_json_logging():
    """
    Настройка JSON логирования для машинной обработки.
    """
    json_log_path = "logs/bot_json.log"
    
    # Создание обработчика для JSON логов
    json_handler = RotatingFileHandler(
        json_log_path,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    
    # Создание форматера для JSON
    class JsonFormatter(logging.Formatter):
        def format(self, record):
            log_data = {
                'timestamp': self.formatTime(record, self.datefmt),
                'name': record.name,
                'level': record.levelname,
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno,
                'message': record.getMessage(),
                'process': record.process,
                'thread': record.threadName
            }
            
            # Добавление исключения, если есть
            if record.exc_info:
                log_data['exception'] = self.formatException(record.exc_info)
            
            return json.dumps(log_data, ensure_ascii=False)
    
    json_handler.setFormatter(JsonFormatter())
    json_handler.setLevel(logging.INFO)
    
    # Добавление обработчика к корневому логгеру
    logging.getLogger().addHandler(json_handler)

def get_logger(name: str) -> logging.Logger:
    """
    Получение именованного логгера.
    
    Args:
        name: Имя логгера
        
    Returns:
        logging.Logger: Объект логгера
    """
    return logging.getLogger(name)

def setup_application_logging():
    """
    Настройка логирования для всего приложения.
    """
    # Основная конфигурация
    setup_logging()
    
    # Специальные обработчики для ошибок
    error_handler = RotatingFileHandler(
        'logs/errors.log',
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    
    # Форматер для ошибок
    error_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s'
    )
    error_handler.setFormatter(error_formatter)
    
    # Добавление обработчика ошибок ко всем логгерам
    for logger_name in ['bot', 'api', 'trading', 'signals', 'monitoring', 'database', 'backup', 'ml', 'web']:
        logging.getLogger(logger_name).addHandler(error_handler)
    
    logging.info("Логирование приложения настроено")

# Инициализация при импорте
setup_application_logging()