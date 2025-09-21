"""
Модуль мониторинга и алертинга торгового бота.
Обеспечивает отслеживание состояния системы, производительности и отправку уведомлений.
"""

import time
import psutil
import requests
import socket
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
import logging
import threading
import schedule
from prometheus_client import Gauge, Counter, Histogram, generate_latest, REGISTRY

from config.settings import (
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
    ALERT_THRESHOLD,
    REQUEST_TIMEOUT
)
from src.utils.logger import setup_logger
from src.bot.api_client import BingXAPI

logger = setup_logger(__name__)


# Prometheus метрики
SYSTEM_CPU_USAGE = Gauge('system_cpu_usage', 'CPU usage percentage')
SYSTEM_MEMORY_USAGE = Gauge('system_memory_usage', 'Memory usage percentage')
SYSTEM_DISK_USAGE = Gauge('system_disk_usage', 'Disk usage percentage')
BOT_UPTIME = Gauge('bot_uptime_seconds', 'Bot uptime in seconds')
API_RESPONSE_TIME = Histogram('api_response_time_seconds', 'API response time')
TRADE_COUNTER = Counter('trades_total', 'Total trades', ['symbol', 'direction'])
SIGNAL_COUNTER = Counter('signals_total', 'Total signals', ['symbol', 'direction'])
ERROR_COUNTER = Counter('errors_total', 'Total errors', ['type'])


class Monitoring:
    """Класс для мониторинга состояния торгового бота и системы"""
    
    def __init__(self, api_client: BingXAPI = None):
        """
        Инициализация системы мониторинга.
        
        Args:
            api_client: Клиент API для проверки соединения с биржей
        """
        self.api_client = api_client
        self.start_time = time.time()
        self.health_checks = {}
        self.metrics = {}
        self.alert_history = []
        self.alert_cooldown = {}
        
        # Инициализация метрик
        self._init_metrics()
        
        # Регистрация health checks
        self._register_health_checks()
        
        logger.info("Система мониторинга инициализирована")
    
    def _init_metrics(self):
        """Инициализация метрик мониторинга"""
        self.metrics = {
            'system': {
                'cpu_usage': 0,
                'memory_usage': 0,
                'disk_usage': 0,
                'uptime': 0
            },
            'api': {
                'last_response_time': 0,
                'success_rate': 100,
                'last_error': None
            },
            'trading': {
                'total_trades': 0,
                'successful_trades': 0,
                'failed_trades': 0,
                'active_positions': 0
            },
            'performance': {
                'signals_generated': 0,
                'signals_executed': 0,
                'profit_loss': 0
            }
        }
    
    def _register_health_checks(self):
        """Регистрация health checks системы"""
        self.health_checks = {
            'system_resources': {
                'check': self.check_system_resources,
                'threshold': 90,  # 90% usage
                'severity': 'warning'
            },
            'api_connectivity': {
                'check': self.check_api_connectivity,
                'threshold': 5000,  # 5 seconds response time
                'severity': 'critical'
            },
            'bot_activity': {
                'check': self.check_bot_activity,
                'threshold': 300,  # 5 minutes inactivity
                'severity': 'critical'
            },
            'disk_space': {
                'check': self.check_disk_space,
                'threshold': 10,  # 10% free space
                'severity': 'warning'
            }
        }
    
    def update_system_metrics(self):
        """Обновление метрик системы"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics['system']['cpu_usage'] = cpu_percent
            SYSTEM_CPU_USAGE.set(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.metrics['system']['memory_usage'] = memory.percent
            SYSTEM_MEMORY_USAGE.set(memory.percent)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            self.metrics['system']['disk_usage'] = disk.percent
            SYSTEM_DISK_USAGE.set(disk.percent)
            
            # Uptime
            uptime = time.time() - self.start_time
            self.metrics['system']['uptime'] = uptime
            BOT_UPTIME.set(uptime)
            
            logger.debug("Метрики системы обновлены")
            
        except Exception as e:
            logger.error(f"Ошибка обновления метрик системы: {e}")
            ERROR_COUNTER.labels(type='system_metrics').inc()
    
    def update_api_metrics(self, response_time: float, success: bool = True, error: str = None):
        """
        Обновление метрик API.
        
        Args:
            response_time: Время ответа API в секундах
            success: Успешен ли запрос
            error: Сообщение об ошибке (если есть)
        """
        try:
            self.metrics['api']['last_response_time'] = response_time
            API_RESPONSE_TIME.observe(response_time)
            
            if not success:
                self.metrics['api']['last_error'] = error
                ERROR_COUNTER.labels(type='api_request').inc()
            
            logger.debug(f"Метрики API обновлены: {response_time:.3f}s, success: {success}")
            
        except Exception as e:
            logger.error(f"Ошибка обновления метрик API: {e}")
            ERROR_COUNTER.labels(type='api_metrics').inc()
    
    def update_trading_metrics(self, symbol: str, direction: str, success: bool = True, pnl: float = 0):
        """
        Обновление метрик торговли.
        
        Args:
            symbol: Торговая пара
            direction: Направление сделки (BUY/SELL)
            success: Успешна ли сделка
            pnl: Прибыль/убыток от сделки
        """
        try:
            self.metrics['trading']['total_trades'] += 1
            
            if success:
                self.metrics['trading']['successful_trades'] += 1
                self.metrics['performance']['profit_loss'] += pnl
            else:
                self.metrics['trading']['failed_trades'] += 1
            
            TRADE_COUNTER.labels(symbol=symbol, direction=direction).inc()
            
            logger.debug(f"Метрики торговли обновлены: {symbol} {direction}, success: {success}")
            
        except Exception as e:
            logger.error(f"Ошибка обновления метрик торговли: {e}")
            ERROR_COUNTER.labels(type='trading_metrics').inc()
    
    def update_signal_metrics(self, symbol: str, direction: str, executed: bool = False):
        """
        Обновление метрик сигналов.
        
        Args:
            symbol: Торговая пара
            direction: Направление сигнала (LONG/SHORT)
            executed: Был ли сигнал исполнен
        """
        try:
            self.metrics['performance']['signals_generated'] += 1
            
            if executed:
                self.metrics['performance']['signals_executed'] += 1
            
            SIGNAL_COUNTER.labels(symbol=symbol, direction=direction).inc()
            
            logger.debug(f"Метрики сигналов обновлены: {symbol} {direction}, executed: {executed}")
            
        except Exception as e:
            logger.error(f"Ошибка обновления метрик сигналов: {e}")
            ERROR_COUNTER.labels(type='signal_metrics').inc()
    
    def check_system_resources(self) -> Dict[str, Any]:
        """
        Проверка использования системных ресурсов.
        
        Returns:
            Dict: Результаты проверки
        """
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            
            # Disk usage
            disk = psutil.disk_usage('/')
            
            result = {
                'cpu_usage': cpu_percent,
                'memory_usage': memory.percent,
                'disk_usage': disk.percent,
                'status': 'healthy',
                'message': 'Системные ресурсы в норме'
            }
            
            # Проверка превышения порогов
            if cpu_percent > self.health_checks['system_resources']['threshold']:
                result['status'] = 'warning'
                result['message'] = f'Высокая загрузка CPU: {cpu_percent}%'
            
            if memory.percent > self.health_checks['system_resources']['threshold']:
                result['status'] = 'warning'
                result['message'] = f'Высокая загрузка памяти: {memory.percent}%'
            
            if disk.percent > (100 - self.health_checks['disk_space']['threshold']):
                result['status'] = 'warning'
                result['message'] = f'Мало свободного места на диске: {100 - disk.percent}% свободно'
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка проверки системных ресурсов: {e}")
            ERROR_COUNTER.labels(type='system_check').inc()
            return {
                'status': 'error',
                'message': f'Ошибка проверки системных ресурсов: {e}'
            }
    
    def check_api_connectivity(self) -> Dict[str, Any]:
        """
        Проверка подключения к API биржи.
        
        Returns:
            Dict: Результаты проверки
        """
        if not self.api_client:
            return {
                'status': 'unknown',
                'message': 'API клиент не настроен'
            }
        
        try:
            start_time = time.time()
            
            # Простой запрос для проверки подключения
            result = self.api_client.get_klines("BTC-USDT", "15m", limit=1)
            response_time = (time.time() - start_time) * 1000  # в миллисекундах
            
            if result and 'data' in result:
                status = 'healthy'
                message = f'API подключение стабильно, время ответа: {response_time:.2f}ms'
            else:
                status = 'critical'
                message = 'Ошибка подключения к API'
            
            return {
                'status': status,
                'message': message,
                'response_time': response_time
            }
            
        except Exception as e:
            logger.error(f"Ошибка проверки подключения к API: {e}")
            ERROR_COUNTER.labels(type='api_check').inc()
            return {
                'status': 'critical',
                'message': f'Ошибка подключения к API: {e}',
                'response_time': 0
            }
    
    def check_bot_activity(self, last_activity_time: float = None) -> Dict[str, Any]:
        """
        Проверка активности бота.
        
        Args:
            last_activity_time: Время последней активности
            
        Returns:
            Dict: Результаты проверки
        """
        try:
            if last_activity_time is None:
                # Используем время последнего обновления метрик как индикатор активности
                last_activity_time = max(
                    self.metrics['system'].get('last_update', 0),
                    self.metrics['api'].get('last_update', 0),
                    self.metrics['trading'].get('last_update', 0)
                )
            
            inactivity_time = time.time() - last_activity_time
            threshold = self.health_checks['bot_activity']['threshold']
            
            if inactivity_time > threshold:
                return {
                    'status': 'critical',
                    'message': f'Бот неактивен в течение {inactivity_time:.0f} секунд',
                    'inactivity_time': inactivity_time
                }
            else:
                return {
                    'status': 'healthy',
                    'message': 'Бот активен',
                    'inactivity_time': inactivity_time
                }
            
        except Exception as e:
            logger.error(f"Ошибка проверки активности бота: {e}")
            ERROR_COUNTER.labels(type='activity_check').inc()
            return {
                'status': 'error',
                'message': f'Ошибка проверки активности бота: {e}'
            }
    
    def check_disk_space(self) -> Dict[str, Any]:
        """
        Проверка свободного места на диске.
        
        Returns:
            Dict: Результаты проверки
        """
        try:
            disk = psutil.disk_usage('/')
            free_space_percent = 100 - disk.percent
            threshold = self.health_checks['disk_space']['threshold']
            
            if free_space_percent < threshold:
                return {
                    'status': 'warning',
                    'message': f'Мало свободного места на диске: {free_space_percent:.1f}%',
                    'free_space_percent': free_space_percent
                }
            else:
                return {
                    'status': 'healthy',
                    'message': f'Свободного места на диске: {free_space_percent:.1f}%',
                    'free_space_percent': free_space_percent
                }
            
        except Exception as e:
            logger.error(f"Ошибка проверки свободного места на диске: {e}")
            ERROR_COUNTER.labels(type='disk_check').inc()
            return {
                'status': 'error',
                'message': f'Ошибка проверки свободного места на диске: {e}'
            }
    
    def run_health_checks(self) -> Dict[str, Any]:
        """
        Запуск всех health checks.
        
        Returns:
            Dict: Результаты всех проверок
        """
        results = {}
        
        for check_name, check_config in self.health_checks.items():
            try:
                results[check_name] = check_config['check']()
                
                # Проверка необходимости отправки уведомления
                if results[check_name]['status'] in ['warning', 'critical', 'error']:
                    self.check_alert_condition(check_name, results[check_name])
                    
            except Exception as e:
                logger.error(f"Ошибка выполнения health check '{check_name}': {e}")
                ERROR_COUNTER.labels(type='health_check').inc()
                results[check_name] = {
                    'status': 'error',
                    'message': f'Ошибка выполнения проверки: {e}'
                }
        
        return results
    
    def check_alert_condition(self, check_name: str, check_result: Dict[str, Any]):
        """
        Проверка условий для отправки уведомления.
        
        Args:
            check_name: Название проверки
            check_result: Результат проверки
        """
        try:
            # Проверка кулдауна для этого типа уведомлений
            current_time = time.time()
            last_alert_time = self.alert_cooldown.get(check_name, 0)
            alert_cooldown = 300  # 5 минут между уведомлениями одного типа
            
            if current_time - last_alert_time < alert_cooldown:
                return
            
            # Проверка порога для критических уведомлений
            if check_result['status'] == 'critical':
                self.send_alert(check_name, check_result)
                self.alert_cooldown[check_name] = current_time
                
            # Проверка порога для предупреждений
            elif check_result['status'] == 'warning':
                # Увеличиваем счетчик предупреждений
                warning_count = self.alert_cooldown.get(f"{check_name}_warnings", 0) + 1
                self.alert_cooldown[f"{check_name}_warnings"] = warning_count
                
                # Отправляем уведомление после определенного количества предупреждений
                if warning_count >= ALERT_THRESHOLD:
                    self.send_alert(check_name, check_result)
                    self.alert_cooldown[check_name] = current_time
                    self.alert_cooldown[f"{check_name}_warnings"] = 0  # Сброс счетчика
            
        except Exception as e:
            logger.error(f"Ошибка проверки условий уведомления: {e}")
            ERROR_COUNTER.labels(type='alert_check').inc()
    
    def send_alert(self, check_name: str, check_result: Dict[str, Any]):
        """
        Отправка уведомления о проблеме.
        
        Args:
            check_name: Название проверки
            check_result: Результат проверки
        """
        try:
            # Форматирование сообщения
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            severity = check_result['status'].upper()
            message = check_result['message']
            
            alert_message = (
                f"🚨 *{severity} ALERT* 🚨\n"
                f"*Time:* {timestamp}\n"
                f"*Check:* {check_name}\n"
                f"*Status:* {severity}\n"
                f"*Message:* {message}\n"
                f"*Bot Uptime:* {self.metrics['system']['uptime']:.0f}s"
            )
            
            # Отправка в Telegram
            if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                self.send_telegram_alert(alert_message)
            
            # Логирование уведомления
            logger.warning(f"ALERT: {check_name} - {message}")
            
            # Сохранение в историю уведомлений
            self.alert_history.append({
                'timestamp': timestamp,
                'check_name': check_name,
                'severity': severity,
                'message': message,
                'sent': True
            })
            
            # Ограничение размера истории
            if len(self.alert_history) > 100:
                self.alert_history = self.alert_history[-100:]
            
        except Exception as e:
            logger.error(f"Ошибка отправки уведомления: {e}")
            ERROR_COUNTER.labels(type='alert_send').inc()
    
    def send_telegram_alert(self, message: str):
        """
        Отправка уведомления в Telegram.
        
        Args:
            message: Текст сообщения
        """
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            payload = {
                'chat_id': TELEGRAM_CHAT_ID,
                'text': message,
                'parse_mode': 'Markdown'
            }
            
            response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            
            logger.info("Уведомление отправлено в Telegram")
            
        except Exception as e:
            logger.error(f"Ошибка отправки уведомления в Telegram: {e}")
            ERROR_COUNTER.labels(type='telegram_alert').inc()
    
    def get_prometheus_metrics(self) -> str:
        """
        Получение метрик в формате Prometheus.
        
        Returns:
            str: Метрики в текстовом формате
        """
        try:
            return generate_latest(REGISTRY).decode('utf-8')
        except Exception as e:
            logger.error(f"Ошибка получения метрик Prometheus: {e}")
            ERROR_COUNTER.labels(type='prometheus_metrics').inc()
            return ""
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Получение общего статуса здоровья системы.
        
        Returns:
            Dict: Статус здоровья системы
        """
        health_checks = self.run_health_checks()
        
        # Определение общего статуса
        overall_status = 'healthy'
        for check_name, check_result in health_checks.items():
            if check_result['status'] == 'critical':
                overall_status = 'critical'
                break
            elif check_result['status'] == 'warning' and overall_status == 'healthy':
                overall_status = 'warning'
            elif check_result['status'] == 'error' and overall_status == 'healthy':
                overall_status = 'error'
        
        return {
            'status': overall_status,
            'timestamp': datetime.now().isoformat(),
            'metrics': self.metrics,
            'health_checks': health_checks,
            'uptime': self.metrics['system']['uptime'],
            'alerts': {
                'total': len(self.alert_history),
                'last_24h': len([a for a in self.alert_history 
                               if datetime.now() - datetime.fromisoformat(a['timestamp'].replace(' ', 'T')) < timedelta(hours=24)])
            }
        }
    
    def start_monitoring_loop(self, interval: int = 60):
        """
        Запуск цикла мониторинга.
        
        Args:
            interval: Интервал мониторинга в секундах
        """
        def monitoring_loop():
            while True:
                try:
                    # Обновление метрик системы
                    self.update_system_metrics()
                    
                    # Запуск health checks
                    self.run_health_checks()
                    
                    # Пауза перед следующей итерацией
                    time.sleep(interval)
                    
                except Exception as e:
                    logger.error(f"Ошибка в цикле мониторинга: {e}")
                    ERROR_COUNTER.labels(type='monitoring_loop').inc()
                    time.sleep(interval)  # Пауза при ошибке
        
        # Запуск цикла мониторинга в отдельном потоке
        monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitor_thread.start()
        
        logger.info(f"Цикл мониторинга запущен с интервалом {interval} секунд")
    
    def schedule_daily_report(self):
        """Планирование ежедневного отчета"""
        try:
            # Ежедневный отчет в 09:00
            schedule.every().day.at("09:00").do(self.send_daily_report)
            
            # Запуск планировщика в отдельном потоке
            def scheduler_loop():
                while True:
                    schedule.run_pending()
                    time.sleep(60)
            
            scheduler_thread = threading.Thread(target=scheduler_loop, daemon=True)
            scheduler_thread.start()
            
            logger.info("Ежедневный отчет запланирован на 09:00")
            
        except Exception as e:
            logger.error(f"Ошибка планирования ежедневного отчета: {e}")
            ERROR_COUNTER.labels(type='scheduler').inc()
    
    def send_daily_report(self):
        """Отправка ежедневного отчета"""
        try:
            # Получение статуса здоровья
            health_status = self.get_health_status()
            
            # Форматирование отчета
            timestamp = datetime.now().strftime("%Y-%m-%d")
            report_message = (
                f"📊 *Ежедневный отчет торгового бота* 📊\n"
                f"*Дата:* {timestamp}\n"
                f"*Общий статус:* {health_status['status'].upper()}\n"
                f"*Аптайм:* {health_status['uptime']:.0f} секунд\n"
                f"*Сделок за день:* {self.metrics['trading']['total_trades']}\n"
                f"*Сигналов за день:* {self.metrics['performance']['signals_generated']}\n"
                f"*P/L за день:* {self.metrics['performance']['profit_loss']:.2f} USDT\n"
                f"*Уведомлений за день:* {health_status['alerts']['last_24h']}\n"
                f"*CPU использование:* {self.metrics['system']['cpu_usage']:.1f}%\n"
                f"*Память использование:* {self.metrics['system']['memory_usage']:.1f}%\n"
                f"*Диск использование:* {self.metrics['system']['disk_usage']:.1f}%"
            )
            
            # Отправка в Telegram
            if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                self.send_telegram_alert(report_message)
            
            logger.info("Ежедневный отчет отправлен")
            
        except Exception as e:
            logger.error(f"Ошибка отправки ежедневного отчета: {e}")
            ERROR_COUNTER.labels(type='daily_report').inc()


# Создание глобального экземпляра мониторинга
monitoring_instance = None

def init_monitoring(api_client: BingXAPI = None) -> Monitoring:
    """
    Инициализация глобального экземпляра мониторинга.
    
    Args:
        api_client: Клиент API для проверки соединения
        
    Returns:
        Monitoring: Экземпляр мониторинга
    """
    global monitoring_instance
    monitoring_instance = Monitoring(api_client)
    return monitoring_instance

def get_monitoring() -> Monitoring:
    """
    Получение глобального экземпляра мониторинга.
    
    Returns:
        Monitoring: Экземпляр мониторинга
    """
    global monitoring_instance
    if monitoring_instance is None:
        monitoring_instance = Monitoring()
    return monitoring_instance


if __name__ == "__main__":
    # Пример использования
    monitoring = Monitoring()
    
    # Обновление метрик
    monitoring.update_system_metrics()
    monitoring.update_api_metrics(0.15, True)
    monitoring.update_trading_metrics("BTC-USDT", "BUY", True, 50.0)
    monitoring.update_signal_metrics("BTC-USDT", "LONG", True)
    
    # Запуск health checks
    health_status = monitoring.run_health_checks()
    print("Health Status:", json.dumps(health_status, indent=2))
    
    # Получение метрик Prometheus
    prometheus_metrics = monitoring.get_prometheus_metrics()
    print("Prometheus Metrics:", prometheus_metrics[:200] + "...")
    
    # Запуск цикла мониторинга
    monitoring.start_monitoring_loop(interval=30)
    
    # Демонстрация работы
    time.sleep(60)