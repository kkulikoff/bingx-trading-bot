"""
Основной модуль BingX Trading Bot.

Координирует работу всех компонентов системы: сбор данных, анализ,
генерацию сигналов, исполнение сделок и мониторинг.
"""

import asyncio
import time
import schedule
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from src.bot.api_client import BingXAPI
from src.bot.risk_manager import RiskManager
from src.bot.signal_generator import SignalGenerator
from src.bot.trading_engine import TradingEngine
from src.bot.ml_model import MLModel
from src.bot.backtester import Backtester
from src.data.backup_manager import BackupManager
from src.utils.logger import setup_logger
from src.utils.helpers import safe_round, generate_hash, retry_on_exception

logger = setup_logger(__name__)

class AdvancedBingXBot:
    """
    Главный класс торгового бота, координирующий все компоненты системы.
    """
    
    def __init__(self):
        """
        Инициализация торгового бота с всеми компонентами.
        """
        # Компоненты бота
        self.api_client = None
        self.risk_manager = None
        self.signal_generator = None
        self.trading_engine = None
        self.ml_model = None
        self.backtester = None
        self.backup_manager = None
        
        # Состояние бота
        self.is_running = False
        self.is_initialized = False
        self.start_time = None
        self.iteration_count = 0
        
        # Данные
        self.historical_data = {}
        self.signals_history = []
        self.performance_metrics = {}
        
        # Конфигурация
        self.config = self._load_config()
        
        # Планировщик задач
        self.scheduled_tasks = []
        
        logger.info("Расширенный торговый бот инициализирован")
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Загрузка конфигурации из настроек.
        
        Returns:
            Dict: Конфигурация бота
        """
        try:
            from config.settings import (
                SYMBOLS, TIMEFRAMES, INITIAL_BALANCE, RISK_PER_TRADE,
                BINGX_API_KEY, BINGX_SECRET_KEY, ML_MODEL_PATH, ML_SCALER_PATH,
                BACKUP_SCHEDULE, MAX_BACKUPS
            )
            
            symbols = [s.strip() for s in SYMBOLS.split(",")]
            timeframes = [t.strip() for t in TIMEFRAMES.split(",")]
            
            return {
                'symbols': symbols,
                'timeframes': timeframes,
                'initial_balance': INITIAL_BALANCE,
                'risk_per_trade': RISK_PER_TRADE,
                'api_key': BINGX_API_KEY,
                'api_secret': BINGX_SECRET_KEY,
                'model_path': ML_MODEL_PATH,
                'scaler_path': ML_SCALER_PATH,
                'backup_schedule': BACKUP_SCHEDULE,
                'max_backups': MAX_BACKUPS
            }
            
        except Exception as e:
            logger.error(f"Ошибка загрузки конфигурации: {e}")
            # Возвращаем конфигурацию по умолчанию
            return {
                'symbols': ["BTC-USDT", "ETH-USDT", "SOL-USDT", "XRP-USDT"],
                'timeframes': ["15m", "1h", "4h"],
                'initial_balance': 10000,
                'risk_per_trade': 0.02,
                'api_key': "",
                'api_secret': "",
                'model_path': "data/models/trading_model.pkl",
                'scaler_path': "data/models/scaler.pkl",
                'backup_schedule': "daily",
                'max_backups': 30
            }
    
    async def initialize(self) -> bool:
        """
        Инициализация всех компонентов бота.
        
        Returns:
            bool: True если инициализация успешна, иначе False
        """
        try:
            logger.info("Начало инициализации торгового бота")
            
            # Инициализация компонентов
            self.api_client = BingXAPI(
                self.config['api_key'], 
                self.config['api_secret']
            )
            
            self.risk_manager = RiskManager(
                self.config['initial_balance'],
                self.config['risk_per_trade']
            )
            
            self.signal_generator = SignalGenerator()
            self.trading_engine = TradingEngine(self.api_client, self.risk_manager)
            self.ml_model = MLModel(
                self.config['model_path'],
                self.config['scaler_path']
            )
            
            self.backtester = Backtester(self.config['initial_balance'])
            self.backup_manager = BackupManager()
            
            # Инициализация API клиента
            await self.api_client.initialize()
            
            # Загрузка исторических данных
            await self._load_historical_data()
            
            # Инициализация ML модели
            await self.ml_model.initialize(self.historical_data)
            
            # Инициализация генератора сигналов с ML моделью
            await self.signal_generator.initialize(self.ml_model)
            
            # Настройка планировщика задач
            self._setup_scheduler()
            
            # Создание первоначальной резервной копии
            await self.backup_manager.create_backup()
            
            self.is_initialized = True
            logger.info("Торговый бот успешно инициализирован")
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка инициализации бота: {e}")
            await self.shutdown()
            return False
    
    async def _load_historical_data(self) -> None:
        """
        Загрузка исторических данных для всех символов и таймфреймов.
        """
        logger.info("Загрузка исторических данных")
        
        for symbol in self.config['symbols']:
            self.historical_data[symbol] = {}
            
            for timeframe in self.config['timeframes']:
                try:
                    data = await self.api_client.get_historical_data(
                        symbol, timeframe, limit=500
                    )
                    
                    if data and len(data) > 0:
                        # Преобразование в DataFrame
                        df = pd.DataFrame(data, columns=[
                            'timestamp', 'open', 'high', 'low', 'close', 'volume'
                        ])
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        df.set_index('timestamp', inplace=True)
                        
                        # Преобразование типов данных
                        for col in ['open', 'high', 'low', 'close', 'volume']:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        
                        df.dropna(inplace=True)
                        self.historical_data[symbol][timeframe] = df
                        
                        logger.info(f"Загружены данные для {symbol} ({timeframe}): {len(df)} записей")
                    else:
                        logger.warning(f"Не удалось загрузить данные для {symbol} ({timeframe})")
                        
                except Exception as e:
                    logger.error(f"Ошибка загрузки данных для {symbol} ({timeframe}): {e}")
    
    def _setup_scheduler(self) -> None:
        """
        Настройка планировщика задач для регулярных операций.
        """
        # Ежедневное резервное копирование в 2:00
        schedule.every().day.at("02:00").do(
            self._run_scheduled_task, self.backup_manager.create_backup
        )
        
        # Очистка старых резервных копий в 3:00
        schedule.every().day.at("03:00").do(
            self._run_scheduled_task, self.backup_manager.cleanup_old_backups
        )
        
        # Ежечасный бэктестинг
        schedule.every().hour.do(
            self._run_scheduled_task, self._run_backtesting
        )
        
        # Ежедневное обновление ML модели
        schedule.every().day.at("04:00").do(
            self._run_scheduled_task, self._retrain_ml_model
        )
        
        # Еженедельная оптимизация параметров
        schedule.every().sunday.at("05:00").do(
            self._run_scheduled_task, self._optimize_parameters
        )
        
        logger.info("Планировщик задач настроен")
    
    async def _run_scheduled_task(self, task_func) -> None:
        """
        Запуск запланированной задачи с обработкой ошибок.
        
        Args:
            task_func: Функция задачи для выполнения
        """
        try:
            if asyncio.iscoroutinefunction(task_func):
                await task_func()
            else:
                task_func()
        except Exception as e:
            logger.error(f"Ошибка выполнения запланированной задачи: {e}")
    
    async def run(self) -> None:
        """
        Основной цикл работы торгового бота.
        """
        if not self.is_initialized:
            logger.error("Бот не инициализирован. Запуск невозможен.")
            return
        
        self.is_running = True
        self.start_time = datetime.now()
        
        logger.info("Запуск основного цикла торгового бота")
        
        try:
            while self.is_running:
                try:
                    # Выполнение запланированных задач
                    schedule.run_pending()
                    
                    # Обновление рыночных данных
                    await self._update_market_data()
                    
                    # Генерация и исполнение сигналов
                    await self._generate_and_execute_signals()
                    
                    # Обновление информации о портфеле
                    if self.iteration_count % 5 == 0:  # Каждые 5 итераций
                        await self._update_portfolio()
                    
                    # Проверка статуса ордеров
                    if self.iteration_count % 3 == 0:  # Каждые 3 итерации
                        await self.trading_engine.check_orders_status()
                    
                    # Логирование состояния
                    if self.iteration_count % 10 == 0:  # Каждые 10 итераций
                        self._log_bot_status()
                    
                    # Увеличение счетчика итераций
                    self.iteration_count += 1
                    
                    # Пауза между итерациями
                    await asyncio.sleep(60)  # 1 минута
                    
                except Exception as e:
                    logger.error(f"Ошибка в основной итерации: {e}")
                    await asyncio.sleep(300)  # 5 минут при ошибке
                    
        except KeyboardInterrupt:
            logger.info("Получен сигнал завершения работы")
        except Exception as e:
            logger.error(f"Критическая ошибка в основном цикле: {e}")
        finally:
            await self.shutdown()
    
    async def _update_market_data(self) -> None:
        """
        Обновление рыночных данных для всех символов и таймфреймов.
        """
        try:
            for symbol in self.config['symbols']:
                for timeframe in self.config['timeframes']:
                    # Получение новых данных
                    new_data = await self.api_client.get_historical_data(
                        symbol, timeframe, limit=100
                    )
                    
                    if new_data and len(new_data) > 0:
                        # Преобразование в DataFrame
                        df = pd.DataFrame(new_data, columns=[
                            'timestamp', 'open', 'high', 'low', 'close', 'volume'
                        ])
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        df.set_index('timestamp', inplace=True)
                        
                        # Преобразование типов данных
                        for col in ['open', 'high', 'low', 'close', 'volume']:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        
                        df.dropna(inplace=True)
                        
                        # Обновление исторических данных
                        if symbol in self.historical_data and timeframe in self.historical_data[symbol]:
                            # Объединение с существующими данными
                            old_df = self.historical_data[symbol][timeframe]
                            combined_df = pd.concat([old_df, df])
                            
                            # Удаление дубликатов
                            combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                            
                            # Сохранение только последних 500 записей
                            self.historical_data[symbol][timeframe] = combined_df.tail(500)
                        else:
                            self.historical_data[symbol][timeframe] = df
                        
                        logger.debug(f"Данные обновлены для {symbol} ({timeframe})")
            
            logger.info("Рыночные данные успешно обновлены")
            
        except Exception as e:
            logger.error(f"Ошибка обновления рыночных данных: {e}")
    
    async def _generate_and_execute_signals(self) -> None:
        """
        Генерация и исполнение торговых сигналов.
        """
        try:
            signals = []
            
            # Генерация сигналов для всех символов и таймфреймов
            for symbol in self.config['symbols']:
                for timeframe in self.config['timeframes']:
                    if (symbol in self.historical_data and 
                        timeframe in self.historical_data[symbol]):
                        
                        data = self.historical_data[symbol][timeframe]
                        signal = await self.signal_generator.generate_signal(
                            data, symbol, timeframe
                        )
                        
                        if signal:
                            signals.append(signal)
            
            # Исполнение сигналов
            executed_signals = []
            for signal in signals:
                if signal['confidence'] >= 0.7:  # Порог уверенности
                    result = await self.trading_engine.execute_signal(signal)
                    
                    if result.get('success', False):
                        executed_signals.append(signal)
                        self.signals_history.append(signal)
            
            # Логирование результатов
            if executed_signals:
                logger.info(f"Исполнено {len(executed_signals)} сигналов")
            else:
                logger.debug("Нет сигналов для исполнения")
                
        except Exception as e:
            logger.error(f"Ошибка генерации и исполнения сигналов: {e}")
    
    async def _update_portfolio(self) -> None:
        """
        Обновление информации о портфеле.
        """
        try:
            portfolio = await self.api_client.get_account_balance()
            
            if portfolio and 'balances' in portfolio:
                # Обновление баланса в risk manager
                total_balance = 0
                
                for asset in portfolio['balances']:
                    free = float(asset.get('free', 0))
                    locked = float(asset.get('locked', 0))
                    total = free + locked
                    
                    if total > 0:
                        total_balance += total
                
                # Здесь можно добавить логику конвертации в USDT
                # для расчета общего баланса
                
                logger.info(f"Портфель обновлен. Активов: {len(portfolio['balances'])}")
            
        except Exception as e:
            logger.error(f"Ошибка обновления портфеля: {e}")
    
    async def _run_backtesting(self) -> None:
        """
        Запуск бэктестинга стратегий.
        """
        try:
            logger.info("Запуск бэктестинга стратегий")
            
            for symbol in self.config['symbols'][:2]:  # Первые 2 символа
                for timeframe in self.config['timeframes'][:1]:  # Первый таймфрейм
                    if (symbol in self.historical_data and 
                        timeframe in self.historical_data[symbol]):
                        
                        data = self.historical_data[symbol][timeframe]
                        results = await self.backtester.run_backtest(
                            data, self.signal_generator, symbol, timeframe
                        )
                        
                        if results:
                            # Сохранение результатов
                            self.performance_metrics[f"{symbol}_{timeframe}"] = results
                            logger.info(f"Бэктест завершен для {symbol} ({timeframe}): "
                                      f"ROI: {results.get('total_return', 0):.2f}%")
            
            logger.info("Бэктестинг завершен")
            
        except Exception as e:
            logger.error(f"Ошибка бэктестинга: {e}")
    
    async def _retrain_ml_model(self) -> None:
        """
        Переобучение ML модели на новых данных.
        """
        try:
            logger.info("Переобучение ML модели")
            
            success = await self.ml_model.train_model(self.historical_data)
            
            if success:
                logger.info("ML модель успешно переобучена")
                
                # Обновление генератора сигналов с новой моделью
                await self.signal_generator.initialize(self.ml_model)
            else:
                logger.warning("Не удалось переобучить ML модель")
                
        except Exception as e:
            logger.error(f"Ошибка переобучения ML модели: {e}")
    
    async def _optimize_parameters(self) -> None:
        """
        Оптимизация параметров торговой стратегии.
        """
        try:
            logger.info("Оптимизация параметров стратегии")
            
            # Здесь можно реализовать оптимизацию параметров индикаторов
            # на основе результатов бэктестинга
            
            # Пример: оптимизация периодов индикаторов
            best_params = await self._optimize_indicator_parameters()
            
            if best_params:
                # Обновление параметров генератора сигналов
                self.signal_generator.indicator_settings.update(best_params)
                logger.info(f"Параметры стратегии оптимизированы: {best_params}")
            else:
                logger.info("Оптимизация параметров не дала улучшений")
                
        except Exception as e:
            logger.error(f"Ошибка оптимизации параметров: {e}")
    
    async def _optimize_indicator_parameters(self) -> Dict[str, Any]:
        """
        Оптимизация параметров индикаторов на основе исторических данных.
        
        Returns:
            Dict: Оптимальные параметры индикаторов
        """
        # Заглушка для оптимизации параметров
        # В реальной реализации здесь будет алгоритм оптимизации
        # (Grid Search, Genetic Algorithm, etc.)
        
        return {}  # Возвращаем пустой словарь, если оптимизация не реализована
    
    def _log_bot_status(self) -> None:
        """
        Логирование текущего состояния бота.
        """
        try:
            status = {
                'running_time': str(datetime.now() - self.start_time),
                'iteration_count': self.iteration_count,
                'symbols_monitored': len(self.config['symbols']),
                'timeframes_monitored': len(self.config['timeframes']),
                'signals_generated': len(self.signals_history),
                'portfolio_balance': safe_round(self.risk_manager.balance, 2),
                'open_orders': len(self.trading_engine.open_orders),
                'open_positions': len(self.trading_engine.positions)
            }
            
            logger.info(f"Статус бота: {status}")
            
            # Логирование производительности
            if self.performance_metrics:
                total_return = np.mean([
                    metrics.get('total_return', 0) 
                    for metrics in self.performance_metrics.values()
                ])
                logger.info(f"Средняя доходность стратегий: {total_return:.2f}%")
                
        except Exception as e:
            logger.error(f"Ошибка логирования статуса: {e}")
    
    async def shutdown(self) -> None:
        """
        Корректное завершение работы торгового бота.
        """
        logger.info("Завершение работы торгового бота")
        
        self.is_running = False
        
        try:
            # Остановка компонентов
            if self.trading_engine:
                await self.trading_engine.shutdown()
            
            if self.api_client:
                await self.api_client.close()
            
            # Создание финальной резервной копии
            if self.backup_manager:
                await self.backup_manager.create_backup()
            
            # Логирование финальной статистики
            logger.info(f"Итоговый баланс: {safe_round(self.risk_manager.balance, 2)} USDT")
            logger.info(f"Всего итераций: {self.iteration_count}")
            logger.info(f"Всего сигналов: {len(self.signals_history)}")
            
            logger.info("Торговый бот успешно остановлен")
            
        except Exception as e:
            logger.error(f"Ошибка завершения работы: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Получение текущего статуса бота.
        
        Returns:
            Dict: Статус бота
        """
        return {
            'is_running': self.is_running,
            'is_initialized': self.is_initialized,
            'start_time': self.start_time,
            'iteration_count': self.iteration_count,
            'symbols': self.config['symbols'],
            'timeframes': self.config['timeframes'],
            'portfolio_balance': safe_round(self.risk_manager.balance, 2),
            'open_orders': len(self.trading_engine.open_orders) if self.trading_engine else 0,
            'open_positions': len(self.trading_engine.positions) if self.trading_engine else 0,
            'signals_history': len(self.signals_history),
            'performance_metrics': self.performance_metrics
        }
    
    async def force_signal_generation(self, symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        """
        Принудительная генерация сигнала для указанного символа и таймфрейма.
        
        Args:
            symbol: Торговая пара
            timeframe: Таймфрейм
            
        Returns:
            Optional[Dict]: Сгенерированный сигнал или None
        """
        try:
            if (symbol in self.historical_data and 
                timeframe in self.historical_data[symbol]):
                
                data = self.historical_data[symbol][timeframe]
                signal = await self.signal_generator.generate_signal(data, symbol, timeframe)
                
                if signal:
                    logger.info(f"Принудительно сгенерирован сигнал для {symbol} ({timeframe})")
                    return signal
                else:
                    logger.info(f"Не удалось сгенерировать сигнал для {symbol} ({timeframe})")
                    return None
            else:
                logger.warning(f"Нет данных для {symbol} ({timeframe})")
                return None
                
        except Exception as e:
            logger.error(f"Ошибка принудительной генерации сигнала: {e}")
            return None
    
    async def manual_trade(self, symbol: str, direction: str, quantity: float, 
                          price: Optional[float] = None) -> Dict[str, Any]:
        """
        Выполнение ручной торговой операции.
        
        Args:
            symbol: Торговая пара
            direction: Направление (BUY/SELL)
            quantity: Количество
            price: Цена (опционально, для лимитных ордеров)
            
        Returns:
            Dict: Результат операции
        """
        try:
            # Создание искусственного сигнала для ручной торговли
            signal = {
                'symbol': symbol,
                'direction': 'LONG' if direction.upper() == 'BUY' else 'SHORT',
                'price': price or await self._get_current_price(symbol),
                'quantity': quantity,
                'confidence': 1.0,
                'reasons': ['Ручная торговая операция'],
                'timestamp': datetime.now()
            }
            
            # Исполнение ордера
            result = await self.trading_engine.execute_signal(signal)
            
            if result.get('success', False):
                logger.info(f"Ручная торговая операция выполнена: {symbol} {direction} {quantity}")
            else:
                logger.error(f"Ошибка ручной торговой операции: {result.get('error', 'Неизвестная ошибка')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка ручной торговой операции: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _get_current_price(self, symbol: str) -> float:
        """
        Получение текущей цены для символа.
        
        Args:
            symbol: Торговая пара
            
        Returns:
            float: Текущая цена
        """
        try:
            # Получение последних данных
            data = await self.api_client.get_historical_data(symbol, "1m", limit=1)
            
            if data and len(data) > 0:
                return float(data[0][4])  # close price
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Ошибка получения текущей цены: {e}")
            return 0.0


# Утилитарные функции для работы с ботом

async def create_and_run_bot() -> AdvancedBingXBot:
    """
    Создание и запуск торгового бота.
    
    Returns:
        AdvancedBingXBot: Экземпляр торгового бота
    """
    bot = AdvancedBingXBot()
    
    try:
        # Инициализация бота
        success = await bot.initialize()
        
        if success:
            # Запуск основного цикла в отдельной задаче
            asyncio.create_task(bot.run())
            logger.info("Торговый бот запущен")
        else:
            logger.error("Не удалось инициализировать торговый бот")
        
        return bot
        
    except Exception as e:
        logger.error(f"Ошибка создания и запуска бота: {e}")
        await bot.shutdown()
        raise


async def run_bot() -> None:
    """
    Основная функция для запуска торгового бота.
    """
    bot = None
    
    try:
        bot = await create_and_run_bot()
        
        # Бесконечный цикл ожидания
        while True:
            await asyncio.sleep(3600)  # Спим 1 час
            
    except KeyboardInterrupt:
        logger.info("Получен сигнал завершения работы")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
    finally:
        if bot:
            await bot.shutdown()


if __name__ == "__main__":
    # Запуск бота при прямом выполнении файла
    asyncio.run(run_bot())