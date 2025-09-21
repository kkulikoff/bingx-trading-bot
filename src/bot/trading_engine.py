"""
Модуль исполнения торговых операций для BingX Trading Bot.

Обеспечивает взаимодействие с API биржи для исполнения сделок,
управление ордерами и отслеживание позиций.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple
import logging
from decimal import Decimal, ROUND_DOWN
import json

from src.bot.api_client import BingXAPI
from src.bot.risk_manager import RiskManager
from src.utils.logger import setup_logger
from src.utils.helpers import safe_round, generate_hash, retry_on_exception

logger = setup_logger(__name__)

class TradingEngine:
    """Класс для исполнения торговых операций и управления ордерами."""
    
    def __init__(self, api_client: BingXAPI, risk_manager: RiskManager):
        """
        Инициализация торгового движка.
        
        Args:
            api_client: Клиент API для взаимодействия с биржей
            risk_manager: Менеджер рисков для расчета позиций
        """
        self.api_client = api_client
        self.risk_manager = risk_manager
        self.open_orders = {}
        self.order_history = []
        self.positions = {}
        self.last_order_check = 0
        self.order_check_interval = 30  # секунды
        
    async def initialize(self):
        """Инициализация торгового движка."""
        logger.info("Инициализация торгового движка")
        
        # Загрузка открытых ордеров и позиций
        await self._load_open_orders()
        await self._load_positions()
        
        logger.info("Торговый двибок инициализирован")
    
    @retry_on_exception(max_retries=3, delay=1.0)
    async def execute_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Исполнение торгового сигнала.
        
        Args:
            signal: Торговый сигнал с деталями сделки
            
        Returns:
            Dict: Результат исполнения сигнала
        """
        try:
            symbol = signal.get('symbol')
            direction = signal.get('direction')
            price = signal.get('price')
            stop_loss = signal.get('stop_loss')
            take_profit = signal.get('take_profit')
            
            if not all([symbol, direction, price]):
                logger.error(f"Неполный сигнал: {signal}")
                return {
                    'success': False,
                    'error': 'Неполный сигнал',
                    'signal': signal
                }
            
            # Расчет размера позиции
            position_size = self.risk_manager.calculate_position_size(
                price, stop_loss, symbol
            )
            
            if position_size <= 0:
                logger.warning(f"Нулевой размер позиции для {symbol}")
                return {
                    'success': False,
                    'error': 'Нулевой размер позиции',
                    'signal': signal
                }
            
            # Проверка возможности совершения сделки
            if not self._validate_trade(symbol, position_size, price, direction):
                return {
                    'success': False,
                    'error': 'Не прошло валидацию',
                    'signal': signal
                }
            
            # Определение типа ордера и параметров
            order_type = "MARKET"  # Можно расширить для лимитных ордеров
            side = "BUY" if direction.upper() == "LONG" else "SELL"
            
            # Размещение ордера
            order_result = await self._place_order(
                symbol, side, order_type, position_size, price
            )
            
            if order_result and order_result.get('success', False):
                order_id = order_result.get('order_id')
                
                # Сохранение информации об ордере
                order_info = {
                    'order_id': order_id,
                    'symbol': symbol,
                    'direction': direction,
                    'side': side,
                    'order_type': order_type,
                    'price': price,
                    'quantity': position_size,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'timestamp': time.time(),
                    'signal': signal,
                    'status': 'OPEN'
                }
                
                self.open_orders[order_id] = order_info
                self.order_history.append(order_info)
                
                logger.info(f"Ордер размещен: {symbol} {side} {position_size} по цене {price}")
                
                # Добавление в историю risk manager
                trade_data = {
                    'symbol': symbol,
                    'direction': direction,
                    'entry_price': price,
                    'quantity': position_size,
                    'timestamp': time.time(),
                    'order_id': order_id,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit
                }
                
                self.risk_manager.add_trade(trade_data)
                
                return {
                    'success': True,
                    'order_id': order_id,
                    'symbol': symbol,
                    'direction': direction,
                    'quantity': position_size,
                    'price': price,
                    'order_info': order_info
                }
            else:
                error_msg = order_result.get('error', 'Неизвестная ошибка') if order_result else 'Ошибка API'
                logger.error(f"Ошибка размещения ордера для {symbol} {direction}: {error_msg}")
                
                return {
                    'success': False,
                    'error': error_msg,
                    'signal': signal
                }
                
        except Exception as e:
            logger.error(f"Критическая ошибка исполнения сигнала: {e}")
            return {
                'success': False,
                'error': str(e),
                'signal': signal
            }
    
    @retry_on_exception(max_retries=3, delay=1.0)
    async def _place_order(self, symbol: str, side: str, order_type: str, 
                          quantity: float, price: float = None) -> Dict[str, Any]:
        """
        Размещение ордера на бирже.
        
        Args:
            symbol: Торговая пара
            side: Направление (BUY/SELL)
            order_type: Тип ордера (MARKET/LIMIT)
            quantity: Количество
            price: Цена (для лимитных ордеров)
            
        Returns:
            Dict: Результат размещения ордера
        """
        try:
            # Конвертация в строку с правильным форматированием
            quantity_str = self._format_quantity(symbol, quantity)
            
            if order_type.upper() == "MARKET":
                result = await self.api_client.place_order(
                    symbol, side, quantity_str, None, order_type
                )
            else:
                result = await self.api_client.place_order(
                    symbol, side, quantity_str, price, order_type
                )
            
            if result and 'orderId' in result:
                return {
                    'success': True,
                    'order_id': result['orderId'],
                    'result': result
                }
            else:
                error_msg = result.get('msg', 'Неизвестная ошибка') if result else 'Пустой ответ от API'
                return {
                    'success': False,
                    'error': error_msg,
                    'result': result
                }
                
        except Exception as e:
            logger.error(f"Ошибка размещения ордера: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _validate_trade(self, symbol: str, quantity: float, price: float, direction: str) -> bool:
        """
        Проверка возможности совершения сделки.
        
        Args:
            symbol: Торговая пара
            quantity: Количество
            price: Цена
            direction: Направление (LONG/SHORT)
            
        Returns:
            bool: True если сделка возможна, False иначе
        """
        # Проверка достаточности средств
        trade_cost = quantity * price
        
        if trade_cost > self.risk_manager.balance * self.risk_manager.max_position_size:
            logger.warning(f"Недостаточно средств для сделки {symbol}. "
                          f"Нужно: {trade_cost:.2f}, "
                          f"доступно: {self.risk_manager.balance * self.risk_manager.max_position_size:.2f}")
            return False
        
        # Проверка на существующую позицию (можно расширить логику)
        if symbol in self.positions:
            logger.warning(f"Позиция по {symbol} уже открыта")
            # Здесь можно добавить логику добавления к позиции
            return False
        
        # Проверка минимального размера ордера
        min_order_size = self._get_min_order_size(symbol)
        if quantity * price < min_order_size:
            logger.warning(f"Размер ордера {symbol} меньше минимального: "
                          f"{quantity * price:.2f} < {min_order_size:.2f}")
            return False
        
        return True
    
    def _format_quantity(self, symbol: str, quantity: float) -> str:
        """
        Форматирование количества в соответствии с требованиями биржи.
        
        Args:
            symbol: Торговая пара
            quantity: Количество
            
        Returns:
            str: Отформатированное количество
        """
        # Правила округления для разных символов
        rounding_rules = {
            'BTC': 6,
            'ETH': 5,
            'SOL': 2,
            'XRP': 0,
            'default': 4
        }
        
        # Определение точности округления
        precision = rounding_rules.get('default')
        for asset, prec in rounding_rules.items():
            if asset in symbol:
                precision = prec
                break
        
        # Округление и преобразование в строку
        formatted = safe_round(quantity, precision)
        return str(formatted)
    
    def _get_min_order_size(self, symbol: str) -> float:
        """
        Получение минимального размера ордера для символа.
        
        Args:
            symbol: Торговая пара
            
        Returns:
            float: Минимальный размер ордера в USDT
        """
        # Минимальные размеры ордеров для разных символов
        min_sizes = {
            'BTC-USDT': 10.0,   # 10 USDT
            'ETH-USDT': 10.0,   # 10 USDT
            'SOL-USDT': 5.0,    # 5 USDT
            'XRP-USDT': 5.0,    # 5 USDT
            'default': 10.0     # 10 USDT по умолчанию
        }
        
        return min_sizes.get(symbol, min_sizes['default'])
    
    async def cancel_order(self, order_id: str, symbol: str = None) -> bool:
        """
        Отмена ордера.
        
        Args:
            order_id: ID ордера
            symbol: Торговая пара (опционально)
            
        Returns:
            bool: True если ордер отменен успешно, False иначе
        """
        try:
            # Если symbol не указан, пытаемся найти его в открытых ордерах
            if not symbol and order_id in self.open_orders:
                symbol = self.open_orders[order_id]['symbol']
            
            if not symbol:
                logger.error(f"Не удалось определить символ для отмены ордера {order_id}")
                return False
            
            # Отмена ордера через API
            result = await self.api_client.cancel_order(symbol, order_id)
            
            if result and result.get('code', -1) == 0:
                # Обновление статуса ордера
                if order_id in self.open_orders:
                    self.open_orders[order_id]['status'] = 'CANCELLED'
                    self.open_orders[order_id]['cancel_time'] = time.time()
                
                logger.info(f"Ордер {order_id} отменен успешно")
                return True
            else:
                error_msg = result.get('msg', 'Неизвестная ошибка') if result else 'Пустой ответ от API'
                logger.error(f"Ошибка отмены ордера {order_id}: {error_msg}")
                return False
                
        except Exception as e:
            logger.error(f"Критическая ошибка отмены ордера {order_id}: {e}")
            return False
    
    async def get_order_status(self, order_id: str, symbol: str = None) -> Optional[Dict[str, Any]]:
        """
        Получение статуса ордера.
        
        Args:
            order_id: ID ордера
            symbol: Торговая пара (опционально)
            
        Returns:
            Optional[Dict]: Информация о статусе ордера или None
        """
        try:
            # Если symbol не указан, пытаемся найти его в открытых ордерах
            if not symbol and order_id in self.open_orders:
                symbol = self.open_orders[order_id]['symbol']
            
            if not symbol:
                logger.error(f"Не удалось определить символ для проверки ордера {order_id}")
                return None
            
            # Получение статуса ордера через API
            result = await self.api_client.get_order_status(symbol, order_id)
            
            if result and 'data' in result:
                return result['data']
            else:
                logger.warning(f"Не удалось получить статус ордера {order_id}")
                return None
                
        except Exception as e:
            logger.error(f"Ошибка получения статуса ордера {order_id}: {e}")
            return None
    
    async def check_orders_status(self):
        """
        Проверка статусов всех открытых ордеров.
        """
        current_time = time.time()
        
        # Проверяем ордера не чаще чем раз в order_check_interval секунд
        if current_time - self.last_order_check < self.order_check_interval:
            return
        
        self.last_order_check = current_time
        
        logger.info("Проверка статусов открытых ордеров")
        
        # Создаем копию списка ордеров для итерации
        order_ids = list(self.open_orders.keys())
        
        for order_id in order_ids:
            order_info = self.open_orders[order_id]
            
            # Пропускаем уже закрытые или отмененные ордера
            if order_info.get('status') in ['FILLED', 'CANCELLED', 'REJECTED']:
                continue
            
            # Получаем текущий статус ордера
            status_data = await self.get_order_status(order_id, order_info['symbol'])
            
            if status_data:
                new_status = status_data.get('status')
                executed_qty = float(status_data.get('executedQty', 0))
                
                # Обновляем информацию об ордере
                order_info['status'] = new_status
                order_info['executed_quantity'] = executed_qty
                
                # Если ордер исполнен полностью
                if new_status == 'FILLED' and executed_qty >= float(order_info['quantity']) * 0.99:
                    order_info['fill_time'] = time.time()
                    logger.info(f"Ордер {order_id} исполнен полностью")
                    
                    # Создаем позицию
                    await self._create_position_from_order(order_info)
                
                # Если ордер исполнен частично
                elif executed_qty > 0:
                    logger.info(f"Ордер {order_id} исполнен частично: {executed_qty}/{order_info['quantity']}")
    
    async def _create_position_from_order(self, order_info: Dict[str, Any]):
        """
        Создание позиции на основе исполненного ордера.
        
        Args:
            order_info: Информация об ордере
        """
        position_id = generate_hash(f"{order_info['order_id']}_{time.time()}")
        
        position = {
            'position_id': position_id,
            'symbol': order_info['symbol'],
            'direction': order_info['direction'],
            'entry_price': order_info['price'],
            'quantity': order_info['quantity'],
            'stop_loss': order_info.get('stop_loss'),
            'take_profit': order_info.get('take_profit'),
            'entry_time': time.time(),
            'order_id': order_info['order_id']
        }
        
        self.positions[position_id] = position
        logger.info(f"Создана новая позиция {position_id} для {order_info['symbol']}")
    
    async def close_position(self, position_id: str, reason: str = "manual") -> bool:
        """
        Закрытие позиции.
        
        Args:
            position_id: ID позиции
            reason: Причина закрытия
            
        Returns:
            bool: True если позиция закрыта успешно, False иначе
        """
        try:
            if position_id not in self.positions:
                logger.error(f"Позиция {position_id} не найдена")
                return False
            
            position = self.positions[position_id]
            symbol = position['symbol']
            
            # Определяем направление для закрытия
            side = "SELL" if position['direction'].upper() == "LONG" else "BUY"
            
            # Размещаем ордер на закрытие
            order_result = await self._place_order(
                symbol, side, "MARKET", position['quantity']
            )
            
            if order_result and order_result.get('success', False):
                # Обновляем информацию о позиции
                position['exit_time'] = time.time()
                position['exit_price'] = await self._get_current_price(symbol)
                position['close_reason'] = reason
                position['status'] = 'CLOSED'
                
                # Расчет PnL
                entry_price = position['entry_price']
                exit_price = position['exit_price']
                quantity = position['quantity']
                
                if position['direction'].upper() == "LONG":
                    pnl = (exit_price - entry_price) * quantity
                else:  # SHORT
                    pnl = (entry_price - exit_price) * quantity
                
                position['pnl'] = pnl
                position['pnl_percent'] = (pnl / (entry_price * quantity)) * 100
                
                # Обновляем баланс
                self.risk_manager.update_balance(pnl)
                
                logger.info(f"Позиция {position_id} закрыта. PnL: {pnl:.2f} USDT ({position['pnl_percent']:.2f}%)")
                
                return True
            else:
                error_msg = order_result.get('error', 'Неизвестная ошибка') if order_result else 'Ошибка API'
                logger.error(f"Ошибка закрытия позиции {position_id}: {error_msg}")
                return False
                
        except Exception as e:
            logger.error(f"Критическая ошибка закрытия позиции {position_id}: {e}")
            return False
    
    async def _get_current_price(self, symbol: str) -> float:
        """
        Получение текущей цены для символа.
        
        Args:
            symbol: Торговая пара
            
        Returns:
            float: Текущая цена
        """
        try:
            # Здесь можно реализовать получение цены из кэша или API
            # Временно возвращаем 0, нужно реализовать properly
            return 0.0
        except Exception as e:
            logger.error(f"Ошибка получения текущей цены для {symbol}: {e}")
            return 0.0
    
    async def _load_open_orders(self):
        """Загрузка открытых ордеров с биржи."""
        try:
            # Получение открытых ордеров через API
            orders_result = await self.api_client.get_open_orders()
            
            if orders_result and 'data' in orders_result:
                for order_data in orders_result['data']:
                    order_id = order_data.get('orderId')
                    
                    if order_id:
                        self.open_orders[order_id] = {
                            'order_id': order_id,
                            'symbol': order_data.get('symbol'),
                            'side': order_data.get('side'),
                            'order_type': order_data.get('type'),
                            'price': float(order_data.get('price', 0)),
                            'quantity': float(order_data.get('origQty', 0)),
                            'executed_quantity': float(order_data.get('executedQty', 0)),
                            'status': order_data.get('status'),
                            'timestamp': order_data.get('time'),
                            'api_data': order_data
                        }
                
                logger.info(f"Загружено {len(self.open_orders)} открытых ордеров")
            else:
                logger.warning("Не удалось загрузить открытые ордера")
                
        except Exception as e:
            logger.error(f"Ошибка загрузки открытых ордеров: {e}")
    
    async def _load_positions(self):
        """Загрузка открытых позиций с биржи."""
        try:
            # Для BingX可能需要 специальный API для получения позиций
            # Временная заглушка - позиции будут создаваться при исполнении ордеров
            logger.info("Загрузка позиций не реализована для BingX")
                
        except Exception as e:
            logger.error(f"Ошибка загрузки позиций: {e}")
    
    async def shutdown(self):
        """Корректное завершение работы торгового движка."""
        logger.info("Завершение работы торгового движка")
        
        # Отмена всех открытых ордеров
        for order_id in list(self.open_orders.keys()):
            await self.cancel_order(order_id)
        
        # Закрытие всех позиций (опционально, зависит от стратегии)
        # for position_id in list(self.positions.keys()):
        #     await self.close_position(position_id, "shutdown")
        
        logger.info("Торговый двибок остановлен")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Получение статуса торгового движка.
        
        Returns:
            Dict: Статус движка
        """
        return {
            'open_orders': len(self.open_orders),
            'order_history': len(self.order_history),
            'positions': len(self.positions),
            'last_check': self.last_order_check
        }