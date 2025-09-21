"""
Модуль для работы с API BingX.
Обеспечивает безопасное взаимодействие с биржевым API.
"""
import aiohttp
import asyncio
import time
import hmac
import hashlib
import urllib.parse
from typing import Dict, List, Optional, Any
import logging

from src.config.settings import BINGX_API_KEY, BINGX_SECRET_KEY, REQUEST_TIMEOUT, API_RATE_LIMIT
from src.utils.logger import setup_logging

logger = setup_logging(__name__)

class BingXAPI:
    """Класс для работы с API BingX с улучшенной безопасностью"""
    
    def __init__(self):
        self.api_key = BINGX_API_KEY
        self.secret_key = BINGX_SECRET_KEY
        self.session = None
        self.last_request_time = 0
        self.request_count = 0
        self.reset_time = time.time()
        self.base_url = "https://open-api.bingx.com"
        self.api_path = "/openApi"

    async def initialize(self):
        """Инициализация API клиента"""
        try:
            self.session = aiohttp.ClientSession()
            logger.info("BingX API клиент инициализирован")
        except Exception as e:
            logger.error(f"Ошибка инициализации API клиента: {e}")
            raise

    async def _rate_limit(self):
        """Ограничение частоты запросов для соблюдения лимитов API"""
        current_time = time.time()
        elapsed = current_time - self.reset_time
        
        # Сброс счетчика каждую минуту
        if elapsed >= 60:
            self.request_count = 0
            self.reset_time = current_time
        
        # Проверка лимита запросов
        if self.request_count >= API_RATE_LIMIT:
            sleep_time = 60 - elapsed
            if sleep_time > 0:
                logger.warning(f"Достигнут лимит запросов. Ожидание {sleep_time:.2f} сек.")
                await asyncio.sleep(sleep_time)
            self.request_count = 0
            self.reset_time = time.time()
        
        # Задержка между запросами
        since_last_request = current_time - self.last_request_time
        if since_last_request < 0.1:  # 100ms между запросами
            await asyncio.sleep(0.1 - since_last_request)
        
        self.last_request_time = time.time()
        self.request_count += 1

    def _generate_signature(self, params: Dict[str, Any]) -> str:
        """Генерация подписи для запросов к API"""
        query_string = urllib.parse.urlencode(params)
        signature = hmac.new(
            self.secret_key.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature

    async def _make_request(self, method: str, endpoint: str, params: Dict[str, Any] = None, signed: bool = False) -> Optional[Dict[str, Any]]:
        """Выполнение запроса к API BingX"""
        await self._rate_limit()
        
        try:
            url = f"{self.base_url}{self.api_path}{endpoint}"
            
            if signed:
                if params is None:
                    params = {}
                params['timestamp'] = int(time.time() * 1000)
                params['signature'] = self._generate_signature(params)
            
            headers = {
                'X-BX-APIKEY': self.api_key,
                'User-Agent': 'BingX-Trading-Bot/1.0'
            }
            
            timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
            logger.debug(f"API Request: {method} {url}, params: {params}")
            
            if method == 'GET':
                async with self.session.get(url, params=params, headers=headers, timeout=timeout) as response:
                    return await self._process_response(response, url, params)
            elif method == 'POST':
                async with self.session.post(url, params=params, headers=headers, timeout=timeout) as response:
                    return await self._process_response(response, url, params)
            elif method == 'DELETE':
                async with self.session.delete(url, params=params, headers=headers, timeout=timeout) as response:
                    return await self._process_response(response, url, params)
                    
        except asyncio.TimeoutError:
            logger.error(f"Таймаут запроса к {url}")
            return None
        except Exception as e:
            logger.error(f"Ошибка запроса к {url}: {e}")
            return None

    async def _process_response(self, response, url: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Обработка ответа от API"""
        try:
            data = await response.json()
            
            # Логирование запроса (без секретных данных)
            safe_params = {k: v for k, v in params.items() if k not in ['signature', 'apiKey', 'secretKey']}
            logger.info(f"API Request: {url}, Params: {safe_params}, Status: {response.status}")
            
            # Проверка статуса ответа
            if response.status != 200:
                logger.error(f"Ошибка API: статус {response.status}, ответ: {data}")
                return None
            
            # Проверка структуры ответа
            if not isinstance(data, dict):
                logger.error(f"Некорректный формат ответа: {data}")
                return None
            
            # Проверка кода ошибки в ответе
            if 'code' in data and data['code'] != 0:
                logger.error(f"Ошибка API: код {data['code']}, сообщение: {data.get('msg', 'Нет сообщения')}")
                return None
            
            return data
            
        except Exception as e:
            logger.error(f"Ошибка обработки ответа: {e}")
            return None

    async def get_account_balance(self) -> Optional[Dict[str, Any]]:
        """Получение баланса аккаунта"""
        endpoint = "/spot/v1/account/balance"
        params = {}
        return await self._make_request('GET', endpoint, params, signed=True)

    async def get_klines(self, symbol: str, interval: str, limit: int = 500) -> Optional[Dict[str, Any]]:
        """Получение исторических данных (свечи)"""
        endpoint = "/spot/v1/market/kline"
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        return await self._make_request('GET', endpoint, params)

    async def place_order(self, symbol: str, side: str, quantity: float, price: Optional[float] = None, order_type: str = "MARKET") -> Optional[Dict[str, Any]]:
        """Размещение ордера"""
        endpoint = "/spot/v1/trade/order"
        params = {
            'symbol': symbol,
            'side': side,
            'type': order_type,
            'quantity': quantity
        }
        if price is not None:
            params['price'] = price
        
        return await self._make_request('POST', endpoint, params, signed=True)

    async def get_open_orders(self, symbol: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Получение списка открытых ордеров"""
        endpoint = "/spot/v1/trade/openOrders"
        params = {}
        if symbol:
            params['symbol'] = symbol
        
        return await self._make_request('GET', endpoint, params, signed=True)

    async def cancel_order(self, symbol: str, order_id: str) -> Optional[Dict[str, Any]]:
        """Отмена ордера"""
        endpoint = "/spot/v1/trade/cancel"
        params = {
            'symbol': symbol,
            'orderId': order_id
        }
        return await self._make_request('DELETE', endpoint, params, signed=True)

    async def get_order_status(self, symbol: str, order_id: str) -> Optional[Dict[str, Any]]:
        """Проверка статуса ордера"""
        endpoint = "/spot/v1/trade/query"
        params = {
            'symbol': symbol,
            'orderId': order_id
        }
        return await self._make_request('GET', endpoint, params, signed=True)

    async def get_historical_data(self, symbol: str, interval: str, limit: int = 500) -> Optional[List[Dict[str, Any]]]:
        """Получение исторических данных в формате для pandas"""
        data = await self.get_klines(symbol, interval, limit)
        if data and 'data' in data:
            return data['data']
        return None

    async def close(self):
        """Закрытие сессии"""
        if self.session:
            await self.session.close()
            logger.info("BingX API сессия закрыта")