"""
Cache module for BingX Trading Bot.
Handles Redis caching for performance optimization.
"""

import json
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import timedelta

import redis.asyncio as redis
from config.settings import REDIS_URL

logger = logging.getLogger(__name__)

# Global Redis connection
redis_client = None

def init_cache():
    """Initialize Redis cache connection"""
    global redis_client
    
    try:
        if REDIS_URL:
            redis_client = redis.from_url(REDIS_URL, decode_responses=True)
            logger.info("Redis cache initialized")
        else:
            logger.warning("REDIS_URL not set, cache disabled")
            
    except Exception as e:
        logger.error(f"Error initializing Redis cache: {e}")
        redis_client = None

def close_cache():
    """Close Redis cache connection"""
    global redis_client
    
    if redis_client:
        redis_client.close()
        logger.info("Redis cache connection closed")

async def get_cached_data(key: str) -> Optional[Any]:
    """Get data from cache"""
    if not redis_client:
        return None
    
    try:
        data = await redis_client.get(key)
        if data:
            return json.loads(data)
        return None
        
    except Exception as e:
        logger.error(f"Error getting cached data for key {key}: {e}")
        return None

async def set_cached_data(key: str, data: Any, expire: int = 3600) -> bool:
    """Set data in cache with expiration"""
    if not redis_client:
        return False
    
    try:
        await redis_client.setex(key, timedelta(seconds=expire), json.dumps(data))
        return True
        
    except Exception as e:
        logger.error(f"Error setting cached data for key {key}: {e}")
        return False

async def delete_cached_data(key: str) -> bool:
    """Delete data from cache"""
    if not redis_client:
        return False
    
    try:
        await redis_client.delete(key)
        return True
        
    except Exception as e:
        logger.error(f"Error deleting cached data for key {key}: {e}")
        return False

async def cache_historical_data(symbol: str, timeframe: str, data: List[Dict[str, Any]]) -> bool:
    """Cache historical market data"""
    key = f"historical:{symbol}:{timeframe}"
    return await set_cached_data(key, data, expire=300)  # 5 minutes

async def get_cached_historical_data(symbol: str, timeframe: str) -> Optional[List[Dict[str, Any]]]:
    """Get cached historical market data"""
    key = f"historical:{symbol}:{timeframe}"
    return await get_cached_data(key)

async def cache_indicator(symbol: str, timeframe: str, indicator: str, value: Any) -> bool:
    """Cache technical indicator value"""
    key = f"indicator:{symbol}:{timeframe}:{indicator}"
    return await set_cached_data(key, value, expire=180)  # 3 minutes

async def get_cached_indicator(symbol: str, timeframe: str, indicator: str) -> Optional[Any]:
    """Get cached technical indicator value"""
    key = f"indicator:{symbol}:{timeframe}:{indicator}"
    return await get_cached_data(key)

async def cache_signal(signal_data: Dict[str, Any]) -> bool:
    """Cache trading signal"""
    key = f"signal:{signal_data.get('symbol')}:{signal_data.get('timestamp')}"
    return await set_cached_data(key, signal_data, expire=600)  # 10 minutes

async def increment_api_counter() -> int:
    """Increment API call counter for rate limiting"""
    if not redis_client:
        return 0
    
    try:
        return await redis_client.incr('api:calls')
        
    except Exception as e:
        logger.error(f"Error incrementing API counter: {e}")
        return 0

async def reset_api_counter() -> bool:
    """Reset API call counter"""
    if not redis_client:
        return False
    
    try:
        await redis_client.delete('api:calls')
        return True
        
    except Exception as e:
        logger.error(f"Error resetting API counter: {e}")
        return False