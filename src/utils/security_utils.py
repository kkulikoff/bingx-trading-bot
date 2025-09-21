"""
Модуль утилит безопасности для торгового бота.
Содержит функции для шифрования, хеширования и обеспечения безопасности.
"""

import hashlib
import hmac
import base64
import secrets
import string
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import jwt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import os
import logging

from config.settings import ENCRYPTION_KEY

logger = logging.getLogger(__name__)

class SecurityUtils:
    """Класс утилит для обеспечения безопасности приложения"""
    
    def __init__(self):
        self.encryption_key = ENCRYPTION_KEY.encode() if ENCRYPTION_KEY else None
        self.jwt_secret = self._generate_jwt_secret()
    
    def _generate_jwt_secret(self) -> str:
        """Генерация секретного ключа для JWT"""
        # Если ключ уже есть в переменных окружения, используем его
        if os.getenv('JWT_SECRET'):
            return os.getenv('JWT_SECRET')
        
        # Иначе генерируем новый и сохраняем в переменные окружения
        secret = secrets.token_urlsafe(64)
        os.environ['JWT_SECRET'] = secret
        return secret
    
    def derive_key(self, password: str, salt: bytes = None) -> Tuple[bytes, bytes]:
        """
        Derive a cryptographic key from a password using PBKDF2.
        
        Args:
            password: Пароль для derivation
            salt: Соль (если None, генерируется новая)
            
        Returns:
            Tuple: (key, salt) где key - bytes и salt - bytes
        """
        if salt is None:
            salt = os.urandom(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key, salt
    
    def encrypt_data(self, data: str, key: str = None) -> Dict[str, str]:
        """
        Шифрование данных с использованием Fernet.
        
        Args:
            data: Данные для шифрования
            key: Ключ шифрования (если None, используется ключ из настроек)
            
        Returns:
            Dict: Словарь с зашифрованными данными и солью
        """
        try:
            if key is None:
                key = self.encryption_key
            if not key:
                raise ValueError("Encryption key is required")
            
            # Derive key from password
            fernet_key, salt = self.derive_key(key)
            cipher_suite = Fernet(fernet_key)
            
            # Encrypt data
            encrypted_data = cipher_suite.encrypt(data.encode())
            
            return {
                'data': base64.urlsafe_b64encode(encrypted_data).decode(),
                'salt': base64.urlsafe_b64encode(salt).decode()
            }
        except Exception as e:
            logger.error(f"Ошибка шифрования данных: {e}")
            raise
    
    def decrypt_data(self, encrypted_data: Dict[str, str], key: str = None) -> str:
        """
        Дешифрование данных с использованием Fernet.
        
        Args:
            encrypted_data: Словарь с зашифрованными данными и солью
            key: Ключ шифрования (если None, используется ключ из настроек)
            
        Returns:
            str: Расшифрованные данные
        """
        try:
            if key is None:
                key = self.encryption_key
            if not key:
                raise ValueError("Encryption key is required")
            
            # Decode salt and data
            salt = base64.urlsafe_b64decode(encrypted_data['salt'].encode())
            data = base64.urlsafe_b64decode(encrypted_data['data'].encode())
            
            # Derive key from password
            fernet_key, _ = self.derive_key(key, salt)
            cipher_suite = Fernet(fernet_key)
            
            # Decrypt data
            decrypted_data = cipher_suite.decrypt(data)
            return decrypted_data.decode()
        except Exception as e:
            logger.error(f"Ошибка дешифрования данных: {e}")
            raise
    
    def generate_secure_hash(self, data: str, secret: str = None) -> str:
        """
        Генерация безопасного HMAC-SHA256 хеша данных.
        
        Args:
            data: Данные для хеширования
            secret: Секретный ключ (если None, используется ключ из настроек)
            
        Returns:
            str: Base64 encoded HMAC-SHA256 hash
        """
        try:
            if secret is None:
                secret = self.encryption_key.decode() if self.encryption_key else ""
            
            # Create HMAC-SHA256 hash
            hmac_hash = hmac.new(
                secret.encode(),
                data.encode(),
                hashlib.sha256
            )
            
            # Return base64 encoded hash
            return base64.b64encode(hmac_hash.digest()).decode()
        except Exception as e:
            logger.error(f"Ошибка генерации хеша: {e}")
            raise
    
    def generate_api_signature(self, params: Dict[str, str], secret: str) -> str:
        """
        Генерация подписи для API запросов к BingX.
        
        Args:
            params: Параметры запроса
            secret: Секретный ключ API
            
        Returns:
            str: Подпись для API запроса
        """
        try:
            # Сортировка параметров по ключу
            sorted_params = sorted(params.items(), key=lambda x: x[0])
            
            # Создание query string
            query_string = '&'.join([f"{k}={v}" for k, v in sorted_params])
            
            # Генерация подписи
            signature = hmac.new(
                secret.encode('utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            return signature
        except Exception as e:
            logger.error(f"Ошибка генерации подписи API: {e}")
            raise
    
    def generate_jwt_token(self, payload: Dict, expires_in: int = 3600) -> str:
        """
        Генерация JWT токена.
        
        Args:
            payload: Данные для включения в токен
            expires_in: Время жизни токена в секундах
            
        Returns:
            str: JWT токен
        """
        try:
            # Добавление времени expiration
            payload['exp'] = datetime.utcnow() + timedelta(seconds=expires_in)
            payload['iat'] = datetime.utcnow()
            
            # Генерация токена
            token = jwt.encode(payload, self.jwt_secret, algorithm='HS256')
            return token
        except Exception as e:
            logger.error(f"Ошибка генерации JWT токена: {e}")
            raise
    
    def verify_jwt_token(self, token: str) -> Optional[Dict]:
        """
        Верификация JWT токена.
        
        Args:
            token: JWT токен для верификации
            
        Returns:
            Optional[Dict]: Данные из токена или None если токен невалиден
        """
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("JWT токен истек")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Невалидный JWT токен")
            return None
        except Exception as e:
            logger.error(f"Ошибка верификации JWT токена: {e}")
            return None
    
    def generate_secure_password(self, length: int = 16) -> str:
        """
        Генерация безопасного пароля.
        
        Args:
            length: Длина пароля
            
        Returns:
            str: Сгенерированный пароль
        """
        try:
            # Комбинация символов для пароля
            characters = string.ascii_letters + string.digits + string.punctuation
            
            # Генерация пароля
            password = ''.join(secrets.choice(characters) for _ in range(length))
            return password
        except Exception as e:
            logger.error(f"Ошибка генерации пароля: {e}")
            raise
    
    def validate_password_strength(self, password: str) -> Tuple[bool, str]:
        """
        Проверка сложности пароля.
        
        Args:
            password: Пароль для проверки
            
        Returns:
            Tuple: (is_valid, message) где is_valid - булево значение, message - сообщение
        """
        if len(password) < 8:
            return False, "Пароль должен содержать минимум 8 символов"
        
        if not any(char.isdigit() for char in password):
            return False, "Пароль должен содержать хотя бы одну цифру"
        
        if not any(char.isupper() for char in password):
            return False, "Пароль должен содержать хотя бы одну заглавную букву"
        
        if not any(char.islower() for char in password):
            return False, "Пароль должен содержать хотя бы одну строчную букву"
        
        if not any(char in string.punctuation for char in password):
            return False, "Пароль должен содержать хотя бы один специальный символ"
        
        return True, "Пароль соответствует требованиям безопасности"
    
    def sanitize_input(self, input_str: str, max_length: int = 255) -> str:
        """
        Санитизация пользовательского ввода.
        
        Args:
            input_str: Входная строка
            max_length: Максимальная длина строки
            
        Returns:
            str: Санитизированная строка
        """
        if not input_str:
            return ""
        
        # Обрезка до максимальной длины
        if len(input_str) > max_length:
            input_str = input_str[:max_length]
        
        # Удаление потенциально опасных символов
        sanitized = input_str.replace('<', '&lt;').replace('>', '&gt;')
        sanitized = sanitized.replace('"', '&quot;').replace("'", '&#x27;')
        sanitized = sanitized.replace('/', '&#x2F;').replace('\\', '&#x5C;')
        
        return sanitized
    
    def rate_limit_check(self, identifier: str, max_requests: int, time_window: int) -> bool:
        """
        Проверка ограничения частоты запросов.
        
        Args:
            identifier: Идентификатор (IP, user_id, etc.)
            max_requests: Максимальное количество запросов
            time_window: Временное окно в секундах
            
        Returns:
            bool: True если запрос разрешен, False если превышен лимит
        """
        # В реальной реализации здесь будет использоваться Redis или другая быстрая БД
        # Для простоты используем in-memory кэш (в production заменить на Redis)
        if not hasattr(self, '_rate_limit_cache'):
            self._rate_limit_cache = {}
        
        current_time = datetime.now().timestamp()
        key = f"rate_limit:{identifier}"
        
        if key not in self._rate_limit_cache:
            self._rate_limit_cache[key] = {
                'count': 1,
                'window_start': current_time
            }
            return True
        
        cache_data = self._rate_limit_cache[key]
        
        # Проверка, истекло ли временное окно
        if current_time - cache_data['window_start'] > time_window:
            cache_data['count'] = 1
            cache_data['window_start'] = current_time
            return True
        
        # Проверка количества запросов
        if cache_data['count'] >= max_requests:
            return False
        
        cache_data['count'] += 1
        return True
    
    def generate_csrf_token(self) -> str:
        """
        Генерация CSRF токена.
        
        Returns:
            str: CSRF токен
        """
        return secrets.token_urlsafe(32)
    
    def validate_csrf_token(self, token: str, expected_token: str) -> bool:
        """
        Валидация CSRF токена.
        
        Args:
            token: Токен для проверки
            expected_token: Ожидаемый токен
            
        Returns:
            bool: True если токен валиден
        """
        return hmac.compare_digest(token, expected_token)


# Создание глобального экземпляра для использования во всем приложении
security_utils = SecurityUtils()

# Функции для обратной совместимости
def encrypt_data(data: str, key: str = None) -> Dict[str, str]:
    """Шифрование данных (обертка для обратной совместимости)"""
    return security_utils.encrypt_data(data, key)

def decrypt_data(encrypted_data: Dict[str, str], key: str = None) -> str:
    """Дешифрование данных (обертка для обратной совместимости)"""
    return security_utils.decrypt_data(encrypted_data, key)

def generate_secure_hash(data: str, secret: str = None) -> str:
    """Генерация безопасного хеша (обертка для обратной совместимости)"""
    return security_utils.generate_secure_hash(data, secret)

def generate_api_signature(params: Dict[str, str], secret: str) -> str:
    """Генерация подписи API (обертка для обратной совместимости)"""
    return security_utils.generate_api_signature(params, secret)