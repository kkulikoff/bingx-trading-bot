"""
Модуль безопасности для торгового бота.
Содержит функции для шифрования, хэширования и валидации.
"""
import hashlib
import hmac
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import os
import logging

from .settings import ENCRYPTION_KEY

logger = logging.getLogger(__name__)

def derive_key(password: str, salt: bytes = None) -> tuple:
    """Derive a cryptographic key from a password using PBKDF2"""
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

def encrypt_data(data: str, key: str = None) -> dict:
    """Encrypt data using Fernet symmetric encryption"""
    try:
        if key is None:
            key = ENCRYPTION_KEY
        
        fernet_key, salt = derive_key(key)
        cipher_suite = Fernet(fernet_key)
        encrypted_data = cipher_suite.encrypt(data.encode())
        
        return {
            'data': base64.urlsafe_b64encode(encrypted_data).decode(),
            'salt': base64.urlsafe_b64encode(salt).decode()
        }
    except Exception as e:
        logger.error(f"Error encrypting data: {e}")
        raise

def decrypt_data(encrypted_data: dict, key: str = None) -> str:
    """Decrypt data using Fernet symmetric encryption"""
    try:
        if key is None:
            key = ENCRYPTION_KEY
        
        salt = base64.urlsafe_b64decode(encrypted_data['salt'].encode())
        data = base64.urlsafe_b64decode(encrypted_data['data'].encode())
        
        fernet_key, _ = derive_key(key, salt)
        cipher_suite = Fernet(fernet_key)
        decrypted_data = cipher_suite.decrypt(data)
        
        return decrypted_data.decode()
    except Exception as e:
        logger.error(f"Error decrypting data: {e}")
        raise

def validate_api_keys(api_key: str, secret_key: str) -> bool:
    """Validate BingX API keys format"""
    if not api_key or not secret_key:
        return False
    if len(api_key) < 10 or len(secret_key) < 10:
        return False
    return True

def generate_secure_hash(data: str, secret: str) -> str:
    """Generate a secure HMAC-SHA256 hash of data"""
    try:
        hmac_hash = hmac.new(
            secret.encode(),
            data.encode(),
            hashlib.sha256
        )
        return base64.b64encode(hmac_hash.digest()).decode()
    except Exception as e:
        logger.error(f"Error generating secure hash: {e}")
        raise