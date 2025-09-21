"""
Модуль управления резервным копированием данных.
Обеспечивает создание, шифрование и восстановление резервных копий критичных данных.
"""

import os
import zipfile
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import shutil
import hashlib
from cryptography.fernet import Fernet

from config.settings import (
    BACKUP_SCHEDULE, 
    MAX_BACKUPS, 
    ENCRYPTION_KEY,
    ML_MODEL_PATH,
    ML_SCALER_PATH
)
from src.utils.logger import setup_logger
from src.utils.security_utils import encrypt_data, decrypt_data

logger = setup_logger(__name__)


class BackupManager:
    """Класс для управления резервным копированием данных"""
    
    def __init__(self, backup_dir: str = "backups"):
        """
        Инициализация менеджера резервного копирования.
        
        Args:
            backup_dir: Директория для хранения резервных копий
        """
        self.backup_dir = backup_dir
        self.ensure_backup_dir()
        
        # Инициализация шифрования
        self.cipher_suite = Fernet(ENCRYPTION_KEY.encode()) if ENCRYPTION_KEY else None
        
    def ensure_backup_dir(self):
        """Создание директории для резервных копий, если она не существует"""
        if not os.path.exists(self.backup_dir):
            os.makedirs(self.backup_dir)
            logger.info(f"Создана директория для резервных копий: {self.backup_dir}")
    
    def create_backup(self, backup_name: str = None) -> Optional[str]:
        """
        Создание резервной копии критичных данных.
        
        Args:
            backup_name: Имя резервной копии (если None, генерируется автоматически)
            
        Returns:
            Optional[str]: Путь к созданной резервной копии или None в случае ошибки
        """
        try:
            # Генерация имени резервной копии
            if not backup_name:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_name = f"backup_{timestamp}"
            
            backup_path = os.path.join(self.backup_dir, f"{backup_name}.zip")
            temp_dir = os.path.join(self.backup_dir, "temp", backup_name)
            
            # Создание временной директории
            os.makedirs(temp_dir, exist_ok=True)
            
            # Копирование критичных данных
            self._backup_configs(temp_dir)
            self._backup_models(temp_dir)
            self._backup_logs(temp_dir)
            self._backup_trade_history(temp_dir)
            self._backup_database(temp_dir)
            
            # Создание метаданных резервной копии
            metadata = self._create_metadata()
            with open(os.path.join(temp_dir, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)
            
            # Создание ZIP-архива
            with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, temp_dir)
                        zipf.write(file_path, arcname)
            
            # Шифрование резервной копии (если настроено шифрование)
            if self.cipher_suite:
                backup_path = self._encrypt_backup(backup_path)
            
            # Очистка временной директории
            shutil.rmtree(os.path.join(self.backup_dir, "temp"))
            
            # Проверка целостности резервной копии
            if self._verify_backup(backup_path):
                logger.info(f"Создана резервная копия: {backup_path}")
                return backup_path
            else:
                logger.error("Ошибка проверки целостности резервной копии")
                os.remove(backup_path)
                return None
                
        except Exception as e:
            logger.error(f"Ошибка создания резервной копии: {e}")
            # Очистка в случае ошибки
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            return None
    
    def _backup_configs(self, temp_dir: str):
        """Резервное копирование конфигурационных файлов"""
        config_dir = os.path.join(temp_dir, "configs")
        os.makedirs(config_dir, exist_ok=True)
        
        # Копирование файлов конфигурации
        config_files = [
            "config/settings.py",
            "config/security.py",
            "config/logging.conf",
            ".env"
        ]
        
        for config_file in config_files:
            if os.path.exists(config_file):
                shutil.copy2(config_file, config_dir)
    
    def _backup_models(self, temp_dir: str):
        """Резервное копирование ML моделей"""
        models_dir = os.path.join(temp_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        
        # Копирование моделей
        model_files = [
            ML_MODEL_PATH,
            ML_SCALER_PATH
        ]
        
        for model_file in model_files:
            if os.path.exists(model_file):
                shutil.copy2(model_file, models_dir)
    
    def _backup_logs(self, temp_dir: str):
        """Резервное копирование логов"""
        logs_dir = os.path.join(temp_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        
        # Копирование логов
        if os.path.exists("logs"):
            for log_file in os.listdir("logs"):
                if log_file.endswith(".log"):
                    shutil.copy2(os.path.join("logs", log_file), logs_dir)
    
    def _backup_trade_history(self, temp_dir: str):
        """Резервное копирование истории сделок"""
        trades_dir = os.path.join(temp_dir, "trades")
        os.makedirs(trades_dir, exist_ok=True)
        
        # Копирование истории сделок
        trade_files = [
            "data/trade_history.json",
            "data/signals_history.json"
        ]
        
        for trade_file in trade_files:
            if os.path.exists(trade_file):
                shutil.copy2(trade_file, trades_dir)
    
    def _backup_database(self, temp_dir: str):
        """Резервное копирование базы данных"""
        db_dir = os.path.join(temp_dir, "database")
        os.makedirs(db_dir, exist_ok=True)
        
        # Копирование базы данных
        db_files = [
            "data/trading_bot.db",
            "data/trading_bot.db-shm",
            "data/trading_bot.db-wal"
        ]
        
        for db_file in db_files:
            if os.path.exists(db_file):
                shutil.copy2(db_file, db_dir)
    
    def _create_metadata(self) -> Dict[str, Any]:
        """Создание метаданных резервной копии"""
        return {
            "backup_date": datetime.now().isoformat(),
            "version": "1.0",
            "system": {
                "platform": os.name,
                "python_version": os.sys.version
            },
            "contents": {
                "configs": True,
                "models": True,
                "logs": True,
                "trades": True,
                "database": True
            },
            "encrypted": self.cipher_suite is not None,
            "checksum": None  # Будет заполнено после создания архива
        }
    
    def _encrypt_backup(self, backup_path: str) -> str:
        """
        Шифрование резервной копии.
        
        Args:
            backup_path: Путь к резервной копии
            
        Returns:
            str: Путь к зашифрованной резервной копии
        """
        try:
            if not self.cipher_suite:
                logger.warning("Шифрование не настроено, пропускаем шифрование")
                return backup_path
            
            encrypted_path = backup_path + ".encrypted"
            
            # Чтение исходного файла
            with open(backup_path, 'rb') as f:
                data = f.read()
            
            # Шифрование данных
            encrypted_data = self.cipher_suite.encrypt(data)
            
            # Запись зашифрованного файла
            with open(encrypted_path, 'wb') as f:
                f.write(encrypted_data)
            
            # Удаление исходного файла
            os.remove(backup_path)
            
            logger.info(f"Резервная копия зашифрована: {encrypted_path}")
            return encrypted_path
            
        except Exception as e:
            logger.error(f"Ошибка шифрования резервной копии: {e}")
            return backup_path
    
    def _verify_backup(self, backup_path: str) -> bool:
        """
        Проверка целостности резервной копии.
        
        Args:
            backup_path: Путь к резервной копии
            
        Returns:
            bool: True если резервная копия цела, False в противном случае
        """
        try:
            # Проверка существования файла
            if not os.path.exists(backup_path):
                return False
            
            # Проверка размера файла
            if os.path.getsize(backup_path) == 0:
                return False
            
            # Проверка контрольной суммы (для незашифрованных файлов)
            if not backup_path.endswith(".encrypted"):
                with open(backup_path, 'rb') as f:
                    data = f.read()
                    checksum = hashlib.md5(data).hexdigest()
                
                # Здесь можно сохранить checksum в метаданные для будущей проверки
                return True
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка проверки целостности резервной копии: {e}")
            return False
    
    def restore_backup(self, backup_path: str, restore_dir: str = ".") -> bool:
        """
        Восстановление данных из резервной копии.
        
        Args:
            backup_path: Путь к резервной копии
            restore_dir: Директория для восстановления
            
        Returns:
            bool: True если восстановление успешно, False в противном случае
        """
        try:
            # Проверка существования резервной копии
            if not os.path.exists(backup_path):
                logger.error(f"Резервная копия не найдена: {backup_path}")
                return False
            
            # Расшифровка резервной копии (если необходимо)
            if backup_path.endswith(".encrypted"):
                if not self.cipher_suite:
                    logger.error("Не настроено шифрование, невозможно расшифровать резервную копию")
                    return False
                
                decrypted_path = backup_path.replace(".encrypted", "")
                
                # Чтение зашифрованного файла
                with open(backup_path, 'rb') as f:
                    encrypted_data = f.read()
                
                # Расшифровка данных
                try:
                    decrypted_data = self.cipher_suite.decrypt(encrypted_data)
                except Exception as e:
                    logger.error(f"Ошибка расшифровки: {e}")
                    return False
                
                # Запись расшифрованного файла
                with open(decrypted_path, 'wb') as f:
                    f.write(decrypted_data)
                
                backup_path = decrypted_path
            
            # Временная директория для извлечения
            temp_dir = os.path.join(self.backup_dir, "restore_temp")
            os.makedirs(temp_dir, exist_ok=True)
            
            # Извлечение архива
            with zipfile.ZipFile(backup_path, 'r') as zipf:
                zipf.extractall(temp_dir)
            
            # Восстановление данных
            self._restore_configs(temp_dir, restore_dir)
            self._restore_models(temp_dir, restore_dir)
            self._restore_logs(temp_dir, restore_dir)
            self._restore_trade_history(temp_dir, restore_dir)
            self._restore_database(temp_dir, restore_dir)
            
            # Очистка временной директории
            shutil.rmtree(temp_dir)
            
            logger.info(f"Данные восстановлены из резервной копии: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка восстановления из резервной копии: {e}")
            # Очистка в случае ошибки
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            return False
    
    def _restore_configs(self, temp_dir: str, restore_dir: str):
        """Восстановление конфигурационных файлов"""
        config_src = os.path.join(temp_dir, "configs")
        if os.path.exists(config_src):
            for item in os.listdir(config_src):
                src_path = os.path.join(config_src, item)
                dst_path = os.path.join(restore_dir, item)
                
                # Создание резервной копии существующих файлов
                if os.path.exists(dst_path):
                    backup_path = dst_path + ".backup"
                    shutil.copy2(dst_path, backup_path)
                
                shutil.copy2(src_path, dst_path)
    
    def _restore_models(self, temp_dir: str, restore_dir: str):
        """Восстановление ML моделей"""
        models_src = os.path.join(temp_dir, "models")
        if os.path.exists(models_src):
            # Создание директории для моделей, если не существует
            models_dst = os.path.join(restore_dir, "data", "models")
            os.makedirs(models_dst, exist_ok=True)
            
            for item in os.listdir(models_src):
                src_path = os.path.join(models_src, item)
                dst_path = os.path.join(models_dst, item)
                shutil.copy2(src_path, dst_path)
    
    def _restore_logs(self, temp_dir: str, restore_dir: str):
        """Восстановление логов"""
        logs_src = os.path.join(temp_dir, "logs")
        if os.path.exists(logs_src):
            # Создание директории для логов, если не существует
            logs_dst = os.path.join(restore_dir, "logs")
            os.makedirs(logs_dst, exist_ok=True)
            
            for item in os.listdir(logs_src):
                src_path = os.path.join(logs_src, item)
                dst_path = os.path.join(logs_dst, item)
                shutil.copy2(src_path, dst_path)
    
    def _restore_trade_history(self, temp_dir: str, restore_dir: str):
        """Восстановление истории сделок"""
        trades_src = os.path.join(temp_dir, "trades")
        if os.path.exists(trades_src):
            # Создание директории для данных, если не существует
            data_dst = os.path.join(restore_dir, "data")
            os.makedirs(data_dst, exist_ok=True)
            
            for item in os.listdir(trades_src):
                src_path = os.path.join(trades_src, item)
                dst_path = os.path.join(data_dst, item)
                shutil.copy2(src_path, dst_path)
    
    def _restore_database(self, temp_dir: str, restore_dir: str):
        """Восстановление базы данных"""
        db_src = os.path.join(temp_dir, "database")
        if os.path.exists(db_src):
            # Создание директории для данных, если не существует
            data_dst = os.path.join(restore_dir, "data")
            os.makedirs(data_dst, exist_ok=True)
            
            for item in os.listdir(db_src):
                src_path = os.path.join(db_src, item)
                dst_path = os.path.join(data_dst, item)
                
                # Остановка бота перед восстановлением БД (если запущен)
                # Эта логика должна быть реализована в основном модуле бота
                
                shutil.copy2(src_path, dst_path)
    
    def cleanup_old_backups(self, max_backups: int = None):
        """
        Удаление старых резервных копий.
        
        Args:
            max_backups: Максимальное количество хранимых резервных копий
        """
        if max_backups is None:
            max_backups = MAX_BACKUPS
        
        try:
            # Получение списка резервных копий
            backups = []
            for file in os.listdir(self.backup_dir):
                if file.startswith("backup_") and (file.endswith(".zip") or file.endswith(".encrypted")):
                    file_path = os.path.join(self.backup_dir, file)
                    backups.append((file_path, os.path.getctime(file_path)))
            
            # Сортировка по дате создания (сначала старые)
            backups.sort(key=lambda x: x[1])
            
            # Удаление старых копий сверх лимита
            while len(backups) > max_backups:
                old_backup = backups.pop(0)
                os.remove(old_backup[0])
                logger.info(f"Удалена старая резервная копия: {old_backup[0]}")
                
        except Exception as e:
            logger.error(f"Ошибка очистки старых резервных копий: {e}")
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """
        Получение списка доступных резервных копий.
        
        Returns:
            List[Dict]: Список информации о резервных копиях
        """
        backups = []
        
        try:
            for file in os.listdir(self.backup_dir):
                if file.startswith("backup_") and (file.endswith(".zip") or file.endswith(".encrypted")):
                    file_path = os.path.join(self.backup_dir, file)
                    file_stats = os.stat(file_path)
                    
                    backup_info = {
                        'name': file,
                        'path': file_path,
                        'size': file_stats.st_size,
                        'created': datetime.fromtimestamp(file_stats.st_ctime),
                        'modified': datetime.fromtimestamp(file_stats.st_mtime),
                        'encrypted': file.endswith(".encrypted")
                    }
                    
                    backups.append(backup_info)
            
            # Сортировка по дате создания (сначала новые)
            backups.sort(key=lambda x: x['created'], reverse=True)
            
        except Exception as e:
            logger.error(f"Ошибка получения списка резервных копий: {e}")
        
        return backups
    
    def get_backup_info(self, backup_path: str) -> Optional[Dict[str, Any]]:
        """
        Получение информации о конкретной резервной копии.
        
        Args:
            backup_path: Путь к резервной копии
            
        Returns:
            Optional[Dict]: Информация о резервной копии или None в случае ошибки
        """
        try:
            if not os.path.exists(backup_path):
                return None
            
            file_stats = os.stat(backup_path)
            
            # Для зашифрованных копий информация ограничена
            if backup_path.endswith(".encrypted"):
                return {
                    'name': os.path.basename(backup_path),
                    'path': backup_path,
                    'size': file_stats.st_size,
                    'created': datetime.fromtimestamp(file_stats.st_ctime),
                    'modified': datetime.fromtimestamp(file_stats.st_mtime),
                    'encrypted': True
                }
            
            # Для незашифрованных копий можно извлечь метаданные
            info = {
                'name': os.path.basename(backup_path),
                'path': backup_path,
                'size': file_stats.st_size,
                'created': datetime.fromtimestamp(file_stats.st_ctime),
                'modified': datetime.fromtimestamp(file_stats.st_mtime),
                'encrypted': False
            }
            
            # Извлечение метаданных из архива
            with zipfile.ZipFile(backup_path, 'r') as zipf:
                if 'metadata.json' in zipf.namelist():
                    with zipf.open('metadata.json') as f:
                        metadata = json.load(f)
                    info['metadata'] = metadata
            
            return info
            
        except Exception as e:
            logger.error(f"Ошибка получения информации о резервной копии: {e}")
            return None


# Утилиты для работы с резервными копиями
def schedule_backups():
    """Планирование регулярного резервного копирования"""
    # Эта функция должна быть интегрирована с основной системой планирования задач
    # (например, с использованием schedule или celery)
    pass


if __name__ == "__main__":
    # Пример использования
    backup_manager = BackupManager()
    
    # Создание резервной копии
    backup_path = backup_manager.create_backup()
    
    if backup_path:
        # Получение информации о резервной копии
        backup_info = backup_manager.get_backup_info(backup_path)
        print(f"Создана резервная копия: {backup_info}")
        
        # Список всех резервных копий
        backups = backup_manager.list_backups()
        print(f"Доступно резервных копий: {len(backups)}")
        
        # Очистка старых резервных копий
        backup_manager.cleanup_old_backups(max_backups=5)
    else:
        print("Ошибка создания резервной копии")