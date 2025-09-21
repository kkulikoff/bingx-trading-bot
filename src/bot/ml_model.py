"""
Модуль машинного обучения для BingX Trading Bot.

Содержит реализации ML моделей для прогнозирования движения цен,
генерации признаков и обработки данных для обучения и предсказаний.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
import joblib
import talib
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

from src.utils.logger import setup_logger
from src.utils.helpers import safe_round, generate_hash

logger = setup_logger(__name__)

class MLModel:
    """
    Класс для машинного обучения и прогнозирования движения цен.
    """
    
    def __init__(self, model_path: str = None, scaler_path: str = None):
        """
        Инициализация ML модели.
        
        Args:
            model_path: Путь для сохранения/загрузки модели
            scaler_path: Путь для сохранения/загрузки scaler
        """
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importances_ = None
        self.model_path = model_path or "data/models/trading_model.pkl"
        self.scaler_path = scaler_path or "data/models/scaler.pkl"
        self.is_trained = False
        self.training_date = None
        self.model_metrics = {}
        
        # Настройки по умолчанию
        self.default_params = {
            'n_estimators': 100,
            'max_depth': 5,
            'random_state': 42,
            'n_jobs': -1
        }
        
        logger.info("ML модель инициализирована")
    
    async def initialize(self, historical_data: Dict[str, pd.DataFrame] = None):
        """
        Инициализация модели с возможностью загрузки или обучения.
        
        Args:
            historical_data: Исторические данные для обучения
        """
        try:
            # Попытка загрузить существующую модель
            if await self.load_model():
                logger.info("ML модель успешно загружена")
                return True
            
            # Если данные предоставлены и модель не загружена, обучаем
            if historical_data is not None:
                logger.info("Обучение новой ML модели на предоставленных данных")
                return await self.train_model(historical_data)
            
            logger.warning("ML модель не загружена и не обучена")
            return False
            
        except Exception as e:
            logger.error(f"Ошибка инициализации ML модели: {e}")
            return False
    
    def prepare_features(self, df: pd.DataFrame, future_bars: int = 5) -> pd.DataFrame:
        """
        Подготовка признаков для ML модели из ценовых данных.
        
        Args:
            df: DataFrame с ценовыми данными
            future_bars: Количество баров вперед для прогнозирования
            
        Returns:
            pd.DataFrame: Признаки и целевая переменная
        """
        try:
            if df is None or len(df) < 50:
                logger.warning("Недостаточно данных для подготовки признаков")
                return pd.DataFrame()
            
            # Создаем копию данных
            data = df.copy()
            
            # Базовые ценовые признаки
            features = pd.DataFrame(index=data.index)
            
            # 1. Returns and volatility
            features['returns'] = data['close'].pct_change()
            features['volatility'] = features['returns'].rolling(20).std()
            features['log_returns'] = np.log(data['close'] / data['close'].shift(1))
            
            # 2. RSI
            features['rsi'] = talib.RSI(data['close'], timeperiod=14)
            
            # 3. MACD
            macd, macd_signal, macd_hist = talib.MACD(
                data['close'], 
                fastperiod=12, 
                slowperiod=26, 
                signalperiod=9
            )
            features['macd'] = macd
            features['macd_signal'] = macd_signal
            features['macd_hist'] = macd_hist
            
            # 4. Moving averages
            features['sma_20'] = talib.SMA(data['close'], timeperiod=20)
            features['sma_50'] = talib.SMA(data['close'], timeperiod=50)
            features['ema_12'] = talib.EMA(data['close'], timeperiod=12)
            features['ema_26'] = talib.EMA(data['close'], timeperiod=26)
            
            # 5. Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(
                data['close'], 
                timeperiod=20, 
                nbdevup=2, 
                nbdevdn=2
            )
            features['bb_upper'] = bb_upper
            features['bb_middle'] = bb_middle
            features['bb_lower'] = bb_lower
            features['bb_width'] = (bb_upper - bb_lower) / bb_middle
            features['bb_position'] = (data['close'] - bb_lower) / (bb_upper - bb_lower)
            
            # 6. Volatility indicators
            features['atr'] = talib.ATR(
                data['high'], 
                data['low'], 
                data['close'], 
                timeperiod=14
            )
            features['adx'] = talib.ADX(
                data['high'], 
                data['low'], 
                data['close'], 
                timeperiod=14
            )
            
            # 7. Momentum indicators
            features['stoch_k'], features['stoch_d'] = talib.STOCH(
                data['high'], 
                data['low'], 
                data['close'], 
                fastk_period=14, 
                slowk_period=3, 
                slowk_matype=0, 
                slowd_period=3, 
                slowd_matype=0
            )
            features['cci'] = talib.CCI(
                data['high'], 
                data['low'], 
                data['close'], 
                timeperiod=14
            )
            features['williams_r'] = talib.WILLR(
                data['high'], 
                data['low'], 
                data['close'], 
                timeperiod=14
            )
            
            # 8. Volume indicators
            features['volume_sma'] = talib.SMA(data['volume'], timeperiod=20)
            features['volume_ratio'] = data['volume'] / features['volume_sma']
            features['obv'] = talib.OBV(data['close'], data['volume'])
            
            # 9. Price patterns
            features['price_sma_20_ratio'] = data['close'] / features['sma_20']
            features['price_sma_50_ratio'] = data['close'] / features['sma_50']
            features['high_low_ratio'] = data['high'] / data['low']
            features['close_open_ratio'] = data['close'] / data['open']
            
            # 10. Additional engineered features
            features['momentum'] = data['close'] / data['close'].shift(5) - 1
            features['volatility_ratio'] = features['volatility'] / features['volatility'].rolling(50).mean()
            
            # 11. Lag features
            for lag in [1, 2, 3, 5, 10]:
                features[f'returns_lag_{lag}'] = features['returns'].shift(lag)
                features[f'volume_lag_{lag}'] = data['volume'].shift(lag)
            
            # 12. Rolling features
            features['returns_rolling_mean_5'] = features['returns'].rolling(5).mean()
            features['returns_rolling_std_5'] = features['returns'].rolling(5).std()
            features['volume_rolling_mean_5'] = data['volume'].rolling(5).mean()
            
            # Целевая переменная - движение цены в будущем
            # 1 - цена вырастет на 1% в течение future_bars баров
            # 0 - цена упадет или останется без значимых изменений
            future_price = data['close'].shift(-future_bars)
            price_change = (future_price / data['close'] - 1) * 100
            features['target'] = (price_change > 1.0).astype(int)
            
            # Удаляем строки с NaN значениями
            features.dropna(inplace=True)
            
            logger.debug(f"Подготовлено {len(features.columns)} признаков из {len(features)} образцов")
            return features
            
        except Exception as e:
            logger.error(f"Ошибка подготовки признаков: {e}")
            return pd.DataFrame()
    
    async def train_model(self, historical_data: Dict[str, pd.DataFrame], 
                         test_size: float = 0.2, optimize: bool = False) -> bool:
        """
        Обучение ML модели на исторических данных.
        
        Args:
            historical_data: Словарь с историческими данными
            test_size: Размер тестовой выборки
            optimize: Флаг оптимизации гиперпараметров
            
        Returns:
            bool: True если обучение успешно, иначе False
        """
        try:
            # Подготовка данных для обучения
            all_features = []
            all_targets = []
            
            for symbol, data in historical_data.items():
                if data is not None and len(data) > 100:
                    features = self.prepare_features(data)
                    if not features.empty:
                        all_features.append(features.drop('target', axis=1))
                        all_targets.append(features['target'])
            
            if not all_features:
                logger.error("Не удалось подготовить признаки для обучения")
                return False
            
            # Объединение данных всех символов
            X = pd.concat(all_features, axis=0)
            y = pd.concat(all_targets, axis=0)
            
            # Проверка баланса классов
            class_counts = y.value_counts()
            logger.info(f"Баланс классов: {dict(class_counts)}")
            
            if len(class_counts) < 2:
                logger.error("Недостаточно классов для обучения")
                return False
            
            # Разделение на обучающую и тестовую выборки
            # Используем временное разделение для временных рядов
            tscv = TimeSeriesSplit(n_splits=5)
            
            best_score = 0
            best_model = None
            
            for train_index, test_index in tscv.split(X):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                
                # Обработка дисбаланса классов
                sampler = SMOTE(sampling_strategy='auto', random_state=42)
                X_train_res, y_train_res = sampler.fit_resample(X_train, y_train)
                
                # Масштабирование признаков
                self.scaler.fit(X_train_res)
                X_train_scaled = self.scaler.transform(X_train_res)
                X_test_scaled = self.scaler.transform(X_test)
                
                # Обучение модели
                if optimize:
                    model = self._optimize_hyperparameters(X_train_scaled, y_train_res)
                else:
                    model = RandomForestClassifier(**self.default_params)
                    model.fit(X_train_scaled, y_train_res)
                
                # Оценка модели
                y_pred = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                
                if accuracy > best_score:
                    best_score = accuracy
                    best_model = model
            
            if best_model is None:
                logger.error("Не удалось обучить модель")
                return False
            
            self.model = best_model
            self.is_trained = True
            self.training_date = datetime.now()
            self.feature_importances_ = self._get_feature_importances(X.columns)
            
            # Сохранение модели и scaler
            await self.save_model()
            
            # Финальная оценка модели
            X_scaled = self.scaler.transform(X)
            y_pred = self.model.predict(X_scaled)
            
            self.model_metrics = {
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred),
                'recall': recall_score(y, y_pred),
                'f1': f1_score(y, y_pred),
                'training_samples': len(X),
                'feature_count': len(X.columns),
                'class_balance': dict(class_counts)
            }
            
            logger.info(f"Модель обучена успешно. Точность: {self.model_metrics['accuracy']:.3f}")
            logger.info(f"Метрики модели: {self.model_metrics}")
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка обучения модели: {e}")
            return False
    
    def _optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Оптимизация гиперпараметров модели.
        
        Args:
            X_train: Признаки обучающей выборки
            y_train: Целевая переменная обучающей выборки
            
        Returns:
            Оптимизированная модель
        """
        try:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            model = RandomForestClassifier(random_state=42)
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=3,
                scoring='f1',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            
            logger.info(f"Лучшие параметры: {grid_search.best_params_}")
            logger.info(f"Лучшая оценка: {grid_search.best_score_:.3f}")
            
            return grid_search.best_estimator_
            
        except Exception as e:
            logger.error(f"Ошибка оптимизации гиперпараметров: {e}")
            return RandomForestClassifier(**self.default_params)
    
    def _get_feature_importances(self, feature_names: pd.Index) -> Dict[str, float]:
        """
        Получение важности признаков модели.
        
        Args:
            feature_names: Названия признаков
            
        Returns:
            Dict: Важность признаков
        """
        if self.model is None or not hasattr(self.model, 'feature_importances_'):
            return {}
        
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        feature_importance = {}
        for i, idx in enumerate(indices):
            if i < 20:  # Топ-20 признаков
                feature_importance[feature_names[idx]] = importances[idx]
        
        return feature_importance
    
    async def predict(self, df: pd.DataFrame) -> float:
        """
        Прогнозирование движения цены на основе данных.
        
        Args:
            df: DataFrame с ценовыми данными
            
        Returns:
            float: Вероятность роста цены (0-1)
        """
        if not self.is_trained or self.model is None:
            logger.warning("Модель не обучена, возвращаем нейтральный прогноз")
            return 0.5
        
        try:
            # Подготовка признаков
            features = self.prepare_features(df)
            
            if features.empty or len(features) == 0:
                logger.warning("Не удалось подготовить признаки для прогнозирования")
                return 0.5
            
            # Берем последний доступный образец
            X = features.drop('target', axis=1).iloc[-1:]
            
            # Масштабирование признаков
            X_scaled = self.scaler.transform(X)
            
            # Прогнозирование
            probability = self.model.predict_proba(X_scaled)[0][1]
            
            logger.debug(f"Прогноз модели: {probability:.3f}")
            return probability
            
        except Exception as e:
            logger.error(f"Ошибка прогнозирования: {e}")
            return 0.5
    
    async def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Оценка производительности модели на тестовых данных.
        
        Args:
            X_test: Признаки тестовой выборки
            y_test: Целевая переменная тестовой выборки
            
        Returns:
            Dict: Метрики производительности
        """
        if not self.is_trained or self.model is None:
            logger.error("Модель не обучена для оценки")
            return {}
        
        try:
            # Масштабирование признаков
            X_test_scaled = self.scaler.transform(X_test)
            
            # Прогнозирование
            y_pred = self.model.predict(X_test_scaled)
            y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
            
            # Расчет метрик
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'roc_auc': None,  # Можно добавить roc_auc_score
                'confusion_matrix': None  # Можно добавить confusion_matrix
            }
            
            # Детальный отчет
            logger.info("Отчет классификации:")
            logger.info(classification_report(y_test, y_pred))
            
            # Важность признаков
            logger.info("Важность признаков (топ-10):")
            for feature, importance in list(self.feature_importances_.items())[:10]:
                logger.info(f"  {feature}: {importance:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Ошибка оценки модели: {e}")
            return {}
    
    async def save_model(self) -> bool:
        """
        Сохранение модели и scaler в файлы.
        
        Returns:
            bool: True если сохранение успешно, иначе False
        """
        try:
            if self.model is None:
                logger.error("Нет модели для сохранения")
                return False
            
            # Создание директорий если не существуют
            import os
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.scaler_path), exist_ok=True)
            
            # Сохранение модели и scaler
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            
            logger.info(f"Модель сохранена: {self.model_path}")
            logger.info(f"Scaler сохранен: {self.scaler_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка сохранения модели: {e}")
            return False
    
    async def load_model(self) -> bool:
        """
        Загрузка модели и scaler из файлов.
        
        Returns:
            bool: True если загрузка успешна, иначе False
        """
        try:
            import os
            if not os.path.exists(self.model_path) or not os.path.exists(self.scaler_path):
                logger.warning("Файлы модели или scaler не найдены")
                return False
            
            # Загрузка модели и scaler
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            self.is_trained = True
            
            # Восстановление feature importances если возможно
            if hasattr(self.model, 'feature_importances_'):
                # Создаем фиктивные имена признаков
                dummy_features = [f'feature_{i}' for i in range(len(self.model.feature_importances_))]
                self.feature_importances_ = self._get_feature_importances(dummy_features)
            
            logger.info(f"Модель загружена: {self.model_path}")
            logger.info(f"Scaler загружен: {self.scaler_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Получение информации о модели.
        
        Returns:
            Dict: Информация о модели
        """
        if not self.is_trained:
            return {'status': 'not_trained'}
        
        info = {
            'status': 'trained',
            'training_date': self.training_date.isoformat() if self.training_date else None,
            'model_type': type(self.model).__name__,
            'metrics': self.model_metrics,
            'feature_importances': self.feature_importances_,
            'model_path': self.model_path,
            'scaler_path': self.scaler_path
        }
        
        return info
    
    async def retrain_model(self, new_data: Dict[str, pd.DataFrame], 
                           incremental: bool = False) -> bool:
        """
        Переобучение модели на новых данных.
        
        Args:
            new_data: Новые данные для обучения
            incremental: Флаг инкрементального обучения
            
        Returns:
            bool: True если переобучение успешно, иначе False
        """
        try:
            if incremental and self.is_trained:
                # Инкрементальное обучение (требует специальной реализации)
                logger.info("Начинаем инкрементальное обучение")
                return await self._incremental_train(new_data)
            else:
                # Полное переобучение
                logger.info("Начинаем полное переобучение модели")
                return await self.train_model(new_data)
                
        except Exception as e:
            logger.error(f"Ошибка переобучения модели: {e}")
            return False
    
    async def _incremental_train(self, new_data: Dict[str, pd.DataFrame]) -> bool:
        """
        Инкрементальное обучение модели на новых данных.
        
        Args:
            new_data: Новые данные для обучения
            
        Returns:
            bool: True если инкрементальное обучение успешно, иначе False
        """
        # Заглушка для инкрементального обучения
        # В реальной реализации здесь будет код для обновления модели
        # без полного переобучения на всех данных
        
        logger.warning("Инкрементальное обучение не реализовано, выполняется полное переобучение")
        return await self.train_model(new_data)
    
    async def shutdown(self):
        """
        Корректное завершение работы ML модели.
        """
        logger.info("Завершение работы ML модели")
        
        # Сохранение модели перед выходом
        await self.save_model()
        
        logger.info("ML модель остановлена")


# Дополнительные классы моделей для расширения функциональности

class GradientBoostingModel(MLModel):
    """
    Модель на основе Gradient Boosting для сравнения производительности.
    """
    
    def __init__(self, model_path: str = None, scaler_path: str = None):
        super().__init__(model_path, scaler_path)
        self.default_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'random_state': 42
        }
    
    async def train_model(self, historical_data: Dict[str, pd.DataFrame], 
                         test_size: float = 0.2, optimize: bool = False) -> bool:
        """
        Обучение Gradient Boosting модели.
        """
        try:
            # Подготовка данных (аналогично базовому классу)
            all_features = []
            all_targets = []
            
            for symbol, data in historical_data.items():
                if data is not None and len(data) > 100:
                    features = self.prepare_features(data)
                    if not features.empty:
                        all_features.append(features.drop('target', axis=1))
                        all_targets.append(features['target'])
            
            if not all_features:
                logger.error("Не удалось подготовить признаки для обучения")
                return False
            
            X = pd.concat(all_features, axis=0)
            y = pd.concat(all_targets, axis=0)
            
            # Разделение на обучающую и тестовую выборки
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, shuffle=False
            )
            
            # Обработка дисбаланса классов
            sampler = SMOTE(sampling_strategy='auto', random_state=42)
            X_train_res, y_train_res = sampler.fit_resample(X_train, y_train)
            
            # Масштабирование признаков
            self.scaler.fit(X_train_res)
            X_train_scaled = self.scaler.transform(X_train_res)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Обучение Gradient Boosting модели
            if optimize:
                model = self._optimize_hyperparameters(X_train_scaled, y_train_res)
            else:
                model = GradientBoostingClassifier(**self.default_params)
                model.fit(X_train_scaled, y_train_res)
            
            self.model = model
            self.is_trained = True
            self.training_date = datetime.now()
            self.feature_importances_ = self._get_feature_importances(X.columns)
            
            # Сохранение модели
            await self.save_model()
            
            # Оценка модели
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.model_metrics = {
                'accuracy': accuracy,
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred)
            }
            
            logger.info(f"Gradient Boosting модель обучена. Точность: {accuracy:.3f}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка обучения Gradient Boosting модели: {e}")
            return False


class EnsembleModel(MLModel):
    """
    Ансамблевая модель, сочетающая несколько алгоритмов.
    """
    
    def __init__(self, model_path: str = None, scaler_path: str = None):
        super().__init__(model_path, scaler_path)
        self.models = {}
    
    async def train_model(self, historical_data: Dict[str, pd.DataFrame], 
                         test_size: float = 0.2, optimize: bool = False) -> bool:
        """
        Обучение ансамблевой модели.
        """
        # Реализация ансамблевого обучения
        # Можно комбинировать Random Forest, Gradient Boosting, etc.
        logger.info("Ансамблевое обучение не реализовано")
        return await super().train_model(historical_data, test_size, optimize)