"""
Модуль генерации торговых сигналов для BingX Trading Bot.

Содержит логику технического анализа, генерации сигналов
и интеграцию с ML-моделями для принятия торговых решений.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
import talib
from datetime import datetime

from src.bot.ml_model import MLModel
from src.utils.logger import setup_logger
from src.utils.helpers import safe_round, calculate_volatility

logger = setup_logger(__name__)

class SignalGenerator:
    """
    Класс для генерации торговых сигналов на основе технического анализа
    и машинного обучения.
    """
    
    def __init__(self):
        """
        Инициализация генератора сигналов с настройками индикаторов.
        """
        # Настройки индикаторов
        self.indicator_settings = {
            'rsi': {'period': 14, 'oversold': 30, 'overbought': 70},
            'macd': {'fast': 12, 'slow': 26, 'signal': 9},
            'bollinger': {'period': 20, 'std_dev': 2},
            'atr': {'period': 14},
            'ema': {'short': 20, 'medium': 50, 'long': 100},
            'stoch': {'k_period': 14, 'd_period': 3, 'slow': 3}
        }
        
        # Пороги уверенности для сигналов
        self.confidence_thresholds = {
            'low': 0.6,
            'medium': 0.7,
            'high': 0.8
        }
        
        # Весовые коэффициенты для индикаторов
        self.indicator_weights = {
            'rsi': 0.2,
            'macd': 0.2,
            'bollinger': 0.15,
            'trend': 0.15,
            'volume': 0.1,
            'volatility': 0.1,
            'ml': 0.1
        }
        
        self.ml_model = None
        logger.info("Генератор сигналов инициализирован")
    
    async def initialize(self, ml_model: MLModel = None):
        """
        Инициализация генератора сигналов с ML моделью.
        
        Args:
            ml_model: Обученная ML модель для прогнозирования
        """
        self.ml_model = ml_model
        if self.ml_model:
            logger.info("ML модель подключена к генератору сигналов")
        else:
            logger.warning("Генератор сигналов работает без ML модели")
    
    async def generate_signal(self, data: pd.DataFrame, symbol: str, 
                            timeframe: str) -> Optional[Dict[str, Any]]:
        """
        Генерация торгового сигнала на основе данных.
        
        Args:
            data: DataFrame с ценовыми данными
            symbol: Торговая пара
            timeframe: Таймфрейм анализа
            
        Returns:
            Optional[Dict]: Торговый сигнал или None
        """
        try:
            if data is None or len(data) < 50:
                logger.warning(f"Недостаточно данных для анализа {symbol} {timeframe}")
                return None
            
            # Расчет индикаторов
            indicators = self._calculate_indicators(data)
            
            # Генерация базового сигнала на основе индикаторов
            signal = self._generate_from_indicators(indicators, data, symbol, timeframe)
            
            if not signal:
                return None
            
            # Интеграция с ML моделью
            if self.ml_model:
                ml_confidence = self.ml_model.predict(data)
                if ml_confidence is not None:
                    signal = self._apply_ml_confidence(signal, ml_confidence)
            
            # Фильтрация слабых сигналов
            if signal['confidence'] < self.confidence_thresholds['low']:
                logger.debug(f"Слабый сигнал для {symbol} ({timeframe}): {signal['confidence']:.2f}")
                return None
            
            # Расчет уровней стоп-лосса и тейк-профита
            signal = self._calculate_risk_levels(signal, data)
            
            # Добавление метаданных
            signal['symbol'] = symbol
            signal['timeframe'] = timeframe
            signal['timestamp'] = datetime.now()
            signal['signal_id'] = self._generate_signal_id(symbol, timeframe)
            
            logger.info(f"Сгенерирован сигнал для {symbol} ({timeframe}): "
                       f"{signal['direction']} с уверенностью {signal['confidence']:.2f}")
            
            return signal
            
        except Exception as e:
            logger.error(f"Ошибка генерации сигнала для {symbol} {timeframe}: {e}")
            return None
    
    def _calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Расчет технических индикаторов.
        
        Args:
            data: DataFrame с ценовыми данными
            
        Returns:
            Dict: Рассчитанные индикаторы
        """
        indicators = {}
        
        try:
            # RSI
            rsi_period = self.indicator_settings['rsi']['period']
            indicators['rsi'] = talib.RSI(data['close'], timeperiod=rsi_period)
            
            # MACD
            macd_fast = self.indicator_settings['macd']['fast']
            macd_slow = self.indicator_settings['macd']['slow']
            macd_signal = self.indicator_settings['macd']['signal']
            macd, macd_signal_line, macd_hist = talib.MACD(
                data['close'], 
                fastperiod=macd_fast, 
                slowperiod=macd_slow, 
                signalperiod=macd_signal
            )
            indicators['macd'] = macd
            indicators['macd_signal'] = macd_signal_line
            indicators['macd_hist'] = macd_hist
            
            # Bollinger Bands
            bb_period = self.indicator_settings['bollinger']['period']
            bb_std = self.indicator_settings['bollinger']['std_dev']
            indicators['bb_upper'], indicators['bb_middle'], indicators['bb_lower'] = talib.BBANDS(
                data['close'], 
                timeperiod=bb_period, 
                nbdevup=bb_std, 
                nbdevdn=bb_std
            )
            
            # ATR
            atr_period = self.indicator_settings['atr']['period']
            indicators['atr'] = talib.ATR(
                data['high'], 
                data['low'], 
                data['close'], 
                timeperiod=atr_period
            )
            
            # EMA
            ema_short = self.indicator_settings['ema']['short']
            ema_medium = self.indicator_settings['ema']['medium']
            ema_long = self.indicator_settings['ema']['long']
            indicators['ema_short'] = talib.EMA(data['close'], timeperiod=ema_short)
            indicators['ema_medium'] = talib.EMA(data['close'], timeperiod=ema_medium)
            indicators['ema_long'] = talib.EMA(data['close'], timeperiod=ema_long)
            
            # Stochastic
            stoch_k = self.indicator_settings['stoch']['k_period']
            stoch_d = self.indicator_settings['stoch']['d_period']
            stoch_slow = self.indicator_settings['stoch']['slow']
            indicators['stoch_k'], indicators['stoch_d'] = talib.STOCH(
                data['high'], 
                data['low'], 
                data['close'], 
                fastk_period=stoch_k, 
                slowk_period=stoch_slow, 
                slowk_matype=0, 
                slowd_period=stoch_d, 
                slowd_matype=0
            )
            
            # Volume indicators
            indicators['volume_sma'] = talib.SMA(data['volume'], timeperiod=20)
            indicators['obv'] = talib.OBV(data['close'], data['volume'])
            
            logger.debug("Технические индикаторы успешно рассчитаны")
            
        except Exception as e:
            logger.error(f"Ошибка расчета индикаторов: {e}")
        
        return indicators
    
    def _generate_from_indicators(self, indicators: Dict[str, Any], 
                                data: pd.DataFrame, symbol: str, 
                                timeframe: str) -> Optional[Dict[str, Any]]:
        """
        Генерация сигнала на основе рассчитанных индикаторов.
        
        Args:
            indicators: Рассчитанные индикаторы
            data: Исходные данные
            symbol: Торговая пара
            timeframe: Таймфрейм
            
        Returns:
            Optional[Dict]: Торговый сигнал
        """
        try:
            # Получение последних значений
            current_price = data['close'].iloc[-1]
            current_volume = data['volume'].iloc[-1]
            
            # Инициализация сигнала
            signal = {
                'price': current_price,
                'direction': None,
                'confidence': 0.0,
                'reasons': [],
                'indicators': {}
            }
            
            # Анализ каждого индикатора
            rsi_analysis = self._analyze_rsi(indicators.get('rsi'), signal)
            macd_analysis = self._analyze_macd(indicators.get('macd'), 
                                             indicators.get('macd_signal'), signal)
            bb_analysis = self._analyze_bollinger_bands(indicators.get('bb_upper'),
                                                      indicators.get('bb_lower'),
                                                      current_price, signal)
            trend_analysis = self._analyze_trend(indicators.get('ema_short'),
                                               indicators.get('ema_medium'),
                                               indicators.get('ema_long'),
                                               current_price, signal)
            volume_analysis = self._analyze_volume(current_volume,
                                                 indicators.get('volume_sma'),
                                                 indicators.get('obv'), signal)
            volatility_analysis = self._analyze_volatility(data, signal)
            
            # Совокупный анализ
            if signal['direction']:
                # Расчет общей уверенности
                total_confidence = (
                    rsi_analysis * self.indicator_weights['rsi'] +
                    macd_analysis * self.indicator_weights['macd'] +
                    bb_analysis * self.indicator_weights['bollinger'] +
                    trend_analysis * self.indicator_weights['trend'] +
                    volume_analysis * self.indicator_weights['volume'] +
                    volatility_analysis * self.indicator_weights['volatility']
                )
                
                signal['confidence'] = min(1.0, max(0.0, total_confidence))
                
                # Сохранение значений индикаторов
                self._save_indicator_values(indicators, signal)
                
                return signal
            
            return None
            
        except Exception as e:
            logger.error(f"Ошибка генерации сигнала из индикаторов: {e}")
            return None
    
    def _analyze_rsi(self, rsi: pd.Series, signal: Dict[str, Any]) -> float:
        """
        Анализ RSI индикатора.
        
        Args:
            rsi: Значения RSI
            signal: Сигнал для обновления
            
        Returns:
            float: Уверенность от RSI анализа
        """
        if rsi is None or len(rsi) < 2:
            return 0.0
        
        current_rsi = rsi.iloc[-1]
        previous_rsi = rsi.iloc[-2]
        
        # Сохранение значения индикатора
        signal['indicators']['rsi'] = safe_round(current_rsi, 2)
        
        oversold = self.indicator_settings['rsi']['oversold']
        overbought = self.indicator_settings['rsi']['overbought']
        
        confidence = 0.0
        
        if current_rsi < oversold:
            if signal['direction'] is None or signal['direction'] == 'LONG':
                signal['direction'] = 'LONG'
                signal['reasons'].append(f"RSI в зоне перепроданности ({current_rsi:.1f})")
                
                # Расчет уверенности на основе глубины перепроданности
                confidence = min(1.0, (oversold - current_rsi) / oversold)
        
        elif current_rsi > overbought:
            if signal['direction'] is None or signal['direction'] == 'SHORT':
                signal['direction'] = 'SHORT'
                signal['reasons'].append(f"RSI в зоне перекупленности ({current_rsi:.1f})")
                
                # Расчет уверенности на основе глубины перекупленности
                confidence = min(1.0, (current_rsi - overbought) / (100 - overbought))
        
        return confidence
    
    def _analyze_macd(self, macd: pd.Series, macd_signal: pd.Series, 
                     signal: Dict[str, Any]) -> float:
        """
        Анализ MACD индикатора.
        
        Args:
            macd: Значения MACD
            macd_signal: Значения сигнальной линии MACD
            signal: Сигнал для обновления
            
        Returns:
            float: Уверенность от MACD анализа
        """
        if macd is None or macd_signal is None or len(macd) < 3 or len(macd_signal) < 3:
            return 0.0
        
        current_macd = macd.iloc[-1]
        current_signal = macd_signal.iloc[-1]
        previous_macd = macd.iloc[-2]
        previous_signal = macd_signal.iloc[-2]
        
        # Сохранение значений индикаторов
        signal['indicators']['macd'] = safe_round(current_macd, 4)
        signal['indicators']['macd_signal'] = safe_round(current_signal, 4)
        
        confidence = 0.0
        
        # Бычье пересечение
        if current_macd > current_signal and previous_macd <= previous_signal:
            if signal['direction'] is None or signal['direction'] == 'LONG':
                signal['direction'] = 'LONG'
                signal['reasons'].append("MACD пересек сигнальную линию снизу вверх")
                confidence = 0.7
        
        # Медвежье пересечение
        elif current_macd < current_signal and previous_macd >= previous_signal:
            if signal['direction'] is None or signal['direction'] == 'SHORT':
                signal['direction'] = 'SHORT'
                signal['reasons'].append("MACD пересек сигнальную линию сверху вниз")
                confidence = 0.7
        
        # Расхождение линий
        elif abs(current_macd - current_signal) > abs(previous_macd - previous_signal):
            if current_macd > current_signal and signal['direction'] == 'LONG':
                confidence = 0.3
                signal['reasons'].append("MACD продолжает расходиться в бычьем направлении")
            elif current_macd < current_signal and signal['direction'] == 'SHORT':
                confidence = 0.3
                signal['reasons'].append("MACD продолжает расходиться в медвежьем направлении")
        
        return confidence
    
    def _analyze_bollinger_bands(self, bb_upper: pd.Series, bb_lower: pd.Series,
                               current_price: float, signal: Dict[str, Any]) -> float:
        """
        Анализ полос Боллинджера.
        
        Args:
            bb_upper: Верхняя полоса Боллинджера
            bb_lower: Нижняя полоса Боллинджера
            current_price: Текущая цена
            signal: Сигнал для обновления
            
        Returns:
            float: Уверенность от анализа Bollinger Bands
        """
        if bb_upper is None or bb_lower is None or len(bb_upper) < 1 or len(bb_lower) < 1:
            return 0.0
        
        current_upper = bb_upper.iloc[-1]
        current_lower = bb_lower.iloc[-1]
        
        # Сохранение значений индикаторов
        signal['indicators']['bb_upper'] = safe_round(current_upper, 2)
        signal['indicators']['bb_lower'] = safe_round(current_lower, 2)
        
        confidence = 0.0
        bb_width = current_upper - current_lower
        
        if bb_width == 0:  # Избегаем деления на ноль
            return 0.0
        
        # Цена около нижней полосы
        if current_price <= current_lower:
            if signal['direction'] is None or signal['direction'] == 'LONG':
                signal['direction'] = 'LONG'
                signal['reasons'].append("Цена ниже нижней полосы Боллинджера")
                
                # Расчет уверенности на основе расстояния от полосы
                distance = (current_lower - current_price) / bb_width
                confidence = min(1.0, distance * 2)
        
        # Цена около верхней полосы
        elif current_price >= current_upper:
            if signal['direction'] is None or signal['direction'] == 'SHORT':
                signal['direction'] = 'SHORT'
                signal['reasons'].append("Цена выше верхней полосы Боллинджера")
                
                # Расчет уверенности на основе расстояния от полосы
                distance = (current_price - current_upper) / bb_width
                confidence = min(1.0, distance * 2)
        
        return confidence
    
    def _analyze_trend(self, ema_short: pd.Series, ema_medium: pd.Series,
                      ema_long: pd.Series, current_price: float,
                      signal: Dict[str, Any]) -> float:
        """
        Анализ тренда на основе скользящих средних.
        
        Args:
            ema_short: Короткая EMA
            ema_medium: Средняя EMA
            ema_long: Длинная EMA
            current_price: Текущая цена
            signal: Сигнал для обновления
            
        Returns:
            float: Уверенность от анализа тренда
        """
        if any(x is None for x in [ema_short, ema_medium, ema_long]) or \
           any(len(x) < 1 for x in [ema_short, ema_medium, ema_long]):
            return 0.0
        
        current_short = ema_short.iloc[-1]
        current_medium = ema_medium.iloc[-1]
        current_long = ema_long.iloc[-1]
        
        # Сохранение значений индикаторов
        signal['indicators']['ema_short'] = safe_round(current_short, 2)
        signal['indicators']['ema_medium'] = safe_round(current_medium, 2)
        signal['indicators']['ema_long'] = safe_round(current_long, 2)
        
        confidence = 0.0
        
        # Сильный восходящий тренд
        if current_price > current_short > current_medium > current_long:
            if signal['direction'] == 'LONG':
                confidence = 0.8
                signal['reasons'].append("Сильный восходящий тренд (цена > EMA20 > EMA50 > EMA100)")
            elif signal['direction'] is None:
                signal['direction'] = 'LONG'
                confidence = 0.6
                signal['reasons'].append("Начинается восходящий тренд")
        
        # Сильный нисходящий тренд
        elif current_price < current_short < current_medium < current_long:
            if signal['direction'] == 'SHORT':
                confidence = 0.8
                signal['reasons'].append("Сильный нисходящий тренд (цена < EMA20 < EMA50 < EMA100)")
            elif signal['direction'] is None:
                signal['direction'] = 'SHORT'
                confidence = 0.6
                signal['reasons'].append("Начинается нисходящий тренд")
        
        # Пересечение скользящих средних
        elif current_short > current_medium and signal['direction'] == 'LONG':
            confidence = 0.4
            signal['reasons'].append("EMA20 пересекла EMA50 снизу вверх")
        elif current_short < current_medium and signal['direction'] == 'SHORT':
            confidence = 0.4
            signal['reasons'].append("EMA20 пересекла EMA50 сверху вниз")
        
        return confidence
    
    def _analyze_volume(self, current_volume: float, volume_sma: pd.Series,
                       obv: pd.Series, signal: Dict[str, Any]) -> float:
        """
        Анализ объема торгов.
        
        Args:
            current_volume: Текущий объем
            volume_sma: SMA объема
            obv: On Balance Volume
            signal: Сигнал для обновления
            
        Returns:
            float: Уверенность от анализа объема
        """
        if volume_sma is None or obv is None or len(volume_sma) < 2 or len(obv) < 2:
            return 0.0
        
        current_volume_sma = volume_sma.iloc[-1] if len(volume_sma) > 0 else 0
        previous_obv = obv.iloc[-2] if len(obv) > 1 else 0
        current_obv = obv.iloc[-1] if len(obv) > 0 else 0
        
        # Сохранение значений индикаторов
        signal['indicators']['volume'] = current_volume
        signal['indicators']['volume_sma'] = safe_round(current_volume_sma, 2)
        signal['indicators']['obv'] = safe_round(current_obv, 2)
        
        confidence = 0.0
        
        # Высокий объем относительно среднего
        if current_volume > current_volume_sma * 1.5:
            confidence = 0.3
            signal['reasons'].append("Объем торгов выше среднего")
        
        # Рост OBV
        if current_obv > previous_obv and signal['direction'] == 'LONG':
            confidence = max(confidence, 0.2)
            signal['reasons'].append("On Balance Volume растет")
        elif current_obv < previous_obv and signal['direction'] == 'SHORT':
            confidence = max(confidence, 0.2)
            signal['reasons'].append("On Balance Volume падает")
        
        return confidence
    
    def _analyze_volatility(self, data: pd.DataFrame, signal: Dict[str, Any]) -> float:
        """
        Анализ волатильности рынка.
        
        Args:
            data: Данные с ценами
            signal: Сигнал для обновления
            
        Returns:
            float: Уверенность от анализа волатильности
        """
        if len(data) < 20:
            return 0.0
        
        # Расчет волатильности
        volatility = calculate_volatility(data['close'].tolist(), period=20)
        
        # Сохранение значения волатильности
        signal['indicators']['volatility'] = safe_round(volatility, 2)
        
        confidence = 0.0
        
        # Высокая волатильность
        if volatility > 3.0:  # 3%
            confidence = 0.3
            signal['reasons'].append(f"Высокая волатильность ({volatility:.1f}%)")
        # Низкая волатильность
        elif volatility < 0.5:  # 0.5%
            confidence = -0.2  # Штраф за низкую волатильность
            signal['reasons'].append(f"Низкая волатильность ({volatility:.1f}%)")
        
        return max(0.0, confidence)  # Не допускаем отрицательные значения
    
    def _apply_ml_confidence(self, signal: Dict[str, Any], ml_confidence: float) -> Dict[str, Any]:
        """
        Применение уверенности от ML модели к сигналу.
        
        Args:
            signal: Торговый сигнал
            ml_confidence: Уверенность от ML модели
            
        Returns:
            Dict: Обновленный сигнал
        """
        # Сохранение уверенности ML модели
        signal['indicators']['ml_confidence'] = safe_round(ml_confidence, 3)
        
        # Корректировка общей уверенности
        if ml_confidence > 0.7:
            signal['confidence'] = min(1.0, signal['confidence'] + ml_confidence * self.indicator_weights['ml'])
            signal['reasons'].append(f"ML модель подтверждает сигнал ({ml_confidence:.1%})")
        elif ml_confidence < 0.3:
            signal['confidence'] = max(0.0, signal['confidence'] - (1 - ml_confidence) * self.indicator_weights['ml'])
            signal['reasons'].append(f"ML модель не подтверждает сигнал ({ml_confidence:.1%})")
        
        return signal
    
    def _calculate_risk_levels(self, signal: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """
        Расчет уровней стоп-лосса и тейк-профита.
        
        Args:
            signal: Торговый сигнал
            data: Данные с ценами
            
        Returns:
            Dict: Сигнал с уровнями риска
        """
        try:
            current_price = signal['price']
            atr = data['atr'].iloc[-1] if 'atr' in data and len(data['atr']) > 0 else 0
            
            if atr <= 0:
                # Если ATR не доступен, используем процентные уровни
                if signal['direction'] == 'LONG':
                    signal['stop_loss'] = current_price * 0.98  # -2%
                    signal['take_profit'] = current_price * 1.04  # +4%
                else:  # SHORT
                    signal['stop_loss'] = current_price * 1.02  # +2%
                    signal['take_profit'] = current_price * 0.96  # -4%
            else:
                # Используем ATR для расчета уровней
                atr_multiplier_sl = 1.5
                atr_multiplier_tp = 3.0
                
                if signal['direction'] == 'LONG':
                    signal['stop_loss'] = current_price - (atr * atr_multiplier_sl)
                    signal['take_profit'] = current_price + (atr * atr_multiplier_tp)
                else:  # SHORT
                    signal['stop_loss'] = current_price + (atr * atr_multiplier_sl)
                    signal['take_profit'] = current_price - (atr * atr_multiplier_tp)
            
            # Расчет соотношения риск/прибыль
            if signal['direction'] == 'LONG':
                risk = current_price - signal['stop_loss']
                reward = signal['take_profit'] - current_price
            else:  # SHORT
                risk = signal['stop_loss'] - current_price
                reward = current_price - signal['take_profit']
            
            if risk > 0:
                signal['risk_reward'] = safe_round(reward / risk, 2)
            else:
                signal['risk_reward'] = 0.0
            
            # Добавление информации о рисках в причины
            signal['reasons'].append(f"R/R соотношение: {signal['risk_reward']}:1")
            
            return signal
            
        except Exception as e:
            logger.error(f"Ошибка расчета уровней риска: {e}")
            
            # Установка значений по умолчанию при ошибке
            if signal['direction'] == 'LONG':
                signal['stop_loss'] = current_price * 0.98
                signal['take_profit'] = current_price * 1.04
            else:  # SHORT
                signal['stop_loss'] = current_price * 1.02
                signal['take_profit'] = current_price * 0.96
            
            signal['risk_reward'] = 2.0
            
            return signal
    
    def _save_indicator_values(self, indicators: Dict[str, Any], signal: Dict[str, Any]):
        """
        Сохранение значений индикаторов в сигнал.
        
        Args:
            indicators: Рассчитанные индикаторы
            signal: Сигнал для обновления
        """
        try:
            # Сохранение последних значений всех индикаторов
            for indicator_name, values in indicators.items():
                if values is not None and len(values) > 0:
                    signal['indicators'][indicator_name] = safe_round(values.iloc[-1], 4)
        except Exception as e:
            logger.error(f"Ошибка сохранения значений индикаторов: {e}")
    
    def _generate_signal_id(self, symbol: str, timeframe: str) -> str:
        """
        Генерация уникального ID для сигнала.
        
        Args:
            symbol: Торговая пара
            timeframe: Таймфрейм
            
        Returns:
            str: Уникальный ID сигнала
        """
        timestamp = int(datetime.now().timestamp() * 1000)
        return f"{symbol}_{timeframe}_{timestamp}"
    
    async def batch_generate_signals(self, data_dict: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Пакетная генерация сигналов для нескольких символов и таймфреймов.
        
        Args:
            data_dict: Словарь с данными в формате {symbol: {timeframe: data}}
            
        Returns:
            Dict: Словарь с сигналами в формате {symbol: [signals]}
        """
        signals = {}
        
        for symbol, timeframe_data in data_dict.items():
            signals[symbol] = []
            
            for timeframe, data in timeframe_data.items():
                signal = await self.generate_signal(data, symbol, timeframe)
                
                if signal:
                    signals[symbol].append(signal)
        
        return signals