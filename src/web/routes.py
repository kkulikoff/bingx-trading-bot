"""
Модуль маршрутов веб-интерфейса торгового бота.
Определяет все API endpoints и веб-страницы для взаимодействия с торговой системой.
"""

from flask import jsonify, render_template, request, abort
from typing import Dict, List, Any, Optional
import logging
import json
from datetime import datetime, timedelta

from src.utils.logger import setup_logger
from src.utils.helpers import validate_api_key, format_timestamp
from src.data.cache import DataCache
from src.bot.core import AdvancedBingXBot

logger = setup_logger(__name__)

def register_routes(app, bot_instance: AdvancedBingXBot):
    """
    Регистрация всех маршрутов в Flask приложении
    
    Args:
        app: Flask приложение
        bot_instance: Экземпляр торгового бота
    """
    
    @app.route('/')
    def index():
        """Главная страница веб-интерфейса"""
        return render_template('dashboard.html')
    
    @app.route('/dashboard')
    def dashboard():
        """Dashboard с основной информацией"""
        return render_template('dashboard.html')
    
    @app.route('/signals')
    def signals_page():
        """Страница с торговыми сигналами"""
        return render_template('signals.html')
    
    @app.route('/portfolio')
    def portfolio_page():
        """Страница с информацией о портфеле"""
        return render_template('portfolio.html')
    
    @app.route('/settings')
    def settings_page():
        """Страница настроек"""
        return render_template('settings.html')
    
    @app.route('/api/health')
    def health_check():
        """API для проверки здоровья системы"""
        try:
            # Проверка основных компонентов системы
            health_status = {
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'components': {
                    'bot': bot_instance.is_running if hasattr(bot_instance, 'is_running') else False,
                    'api': True,
                    'database': True,  # Заглушка, в реальности нужно проверять подключение к БД
                    'cache': True      # Заглушка, в реальности нужно проверять кэш
                },
                'version': '1.0.0'
            }
            return jsonify(health_status)
        except Exception as e:
            logger.error(f"Ошибка проверки здоровья системы: {e}")
            return jsonify({
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500
    
    @app.route('/api/signals')
    def get_signals():
        """API для получения текущих и исторических сигналов"""
        try:
            # Получение параметров запроса
            limit = request.args.get('limit', default=50, type=int)
            symbol = request.args.get('symbol', default=None)
            timeframe = request.args.get('timeframe', default=None)
            
            # Получение сигналов из бота
            signals = getattr(bot_instance, 'signals_history', [])
            
            # Фильтрация по символу и таймфрейму
            if symbol:
                signals = [s for s in signals if s.get('symbol') == symbol]
            if timeframe:
                signals = [s for s in signals if s.get('timeframe') == timeframe]
            
            # Ограничение количества возвращаемых сигналов
            signals = signals[-limit:] if limit > 0 else signals
            
            # Форматирование сигналов для ответа
            formatted_signals = []
            for signal in signals:
                formatted_signal = {
                    'id': signal.get('id', hash(str(signal))),
                    'symbol': signal.get('symbol'),
                    'timeframe': signal.get('timeframe'),
                    'direction': signal.get('direction'),
                    'price': signal.get('price'),
                    'confidence': signal.get('confidence'),
                    'timestamp': format_timestamp(signal.get('timestamp')),
                    'stop_loss': signal.get('stop_loss'),
                    'take_profit': signal.get('take_profit'),
                    'risk_reward': signal.get('risk_reward'),
                    'position_size': signal.get('position_size'),
                    'reasons': signal.get('reasons', [])
                }
                formatted_signals.append(formatted_signal)
            
            return jsonify({
                'success': True,
                'signals': formatted_signals,
                'count': len(formatted_signals),
                'total': len(getattr(bot_instance, 'signals_history', []))
            })
        except Exception as e:
            logger.error(f"Ошибка получения сигналов: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/portfolio')
    def get_portfolio():
        """API для получения информации о портфеле"""
        try:
            # Получение информации о портфеле из risk manager
            risk_manager = getattr(bot_instance, 'risk_manager', None)
            if risk_manager:
                portfolio_status = risk_manager.get_portfolio_status()
                portfolio_metrics = risk_manager.calculate_portfolio_metrics()
                
                response = {
                    'success': True,
                    'portfolio': {
                        'balance': portfolio_status.get('balance', 0),
                        'equity_curve': portfolio_status.get('equity_curve', []),
                        'current_positions': portfolio_status.get('current_positions', 0),
                        'risk_per_trade': portfolio_status.get('risk_per_trade', 0.02)
                    },
                    'performance': portfolio_metrics
                }
            else:
                response = {
                    'success': True,
                    'portfolio': {
                        'balance': 10000,  # Заглушка
                        'equity_curve': [],
                        'current_positions': 0,
                        'risk_per_trade': 0.02
                    },
                    'performance': {}
                }
            
            return jsonify(response)
        except Exception as e:
            logger.error(f"Ошибка получения информации о портфеле: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/performance')
    def get_performance():
        """API для получения данных о производительности и результатах бэктестинга"""
        try:
            # Получение результатов бэктестинга
            backtester = getattr(bot_instance, 'backtester', None)
            performance_data = {}
            
            if backtester and hasattr(backtester, 'results'):
                performance_data = {
                    'backtest_results': backtester.results
                }
            
            # Получение метрик производительности системы
            performance_data['system_metrics'] = {
                'uptime': getattr(bot_instance, 'uptime', 0),
                'signals_generated': len(getattr(bot_instance, 'signals_history', [])),
                'trades_executed': len(getattr(bot_instance, 'trading_engine', {}).get('order_history', [])),
                'api_requests': getattr(bot_instance, 'api_request_count', 0)
            }
            
            return jsonify({
                'success': True,
                'performance': performance_data
            })
        except Exception as e:
            logger.error(f"Ошибка получения данных о производительности: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/execute', methods=['POST'])
    def execute_trade():
        """API для исполнения ручной сделки"""
        try:
            # Проверка аутентификации
            auth_header = request.headers.get('Authorization')
            if not validate_api_key(auth_header):
                return jsonify({
                    'success': False,
                    'error': 'Неавторизованный доступ'
                }), 401
            
            # Получение данных сделки из запроса
            trade_data = request.get_json()
            if not trade_data:
                return jsonify({
                    'success': False,
                    'error': 'Отсутствуют данные сделки'
                }), 400
            
            symbol = trade_data.get('symbol')
            direction = trade_data.get('direction')
            quantity = trade_data.get('quantity')
            
            if not all([symbol, direction, quantity]):
                return jsonify({
                    'success': False,
                    'error': 'Необходимы symbol, direction и quantity'
                }), 400
            
            # Создание сигнала для ручной сделки
            manual_signal = {
                'symbol': symbol,
                'direction': direction.upper(),
                'price': 0,  # Будет получена текущая цена
                'confidence': 1.0,
                'timestamp': datetime.now(),
                'is_manual': True
            }
            
            # Исполнение сделки через торговый движок
            trading_engine = getattr(bot_instance, 'trading_engine', None)
            if trading_engine:
                success = trading_engine.execute_trade(manual_signal)
                
                if success:
                    return jsonify({
                        'success': True,
                        'message': f'Сделка {symbol} {direction} на {quantity} исполнена'
                    })
                else:
                    return jsonify({
                        'success': False,
                        'error': 'Не удалось исполнить сделку'
                    }), 500
            else:
                return jsonify({
                    'success': False,
                    'error': 'Торговый движок не доступен'
                }), 500
                
        except Exception as e:
            logger.error(f"Ошибка исполнения ручной сделки: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/symbols')
    def get_symbols():
        """API для получения списка доступных торговых пар"""
        try:
            symbols = getattr(bot_instance, 'symbols', [])
            return jsonify({
                'success': True,
                'symbols': symbols
            })
        except Exception as e:
            logger.error(f"Ошибка получения списка символов: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/timeframes')
    def get_timeframes():
        """API для получения списка доступных таймфреймов"""
        try:
            timeframes = getattr(bot_instance, 'timeframes', [])
            return jsonify({
                'success': True,
                'timeframes': timeframes
            })
        except Exception as e:
            logger.error(f"Ошибка получения списка таймфреймов: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/system/start', methods=['POST'])
    def start_system():
        """API для запуска торговой системы"""
        try:
            # Проверка аутентификации
            auth_header = request.headers.get('Authorization')
            if not validate_api_key(auth_header):
                return jsonify({
                    'success': False,
                    'error': 'Неавторизованный доступ'
                }), 401
            
            if bot_instance.is_running:
                return jsonify({
                    'success': False,
                    'error': 'Система уже запущена'
                }), 400
            
            # Запуск системы
            # В реальной реализации здесь должен быть вызов метода запуска бота
            bot_instance.is_running = True
            
            return jsonify({
                'success': True,
                'message': 'Торговая система запущена'
            })
        except Exception as e:
            logger.error(f"Ошибка запуска системы: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/system/stop', methods=['POST'])
    def stop_system():
        """API для остановки торговой системы"""
        try:
            # Проверка аутентификации
            auth_header = request.headers.get('Authorization')
            if not validate_api_key(auth_header):
                return jsonify({
                    'success': False,
                    'error': 'Неавторизованный доступ'
                }), 401
            
            if not bot_instance.is_running:
                return jsonify({
                    'success': False,
                    'error': 'Система уже остановлена'
                }), 400
            
            # Остановка системы
            # В реальной реализации здесь должен быть вызов метода остановки бота
            bot_instance.is_running = False
            
            return jsonify({
                'success': True,
                'message': 'Торговая система остановлена'
            })
        except Exception as e:
            logger.error(f"Ошибка остановки системы: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/system/status')
    def system_status():
        """API для получения статуса системы"""
        try:
            status = {
                'is_running': getattr(bot_instance, 'is_running', False),
                'start_time': getattr(bot_instance, 'start_time', None),
                'uptime': getattr(bot_instance, 'uptime', 0),
                'signals_generated': len(getattr(bot_instance, 'signals_history', [])),
                'trades_executed': len(getattr(bot_instance, 'trading_engine', {}).get('order_history', [])),
                'errors_count': getattr(bot_instance, 'errors_count', 0)
            }
            
            return jsonify({
                'success': True,
                'status': status
            })
        except Exception as e:
            logger.error(f"Ошибка получения статуса системы: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/cache/clear', methods=['POST'])
    def clear_cache():
        """API для очистки кэша"""
        try:
            # Проверка аутентификации
            auth_header = request.headers.get('Authorization')
            if not validate_api_key(auth_header):
                return jsonify({
                    'success': False,
                    'error': 'Неавторизованный доступ'
                }), 401
            
            # Очистка кэша
            from src.data.cache import cache_manager
            success = cache_manager.clear()
            
            if success:
                return jsonify({
                    'success': True,
                    'message': 'Кэш очищен'
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Не удалось очистить кэш'
                }), 500
        except Exception as e:
            logger.error(f"Ошибка очистки кэша: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/cache/stats')
    def cache_stats():
        """API для получения статистики кэша"""
        try:
            # Проверка аутентификации
            auth_header = request.headers.get('Authorization')
            if not validate_api_key(auth_header):
                return jsonify({
                    'success': False,
                    'error': 'Неавторизованный доступ'
                }), 401
            
            # Получение статистики кэша
            from src.data.cache import cache_manager
            stats = cache_manager.get_stats()
            
            return jsonify({
                'success': True,
                'stats': stats
            })
        except Exception as e:
            logger.error(f"Ошибка получения статистики кэша: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    # Обработчики ошибок
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            'success': False,
            'error': 'Ресурс не найден'
        }), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({
            'success': False,
            'error': 'Внутренняя ошибка сервера'
        }), 500
    
    logger.info("Все маршруты успешно зарегистрированы")