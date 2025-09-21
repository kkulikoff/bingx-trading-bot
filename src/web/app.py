"""
Модуль веб-интерфейса торгового бота.
Создает Flask приложение с API и dashboard.
"""
from flask import Flask, jsonify, render_template
from prometheus_client import make_wsgi_app
from werkzeug.middleware.dispatcher import DispatcherMiddleware

from src.web.routes import register_routes
from src.utils.prometheus_metrics import setup_metrics

def create_app(bot_instance):
    """Создание и настройка Flask приложения"""
    app = Flask(__name__)
    
    # Настройка конфигурации
    app.config.from_pyfile('../../config/settings.py')
    
    # Регистрация маршрутов
    register_routes(app, bot_instance)
    
    # Настройка метрик Prometheus
    setup_metrics(app)
    
    # Добавление endpoint для метрик Prometheus
    app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {
        '/metrics': make_wsgi_app()
    })
    
    return app

# Если файл запускается напрямую
if __name__ == '__main__':
    from src.bot.core import AdvancedBingXBot
    import asyncio
    
    # Создание экземпляра бота
    bot = AdvancedBingXBot()
    
    # Инициализация бота в event loop
    loop = asyncio.get_event_loop()
    loop.run_until_complete(bot.initialize())
    
    # Создание и запуск веб-приложения
    app = create_app(bot)
    app.run(host='0.0.0.0', port=5000, debug=False)