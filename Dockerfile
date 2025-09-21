FROM python:3.9-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Создание пользователя приложения
RUN useradd --create-home --shell /bin/bash app
USER app
WORKDIR