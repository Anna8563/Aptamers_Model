#!/bin/bash

# Скрипт для генерации .env файла из переменных окружения

set -e

ENV_FILE=".env"

echo "🔧 Генерируем .env файл..."

# Создаем .env файл
PWD=$(pwd)

cat > "$ENV_FILE" << EOF
PYTHONPATH="$PWD"
DATA_PATH="$PWD/data"
OUTPUTS_PATH="$PWD/outputs"
CHECKPOINTS_PATH="$PWD/outputs/checkpoints"
MLRUNS_PATH="$PWD/outputs/mlruns"
CONFIG_PATH="$PWD/conf"
# Generated at: $(date)
EOF

# Проверяем и создаём директории
for var in "${required_vars[@]}"; do
    # Проверка: переменная окружения задана?
    if [[ -z "${!var:-}" ]]; then
        echo "Ошибка: переменная окружения '$var' не установлена или пуста." >&2
        exit 1
    fi

    path="${!var}"

    # Создаём директорию (mkdir -p безопасно при пробелах благодаря кавычкам)
    if mkdir -p -- "$path"; then
        echo "Создано/существует: $path"
    else
        echo "Не удалось создать директорию: $path" >&2
        exit 1
    fi
done

# Устанавливаем безопасные права
chmod 600 "$ENV_FILE"

echo "✅ .env файл создан: $ENV_FILE"
echo "🔒 Права установлены: 600 (только владелец может читать/писать)"

# Показываем содержимое (без секретов)
echo ""
echo "📋 Содержимое .env файла:"
echo "========================"
sed 's/=.*/=***/' "$ENV_FILE"
echo "========================"

