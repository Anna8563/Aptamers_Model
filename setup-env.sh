#!/bin/bash

# Скрипт для генерации .env файла из переменных окружения

set -e

ENV_FILE=".env"

echo "🔧 Генерируем .env файл..."

# Проверяем обязательные переменные
required_vars=(
    "DATA_PATH"
    "OUTPUTS_PATH"
    "MLFLOW_PATH"
)
all_set=1
for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo "❌ Переменная $var не установлена. Генерируем новый .env файл"
        all_set=0
        break;
    fi
done

if [ "$all_set" -eq 1 ]; then 
    echo "Все переменные окружения установлены"
    exit 0
fi

echo "Создание .env файла"

# Создаем .env файл
PWD=$(pwd)

cat > "$ENV_FILE" << EOF
PYTHONPATH="$PWD"
DATA_PATH="$PWD/data"
OUTPUTS_PATH="$PWD/outputs"
CHECKPOINTS_PATH="$PWD/outputs/checkpoints"
MLRUNS_PATH="$PWD/outputs/mlruns"
# Generated at: $(date)
EOF

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

