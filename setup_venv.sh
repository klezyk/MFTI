#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "Создание venv с Python 3.11..."
/opt/homebrew/bin/python3.11 -m venv .venv

echo "Активация venv..."
source .venv/bin/activate

echo "Обновление pip..."
pip install --upgrade pip

echo "Установка зависимостей..."
pip install -r requirements.txt

echo "Регистрация Jupyter kernel..."
python -m ipykernel install --user --name mfti --display-name "MFTI DLS"

echo ""
echo "=== venv готов ==="
echo "Активация: source .venv/bin/activate"
echo "Jupyter kernel: MFTI DLS"
