#!/bin/bash

# Exit on error
set -e

echo "================================================================="
echo "   HiReF - Hierarchical Regression Forecasting Startup Script"
echo "================================================================="

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "Error: 'uv' is not installed. Please install it first."
    echo "Visit https://github.com/astral-sh/uv for installation instructions."
    exit 1
fi

echo "[1/6] Creating virtual environment..."
uv venv

echo "[2/6] Installing dependencies..."
source .venv/bin/activate
uv pip install -r requirements.txt

echo "[3/6] Creating directories..."
mkdir -p sample_data outputs/results

echo "[4/6] Generating data..."
python data_generator/data_generator.py

echo "[5/6] Training model..."
python scripts/train_model.py

echo "[6/6] Analyzing results..."
python scripts/analyze_results.py

echo "================================================================="
echo "   Setup Complete! Launching Dashboard..."
echo "================================================================="
python app/app.py
