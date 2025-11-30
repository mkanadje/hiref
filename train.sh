#!/bin/bash

# Exit on error
set -e

echo "================================================================="
echo "   HiReF - Hierarchical Regression Forecasting Startup Script"
echo "================================================================="

echo "[1/2] Training model..."
python scripts/train_model.py

echo "[2/2] Analyzing results..."
python scripts/analyze_results.py

echo "================================================================="
echo "   Setup Complete! Launching Dashboard..."
echo "================================================================="
python app/app.py
