#!/bin/bash
set -e

echo "=== ARCH_AGENT Setup ==="

# Python dependencies
echo ""
echo "[1/3] Installing Python dependencies..."
pip3 install -r requirements.txt

# Ollama
echo ""
echo "[2/3] Checking Ollama..."
if ! command -v ollama &> /dev/null; then
    echo "  Ollama not found. Installing via Homebrew (macOS)..."
    brew install ollama
else
    echo "  Ollama already installed: $(ollama --version)"
fi

# Models
echo ""
echo "[3/3] Pulling required models (this may take several minutes)..."
echo "  Pulling deepseek-r1:14b (~9 GB)..."
ollama pull deepseek-r1:14b

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Quick start:"
echo "  cd 01_stage_analyze"
echo "  python3 LLM_frontend_upgraded.py \"analyze https://github.com/apache/commons-io.git all-time 5 timesteps with deepseek-r1:14b and answer: how did the architecture evolve?\""
