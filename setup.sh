#!/usr/bin/env bash

# One-time setup: creates the venv, installs dependencies, seeds config
# files, and checks external dependencies (Docker/Ollama). Run this once
# (and again any time requirements.txt changes) — then use ./run.sh for
# every actual modernization run.

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo "========================================="
echo "   Setting up Code Modernizer Engine     "
echo "========================================="

# 1. Setup Virtual Environment
if [ ! -d ".venv" ]; then
    echo "=> Creating Python virtual environment in .venv..."
    python3 -m venv .venv
fi

# 2. Activate Virtual Environment
source .venv/bin/activate

# 3. Install Requirements
echo "=> Installing Python dependencies..."
pip install -q -r requirements.txt

# 4. Setup Config Files
if [ ! -f ".env" ]; then
    echo "=> Copying .env.example to .env..."
    cp .env.example .env
fi

if [ ! -f ".modernizer.toml" ]; then
    echo "=> Copying .modernizer.toml.example to .modernizer.toml..."
    cp .modernizer.toml.example .modernizer.toml
fi

# 5. Check external dependencies (Docker & Ollama)
if ! docker info >/dev/null 2>&1; then
    echo "❌ ERROR: Docker does not appear to be running. Please start Docker Desktop/Engine and try again."
    exit 1
fi

if ! command -v ollama &> /dev/null; then
    echo "⚠️  WARNING: Ollama is not installed or not in PATH."
    echo "   Download it from https://ollama.com/ to use local models."
fi

echo "========================================="
echo "   Setup complete.                       "
echo "   Edit .env / .modernizer.toml as needed, then run:"
echo "   ./run.sh <path_to_file_or_directory> [flags]"
echo "========================================="
