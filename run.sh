#!/usr/bin/env bash

# Runs the modernization engine. Assumes ./setup.sh has already been run
# once (venv exists, dependencies installed, .env/.modernizer.toml present).

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

if [ ! -d ".venv" ]; then
    echo "❌ ERROR: .venv not found. Run ./setup.sh first."
    exit 1
fi

source .venv/bin/activate

if [ $# -eq 0 ]; then
    echo "❌ ERROR: You must provide a target path to modernize."
    echo "Usage: ./run.sh <path_to_file_or_directory> [additional flags]"
    echo "Example: ./run.sh ../my-legacy-project --pr"
    exit 1
fi

python main.py "$@"
