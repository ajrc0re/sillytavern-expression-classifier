#!/usr/bin/env bash
# Bootstrap the Python environment for sillytavern-expression-classifier.
# Defaults to the CUDA 13.0 PyTorch wheel index; override in setup_config.sh if needed.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${PROJECT_ROOT}/.venv"
PYTHON_BIN="${PYTHON:-python3}"
SETUP_CONFIG_FILE="${PROJECT_ROOT}/setup_config.sh"
DEFAULT_PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu130"
PYTORCH_INDEX_URL="${DEFAULT_PYTORCH_INDEX_URL}"

# shellcheck disable=SC1090
if [ -f "${SETUP_CONFIG_FILE}" ]; then
  # Allow overriding defaults via setup_config.sh.
  source "${SETUP_CONFIG_FILE}"
fi

PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-${DEFAULT_PYTORCH_INDEX_URL}}"

# Ensure the requested Python interpreter is available before continuing.
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "error: ${PYTHON_BIN} is not installed or not on PATH" >&2
  exit 1
fi

# Create a project-local virtual environment on the first run.
if [ ! -d "${VENV_PATH}" ]; then
  echo "Creating virtual environment at ${VENV_PATH}"
  "${PYTHON_BIN}" -m venv "${VENV_PATH}"
fi

# shellcheck disable=SC1090
# Activate the environment so subsequent installs stay self-contained.
source "${VENV_PATH}/bin/activate"

# Refresh packaging tools to avoid installer edge cases.
python -m pip install --upgrade pip setuptools wheel

echo "Installing torch/torchvision using index: ${PYTORCH_INDEX_URL}"
if ! python -m pip install torch torchvision --index-url "${PYTORCH_INDEX_URL}"; then
  echo "warning: Unable to get torch/torchvision from ${PYTORCH_INDEX_URL}. Falling back to PyPI." >&2
  python -m pip install torch torchvision
fi
# Install remaining dependencies defined by the project.
python -m pip install -r "${PROJECT_ROOT}/requirements.txt"

echo "Environment ready. Activate with: source ${VENV_PATH}/bin/activate"
