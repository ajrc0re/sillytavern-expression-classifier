#!/usr/bin/env bash
# Bootstrap the Python environment for sillytavern-expression-classifier.
# Defaults to the CUDA 13.0 PyTorch wheel index; override in setup_config.sh if needed.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${PROJECT_ROOT}/.venv"
PYTHON_BIN="${PYTHON:-}"
SETUP_CONFIG_FILE="${PROJECT_ROOT}/setup_config.sh"
DEFAULT_PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu130"
PYTORCH_INDEX_URL="${DEFAULT_PYTORCH_INDEX_URL}"

# shellcheck disable=SC1090
if [ -f "${SETUP_CONFIG_FILE}" ]; then
  # Allow overriding defaults via setup_config.sh.
  source "${SETUP_CONFIG_FILE}"
fi

PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-${DEFAULT_PYTORCH_INDEX_URL}}"

# Resolve an available Python interpreter (prefer env override -> python3 -> python).
if [ -n "${PYTHON_BIN}" ]; then
  if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "error: Requested python interpreter '${PYTHON_BIN}' is not on PATH" >&2
    exit 1
  fi
else
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "error: Unable to locate a python interpreter (tried python3, python)" >&2
    exit 1
  fi
fi

# Create a project-local virtual environment on the first run.
if [ ! -d "${VENV_PATH}" ]; then
  echo "Creating virtual environment at ${VENV_PATH}"
  "${PYTHON_BIN}" -m venv "${VENV_PATH}"
fi

# shellcheck disable=SC1090
# Activate the environment so subsequent installs stay self-contained.
if [ -f "${VENV_PATH}/bin/activate" ]; then
  source "${VENV_PATH}/bin/activate"
elif [ -f "${VENV_PATH}/Scripts/activate" ]; then
  # Windows virtualenv layout.
  source "${VENV_PATH}/Scripts/activate"
else
  echo "error: Unable to locate virtual environment activation script in ${VENV_PATH}" >&2
  exit 1
fi

# Refresh packaging tools to avoid installer edge cases.
python -m pip install --upgrade pip setuptools wheel

echo "Installing torch/torchvision using index: ${PYTORCH_INDEX_URL}"
if ! python -m pip install torch torchvision --index-url "${PYTORCH_INDEX_URL}"; then
  echo "warning: Unable to get torch/torchvision from ${PYTORCH_INDEX_URL}. Falling back to PyPI." >&2
  python -m pip install torch torchvision
fi
# Install remaining dependencies defined by the project.
python -m pip install -r "${PROJECT_ROOT}/requirements.txt"

"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe" -ExecutionPolicy Bypass -File "${PROJECT_ROOT}/install-make.ps1"

echo "Environment ready. Activate with: source ${VENV_PATH}/bin/activate"
