#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${1:-${REPO_ROOT}/.venv_eeg}"

if [[ -e "${VENV_PATH}" ]]; then
    echo "Refusing to overwrite existing path: ${VENV_PATH}" >&2
    echo "Pass a new path or remove the old environment yourself." >&2
    exit 1
fi

python3.10 -m venv "${VENV_PATH}"
"${VENV_PATH}/bin/python" -m pip install --upgrade pip
(
    cd "${REPO_ROOT}"
    "${VENV_PATH}/bin/python" -m pip install -r requirements-eeg.txt
)

echo
echo "EEG DAC environment created at ${VENV_PATH}"
echo "Activate it with: source ${VENV_PATH}/bin/activate"
