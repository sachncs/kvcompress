#!/usr/bin/env bash
# setup.sh — create a development venv and install kvfold + optional extras.
#
# Idempotent: re-running with an existing .venv reuses it. Pass
# --recreate to delete and rebuild from scratch. Pass --no-vllm to skip
# the heavy vLLM install (the integration tests then stay skipped).
#
# Usage:
#   ./scripts/setup.sh                # default: venv at .venv/, installs all extras
#   ./scripts/setup.sh --no-vllm      # skip the ~5 GB vLLM install
#   ./scripts/setup.sh --recreate     # nuke .venv/ and rebuild
#
# Exit codes:
#   0 success
#   1 missing python3
#   2 venv creation failed
#   3 pip install failed

set -euo pipefail

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

RECREATE=0
INSTALL_VLLM=1
PYTHON_BIN="${PYTHON_BIN:-python3}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --recreate) RECREATE=1; shift ;;
        --no-vllm)  INSTALL_VLLM=0; shift ;;
        --python)   PYTHON_BIN="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,12p' "$0"
            exit 0
            ;;
        *)  echo "unknown argument: $1" >&2; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Locate the repo root and target venv
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV="${REPO_ROOT}/.venv"
VENV_INFO="${REPO_ROOT}/.venv-info"

cd "${REPO_ROOT}"

# ---------------------------------------------------------------------------
# Validate python version (3.11+ matches pyproject's requires-python)
# ---------------------------------------------------------------------------

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "error: ${PYTHON_BIN} not found on PATH" >&2
    exit 1
fi

PY_VERSION="$(${PYTHON_BIN} -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
PY_MAJOR="${PY_VERSION%%.*}"
PY_MINOR="${PY_VERSION##*.}"

if [[ "${PY_MAJOR}" -lt 3 || ( "${PY_MAJOR}" -eq 3 && "${PY_MINOR}" -lt 11 ) ]]; then
    echo "error: python ${PY_VERSION} is too old; kvfold requires 3.11+" >&2
    exit 1
fi

echo "==> using python ${PY_VERSION}"

# ---------------------------------------------------------------------------
# (Re)create the venv
# ---------------------------------------------------------------------------

if [[ "${RECREATE}" -eq 1 && -d "${VENV}" ]]; then
    echo "==> removing existing venv at ${VENV}"
    rm -rf "${VENV}"
fi

if [[ ! -d "${VENV}" ]]; then
    echo "==> creating venv at ${VENV}"
    if ! "${PYTHON_BIN}" -m venv "${VENV}"; then
        echo "error: venv creation failed" >&2
        exit 2
    fi
fi

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
PY_BIN="${VENV}/bin/python"

# ---------------------------------------------------------------------------
# Upgrade pip + install the package
# ---------------------------------------------------------------------------

echo "==> upgrading pip"
"${PY_BIN}" -m pip install --upgrade pip wheel setuptools >/dev/null

echo "==> installing torch (CPU build)"
"${PY_BIN}" -m pip install --index-url https://download.pytorch.org/whl/cpu \
    torch torchvision >/dev/null

EXTRA="dev"
if [[ "${INSTALL_VLLM}" -eq 1 ]]; then
    EXTRA="${EXTRA},vllm"
fi

echo "==> installing kvfold with extras: ${EXTRA}"
if ! "${PY_BIN}" -m pip install -e ".[${EXTRA}]"; then
    echo "error: pip install failed" >&2
    exit 3
fi

# ---------------------------------------------------------------------------
# Persist environment metadata for debugging
# ---------------------------------------------------------------------------

cat > "${VENV_INFO}" <<EOF
python=${PY_VERSION}
pip=$("${PY_BIN}" -m pip --version | awk '{print $2}')
torch=$("${PY_BIN}" -c 'import torch; print(torch.__version__)' 2>/dev/null || echo missing)
vllm=$("${PY_BIN}" -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo missing)
installed=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
EOF

echo "==> done"
echo "    activate with:  source ${VENV}/bin/activate"
echo "    run tests with:  ${VENV}/bin/python -m pytest tests/unit tests/property"
echo "    env info:       ${VENV_INFO}"