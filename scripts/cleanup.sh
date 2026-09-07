#!/usr/bin/env bash
# cleanup.sh — remove generated artefacts and tear down the dev venv.
#
# Default: interactive (asks before each top-level category). Pass
# --force to skip the prompts; pass --keep-venv to skip the .venv/
# removal only.
#
# Usage:
#   ./scripts/cleanup.sh               # interactive
#   ./scripts/cleanup.sh --force       # non-interactive
#   ./scripts/cleanup.sh --keep-venv   # keep .venv/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

FORCE=0
KEEP_VENV=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)     FORCE=1; shift ;;
        --keep-venv) KEEP_VENV=1; shift ;;
        -h|--help)
            sed -n '2,11p' "$0"
            exit 0
            ;;
        *)  echo "unknown argument: $1" >&2; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

confirm() {
    local prompt="$1"
    if [[ "${FORCE}" -eq 1 ]]; then
        return 0
    fi
    local reply
    read -r -p "${prompt} [y/N] " reply
    [[ "${reply}" =~ ^[Yy]([Ee][Ss])?$ ]]
}

remove_paths() {
    local label="$1"; shift
    local found=0
    for path in "$@"; do
        if [[ -e "${path}" ]]; then
            echo "==> removing ${path}"
            rm -rf "${path}"
            found=1
        fi
    done
    if [[ "${found}" -eq 0 ]]; then
        echo "==> (no ${label} to remove)"
    fi
}

# ---------------------------------------------------------------------------
# Categories
# ---------------------------------------------------------------------------

echo "==> Cleaning kvpress generated artefacts (repo root: ${REPO_ROOT})"

if confirm "Remove Python caches (__pycache__/, *.pyc, .mypy_cache/, .ruff_cache/)?"; then
    remove_paths "Python caches" \
        $(find . -type d -name '__pycache__' -not -path './.venv/*' -not -path './.git/*' 2>/dev/null) \
        .mypy_cache \
        .ruff_cache
    find . -type f -name '*.pyc' -not -path './.venv/*' -not -path './.git/*' -delete 2>/dev/null || true
fi

if confirm "Remove test/coverage artefacts (.coverage, .hypothesis, htmlcov/, .pytest_cache/, .benchmarks/)?"; then
    remove_paths "test artefacts" \
        .coverage \
        .coverage.* \
        htmlcov \
        .pytest_cache \
        .hypothesis \
        .benchmarks
fi

if confirm "Remove build artefacts (dist/, build/, *.egg-info/, results/, benchmarks/output/, benchmarks/plots/)?"; then
    remove_paths "build artefacts" \
        dist \
        build \
        results \
        benchmarks/output \
        benchmarks/plots \
        $(find . -maxdepth 2 -type d -name '*.egg-info' -not -path './.venv/*' 2>/dev/null)
fi

if [[ "${KEEP_VENV}" -eq 0 ]]; then
    if confirm "Remove .venv/ and .venv-info?"; then
        remove_paths "venv" .venv .venv-info
    fi
else
    echo "==> keeping .venv/ (--keep-venv)"
fi

echo "==> cleanup complete"