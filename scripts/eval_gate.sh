#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

: "${OFFLINE_EVAL:=0}"
: "${STRICT_EVAL:=1}"

export OFFLINE_EVAL STRICT_EVAL

if [[ -x .venv/bin/python ]]; then
  PYTHON=.venv/bin/python
else
  PYTHON=python
fi

echo "Running retrieval gate with OFFLINE_EVAL=${OFFLINE_EVAL} STRICT_EVAL=${STRICT_EVAL}"
"${PYTHON}" scripts/eval_smoke.py
