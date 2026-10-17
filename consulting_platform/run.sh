#!/bin/bash
# One-command runner. Creates the virtual environment on first use and always uses the project's own Python.
#   ./run.sh app | check | fetch | index | eval | eval-llm | cli "question"
cd "$(dirname "$0")" || exit 1
PY=".venv/bin/python"

if [ ! -x "$PY" ]; then
  echo "[setup] creating virtual environment (.venv) and installing requirements - first run only ..."
  PYTHON_BIN=$(command -v python3.12 || command -v python3.11 || command -v python3)
  "$PYTHON_BIN" -m venv .venv || { echo "Could not create .venv - install Python 3.10+ from python.org"; exit 1; }
  "$PY" -m pip install --upgrade pip -q
  "$PY" -m pip install -r requirements.txt || { echo "pip install failed"; exit 1; }
fi

case "$1" in
  app|"")   $PY -m streamlit run app.py ;;
  check)    $PY check_llm.py ;;
  fetch)    $PY fetch_sources.py ;;
  index)    $PY build_index.py ;;
  setup)    $PY build_index.py && $PY check_llm.py ;;
  eval)     $PY evaluate.py ;;
  eval-llm) $PY evaluate.py --with-llm --agent-queries 3 --planner-limit 15 ;;
  cli)      shift; $PY cli.py "$@" ;;
  *)        echo "usage: ./run.sh [app] | check | fetch | index | setup | eval | eval-llm | cli \"question\"" ;;
esac
