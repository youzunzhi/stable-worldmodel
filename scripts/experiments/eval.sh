#!/usr/bin/env bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=runtime.sh
source "$SCRIPT_DIR/runtime.sh"
run_python eval.py "$@"
