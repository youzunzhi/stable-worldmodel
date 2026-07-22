set -euo pipefail

EXPERIMENTS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=paths.env
source "$EXPERIMENTS_DIR/paths.env"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

run_python() {
    local script_name=$1
    shift

    local python="$ENV_ROOT/bin/python"
    if [[ ! -x "$python" ]]; then
        echo "Python environment not found: $python" >&2
        exit 1
    fi

    if [[ ! -f "$EXPERIMENTS_DIR/$script_name" ]]; then
        echo "Experiment script not found: $EXPERIMENTS_DIR/$script_name" >&2
        exit 1
    fi

    exec "$python" "$EXPERIMENTS_DIR/$script_name" "$@"
}
