#!/bin/bash
#
# Convenience wrapper to run active learning with the virtual environment
# and parallel processing enabled.
#
# Usage: ./run_parallel_active_learning.sh [args...]
#

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/../.."
VENV_PATH="$PROJECT_ROOT/pystandard"

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo "Error: Virtual environment not found at $VENV_PATH"
    echo "Please create it with: python3 -m venv $VENV_PATH"
    exit 1
fi

# Activate virtual environment and run
source "$VENV_PATH/bin/activate"

echo "=================================================="
echo "Active Learning with Parallel Processing"
echo "=================================================="
echo "Virtual environment: $VENV_PATH"
echo "Python: $(which python)"
echo "CPUs available: $(nproc)"
echo "=================================================="
echo ""

# Run the active learning script with all provided arguments
python "$SCRIPT_DIR/run_active_learning.py" "$@"

EXIT_CODE=$?

# Deactivate virtual environment
deactivate

exit $EXIT_CODE
