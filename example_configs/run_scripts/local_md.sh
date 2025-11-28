#!/bin/bash
# Local Molecular Dynamics Run Script
# For running MD simulations on a local workstation
# Usage: ./local_md.sh <config_file>

# Check arguments
if [ $# -ne 1 ]; then
    echo "Usage: $0 <config_file>"
    echo "Example: $0 ../BCAO/md_emily.param"
    exit 1
fi

CONFIG_FILE=$1
BUILD_DIR="../../build"
EXECUTABLE="${BUILD_DIR}/unified_simulation"

# Check if executable exists
if [ ! -f "$EXECUTABLE" ]; then
    echo "Error: Executable not found at $EXECUTABLE"
    echo "Please build the project first:"
    echo "  cd ../../build && cmake .. && make -j\$(nproc)"
    exit 1
fi

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Get absolute paths
CONFIG_ABS=$(realpath "$CONFIG_FILE")
EXEC_ABS=$(realpath "$EXECUTABLE")

echo "======================================"
echo "  Classical Spin MD Simulation"
echo "======================================"
echo "Config:     $CONFIG_ABS"
echo "Executable: $EXEC_ABS"
echo "Start time: $(date)"
echo "======================================"

# Run simulation
time "$EXEC_ABS" "$CONFIG_ABS"

EXIT_CODE=$?

echo "======================================"
echo "Simulation finished"
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
echo "======================================"

exit $EXIT_CODE
