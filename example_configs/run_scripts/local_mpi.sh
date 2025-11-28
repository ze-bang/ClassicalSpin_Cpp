#!/bin/bash
# Local MPI Run Script for Parallel Tempering or Field Scans
# For running MPI jobs on a local workstation
# Usage: ./local_mpi.sh <num_processes> <config_file>

if [ $# -ne 2 ]; then
    echo "Usage: $0 <num_processes> <config_file>"
    echo "Example: $0 8 ../BCAO/pt_emily.param"
    exit 1
fi

NUM_PROCS=$1
CONFIG_FILE=$2
BUILD_DIR="../../build"
EXECUTABLE="${BUILD_DIR}/unified_simulation"

# Check if executable exists
if [ ! -f "$EXECUTABLE" ]; then
    echo "Error: Executable not found at $EXECUTABLE"
    exit 1
fi

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

CONFIG_ABS=$(realpath "$CONFIG_FILE")
EXEC_ABS=$(realpath "$EXECUTABLE")

echo "======================================"
echo "  MPI Parallel Simulation"
echo "======================================"
echo "Processes:  $NUM_PROCS"
echo "Config:     $CONFIG_ABS"
echo "Executable: $EXEC_ABS"
echo "Start time: $(date)"
echo "======================================"

# Run with MPI
time mpirun -np $NUM_PROCS "$EXEC_ABS" "$CONFIG_ABS"

EXIT_CODE=$?

echo "======================================"
echo "MPI job finished"
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
echo "======================================"

exit $EXIT_CODE
