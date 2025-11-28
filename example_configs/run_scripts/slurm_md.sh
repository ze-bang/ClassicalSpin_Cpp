#!/bin/bash
#SBATCH --job-name=spin_md
#SBATCH --output=spin_md_%j.out
#SBATCH --error=spin_md_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=standard

# SLURM Job Script for Molecular Dynamics
# Usage: sbatch slurm_md.sh <config_file>
# Example: sbatch slurm_md.sh ../BCAO/md_emily.param

# Load required modules (adjust for your cluster)
# module load gcc/11.2.0
# module load hdf5/1.12.1
# module load eigen/3.4.0
# module load boost/1.78.0

CONFIG_FILE=$1
BUILD_DIR="../../build"
EXECUTABLE="${BUILD_DIR}/unified_simulation"

# Check arguments
if [ -z "$CONFIG_FILE" ]; then
    echo "Error: No config file specified"
    echo "Usage: sbatch slurm_md.sh <config_file>"
    exit 1
fi

# Get absolute paths
CONFIG_ABS=$(realpath "$CONFIG_FILE")
EXEC_ABS=$(realpath "$EXECUTABLE")

# Print job information
echo "======================================"
echo "SLURM Job Information"
echo "======================================"
echo "Job ID:       $SLURM_JOB_ID"
echo "Job name:     $SLURM_JOB_NAME"
echo "Node:         $SLURM_NODELIST"
echo "Cores:        $SLURM_CPUS_PER_TASK"
echo "Memory:       $SLURM_MEM_PER_NODE MB"
echo "Start time:   $(date)"
echo "Config:       $CONFIG_ABS"
echo "Executable:   $EXEC_ABS"
echo "======================================"

# Set OpenMP threads if using parallelization
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run simulation
time "$EXEC_ABS" "$CONFIG_ABS"

EXIT_CODE=$?

echo "======================================"
echo "Job finished"
echo "End time:   $(date)"
echo "Exit code:  $EXIT_CODE"
echo "======================================"

exit $EXIT_CODE
