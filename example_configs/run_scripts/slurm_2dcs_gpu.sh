#!/bin/bash
#SBATCH --job-name=spin_2dcs
#SBATCH --output=2dcs_%j.out
#SBATCH --error=2dcs_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=gpu

# SLURM Job Script for 2DCS with GPU Acceleration
# Usage: sbatch slurm_2dcs_gpu.sh <config_file>
# Example: sbatch slurm_2dcs_gpu.sh ../TmFeO3/2dcs_tmfeo3.param

# Load required modules
# module load gcc/11.2.0
# module load cuda/11.7
# module load hdf5/1.12.1
# module load eigen/3.4.0
# module load boost/1.78.0

CONFIG_FILE=$1
BUILD_DIR="../../build"
EXECUTABLE="${BUILD_DIR}/unified_simulation"

if [ -z "$CONFIG_FILE" ]; then
    echo "Error: No config file specified"
    exit 1
fi

CONFIG_ABS=$(realpath "$CONFIG_FILE")
EXEC_ABS=$(realpath "$EXECUTABLE")

echo "======================================"
echo "SLURM 2DCS GPU Job"
echo "======================================"
echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $SLURM_NODELIST"
echo "GPU:          $CUDA_VISIBLE_DEVICES"
echo "Start time:   $(date)"
echo "Config:       $CONFIG_ABS"
echo "======================================"

# Check GPU availability
nvidia-smi

# Run 2DCS with GPU
time "$EXEC_ABS" "$CONFIG_ABS"

EXIT_CODE=$?

echo "======================================"
echo "2DCS simulation finished"
echo "End time:   $(date)"
echo "Exit code:  $EXIT_CODE"
echo "======================================"

exit $EXIT_CODE
