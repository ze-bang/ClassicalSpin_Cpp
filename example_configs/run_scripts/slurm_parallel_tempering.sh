#!/bin/bash
#SBATCH --job-name=spin_pt
#SBATCH --output=spin_pt_%j.out
#SBATCH --error=spin_pt_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=2
#SBATCH --ntasks=48
#SBATCH --ntasks-per-node=24
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=standard

# SLURM Job Script for Parallel Tempering with MPI
# Each MPI rank handles one temperature replica
# Usage: sbatch slurm_parallel_tempering.sh <config_file>
# Example: sbatch slurm_parallel_tempering.sh ../BCAO/pt_emily.param

# Load required modules (adjust for your cluster)
# module load gcc/11.2.0
# module load openmpi/4.1.2
# module load hdf5/1.12.1
# module load eigen/3.4.0
# module load boost/1.78.0

CONFIG_FILE=$1
BUILD_DIR="../../build"
EXECUTABLE="${BUILD_DIR}/unified_simulation"

# Check arguments
if [ -z "$CONFIG_FILE" ]; then
    echo "Error: No config file specified"
    echo "Usage: sbatch slurm_parallel_tempering.sh <config_file>"
    exit 1
fi

# Get absolute paths
CONFIG_ABS=$(realpath "$CONFIG_FILE")
EXEC_ABS=$(realpath "$EXECUTABLE")

# Print job information
echo "======================================"
echo "SLURM Parallel Tempering Job"
echo "======================================"
echo "Job ID:       $SLURM_JOB_ID"
echo "Job name:     $SLURM_JOB_NAME"
echo "Nodes:        $SLURM_NNODES"
echo "Total tasks:  $SLURM_NTASKS"
echo "Tasks/node:   $SLURM_NTASKS_PER_NODE"
echo "Node list:    $SLURM_NODELIST"
echo "Start time:   $(date)"
echo "Config:       $CONFIG_ABS"
echo "Executable:   $EXEC_ABS"
echo "======================================"
echo ""
echo "NOTE: Number of MPI ranks ($SLURM_NTASKS) should match"
echo "      the number of temperature replicas in the config file"
echo ""

# Run parallel tempering with MPI
time mpirun -np $SLURM_NTASKS "$EXEC_ABS" "$CONFIG_ABS"

EXIT_CODE=$?

echo "======================================"
echo "Parallel Tempering job finished"
echo "End time:   $(date)"
echo "Exit code:  $EXIT_CODE"
echo "======================================"

exit $EXIT_CODE
