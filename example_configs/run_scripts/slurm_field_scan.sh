#!/bin/bash
#SBATCH --job-name=spin_field_scan
#SBATCH --output=field_scan_%j.out
#SBATCH --error=field_scan_%j.err
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=standard

# SLURM Job Script for Magnetic Field Scan with MPI
# Each MPI rank handles different field values in parallel
# Usage: sbatch slurm_field_scan.sh <config_file>
# Example: sbatch slurm_field_scan.sh ../Pyrochlore/field_scan.param

# Load required modules
# module load gcc/11.2.0
# module load openmpi/4.1.2
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
echo "SLURM Field Scan Job"
echo "======================================"
echo "Job ID:       $SLURM_JOB_ID"
echo "Total tasks:  $SLURM_NTASKS"
echo "Start time:   $(date)"
echo "Config:       $CONFIG_ABS"
echo "======================================"

# Run field scan - MPI distributes field values
time mpirun -np $SLURM_NTASKS "$EXEC_ABS" "$CONFIG_ABS"

EXIT_CODE=$?

echo "======================================"
echo "Field scan finished"
echo "End time:   $(date)"
echo "Exit code:  $EXIT_CODE"
echo "======================================"

exit $EXIT_CODE
