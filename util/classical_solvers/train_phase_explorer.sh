#!/bin/bash
# =============================================================================
# BCAO Honeycomb Phase Explorer Training Script
# =============================================================================
#
# This script runs the active learning phase exploration workflow.
#
# Usage:
#   ./train_phase_explorer.sh [MODE] [OPTIONS]
#
# Modes:
#   quick     - Fast test run (2 min): 4 initial + 1 iteration
#   short     - Short exploration (10 min): 10 initial + 5 iterations  
#   medium    - Medium exploration (1 hour): 20 initial + 20 iterations
#   full      - Full exploration (4+ hours): 50 initial + 50 iterations
#   custom    - Use custom options passed after mode
#
# Examples:
#   ./train_phase_explorer.sh quick
#   ./train_phase_explorer.sh short --output-dir my_results
#   ./train_phase_explorer.sh custom --n-initial 30 --n-iterations 15
#
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default output directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DEFAULT_OUTPUT="${PROJECT_ROOT}/AL_exploration_${TIMESTAMP}"

# =============================================================================
# Functions
# =============================================================================

print_banner() {
    echo -e "${BLUE}"
    echo "=============================================================="
    echo "  BCAO Honeycomb Phase Explorer"
    echo "  Active Learning for Magnetic Phase Discovery"
    echo "=============================================================="
    echo -e "${NC}"
}

print_usage() {
    echo "Usage: $0 [MODE] [OPTIONS]"
    echo ""
    echo "Modes:"
    echo "  quick   - Fast test (4 initial, 1 iteration, screening mode)"
    echo "  short   - Short run (10 initial, 5 iterations, fast mode)"
    echo "  medium  - Medium run (20 initial, 20 iterations, fast mode)"
    echo "  full    - Full run (50 initial, 50 iterations, accurate mode)"
    echo "  custom  - Pass your own options"
    echo ""
    echo "Common options (pass after mode):"
    echo "  --output-dir DIR      Output directory"
    echo "  --fresh-start         Start fresh (ignore existing data)"
    echo "  --disable-lswt        Skip LSWT pre-screening"
    echo "  --n-jobs N            Number of parallel jobs"
    echo "  --seed N              Random seed for reproducibility"
    echo ""
    echo "Examples:"
    echo "  $0 quick"
    echo "  $0 short --output-dir ./my_exploration"
    echo "  $0 medium --disable-lswt --n-jobs 4"
    echo "  $0 custom --n-initial 15 --n-iterations 10 --screening"
}

check_dependencies() {
    echo -e "${YELLOW}Checking dependencies...${NC}"
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}Error: python3 not found${NC}"
        exit 1
    fi
    
    # Check spin_solver
    SOLVER="${PROJECT_ROOT}/build/spin_solver"
    if [[ ! -x "$SOLVER" ]]; then
        echo -e "${RED}Error: spin_solver not found at ${SOLVER}${NC}"
        echo "Please build the project first: cd build && cmake .. && make"
        exit 1
    fi
    
    # Check required Python modules
    python3 -c "import numpy" 2>/dev/null || {
        echo -e "${RED}Error: numpy not installed${NC}"
        exit 1
    }
    
    echo -e "${GREEN}✓ All dependencies satisfied${NC}"
}

run_exploration() {
    local mode=$1
    shift
    local extra_args="$@"
    
    cd "$SCRIPT_DIR"
    
    case $mode in
        quick)
            echo -e "${YELLOW}Running QUICK exploration (~2 minutes)${NC}"
            echo "  - 4 initial LHS samples + 3 known seeds"
            echo "  - 1 iteration with 2 new points"
            echo "  - Screening mode (L=8, minimal MC steps)"
            echo ""
            python3 run_active_learning.py \
                --n-initial 4 \
                --n-iterations 1 \
                --n-per-iteration 2 \
                --screening \
                --disable-lswt \
                --output-dir "${DEFAULT_OUTPUT}" \
                $extra_args
            ;;
            
        short)
            echo -e "${YELLOW}Running SHORT exploration (~10 minutes)${NC}"
            echo "  - 10 initial LHS samples + 3 known seeds"
            echo "  - 5 iterations with 3 new points each"
            echo "  - Fast mode (L=12, moderate MC steps)"
            echo ""
            python3 run_active_learning.py \
                --n-initial 10 \
                --n-iterations 5 \
                --n-per-iteration 3 \
                --fast-mode \
                --disable-lswt \
                --output-dir "${DEFAULT_OUTPUT}" \
                $extra_args
            ;;
            
        medium)
            echo -e "${YELLOW}Running MEDIUM exploration (~1 hour)${NC}"
            echo "  - 20 initial LHS samples + 3 known seeds"
            echo "  - 20 iterations with 5 new points each"
            echo "  - Fast mode (L=12)"
            echo ""
            python3 run_active_learning.py \
                --n-initial 20 \
                --n-iterations 20 \
                --n-per-iteration 5 \
                --fast-mode \
                --output-dir "${DEFAULT_OUTPUT}" \
                $extra_args
            ;;
            
        full)
            echo -e "${YELLOW}Running FULL exploration (~4+ hours)${NC}"
            echo "  - 50 initial LHS samples + 3 known seeds"
            echo "  - 50 iterations with 5 new points each"
            echo "  - Accurate mode (L=16, extensive MC)"
            echo ""
            python3 run_active_learning.py \
                --n-initial 50 \
                --n-iterations 50 \
                --n-per-iteration 5 \
                --accurate \
                --output-dir "${DEFAULT_OUTPUT}" \
                $extra_args
            ;;
            
        custom)
            echo -e "${YELLOW}Running CUSTOM exploration${NC}"
            echo "  Options: $extra_args"
            echo ""
            python3 run_active_learning.py \
                --output-dir "${DEFAULT_OUTPUT}" \
                $extra_args
            ;;
            
        *)
            echo -e "${RED}Unknown mode: $mode${NC}"
            print_usage
            exit 1
            ;;
    esac
}

# =============================================================================
# Main
# =============================================================================

print_banner

# Check for help
if [[ "$1" == "-h" || "$1" == "--help" || -z "$1" ]]; then
    print_usage
    exit 0
fi

MODE=$1
shift

# Override output dir if provided
for arg in "$@"; do
    if [[ "$prev_arg" == "--output-dir" ]]; then
        DEFAULT_OUTPUT="$arg"
    fi
    prev_arg="$arg"
done

echo "Output directory: ${DEFAULT_OUTPUT}"
echo ""

check_dependencies

echo ""
echo -e "${GREEN}Starting exploration...${NC}"
echo "=============================================================="
echo ""

run_exploration "$MODE" "$@"

echo ""
echo "=============================================================="
echo -e "${GREEN}Exploration complete!${NC}"
echo "Results saved to: ${DEFAULT_OUTPUT}"
echo ""
echo "To analyze results:"
echo "  python3 analyze_exploration.py ${DEFAULT_OUTPUT}"
echo ""
