#!/bin/bash
# =============================================================================
# MERGE ENGINE GPU RUNNER
# =============================================================================
# Runs Python scripts with GPU (cuDF/RAPIDS) support via WSL
# GPU execution is MANDATORY - No CPU fallback (ALGO 95.4)
#
# Usage: ./run_gpu.sh [script.py]
# Default: ./run_gpu.sh intelligent_merge_engine_v27_gpu.py
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}=====================================================${NC}"
echo -e "${CYAN}     MERGE ENGINE GPU RUNNER (ALGO 95.4)             ${NC}"
echo -e "${CYAN}=====================================================${NC}"

# GPU Environment
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
export NUMBA_CUDA_USE_NVIDIA_BINDING=1
export NUMBA_CUDA_DRIVER=/usr/lib/wsl/lib/libcuda.so.1
export CUDA_VISIBLE_DEVICES=0

# Verify GPU
echo -e "\n${YELLOW}[CHECK]${NC} Verifying GPU availability..."
if ! nvidia-smi &> /dev/null; then
    echo -e "${RED}[FATAL]${NC} NVIDIA GPU not detected via nvidia-smi"
    echo -e "${RED}[FATAL]${NC} GPU is MANDATORY per ALGO 95.4 - No CPU fallback allowed"
    exit 1
fi

# Show GPU info
echo -e "${GREEN}[GPU]${NC} NVIDIA GPU detected:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# Check for cuDF
echo -e "\n${YELLOW}[CHECK]${NC} Verifying cuDF/RAPIDS..."
if ! python3 -c "import cudf; print(f'cuDF {cudf.__version__} OK')" 2>/dev/null; then
    echo -e "${RED}[FATAL]${NC} cuDF not available"
    echo -e "${RED}[FATAL]${NC} Install RAPIDS: conda install -c rapidsai cudf"
    exit 1
fi

echo -e "${GREEN}[OK]${NC} GPU environment ready"

# Script selection
SCRIPT=${1:-intelligent_merge_engine_v27_gpu.py}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/${SCRIPT}"

if [[ ! -f "$SCRIPT_PATH" ]]; then
    echo -e "${RED}[ERROR]${NC} Script not found: $SCRIPT_PATH"
    exit 1
fi

# Run
echo -e "\n${CYAN}=====================================================${NC}"
echo -e "${CYAN}     EXECUTING: ${SCRIPT}${NC}"
echo -e "${CYAN}=====================================================${NC}\n"

cd "$SCRIPT_DIR"
python3 "$SCRIPT_PATH"

echo -e "\n${GREEN}[DONE]${NC} Merge Engine completed"
