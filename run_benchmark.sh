#!/bin/bash

# Run FlexAction vs CGA benchmark in conda environment
# This script runs the real GPU benchmark with proper environment setup

echo "=========================================="
echo "FlexAction vs CGA Real GPU Benchmark"
echo "=========================================="

# Activate conda environment
source /root/miniconda3/etc/profile.d/conda.sh
conda activate llmulator

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export TVM_NUM_THREADS=16
export OMP_NUM_THREADS=16
export PYTHONPATH=/root/Heron:/root/Heron104:$PYTHONPATH

# Check environment
echo "Environment Check:"
echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader -i 0)"
echo ""

# Run small test first
echo "Running small GEMM test (20 trials)..."
echo "=========================================="
python /root/Heron104/benchmark_flexaction_vs_cga.py \
    --workload gemm \
    --trials 20 \
    --method both

echo ""
echo "=========================================="
echo "Benchmark completed!"
echo "Results saved in: /root/Heron104/benchmark_results/"
echo "=========================================="