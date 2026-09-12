#!/bin/bash
# ==============================================================================
# Run MNGCL Unified Benchmark on Server (10 Runs, 500 Epochs, Fixed Eval Protocol)
# Supports: clean_to_hit, hit_to_clean, or both (default: both)
# Generates 100% unified results for both Transductive and Strict Inductive modes
# ==============================================================================

set -e

PYTHON_EXEC="${PYTHON_EXEC:-python}"
GPU_ID="${1:-0}"
SPLIT="${2:-both}"  # options: both, clean_to_hit, hit_to_clean

echo "========================================================================"
echo "Running MNGCL Unified Benchmark (Protocol: Fixed 500 Epochs, Native Paper)"
echo "Split          : $SPLIT"
echo "Runs per split : 10"
echo "Device         : GPU $GPU_ID"
echo "Python         : $PYTHON_EXEC"
echo "========================================================================"

mkdir -p result

echo ""
echo ">>> [1/2] Running Transductive MNGCL on [$SPLIT] (10 runs, 500 ep, fixed eval)..."
$PYTHON_EXEC src/run_mngcl_leakage.py \
    --split "$SPLIT" \
    --n_runs 10 \
    --epochs 500 \
    --eval_mode fixed \
    --no_pathway \
    --gpu "$GPU_ID"

echo ""
echo ">>> [2/2] Running Strict Inductive MNGCL on [$SPLIT] (10 runs, 500 ep, fixed eval)..."
$PYTHON_EXEC src/run_mngcl_leakage.py \
    --split "$SPLIT" \
    --n_runs 10 \
    --epochs 500 \
    --eval_mode fixed \
    --no_pathway \
    --inductive \
    --gpu "$GPU_ID"

echo ""
echo "========================================================================"
echo "Packaging generated MNGCL benchmark result files..."
echo "========================================================================"
cd result
tar -czf mngcl_leakage_results.tar.gz \
    mngcl*clean_to_hit*.txt \
    mngcl*clean_to_hit*.csv \
    mngcl*hit_to_clean*.txt \
    mngcl*hit_to_clean*.csv 2>/dev/null || tar -czf mngcl_leakage_results.tar.gz mngcl*clean_to_hit*
cd ..

echo "SUCCESS! Created result/mngcl_leakage_results.tar.gz"
echo "Summary of generated files:"
ls -lh result/mngcl*leakage*
echo "========================================================================"
