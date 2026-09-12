#!/bin/bash
# ==============================================================================
# Run MNGCL Clean -> Hit (10 Runs, 500 Epochs, Fixed Eval Protocol) on Server
# Generates 10-run results for both Transductive and Strict Inductive modes
# ==============================================================================

set -e

PYTHON_EXEC="${PYTHON_EXEC:-python}"
GPU_ID="${1:-0}"

echo "========================================================================"
echo "Running MNGCL on Clean -> Hit Benchmark (10 Runs, 500 Epochs, Fixed Eval)"
echo "Using Python: $PYTHON_EXEC on GPU: $GPU_ID"
echo "========================================================================"

mkdir -p result

echo ""
echo ">>> [1/2] Running Transductive MNGCL on Clean -> Hit (10 runs, fixed 500 ep)..."
$PYTHON_EXEC src/run_mngcl_leakage.py \
    --split clean_to_hit \
    --n_runs 10 \
    --epochs 500 \
    --eval_mode fixed \
    --no_pathway \
    --gpu "$GPU_ID"

echo ""
echo ">>> [2/2] Running Strict Inductive MNGCL on Clean -> Hit (10 runs, fixed 500 ep)..."
$PYTHON_EXEC src/run_mngcl_leakage.py \
    --split clean_to_hit \
    --n_runs 10 \
    --epochs 500 \
    --eval_mode fixed \
    --no_pathway \
    --inductive \
    --gpu "$GPU_ID"

echo ""
echo "========================================================================"
echo "Packaging generated Clean -> Hit result files into archive..."
echo "========================================================================"
cd result
tar -czf mngcl_clean_to_hit_results.tar.gz \
    mngcl_leakage_clean_to_hit_auroc.txt \
    mngcl_leakage_clean_to_hit_auprc.txt \
    mngcl_leakage_clean_to_hit_summary.txt \
    mngcl_leakage_clean_to_hit_test_preds.csv \
    mngcl_inductive_leakage_clean_to_hit_auroc.txt \
    mngcl_inductive_leakage_clean_to_hit_auprc.txt \
    mngcl_inductive_leakage_clean_to_hit_summary.txt \
    mngcl_inductive_leakage_clean_to_hit_test_preds.csv
cd ..

echo "SUCCESS! Created result/mngcl_clean_to_hit_results.tar.gz"
echo "You can transfer this archive back to local, or scp the 8 result/mngcl*clean_to_hit* files."
echo "========================================================================"
