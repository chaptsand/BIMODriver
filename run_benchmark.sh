#!/bin/bash
# ==============================================================================
# Clean <-> Hit Benchmark Reproduction Script
# Evaluates BIMODriver (64-dim gating fusion), DISFusion, and MNGCL across 10 runs
# Supports both Transductive (传导式) and Strict Inductive (严格归纳式) settings
# ==============================================================================

set -e

PYTHON_EXEC="${PYTHON_EXEC:-python}"
MODE="${1:-all}"  # options: all, transductive, inductive
GPU_ID="${2:-0}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"

echo "========================================================================"
echo "BIMODriver Benchmark Suite: Clean <-> Hit Cross-Distribution Evaluation"
echo "Python Executable : $PYTHON_EXEC"
echo "Execution Mode    : $MODE (options: all, transductive, inductive)"
echo "GPU Device        : CUDA_VISIBLE_DEVICES=$GPU_ID"
echo "========================================================================"
mkdir -p result

run_transductive() {
    echo ""
    echo "========================================================================"
    echo "Part 1: Transductive Benchmark (传导式全图设置)"
    echo "========================================================================"

    echo ">>> [1/3] Running BIMODriver (Transductive)..."
    $PYTHON_EXEC BIMODriver/main.py --split clean_to_hit 2>&1 | tee result/bimodriver_clean_to_hit.log
    $PYTHON_EXEC BIMODriver/main.py --split hit_to_clean 2>&1 | tee result/bimodriver_hit_to_clean.log

    echo ">>> [2/3] Running DISFusion (Transductive, 64-dim, 10,362 hyperedges)..."
    $PYTHON_EXEC implement/run_disfusion_leakage.py --split both --n_runs 10 --epochs 200 2>&1 | tee result/disfusion_leakage.log

    echo ">>> [3/3] Running MNGCL (Transductive, 64-dim, 2 views PPI+GO, 500 epochs fixed eval)..."
    $PYTHON_EXEC implement/run_mngcl_leakage.py --split both --n_runs 10 --epochs 500 --eval_mode fixed --no_pathway 2>&1 | tee result/mngcl_leakage.log
}

run_inductive() {
    echo ""
    echo "========================================================================"
    echo "Part 2: Strict Inductive Benchmark (严格归纳式设置: 边切断+特征置零)"
    echo "========================================================================"

    echo ">>> [1/3] Running BIMODriver (Strict Inductive)..."
    $PYTHON_EXEC BIMODriver/main.py --split clean_to_hit --inductive 2>&1 | tee result/bimodriver_clean_to_hit_inductive.log
    $PYTHON_EXEC BIMODriver/main.py --split hit_to_clean --inductive 2>&1 | tee result/bimodriver_hit_to_clean_inductive.log

    echo ">>> [2/3] Running DISFusion (Strict Inductive: cut test edges, zero test features)..."
    $PYTHON_EXEC implement/run_disfusion_leakage.py --split both --n_runs 10 --epochs 200 --inductive 2>&1 | tee result/disfusion_inductive.log

    echo ">>> [3/3] Running MNGCL (Strict Inductive: cut test edges, zero test features, 500 epochs fixed eval)..."
    $PYTHON_EXEC implement/run_mngcl_leakage.py --split both --n_runs 10 --epochs 500 --eval_mode fixed --no_pathway --inductive 2>&1 | tee result/mngcl_inductive.log
}

case "$MODE" in
    transductive)
        run_transductive
        ;;
    inductive)
        run_inductive
        ;;
    all)
        run_transductive
        run_inductive
        ;;
    *)
        echo "Unknown mode: $MODE. Choose 'all', 'transductive', or 'inductive'."
        exit 1
        ;;
esac

echo ""
echo "========================================================================"
echo "Benchmark Finished! Summarizing results..."
echo "========================================================================"
$PYTHON_EXEC implement/summarize_leakage_results.py
