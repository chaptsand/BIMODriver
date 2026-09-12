#!/bin/bash
# ==============================================================================
# Clean <-> Hit Fixed Split Benchmark Run Script (Paper Setting Reproduction)
# Evaluates BIMODriver (gating_layer 64-dim), DISFusion, and MNGCL across 10 runs
# Supports both Transductive (传导式) and Strict Inductive (严格归纳式) settings
# ==============================================================================

set -e

PYTHON_EXEC="${PYTHON_EXEC:-python}"
MODE="${1:-all}"  # options: all, transductive, inductive

echo "Using Python: $PYTHON_EXEC"
echo "Execution Mode: $MODE (options: all, transductive, inductive)"
mkdir -p result

run_transductive() {
    echo "========================================================================"
    echo "Part 1: TRANSDUCTIVE BENCHMARK (传导式设置)"
    echo "========================================================================"

    echo ">>> [1/3] Running BIMODriver (Transductive)..."
    $PYTHON_EXEC BIMODriver/main.py --split clean_to_hit 2>&1 | tee result/server_bimodriver_clean_to_hit.log
    $PYTHON_EXEC BIMODriver/main.py --split hit_to_clean 2>&1 | tee result/server_bimodriver_hit_to_clean.log

    echo ">>> [2/3] Running DISFusion (Transductive, Paper Setting: 64-dim, 10362 hyperedges)..."
    $PYTHON_EXEC src/run_disfusion_leakage.py --split both --n_runs 10 --epochs 200 2>&1 | tee result/server_disfusion_leakage.log

    echo ">>> [3/3] Running MNGCL (Transductive, Paper Setting: 64-dim, 2 views PPI+GO, 500 epochs fixed eval)..."
    $PYTHON_EXEC src/run_mngcl_leakage.py --split both --n_runs 10 --epochs 500 --eval_mode fixed --no_pathway 2>&1 | tee result/server_mngcl_leakage.log
}

run_inductive() {
    echo "========================================================================"
    echo "Part 2: STRICT INDUCTIVE BENCHMARK (严格归纳式设置: 边切断+特征置零)"
    echo "========================================================================"

    echo ">>> [1/3] Running BIMODriver (Strict Inductive)..."
    $PYTHON_EXEC BIMODriver/main.py --split clean_to_hit --inductive 2>&1 | tee result/server_bimodriver_clean_to_hit_inductive.log
    $PYTHON_EXEC BIMODriver/main.py --split hit_to_clean --inductive 2>&1 | tee result/server_bimodriver_hit_to_clean_inductive.log

    echo ">>> [2/3] Running DISFusion (Strict Inductive: cut PPI/Hypergraph test edges, zero test features)..."
    $PYTHON_EXEC src/run_disfusion_leakage.py --split both --n_runs 10 --epochs 200 --inductive 2>&1 | tee result/server_disfusion_inductive.log

    echo ">>> [3/3] Running MNGCL (Strict Inductive: cut PPI/GO test edges, zero test features, 500 epochs fixed eval)..."
    $PYTHON_EXEC src/run_mngcl_leakage.py --split both --n_runs 10 --epochs 500 --eval_mode fixed --no_pathway --inductive 2>&1 | tee result/server_mngcl_inductive.log
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

echo "========================================================================"
echo "Benchmark Finished! Summarizing results..."
echo "========================================================================"
$PYTHON_EXEC src/summarize_leakage_results.py
