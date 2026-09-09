#!/bin/bash
# ==============================================================================
# Clean <-> Hit Fixed Split Benchmark Run Script (Paper Setting Reproduction)
# Evaluates BIMODriver (gating_layer 64-dim), DISFusion, and MNGCL across 10 runs
# ==============================================================================

set -e

PYTHON_EXEC="${PYTHON_EXEC:-python}"
echo "Using Python: $PYTHON_EXEC"
mkdir -p result

echo "========================================================================"
echo "Step 1/3: Running BIMODriver on Clean <-> Hit splits (gating_layer: 64-dim)..."
echo "========================================================================"
$PYTHON_EXEC BIMODriver/main.py --split clean_to_hit 2>&1 | tee result/server_bimodriver_clean_to_hit.log
$PYTHON_EXEC BIMODriver/main.py --split hit_to_clean 2>&1 | tee result/server_bimodriver_hit_to_clean.log

echo "========================================================================"
echo "Step 2/3: Running DISFusion on Clean <-> Hit splits (Paper Setting)..."
echo "  - Features: 64-dim (48 omics + 16 topology)"
echo "  - Hypergraph: 10,362 hyperedges (177 KEGG + 10,185 GO terms)"
echo "========================================================================"
$PYTHON_EXEC src/run_disfusion_leakage.py --split both --n_runs 10 --epochs 200 2>&1 | tee result/server_disfusion_leakage.log

echo "========================================================================"
echo "Step 3/3: Running MNGCL on Clean <-> Hit splits (Paper Setting)..."
echo "  - Features: 64-dim (48 omics + 16 topology)"
echo "  - Views: 2 views (PPI + GO similarity, matching paper criteria)"
echo "========================================================================"
$PYTHON_EXEC src/run_mngcl_leakage.py --split both --n_runs 10 --epochs 1000 --no_pathway 2>&1 | tee result/server_mngcl_leakage.log

echo "========================================================================"
echo "Benchmark Finished! Summarizing results..."
echo "========================================================================"
$PYTHON_EXEC src/summarize_leakage_results.py

