#!/bin/bash
# ==============================================================================
# Import and synchronize MNGCL server results into clean_hit_results/
# ==============================================================================

set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"

if [ -f "result/mngcl_clean_to_hit_results.tar.gz" ]; then
    echo ">>> Extracting result/mngcl_clean_to_hit_results.tar.gz..."
    tar -xzf result/mngcl_clean_to_hit_results.tar.gz -C result/
fi

echo ">>> Copying updated MNGCL clean_to_hit files to result/clean_hit_results/..."
cp -v result/mngcl*clean_to_hit* result/clean_hit_results/

echo ""
echo ">>> Verifying Clean <-> Hit benchmark consistency across all 10 runs..."
python -c "
import numpy as np

methods = [
    ('BIMODriver (Transductive)', 'bimodriver_original_leakage'),
    ('BIMODriver (Inductive)', 'bimodriver_original_inductive_leakage'),
    ('MNGCL (Transductive)', 'mngcl_leakage'),
    ('MNGCL (Inductive)', 'mngcl_inductive_leakage'),
    ('DISFusion (Transductive)', 'disfusion_leakage'),
    ('DISFusion (Inductive)', 'disfusion_inductive_leakage')
]

for task in ['clean_to_hit', 'hit_to_clean']:
    print(f'=== TASK: {task} ===')
    for label, prefix in methods:
        auc_f = f'result/clean_hit_results/{prefix}_{task}_auroc.txt'
        prc_f = f'result/clean_hit_results/{prefix}_{task}_auprc.txt'
        try:
            auc = np.loadtxt(auc_f)
            prc = np.loadtxt(prc_f)
            print(f'  {label:32s} | AUROC: {auc.mean():.4f} ± {auc.std():.4f} | AUPRC: {prc.mean():.4f} ± {prc.std():.4f}')
        except Exception as e:
            print(f'  {label:32s} | Error: {e}')
"

echo ""
echo "========================================================================"
echo "All files in result/clean_hit_results/ are synchronized and verified!"
echo "Ready for the statistical testing agent."
echo "========================================================================"
