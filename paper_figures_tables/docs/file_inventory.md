# Repository File Inventory

This document outlines the active datasets, scripts, notebooks, and result directories in the open-source BIMODriver validation repository.

---

## 1. Input Datasets (`data/`)

- **`data/edge_masking_A19/`**:
  Contains the 10-run × 5-fold cross-validation performance matrices across CPDB and STRING networks for the Table A19 edge-masking and strict inductive graph learning evaluation:
  - `BIMODriver-Original/`: Original transductive benchmark matrices on CPDB and STRING.
  - `BIMODriver-Masking/`: Four edge-masking variants:
    - `严格归纳式/` (Strict inductive)
    - `去掉测试集相连的边/` (Remove edges connected to test genes)
    - `测试节点不参加对比损失/` (Exclude test nodes from contrastive loss)
    - `去掉所有的边/` (Remove all topological edges)
  - `DISFusion/`: Baseline matrices on CPDB and STRING.
  - `MNGCL/`: Baseline matrices on CPDB and STRING.

- **`data/label-like_phrases_masking/`**:
  Contains 10-run × 5-fold cross-validation performance matrices on CPDB pan-cancer for the Table R5 label-like phrases masking ablation evaluation:
  - `cpdb-pancancer.txt`: Original transductive BIMODriver benchmark matrices (AUROC & AUPRC).
  - `pan-cancer_masked_auroc.txt` & `pan-cancer_masked_auprc.txt`: Masked transductive matrices.
  - `pan-cancer_masked_inductive_auroc.txt` & `pan-cancer_masked_inductive_auprc.txt`: Masked inductive matrices.

- **`data/leakage_ablation/`**:
  - `1_pan_cancer_10x5_cv/`: AUROC and AUPRC arrays across 4 protocol variants (Original Transductive, Masked Transductive, Original Strict Inductive, Masked Strict Inductive).
  - `2_clean_hit_leakage_benchmark/`: 10-run fixed-split evaluation scores for BIMODriver, DISFusion, and MNGCL across `clean_to_hit` and `hit_to_clean` tasks.

- **`data/paper_supplementary/`**:
  - `figure_code/pan-cancer/`: Full pan-cancer evaluation matrices on CPDB (`cpdb-pancancer.txt`) and STRING (`string-pancancer.txt`).
  - `figure_code/cancer_specific/`: 10 methods × 15 cancer types 10×5 evaluation matrices.
  - `legacy_pancancer_wilcoxon_bh.csv` & `legacy_full_wilcoxon_bh.csv`: Standardized reference tables for QA cross-checking.

---

## 2. Reproduction Scripts (`scripts/`)

- **`scripts/figures/`**:
  - `plot_figure2_comparison.py`: Generates Paper Figure 2 (Oncogene vs OncoKB comparison scatter plot).
  - `plot_figure3_ablation.py`: Generates Paper Figure 3 (Ablation study bar chart and radar chart).
  - `plot_figure4_smoe_gating.py`: Generates Paper Figure 4 (SMoE gating frequency and KDE distributions).
  - `plot_figure5_candidate_genes.py`: Generates Paper Figure 5 (Candidate cancer driver genes multi-database validation).

- **`scripts/leakage_ablation/`**:
  - `run_clean_hit_evaluation.py`: Computes paired Wilcoxon tests, BH FDR corrections, effect sizes ($r_{rb}$), Hodges-Lehmann differences, and generates the Clean $\leftrightarrow$ Hit benchmark deliverables workbook.
  - `run_table_A19_statistics.py`: Computes empirical metrics and statistical significance tests across 7 edge-masking variants for Table A19.
  - `run_table_R5_statistics.py`: Computes paired Wilcoxon signed-rank tests, BH FDR corrections across $m=4$ comparisons, effect sizes ($r_{rb}$), and Hodges-Lehmann median differences with 95% bootstrap confidence intervals for Table R5 (label-like phrases masking ablation).
  - `run_statistics.py`: Computes the 4-way pan-cancer feature and graph protocol ablation benchmark.
  - `verify_results.py`: Independent mathematical verification of data shapes, metrics, and p-values.

- **`scripts/paper_validation/`**:
  - `reanalyse_statistics.py`: Comprehensive recomputation of CPDB, STRING, 15-cancer benchmarks, across-cancer means, 50-fold sensitivity analyses, and appendix statistical tables (Tables 1, 2, A8, A9, A10, A11).

- **`scripts/workbooks/`**:
  - `standardize_workbooks.py`: Standardizes and verifies the authoritative summary workbook `results/workbooks/Statistical_Significance_Summary.xlsx` (Supplementary Significance Workbook).

---

## 3. Interactive Notebooks (`notebooks/`)

- `Figure2_Figure3_Validation_Ablation.ipynb`: Interactive plotting for Figures 2 & 3.
- `Figure4_SMoE_Gating_Visualization.ipynb`: Gating weight query and distribution visualizations for Figure 4.
- `Figure5_Candidate_Genes_Validation.ipynb`: Venn and multi-database validation explorations for Figure 5.
- `PanCancer_Wilcoxon_Analysis.ipynb`: Interactive pan-cancer Wilcoxon and BH FDR analysis.
- `CancerSpecific_Wilcoxon_Analysis.ipynb`: Interactive cancer-specific statistical tests.

---

## 4. Master Entrypoint

- **`reproduce_all.py`**:
  Master CLI script automating the full reproduction workflow (`--all`, `--tables`, `--figures`, `--workbooks`, `--verify`).
