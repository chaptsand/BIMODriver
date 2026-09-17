# BIMODriver: Statistical Validation & Reproducibility Suite

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository provides the official, open-source statistical validation, paper figure generation, and benchmark reproduction suite for:

> **"BIMODriver: Decoupling Multimodal Feature Interaction and Structural Inductive Bias for Cancer Driver Gene Identification"**

---

## Overview

This suite allows reviewers and researchers to independently reproduce, audit, and visualize all experimental findings reported in the paper, including:

1. **Pan-Cancer & Cancer-Specific Benchmarks (Tables 1 & 2)**: 10 methods evaluated across CPDB and STRING PPI networks, including 15 individual cancer types (10 runs × 5-fold CV).
2. **Detailed Statistical Significance Analysis (Tables A8, A9, A10, A11)**: Exact Wilcoxon signed-rank tests, Benjamini-Hochberg (BH) FDR corrections, rank-biserial effect sizes ($r_{rb}$), and Hodges-Lehmann median differences with 95% bootstrap confidence intervals for pan-cancer and 15 cancer types.
3. **LLM Vocabulary Leakage Benchmark (Clean $\leftrightarrow$ Hit)**: Cross-distribution evaluations (`Clean -> Hit` and `Hit -> Clean`) under both **Transductive** and **Strict Inductive** graph learning paradigms against baselines (MNGCL, DISFusion) for reviewer evaluation.
4. **Edge-Masking Information Exposure Ablation (Table A19)**: Evaluating 4 masking variants and strict inductive graph learning on PPI topological edges.
5. **Label-Like Phrases Masking Ablation (Table R5)**: Evaluating the impact of keyword / phrase masking across transductive and inductive protocols with paired Wilcoxon signed-rank tests and BH FDR control.
6. **All Paper Figures (Figures 2, 3, 4, 5)**: Standalone, publication-ready high-resolution plotting scripts (300 DPI PNG + vector PDF) with candidate gene validation and functional enrichment subplots (Table A12 / 5a, 5b, 5c, and composite).
7. **Standardized Statistical Workbooks**: Multi-tab Excel workbooks containing complete empirical runs, sensitivity tests, and statistical test matrices.

For an exhaustive, item-by-item guide, see [REPRODUCTION_GUIDE.md](REPRODUCTION_GUIDE.md).

---

## Repository Structure

```text
├── reproduce_all.py                 # Master one-click reproduction CLI entrypoint
├── REPRODUCTION_GUIDE.md            # Comprehensive paper item-to-code mapping guide
├── environment.yml                  # Conda environment specification
├── requirements.txt                 # Pip dependencies
│
├── notebooks/                       # Interactive Jupyter Notebooks
│   ├── Figure2_Figure3_Validation_Ablation.ipynb
│   ├── Figure4_SMoE_Gating_Visualization.ipynb
│   ├── Figure5_Candidate_Genes_Validation.ipynb
│   ├── PanCancer_Wilcoxon_Analysis.ipynb
│   └── CancerSpecific_Wilcoxon_Analysis.ipynb
│
├── scripts/                         # Executable Python CLI scripts
│   ├── figures/                     # Standalone figure plotting scripts
│   │   ├── plot_figure2_comparison.py
│   │   ├── plot_figure3_ablation.py
│   │   ├── plot_figure4_smoe_gating.py
│   │   └── plot_figure5_candidate_genes.py
│   ├── paper_validation/            # Pan-cancer and 15 cancer benchmarks (Tables 1 & 2)
│   │   └── reanalyse_statistics.py
│   ├── leakage_ablation/            # Clean <-> Hit, Table A19 & Table R5 benchmarks
│   │   ├── run_clean_hit_evaluation.py
│   │   ├── run_table_A19_statistics.py
│   │   ├── run_table_R5_statistics.py
│   │   ├── run_statistics.py
│   │   └── verify_results.py
│   └── workbooks/                   # Excel workbook generators and standardizers
│       └── standardize_workbooks.py
│
├── data/                            # Raw metric arrays and supplementary matrices
│   ├── edge_masking_A19/            # Table A19 CPDB and STRING 10x5 cross-validation matrices
│   ├── label-like_phrases_masking/  # Table R5 raw CPDB 10x5 cross-validation matrices
│   ├── leakage_ablation/            # Clean-Hit benchmark arrays and 4-way ablation
│   └── paper_supplementary/         # CPDB, STRING, and 15-cancer 10x5 matrices
│
├── results/                         # Generated artifacts and deliverables
│   ├── figures/                     # High-resolution PNG and vector PDF figures
│   ├── clean_hit_results_final/     # Clean-Hit vocabulary leakage statistical results and Excel
│   ├── edge_masking_A19/            # Table A19 Edge-masking statistical results and Excel
│   ├── label_like_phrases_masking/  # Table R5 label masking statistical results and Excel
│   ├── paper_validation/            # Tables 1, 2, A8, A9, A10, A11 summary tables and run means
│   └── workbooks/                   # Multi-sheet comprehensive Excel deliverables
│
└── docs/                            # Statistical methodology specifications and file inventory
    ├── file_inventory.md
    └── statistical_methods.md
```

---

## Quick Start

### 1. Environment Setup

Using **Conda** (recommended):
```bash
conda env create -f environment.yml
conda activate bimodriver-statistics
```

Or using **Pip**:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. One-Click Full Reproduction

Execute the master reproduction script:
```bash
python reproduce_all.py --all
```
This runs the full pipeline end-to-end:
1. Recomputes all statistical tests for Tables 1, 2, A8, A9, A10, A11, A19, and Clean-Hit benchmarks.
2. Renders Figures 2, 3, 4, 5 (PNG 300 DPI + vector PDF).
3. Generates and standardizes multi-sheet Excel workbooks.
4. Performs an automated integrity verification audit across all core deliverables.

---

## Paper Deliverables & Execution Mapping

| Paper Item | Description | Execution Command | Primary Deliverable |
| :--- | :--- | :--- | :--- |
| **Table 1** | Pan-cancer benchmark on CPDB & STRING | `python scripts/paper_validation/reanalyse_statistics.py` | `results/paper_validation/table1_performance_comparison.csv`<br>`results/paper_validation/benchmark_summary.csv` |
| **Table 2** | 15 cancer-specific performance comparisons | `python scripts/paper_validation/reanalyse_statistics.py` | `results/paper_validation/table2_cancer_specific_comparison.csv`<br>`results/paper_validation/cancer_type_means.csv` |
| **Table A8** | Pan-cancer Wilcoxon & BH FDR significance (Legacy reporting protocol: 50-fold, m=9 by-metric; Authoritative unified standard in Table A10) | `python scripts/paper_validation/reanalyse_statistics.py` | `results/paper_validation/table_A8_pancancer_significance.csv` |
| **Table A9** | Cancer-specific Wilcoxon & BH FDR significance (Legacy reporting protocol: 15 cancer means, m=9 by-metric; Authoritative unified standard in across15_stats) | `python scripts/paper_validation/reanalyse_statistics.py` | `results/paper_validation/table_A9_cancer_specific_significance.csv` |
| **Table A10** | Authoritative Pan-cancer detailed significance (10-run paired, n=10, m=18 BH FDR per network, $r_{rb}$, HL diff & 95% CI) | `python scripts/paper_validation/reanalyse_statistics.py` | `results/paper_validation/table_A10_pancancer_significance.csv` |
| **Table A11** | Authoritative Cancer-specific detailed significance (10-run paired, n=10, primary global m=270 BH FDR, supplementary m=135, $r_{rb}$, HL diff & 95% CI) | `python scripts/paper_validation/reanalyse_statistics.py` | `results/paper_validation/table_A11_cancer_specific_significance.csv` |
| **Clean $\leftrightarrow$ Hit** | Cross-distribution vocabulary leakage evaluation (Primary Two-Sided Main Table + Supplementary One-Sided) | `python scripts/leakage_ablation/run_clean_hit_evaluation.py` | `results/clean_hit_results_final/clean_hit_paper_main_table.csv`<br>`results/clean_hit_results_final/Clean_Hit_Statistical_Validation.xlsx` |
| **Table A19** | Strict edge-masking & inductive graph learning | `python scripts/leakage_ablation/run_table_A19_statistics.py` | `results/edge_masking_A19/table_A19_edge_masking_comparison.csv`<br>`results/edge_masking_A19/table_A19_extended_variants.csv` |
| **Table R5** | Label-like phrases masking ablation statistical tests | `python scripts/leakage_ablation/run_table_R5_statistics.py` | `results/label_like_phrases_masking/table_R5_label_masking_formatted.csv`<br>`results/label_like_phrases_masking/Table_R5_Statistical_Validation.xlsx` |
| **Figure 2** | Independent validation comparison scatter plot | `python scripts/figures/plot_figure2_comparison.py` | `results/figures/figure2_independent_validation.png` / `.pdf` |
| **Figure 3** | Component ablation breakdown (Bar & Radar) | `python scripts/figures/plot_figure3_ablation.py` | `results/figures/figure3_ablation_study.png`<br>`results/figures/figure3_ablation_radar.png` |
| **Figure 4** | Sparse Mixture-of-Experts (SMoE) gating KDE | `python scripts/figures/plot_figure4_smoe_gating.py` | `results/figures/figure4_smoe_gating.png` / `.pdf` |
| **Figure 5a** | Overall validation rate & database support | `python scripts/figures/plot_figure5_candidate_genes.py` | `results/figures/figure5a_database_validation.png` / `.pdf` |
| **Figure 5b** | 4-set candidate driver gene Venn diagram | `python scripts/figures/plot_figure5_candidate_genes.py` | `results/figures/figure5b_venn_diagram.png` / `.pdf` |
| **Figure 5c** | KEGG & GO pathway functional enrichment | `python scripts/figures/plot_figure5_candidate_genes.py` | `results/figures/figure5c_enrichment_analysis.png` / `.pdf` |
| **Figure 5** | Composite publication figure (panels a, b, c) | `python scripts/figures/plot_figure5_candidate_genes.py` | `results/figures/figure5_candidate_genes_composite.png` / `.pdf` |
| **Supplementary Workbook** | Statistical significance summary workbook (Tables 1, 2, A8, A9, A10, A11) | `python scripts/workbooks/standardize_workbooks.py` | `results/workbooks/Statistical_Significance_Summary.xlsx` |

---

## Individual Component Reproduction

### 1. Generating Figures Headless (No Jupyter required)

```bash
python scripts/figures/plot_figure2_comparison.py
python scripts/figures/plot_figure3_ablation.py
python scripts/figures/plot_figure4_smoe_gating.py
python scripts/figures/plot_figure5_candidate_genes.py
```
Outputs are written to `results/figures/` in both 300 DPI PNG and vector PDF formats.

### 2. Running Clean $\leftrightarrow$ Hit Vocabulary Leakage Tests

```bash
python scripts/leakage_ablation/run_clean_hit_evaluation.py
```
- **Tests**:
  - One-sided paired Wilcoxon signed-rank tests ($H_1: \text{BIMODriver} > \text{Baseline}$) for superiority evaluation.
  - Two-sided paired Wilcoxon tests ($H_1: \text{Variant}_1 \ne \text{Variant}_2$) for ablation pairs.
  - Benjamini-Hochberg (BH) FDR $q$-values.
  - Hodges-Lehmann median difference with 95% bootstrap confidence intervals (10,000 resamples).
  - Paired rank-biserial correlation ($r_{rb}$).

### 3. Running Edge-Masking Analysis (Table A19)

```bash
python scripts/leakage_ablation/run_table_A19_statistics.py
```
Computes empirical metrics and paired statistical tests for all 7 edge-masking and graph learning variants on CPDB.

### 4. Running Label-Like Phrases Masking Analysis (Table R5)

```bash
python scripts/leakage_ablation/run_table_R5_statistics.py
```
Computes 10-run paired Wilcoxon signed-rank tests, BH FDR corrections across m=4 tests, effect sizes ($r_{rb}$), and Hodges-Lehmann median differences with 95% bootstrap confidence intervals across transductive and inductive masking variants.

### 5. Interactive Exploration via Jupyter

Standardized Jupyter notebooks are available under `notebooks/`:
```bash
jupyter lab notebooks/
```
- [Figure2_Figure3_Validation_Ablation.ipynb](notebooks/Figure2_Figure3_Validation_Ablation.ipynb)
- [Figure4_SMoE_Gating_Visualization.ipynb](notebooks/Figure4_SMoE_Gating_Visualization.ipynb)
- [Figure5_Candidate_Genes_Validation.ipynb](notebooks/Figure5_Candidate_Genes_Validation.ipynb)
- [PanCancer_Wilcoxon_Analysis.ipynb](notebooks/PanCancer_Wilcoxon_Analysis.ipynb)
- [CancerSpecific_Wilcoxon_Analysis.ipynb](notebooks/CancerSpecific_Wilcoxon_Analysis.ipynb)

---

## Verification & Deliverables Audit

You can verify the presence, non-zero size, and integrity of all generated artifacts at any time:

```bash
python reproduce_all.py --verify
```

Expected output:
```text
###########################################################################
# STEP 4: VERIFICATION OF ARTIFACTS & INTEGRITY AUDIT
###########################################################################
Artifact Description                     | Status   |       Size | Path
-----------------------------------------------------------------------------------------------
Table 1 Performance Comparison           | PASS     |        804 B | results/paper_validation/table1_performance_comparison.csv
Table 2 Cancer Specific Comparison       | PASS     |      2,436 B | results/paper_validation/table2_cancer_specific_comparison.csv
Table 1 & 2 Summary                      | PASS     |     15,084 B | results/paper_validation/benchmark_summary.csv
Table 1 & 2 Run Means                    | PASS     |     23,520 B | results/paper_validation/benchmark_run_means.csv
Table 2 Cancer Type Means                | PASS     |     10,255 B | results/paper_validation/cancer_type_means.csv
50-Fold Sensitivity Analysis             | PASS     |     11,085 B | results/paper_validation/sensitivity_fold50.csv
Table A8 Pan-Cancer Significance         | PASS     |        467 B | results/paper_validation/table_A8_pancancer_significance.csv
Table A9 Cancer-Specific Significance    | PASS     |        449 B | results/paper_validation/table_A9_cancer_specific_significance.csv
Table A10 Pan-Cancer Detailed Significance | PASS     |      3,829 B | results/paper_validation/table_A10_pancancer_significance.csv
Table A11 Cancer-Specific Detailed Significance | PASS     |     25,106 B | results/paper_validation/table_A11_cancer_specific_significance.csv
Clean-Hit Paper Main Table (One-Sided)   | PASS     |      1,386 B | results/clean_hit_results_final/clean_hit_paper_main_table.csv
Clean-Hit Main Table (Two-Sided)         | PASS     |      1,371 B | results/clean_hit_results_final/clean_hit_paper_main_table_two_sided.csv
Clean-Hit Statistical Comparison         | PASS     |      6,471 B | results/clean_hit_results_final/clean_hit_statistical_comparison.csv
Clean-Hit Descriptive Summary            | PASS     |      5,409 B | results/clean_hit_results_final/clean_hit_descriptive_summary.csv
Clean-Hit Raw Run Scores                 | PASS     |      9,611 B | results/clean_hit_results_final/clean_hit_raw_scores.csv
Clean-Hit Excel Workbook                 | PASS     |     26,040 B | results/clean_hit_results_final/Clean_Hit_Statistical_Validation.xlsx
Table A19 Edge-Masking Comparison        | PASS     |        385 B | results/edge_masking_A19/table_A19_edge_masking_comparison.csv
Table A19 Extended Variants              | PASS     |        727 B | results/edge_masking_A19/table_A19_extended_variants.csv
Table A19 Extended Summary               | PASS     |      3,836 B | results/edge_masking_A19/table_A19_extended_summary.csv
Table A19 Masking vs Baselines (50-fold) | PASS     |     14,900 B | results/edge_masking_A19/masking_vs_baselines_50fold.csv
Table A19 Masking vs Baselines (10-run)  | PASS     |     13,725 B | results/edge_masking_A19/masking_vs_baselines_10run.csv
Table A19 Masking Ablation (50-fold)     | PASS     |      6,783 B | results/edge_masking_A19/masking_ablation_vs_original_50fold.csv
Table A19 Raw 50-Fold Matrices           | PASS     |     18,071 B | results/edge_masking_A19/raw_50fold_matrices.csv
Table A19 Excel Workbook                 | PASS     |     45,341 B | results/edge_masking_A19/Table_A19_Statistical_Validation.xlsx
Statistical Significance Summary Excel   | PASS     |    111,828 B | results/workbooks/Statistical_Significance_Summary.xlsx
Figure 2 PNG                             | PASS     |    277,895 B | results/figures/figure2_independent_validation.png
Figure 2 PDF                             | PASS     |     40,710 B | results/figures/figure2_independent_validation.pdf
Figure 3 Bar PNG                         | PASS     |    242,417 B | results/figures/figure3_ablation_study.png
Figure 3 Bar PDF                         | PASS     |     32,177 B | results/figures/figure3_ablation_study.pdf
Figure 3 Radar PNG                       | PASS     |    679,446 B | results/figures/figure3_ablation_radar.png
Figure 4 PNG                             | PASS     |  1,178,212 B | results/figures/figure4_smoe_gating.png
Figure 4 PDF                             | PASS     |     86,433 B | results/figures/figure4_smoe_gating.pdf
Figure 5a Database Validation PNG        | PASS     |    205,012 B | results/figures/figure5a_database_validation.png
Figure 5a Database Validation PDF        | PASS     |     29,165 B | results/figures/figure5a_database_validation.pdf
Figure 5b Venn Diagram PNG               | PASS     |    286,008 B | results/figures/figure5b_venn_diagram.png
Figure 5b Venn Diagram PDF               | PASS     |     12,774 B | results/figures/figure5b_venn_diagram.pdf
Figure 5c Enrichment Analysis PNG        | PASS     |  1,646,501 B | results/figures/figure5c_enrichment_analysis.png
Figure 5c Enrichment Analysis PDF        | PASS     |    654,438 B | results/figures/figure5c_enrichment_analysis.pdf
Figure 5 Composite PNG                   | PASS     |  2,315,794 B | results/figures/figure5_candidate_genes_composite.png
Figure 5 Composite PDF                   | PASS     |  1,081,599 B | results/figures/figure5_candidate_genes_composite.pdf
-----------------------------------------------------------------------------------------------
[AUDIT SUCCESS] All 40 expected paper deliverables verified successfully!
```

---

## License

This project is licensed under the MIT License.
