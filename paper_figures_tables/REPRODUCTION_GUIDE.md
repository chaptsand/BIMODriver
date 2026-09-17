# BIMODriver Paper Reproduction Guide

This guide provides detailed, executable instructions to reproduce all tables, figures, statistical significance tests, and Excel workbooks reported in the paper:

> **BIMODriver: Decoupling Multimodal Feature Interaction and Structural Inductive Bias for Cancer Driver Gene Identification**

---

## 1. Paper Deliverables Mapping Table

The following matrix maps every experimental Table, Figure, and Supplementary material in the manuscript to its corresponding input dataset, execution script, notebook, and output artifacts.

| Paper Item | Title / Description | Input Data Path | Script / Entrypoint | Key Output Files |
| :--- | :--- | :--- | :--- | :--- |
| **Table 1** | Pan-Cancer Benchmark on CPDB & STRING Networks (AUROC & AUPRC) | `data/paper_supplementary/figure_code/pan-cancer/` | `scripts/paper_validation/reanalyse_statistics.py` | `results/paper_validation/table1_performance_comparison.csv`<br>`results/paper_validation/benchmark_summary.csv` |
| **Table 2** | 15 Cancer-Specific Performance Benchmark (10 methods × 15 cancers) | `data/paper_supplementary/figure_code/cancer_specific/` | `scripts/paper_validation/reanalyse_statistics.py` | `results/paper_validation/table2_cancer_specific_comparison.csv`<br>`results/paper_validation/cancer_type_means.csv` |
| **Table A8** | Pan-Cancer Wilcoxon & BH FDR Significance (Legacy reporting protocol: 50-fold, m=9 by-metric; Unified standard in Table A10) | `data/paper_supplementary/figure_code/pan-cancer/` | `scripts/paper_validation/reanalyse_statistics.py` | `results/paper_validation/table_A8_pancancer_significance.csv` |
| **Table A9** | Cancer-Specific Wilcoxon & BH FDR Significance (Legacy reporting protocol: 15 cancer means, m=9 by-metric; Unified standard in across15_stats) | `data/paper_supplementary/figure_code/cancer_specific/` | `scripts/paper_validation/reanalyse_statistics.py` | `results/paper_validation/table_A9_cancer_specific_significance.csv` |
| **Table A10** | Authoritative Pan-Cancer Significance Table (10-run paired, n=10, m=18 BH FDR per network, $r_{rb}$, HL diff & 95% CI) | `data/paper_supplementary/figure_code/pan-cancer/` | `scripts/paper_validation/reanalyse_statistics.py` | `results/paper_validation/table_A10_pancancer_significance.csv` |
| **Table A11** | Authoritative Cancer-Specific Significance Table (10-run paired, n=10, primary global m=270 BH FDR, supplementary m=135, $r_{rb}$, HL diff & 95% CI) | `data/paper_supplementary/figure_code/cancer_specific/` | `scripts/paper_validation/reanalyse_statistics.py` | `results/paper_validation/table_A11_cancer_specific_significance.csv` |
| **Table A12 / Figure 5** | Candidate Cancer Driver Gene Prediction & Multi-Database Validation | Multi-database validation & functional enrichment | `scripts/figures/plot_figure5_candidate_genes.py`<br>`notebooks/Figure5_Candidate_Genes_Validation.ipynb` | `results/figures/figure5_candidate_genes_composite.png`<br>`results/figures/figure5a_database_validation.png`<br>`results/figures/figure5b_venn_diagram.png`<br>`results/figures/figure5c_enrichment_analysis.png` |
| **Clean $\leftrightarrow$ Hit Benchmark** | Clean $\leftrightarrow$ Hit Cross-Distribution Vocabulary Leakage Evaluation (Primary Two-Sided Main Table + Supplementary One-Sided) | `results/clean_hit_results_final/` | `scripts/leakage_ablation/run_clean_hit_evaluation.py` | `results/clean_hit_results_final/clean_hit_paper_main_table.csv`<br>`results/clean_hit_results_final/clean_hit_statistical_comparison.csv`<br>`results/clean_hit_results_final/Clean_Hit_Statistical_Validation.xlsx` |
| **Table A19** | Edge-Masking Analysis & Strict Inductive Graph Learning Evaluation | `data/edge_masking_A19/` | `scripts/leakage_ablation/run_table_A19_statistics.py` | `results/edge_masking_A19/table_A19_extended_summary.csv`<br>`results/edge_masking_A19/masking_vs_baselines_50fold.csv`<br>`results/edge_masking_A19/Table_A19_Statistical_Validation.xlsx` |
| **Table R5** | Label-Like Phrases Masking Robustness Evaluation (CPDB Pan-Cancer) | `data/label-like_phrases_masking/` | `scripts/leakage_ablation/run_table_R5_statistics.py` | `results/label_like_phrases_masking/table_R5_label_masking_formatted.csv`<br>`results/label_like_phrases_masking/Table_R5_Statistical_Validation.xlsx` |
| **Ablation Study** | 4-Way Pan-Cancer Protocol & Feature Ablation Benchmark | `data/leakage_ablation/1_pan_cancer_10x5_cv/` | `scripts/leakage_ablation/run_statistics.py` | `results/gl_leakage_ablation/tables/`<br>`results/gl_leakage_ablation/workbooks/` |
| **Figure 2** | Independent Validation Performance Comparison Scatter Plot (Oncogene vs OncoKB) | Self-contained independent test benchmark scores | `scripts/figures/plot_figure2_comparison.py`<br>`notebooks/Figure2_Figure3_Validation_Ablation.ipynb` | `results/figures/figure2_independent_validation.png`<br>`results/figures/figure2_independent_validation.pdf` |
| **Figure 3** | Ablation Study Component Breakdown (Bar Chart & Radar Chart) | Ablation evaluation scores across 8 variants | `scripts/figures/plot_figure3_ablation.py`<br>`notebooks/Figure2_Figure3_Validation_Ablation.ipynb` | `results/figures/figure3_ablation_study.png`<br>`results/figures/figure3_ablation_radar.png` |
| **Figure 4** | Sparse Mixture-of-Experts (SMoE) Gating Visualization & KDE Distributions | SMoE routing activations & gating scores | `scripts/figures/plot_figure4_smoe_gating.py`<br>`notebooks/Figure4_SMoE_Gating_Visualization.ipynb` | `results/figures/figure4_smoe_gating.png`<br>`results/figures/figure4_smoe_gating.pdf` |
| **Figure 5** | Composite Candidate Driver Gene Validation (Panels a, b, c) | Multi-database validation & functional enrichment | `scripts/figures/plot_figure5_candidate_genes.py`<br>`notebooks/Figure5_Candidate_Genes_Validation.ipynb` | `results/figures/figure5_candidate_genes_composite.png` / `.pdf` |
| **Supplementary Workbook** | Comprehensive Statistical Test Workbook (Wilcoxon signed-rank + BH FDR) | All benchmark matrices & test runs | `scripts/workbooks/standardize_workbooks.py` | `results/workbooks/Statistical_Significance_Summary.xlsx` |
| **Sensitivity Analysis** | 50-Fold Sensitivity Analysis across 15 Cancer Types | 50-fold evaluation matrices | `scripts/paper_validation/reanalyse_statistics.py` | `results/paper_validation/sensitivity_fold50.csv` |

---

## 2. Environment Setup

### Option A: Conda (Recommended)

```bash
conda env create -f environment.yml
conda activate bimodriver-statistics
```

### Option B: Pip Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 3. One-Click Reproduction Master Command

To reproduce all tables, figures, Excel workbooks, and run the integrity verification audit in a single command:

```bash
python reproduce_all.py --all
```

### Granular Execution Options:

- **Tables and Statistical Significance Tests only:**
  ```bash
  python reproduce_all.py --tables
  ```
- **Figures only (PNG 300 DPI + Vector PDF):**
  ```bash
  python reproduce_all.py --figures
  ```
- **Excel Workbooks only:**
  ```bash
  python reproduce_all.py --workbooks
  ```
- **Integrity Verification & Deliverables Audit only:**
  ```bash
  python reproduce_all.py --verify
  ```

---

## 4. Step-by-Step Individual Table Reproduction

### Tables 1, 2, A8, A9, A10, A11: Pan-Cancer & 15 Cancer Benchmarks

Reproduces the primary pan-cancer comparisons on CPDB and STRING networks alongside the 15 cancer-specific evaluations (10 methods × 15 cancers × 50 folds), and generates all corresponding appendix statistical significance tables:

```bash
python scripts/paper_validation/reanalyse_statistics.py
```
- **Outputs generated**:
  - `results/paper_validation/table1_performance_comparison.csv`: Formatted Table 1 matching `main.tex`.
  - `results/paper_validation/table2_cancer_specific_comparison.csv`: Formatted Table 2 matching `main.tex`.
  - `results/paper_validation/table_A8_pancancer_significance.csv`: Formatted Table A8 summary p/q values matching `appendix.tex`.
  - `results/paper_validation/table_A9_cancer_specific_significance.csv`: Formatted Table A9 cancer-specific p/q values matching `appendix.tex`.
  - `results/paper_validation/table_A10_pancancer_significance.csv`: Formatted Table A10 detailed 36-row pan-cancer statistical table matching `appendix.tex`.
  - `results/paper_validation/table_A11_cancer_specific_significance.csv`: Formatted Table A11 detailed 270-row cancer-specific statistical table matching `appendix.tex`.
  - `results/paper_validation/benchmark_summary.csv`: Macro and micro Wilcoxon $p$-values, BH FDR $q$-values, and effect sizes.
  - `results/paper_validation/benchmark_run_means.csv`: 10-run paired averages per method and metric.
  - `results/paper_validation/cancer_type_means.csv`: Cancer-specific performance averages for all 15 cancers.
  - `results/paper_validation/sensitivity_fold50.csv`: 50-fold sensitivity analysis across all folds.

### Clean $\leftrightarrow$ Hit Cross-Distribution Leakage Benchmark

Evaluates model robustness against LLM vocabulary leakage across two cross-distribution tasks (`clean_to_hit` and `hit_to_clean`) under both **Transductive** and **Strict Inductive** settings:

```bash
python scripts/leakage_ablation/run_clean_hit_evaluation.py
```
- **Statistical Protocol**:
  - One-sided paired Wilcoxon signed-rank tests ($H_1: \text{BIMODriver} > \text{Baseline}$) for benchmark superiority comparisons.
  - Two-sided paired Wilcoxon tests ($H_1: \text{Variant}_1 \ne \text{Variant}_2$) for internal architectural ablations.
  - Benjamini-Hochberg (BH) False Discovery Rate (FDR) control across hypotheses.
  - Paired rank-biserial correlation $r_{rb}$ and Hodges-Lehmann median difference with 95% bootstrap confidence intervals (10,000 resamples).
- **Outputs generated**:
  - `results/clean_hit_results_final/clean_hit_paper_main_table.csv`
  - `results/clean_hit_results_final/clean_hit_paper_main_table_two_sided.csv`
  - `results/clean_hit_results_final/clean_hit_statistical_comparison.csv`
  - `results/clean_hit_results_final/clean_hit_descriptive_summary.csv`
  - `results/clean_hit_results_final/clean_hit_raw_scores.csv`
  - `results/clean_hit_results_final/Clean_Hit_Statistical_Validation.xlsx`

### Table A19: Edge-Masking & Strict Inductive Graph Learning

Tests the effect of edge masking and information exposure across 4 BIMODriver masking variants and baselines:

```bash
python scripts/leakage_ablation/run_table_A19_statistics.py
```
- **Outputs generated**:
  - `results/edge_masking_A19/table_A19_edge_masking_comparison.csv`
  - `results/edge_masking_A19/table_A19_extended_variants.csv`
  - `results/edge_masking_A19/table_A19_extended_summary.csv`
  - `results/edge_masking_A19/masking_vs_baselines_50fold.csv`
  - `results/edge_masking_A19/masking_vs_baselines_10run.csv`
  - `results/edge_masking_A19/masking_ablation_vs_original_50fold.csv`
  - `results/edge_masking_A19/raw_50fold_matrices.csv`
  - `results/edge_masking_A19/Table_A19_Statistical_Validation.xlsx`

### Table R5: Label-Like Phrases Masking Robustness Evaluation

Evaluates the impact of masking label-like keywords/phrases in textual driver descriptions on CPDB pan-cancer performance across transductive and inductive regimes:

```bash
python scripts/leakage_ablation/run_table_R5_statistics.py
```
- **Statistical Protocol**:
  - Raw inputs: 10-run × 5-fold matrices for `Original Transductive BIMODriver`, `Masked Transductive`, and `Masked Inductive` (`AUROC` and `AUPRC`).
  - Unit of analysis: 10 paired run-level averages ($n=10$, averaging 5 folds per run).
  - Hypothesis testing: Two-sided paired Wilcoxon signed-rank tests across $m=4$ comparisons.
  - Multiple testing correction: Benjamini-Hochberg (BH) FDR correction across all 4 comparisons as a single test family.
  - Effect sizes: Paired rank-biserial correlation ($r_{rb}$) and Hodges-Lehmann location shift (median of Walsh averages) with 95% bootstrap confidence intervals (10,000 resamples, seed=42).
- **Outputs generated**:
  - `results/label_like_phrases_masking/table_R5_label_masking_formatted.csv`: Formatted paper-ready table.
  - `results/label_like_phrases_masking/table_R5_label_masking_full_precision.csv`: Full precision metrics.
  - `results/label_like_phrases_masking/table_R5_descriptive_summary.csv`: 50-fold and 10-run descriptive statistics (Mean ± SD, ddof=0).
  - `results/label_like_phrases_masking/table_R5_run_level_means.csv`: 10 run-level paired values.
  - `results/label_like_phrases_masking/Table_R5_Statistical_Validation.xlsx`: Standardized multi-sheet Excel deliverable.
  - `results/label_like_phrases_masking/table_R5_summary.json`: Complete parameter and checksum archive.

---

## 5. Step-by-Step Figure Reproduction

Each figure script can be executed directly from the terminal without requiring Jupyter:

```bash
# Figure 2: Independent validation scatter plot (Oncogene vs OncoKB)
python scripts/figures/plot_figure2_comparison.py

# Figure 3: Ablation study component breakdown (Bar chart & Radar chart)
python scripts/figures/plot_figure3_ablation.py

# Figure 4: SMoE gating visualization & KDE distribution analysis
python scripts/figures/plot_figure4_smoe_gating.py

# Figure 5: Candidate cancer driver gene prediction & validation (Subplots a, b, c and composite)
python scripts/figures/plot_figure5_candidate_genes.py
```

All figures are automatically exported to `results/figures/` in both 300 DPI high-resolution PNG and publication-ready vector PDF formats:
- `figure2_independent_validation.png` / `.pdf`
- `figure3_ablation_study.png` / `.pdf`
- `figure3_ablation_radar.png` / `.pdf`
- `figure4_smoe_gating.png` / `.pdf`
- `figure5a_database_validation.png` / `.pdf` (Subplot 5a: Donut chart & database support bar chart)
- `figure5b_venn_diagram.png` / `.pdf` (Subplot 5b: 4-set Venn diagram)
- `figure5c_enrichment_analysis.png` / `.pdf` (Subplot 5c: KEGG & GO functional enrichment dotplots)
- `figure5_candidate_genes_composite.png` / `.pdf` (Composite Figure 5 matching the paper)

---

## 6. Interactive Jupyter Notebooks

For interactive exploration and custom styling, standardized notebooks are provided in `notebooks/`:

- `notebooks/Figure2_Figure3_Validation_Ablation.ipynb`: Interactive plotting and custom palettes for Figures 2 and 3.
- `notebooks/Figure4_SMoE_Gating_Visualization.ipynb`: Gating weight queries and distribution explorations for Figure 4.
- `notebooks/Figure5_Candidate_Genes_Validation.ipynb`: Venn diagram and multi-database validation explorations for Figure 5.
- `notebooks/PanCancer_Wilcoxon_Analysis.ipynb`: Interactive pan-cancer Wilcoxon signed-rank and BH FDR analysis.
- `notebooks/CancerSpecific_Wilcoxon_Analysis.ipynb`: Interactive cancer-specific statistical tests across 15 cancer types.

Launch Jupyter with:
```bash
jupyter lab notebooks/
```

---

## 7. Statistical Test Methodology Summary

1. **Paired Comparisons**: All statistical evaluations are strictly paired. Each test corresponds to identical 10-fold or 50-fold cross-validation splits evaluated with matched random seeds.
2. **Hypothesis Directionality**:
   - **One-sided tests** (`greater`): Applied when comparing BIMODriver against external baseline models (e.g. MNGCL, DISFusion) to evaluate whether BIMODriver achieves statistically superior performance ($H_1: \mu_{\text{BIMODriver}} > \mu_{\text{Baseline}}$).
   - **Two-sided tests** (`two-sided`): Applied when comparing self-variants or ablation configurations against each other where no prior superiority direction is assumed.
3. **Multiple Testing Correction**: Benjamini-Hochberg (BH) procedure is applied across test families to control the False Discovery Rate (FDR) at $\alpha = 0.05$.
4. **Effect Sizes**:
   - **Rank-Biserial Correlation ($r_{rb}$)**: Evaluates paired non-parametric effect size bounded in $[-1, 1]$.
   - **Hodges-Lehmann Median Difference**: Computed alongside 95% bootstrap confidence intervals derived from 10,000 resamples.
