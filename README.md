# BIMODriver: Integrating Large Model-Generated Biological Knowledge and Multi-Omics Features for Cancer Driver Gene Identification

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-2.3+-3C2179.svg)](https://pyg.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official PyTorch implementation of **BIMODriver**, a multimodal graph representation learning framework that integrates biological knowledge-guided large language model (LLM) semantics with multi-omics profiles and interaction networks for cancer driver gene identification.

---

## 📌 Overview

Accurate identification of cancer driver genes is essential for understanding tumorigenesis and advancing precision oncology. While existing computational methods primarily rely on multi-omics profiles and protein-protein interaction (PPI) networks, they often overlook rich functional contexts encapsulated in biological literature.

**BIMODriver** addresses this challenge through three key contributions:
1. **Domain Knowledge-Guided Semantic Extraction**: We prompt large language models (e.g., Gemma-2) with Gene Ontology (GO) terms to generate structured, context-aware descriptions at three complementary levels: self-functional profile, local topological neighbors, and collective interactions.
2. **Dual-Channel Contrastive Graph Learning**: We project multi-omics profiles and text embeddings (encoded via BioBERT) into aligned latent spaces, leveraging cross-modal contrastive learning to enhance semantic coherence while preserving modality-specific topology.
3. **Adaptive Gating Fusion**: A dynamic gating mechanism balances the contribution of omics and semantic representations for robust gene prioritization under both transductive and strict inductive regimes.

![BIMODriver Architecture](https://raw.githubusercontent.com/weiba/BIMODriver/master/overview.png) *(or refer to the publication figure)*

---

## 📂 Repository Structure

```text
BIMODriver/
├── BIMODriver/                      # Core BIMODriver framework
│   ├── main.py                      # Main training, cross-validation, and benchmark runner
│   ├── model.py                     # Neural network architectures (ChebConv, contrastive loss, gating fusion)
│   └── gcnPreprocessing.py          # Graph processing and cross-validation utilities
│
├── implement/                       # Benchmark baselines and evaluation suite
│   ├── baselines/                   # Baseline model implementations
│   │   ├── disfusion/               # Official DISFusion model (HGNN + ChebNet + Barlow Twins)
│   │   └── mngcl/                   # Official MNGCL model (Multi-view GCN + Contrastive Learning)
│   ├── alignment_check.py           # Strict dataset and feature alignment assertion script
│   ├── create_shared_splits.py      # Deterministic 10-run split generator for leakage benchmark
│   ├── run_disfusion_cv.py          # DISFusion 10x5 cross-validation reproduction
│   ├── run_disfusion_leakage.py     # DISFusion Clean <-> Hit benchmark runner
│   ├── run_mngcl_cv.py              # MNGCL 10x5 cross-validation reproduction
│   ├── run_mngcl_leakage.py         # MNGCL Clean <-> Hit benchmark runner
│   ├── summarize_leakage_results.py # Aggregator for test AUROC/AUPRC across all methods
│   ├── generate_masked_features.py  # Keyword-masking ablation feature generation utility
│   ├── CPDB_gene_index_id_name_label.csv  # Standard CPDB 13,627 gene index mapping
│   └── Gemma_Vocabulary_Leakage_Audit.xlsx # Complete gene-level vocabulary audit annotations
│
├── LLM/                             # Biological knowledge prompt and statement extraction
│   ├── LLM-go-part.py               # GO-guided prompt pipeline for BP, MF, and CC descriptions
│   ├── LLM-satment.py               # Multi-scale statement generator (self, neighbor, together)
│   ├── extract_embeddings.py        # BioBERT statement and GO embedding extraction CLI
│   ├── contxt_prompt.txt            # Domain knowledge context prompt template
│   ├── go_prompt.txt                # Gene Ontology prompt template
│   └── README.md                    # Detailed documentation for LLM feature generation
│
├── paper_figures_tables/            # Statistical validation & paper figure reproduction suite
│   ├── reproduce_all.py             # Master one-click reproduction CLI entrypoint
│   ├── REPRODUCTION_GUIDE.md        # Comprehensive paper item-to-code mapping guide
│   ├── scripts/                     # Standalone figure plotting and statistical test scripts
│   ├── notebooks/                   # Interactive Jupyter Notebooks for Figures 2-5
│   ├── data/                        # 10x5 cross-validation arrays and supplementary matrices
│   ├── results/                     # Generated publication-ready figures (PNG/PDF) and workbooks
│   └── README.md                    # Dedicated statistical suite documentation
│
├── results/                         # Raw paper benchmark evaluation metric files
│   ├── pan-cancer/                  # Table 1: CPDB & STRING 10x5 CV raw metric arrays for 10 methods
│   ├── cancer_specific/             # Table 2: 15 cancer types 50-value metrics for 10 methods
│   ├── clean_hit/                   # Clean <-> Hit benchmark 10-run metric arrays & predictions
│   └── README.md                    # Detailed documentation for raw results
│
├── data/                            # Multi-omics features and biological network archives
│   ├── data.part01.rar ~ part27.rar # Multi-volume compressed dataset archives
│   └── ...
│
├── run_benchmark.sh                 # One-click script to reproduce the full benchmark suite
└── README.md
```

---

## ⚙️ Environment Setup

### 1. Prerequisites
- Linux OS (Ubuntu 20.04/22.04 recommended)
- Python >= 3.9
- CUDA >= 11.8 with compatible NVIDIA GPU (RTX 3090, RTX 4090, A100, etc.)

### 2. Conda Installation
We recommend using a dedicated Conda environment:

```bash
# Clone the repository
git clone https://github.com/weiba/BIMODriver.git
cd BIMODriver

# Create and activate environment
conda create -n bimodriver python=3.9 -y
conda activate bimodriver

# Install PyTorch and PyG (adjust CUDA version if necessary)
pip install torch==2.0.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric==2.3.1
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.1+cu118.html

# Install additional scientific dependencies
pip install numpy==1.26.4 pandas==2.2.1 scikit-learn==1.4.2 scipy==1.13.0 openpyxl transformers
```

---

## 📦 Data Preparation

The preprocessed datasets (multi-omics profiles, network topologies, and semantic embeddings) are stored as split RAR archives in `data/` to comply with Git file size limitations.

Extract the archives using `unrar`:

```bash
# Extract all multi-volume data archives
unrar x data/data.part01.rar data/
```

After extraction, the following directories will be available under `data/`:
- `data/CPDB/`: CPDB multi-omics features (`CPDB_new_data.pt`), network edge indices (`CPDB_merged_k5_edge_index.pt`), precomputed semantic embeddings (`PAN-CANCER_statement_features.pt`), and 10-run split files.
- `data/STRING/`: STRING dataset profiles and networks.
- `data/DISFusion_data/`: Hypergraph incidence matrices and annotations for DISFusion baseline.

Verify data integrity and feature alignment:
```bash
python implement/alignment_check.py
```

---

## 🚀 Usage & Reproduction

### 1. Standard 10×5 Cross-Validation (Pan-Cancer & Specific Cancers)

To run standard transductive 10-run 5-fold cross-validation on the CPDB benchmark:

```bash
python BIMODriver/main.py --split cv --dataset cpdb --cancerType pan-cancer
```

For specific cancer types (e.g., BRCA, LUAD, GBM):
```bash
python BIMODriver/main.py --split cv --dataset cpdb --cancerType BRCA
```

### 2. Strict Inductive 10×5 Cross-Validation

In strict inductive evaluation, all topological edges incident to test genes are removed during training, test gene features are masked, and full graph topology is only restored at test inference:

```bash
python BIMODriver/main.py --split inductive --dataset cpdb --cancerType pan-cancer
```

### 3. LLM Vocabulary Leakage Benchmark (Clean $\leftrightarrow$ Hit Cross-Distribution)

To evaluate robustness against potential LLM memorization or vocabulary leakage, we partition genes into a Clean group (free of driver keywords) and a Hit group (containing sensitive driver-related vocabulary):

#### A. Transductive Setting
```bash
# Train on Clean, test on Hit
python BIMODriver/main.py --split clean_to_hit

# Train on Hit, test on Clean
python BIMODriver/main.py --split hit_to_clean
```

#### B. Strict Inductive Setting
```bash
# Train on Clean, test on Hit (inductive: cut test edges & mask test features during training)
python BIMODriver/main.py --split clean_to_hit --inductive

# Train on Hit, test on Clean (inductive)
python BIMODriver/main.py --split hit_to_clean --inductive
```

### 4. Full Benchmark Suite (One-Click Reproduction)

We provide an automated runner to execute the full evaluation suite comparing **BIMODriver**, **DISFusion**, and **MNGCL** across 10 runs in both Transductive and Strict Inductive settings:

```bash
# Usage: bash run_benchmark.sh [all|transductive|inductive] [GPU_ID]
bash run_benchmark.sh all 0
```

Results will be automatically summarized and displayed as side-by-side comparison tables.

---

## 📊 Benchmark Results

### 1. Transductive Benchmark on Clean $\leftrightarrow$ Hit Splits (10 Runs)
| Method | Clean $\to$ Hit AUROC | Clean $\to$ Hit AUPRC | Hit $\to$ Clean AUROC | Hit $\to$ Clean AUPRC |
| :--- | :---: | :---: | :---: | :---: |
| **BIMODriver (Ours)** | **0.9288 ± 0.0032** | **0.9014 ± 0.0051** | **0.8986 ± 0.0029** | **0.7764 ± 0.0053** |
| MNGCL (500 ep) | 0.9245 ± 0.0033 | 0.8953 ± 0.0048 | 0.8835 ± 0.0050 | 0.7604 ± 0.0089 |
| DISFusion (200 ep) | 0.9247 ± 0.0019 | 0.8951 ± 0.0046 | 0.8871 ± 0.0067 | 0.7656 ± 0.0122 |

### 2. Strict Inductive Benchmark on Clean $\leftrightarrow$ Hit Splits (10 Runs)
| Method | Clean $\to$ Hit AUROC | Clean $\to$ Hit AUPRC | Hit $\to$ Clean AUROC | Hit $\to$ Clean AUPRC |
| :--- | :---: | :---: | :---: | :---: |
| **BIMODriver (Ours)** | **0.9300 ± 0.0026** | **0.9029 ± 0.0047** | **0.8998 ± 0.0028** | **0.7778 ± 0.0041** |
| MNGCL (500 ep) | 0.9243 ± 0.0031 | 0.8953 ± 0.0049 | 0.8831 ± 0.0051 | 0.7606 ± 0.0076 |
| DISFusion (200 ep) | 0.9255 ± 0.0023 | 0.8948 ± 0.0048 | 0.8904 ± 0.0072 | 0.7711 ± 0.0137 |

Across all scenarios, **BIMODriver achieves superior performance** on both AUROC and AUPRC metrics with statistically significant improvements ($p < 0.01$ over MNGCL, $p < 0.05$ over DISFusion). Complete raw metric files, summaries, and gene-level prediction probability tables are provided in [`results/`](results/).

---

## 📈 Statistical Validation & Figure Reproduction Suite (`paper_figures_tables/`)

To support rigorous scientific auditability and direct peer review, we provide a complete, self-contained reproduction and statistical analysis suite in [`paper_figures_tables/`](paper_figures_tables/).

### 1. Scope & Deliverables
- **Tables 1 & 2**: Pan-cancer (CPDB & STRING) and 15 individual cancer benchmarks (10 runs × 5-fold CV across 10 methods).
- **Statistical Significance (Tables A8–A11)**: Exact Wilcoxon signed-rank tests, Benjamini–Hochberg (BH) FDR corrections, rank-biserial effect sizes ($r_{rb}$), and Hodges–Lehmann median differences with 95% bootstrap confidence intervals.
- **Vocabulary Leakage & Masking Ablations**: Clean $\leftrightarrow$ Hit cross-distribution evaluations, edge-masking information exposure (Table A19), and label-like phrase masking (Table R5).
- **Publication Figures (Figures 2–5)**: Standalone plotting scripts producing publication-grade 300 DPI PNG and vector PDF figures (including Sparse Mixture-of-Experts gating visualizations and candidate driver gene functional validation).
- **Standardized Workbooks**: Multi-tab Excel workbooks containing complete empirical runs, sensitivity matrices, and statistical deliverables.

### 2. Environment Setup
```bash
# Using Conda
conda env create -f paper_figures_tables/environment.yml
conda activate bimodriver-statistics

# Or using Pip
pip install -r paper_figures_tables/requirements.txt
```

### 3. Master One-Click Reproduction
```bash
# Execute the full reproduction pipeline (tables, statistical tests, figures, workbooks, verification)
python paper_figures_tables/reproduce_all.py --all

# Or execute individual components:
python paper_figures_tables/reproduce_all.py --tables     # Tables 1, 2, A8-A11, Clean-Hit, A19, R5
python paper_figures_tables/reproduce_all.py --figures    # Figures 2, 3, 4, 5 (PNG & PDF)
python paper_figures_tables/reproduce_all.py --workbooks  # Standardized multi-sheet Excel deliverables
python paper_figures_tables/reproduce_all.py --verify     # Audit integrity of all generated artifacts
```

For exhaustive item-to-code mapping and methodology specifications, see [`paper_figures_tables/README.md`](paper_figures_tables/README.md) and [`paper_figures_tables/REPRODUCTION_GUIDE.md`](paper_figures_tables/REPRODUCTION_GUIDE.md).

---

## 📜 Citation

If you find our work useful in your research, please cite our paper:

```bibtex
@article{bimodriver2026,
  title={Integrating Large Model-Generated Biological Knowledge and Multi-Omics Features for Cancer Driver Gene Identification},
  author={Wei, Peng and Zhang, Qian and Tao, Yuan and et al.},
  journal={Bioinformatics / IEEE Transactions on Computational Biology and Bioinformatics},
  year={2026}
}
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
