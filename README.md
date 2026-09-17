# BIMODriver: Integrating Large Model-Generated Biological Knowledge and Multi-Omics Features for Cancer Driver Gene Identification

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-2.3+-3C2179.svg)](https://pyg.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official PyTorch implementation of **BIMODriver**, a multimodal graph representation learning framework that integrates biological knowledge-guided large language model (LLM) semantics with multi-omics profiles and interaction networks for cancer driver gene identification.

---

## Overview

Accurate identification of cancer driver genes is essential for understanding tumorigenesis and advancing precision oncology. While existing computational methods primarily rely on multi-omics profiles and protein-protein interaction (PPI) networks, they often overlook rich functional contexts encapsulated in biological literature.

**BIMODriver** addresses this challenge through three key contributions:
1. **Domain Knowledge-Guided Semantic Extraction**: We prompt large language models (e.g., Gemma-2) with Gene Ontology (GO) terms to generate structured, context-aware descriptions at three complementary levels: self-functional profile, local topological neighbors, and collective interactions.
2. **Dual-Channel Contrastive Graph Learning**: We project multi-omics profiles and text embeddings (encoded via BioBERT) into aligned latent spaces, leveraging cross-modal contrastive learning to enhance semantic coherence while preserving modality-specific topology.
3. **Adaptive Gating Fusion**: A dynamic gating mechanism balances the contribution of omics and semantic representations for robust gene prioritization under both transductive and strict inductive regimes.

---

## Repository Structure

```text
BIMODriver/
├── BIMODriver/                      # Core BIMODriver framework (model, training, cross-validation)
│   ├── main.py                      # Main training and evaluation runner
│   ├── model.py                     # Neural network architectures (ChebConv, contrastive loss, gating)
│   ├── gcnPreprocessing.py          # Graph processing and cross-validation utilities
│   └── single/                      # 15 single-cancer training and evaluation pipeline
│       ├── main_15_single_cancers.py
│       └── MOE-config-15cancer.yaml
│
├── implement/                       # Benchmark reproduction suite and baselines
│   ├── baselines/                   # Native implementations of DISFusion and MNGCL
│   ├── alignment_check.py           # Feature and data alignment verification script
│   ├── create_shared_splits.py      # Deterministic 10-run split generator
│   ├── run_disfusion_cv.py          # DISFusion 10x5 cross-validation runner
│   ├── run_disfusion_leakage.py     # DISFusion Clean <-> Hit benchmark runner
│   ├── run_mngcl_cv.py              # MNGCL 10x5 cross-validation runner
│   ├── run_mngcl_leakage.py         # MNGCL Clean <-> Hit benchmark runner
│   └── summarize_leakage_results.py # Metric summarizer across evaluated methods
│
├── LLM/                             # Biological knowledge prompt and statement extraction
│   ├── LLM-go-part.py               # GO-guided prompt pipeline for BP, MF, and CC descriptions
│   ├── LLM-satment.py               # Multi-scale statement generator
│   ├── extract_embeddings.py        # BioBERT statement and GO embedding extraction CLI
│   └── README.md                    # Detailed documentation for the LLM pipeline
│
├── paper_figures_tables/            # Statistical validation and paper figure reproduction suite
│   ├── reproduce_all.py             # Master one-click reproduction script
│   ├── REPRODUCTION_GUIDE.md        # Detailed paper item-to-code mapping guide
│   └── README.md                    # Dedicated reproduction documentation
│
├── results/                         # Raw paper benchmark evaluation metric files
│   ├── pan-cancer/                  # Table 1: CPDB & STRING 10x5 CV raw metric arrays
│   ├── cancer_specific/             # Table 2: 15 cancer types 50-value metrics for 10 methods
│   └── README.md                    # Detailed results directory documentation
│
├── data/                            # Multi-omics features and biological network archives
│   ├── data.part01.rar ~ part27.rar # Multi-volume compressed dataset archives
│   └── ...
│
├── run_benchmark.sh                 # Unified script to reproduce the full benchmark suite
└── README.md
```

---

## Installation

### Prerequisites
- Linux OS (Ubuntu 20.04/22.04 recommended)
- Python >= 3.9
- CUDA >= 11.8 with compatible NVIDIA GPU

### Environment Setup
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

# Install dependencies
pip install numpy==1.26.4 pandas==2.2.1 scikit-learn==1.4.2 scipy==1.13.0 openpyxl transformers
```

---

## Data Preparation

The preprocessed datasets (multi-omics profiles, network topologies, and semantic embeddings) are stored as split RAR archives in `data/`.

Extract the archives using `unrar`:

```bash
unrar x data/data.part01.rar data/
```

Verify data integrity and feature alignment across all models:
```bash
python implement/alignment_check.py
```

---

## Usage

### 1. Standard 10×5 Cross-Validation (Pan-Cancer & Cancer-Specific)
```bash
# Pan-cancer on CPDB
python BIMODriver/main.py --split cv --dataset cpdb --cancerType pan-cancer

# Cancer-specific (e.g., blca, brca, or all 15 cancers sequentially)
python BIMODriver/main.py --split cv --dataset cpdb --cancerType blca
python BIMODriver/main.py --split cv --dataset cpdb --cancerType all_15

# Dedicated 15 single-cancer pipeline
python BIMODriver/single/main_15_single_cancers.py
```

### 2. Strict Inductive Evaluation
All edges incident to test genes are cut during training, and test features are masked:
```bash
python BIMODriver/main.py --split inductive --dataset cpdb --cancerType pan-cancer
```

### 3. Vocabulary Leakage Benchmark (Clean $\leftrightarrow$ Hit)
```bash
# Transductive
python BIMODriver/main.py --split clean_to_hit
python BIMODriver/main.py --split hit_to_clean

# Strict Inductive
python BIMODriver/main.py --split clean_to_hit --inductive
python BIMODriver/main.py --split hit_to_clean --inductive
```

### 4. Full Benchmark Reproduction
To run the automated suite evaluating **BIMODriver**, **DISFusion**, and **MNGCL** across 10 runs in both Transductive and Strict Inductive settings:
```bash
bash run_benchmark.sh all 0
```

Raw evaluation metrics for the paper's main benchmarks (Tables 1 and 2) are archived in [`results/`](results/).

---

## Statistical Validation & Paper Reproduction Suite

For independent verification of all statistical significance tests (Wilcoxon signed-rank tests, Benjamini–Hochberg FDR, effect sizes), ablations, and paper figure generation (Figures 2–5), please refer to the dedicated suite in **[`paper_figures_tables/`](paper_figures_tables/)**.

See [`paper_figures_tables/README.md`](paper_figures_tables/README.md) and [`paper_figures_tables/REPRODUCTION_GUIDE.md`](paper_figures_tables/REPRODUCTION_GUIDE.md) for full reproduction guides.

---

## Citation

If you find our work useful in your research, please cite:

```bibtex
@article{bimodriver2026,
  title={Integrating Large Model-Generated Biological Knowledge and Multi-Omics Features for Cancer Driver Gene Identification},
  author={Wei, Peng and Zhang, Qian and Tao, Yuan and et al.},
  journal={Bioinformatics / IEEE Transactions on Computational Biology and Bioinformatics},
  year={2026}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).
