# Raw Benchmark Evaluation Results

This directory archives the complete, unaggregated empirical metric outputs and prediction files generated across all experimental protocols reported in the paper:

> **"BIMODriver: Integrating Large Model-Generated Biological Knowledge and Multi-Omics Features for Cancer Driver Gene Identification"**

---

## Directory Organization

```text
results/
├── pan-cancer/                      # Table 1: Pan-cancer cross-validation metric arrays
│   ├── cpdb-pancancer.txt           # 10x5 CV AUROC and AUPRC arrays across 10 methods on CPDB
│   └── string-pancancer.txt         # 10x5 CV AUROC and AUPRC arrays across 10 methods on STRING
│
├── cancer_specific/                 # Table 2: 15 cancer-specific benchmark metric arrays
│   ├── BIMODriver/                  # 50-value AUC and AUPRC arrays for 15 cancer types
│   ├── ChebNet/
│   ├── DISFusion/
│   ├── DISHyper/
│   ├── ECD-CDGI/
│   ├── EMOGI/
│   ├── GAT/
│   ├── MNGCL/
│   ├── MTGCN/
│   └── SAGE/
│
└── clean_hit/                       # Clean <-> Hit vocabulary leakage & inductive benchmark
    ├── bimodriver_*                 # 10-run AUROC, AUPRC, summary, and gene prediction CSVs
    ├── disfusion_*                  # Baseline reproduction under paper settings
    ├── mngcl_*                      # Baseline reproduction under fixed-epoch paper protocol
    └── clean_hit_*.csv              # Formatted summary and statistical comparison tables
```

---

## Description of Deliverables

### 1. Pan-Cancer Benchmark (`pan-cancer/`)
- **`cpdb-pancancer.txt`**: Contains 10 runs × 5 folds (50 evaluation values) of AUROC and AUPRC for **BIMODriver** and 9 state-of-the-art baseline models on the CPDB interaction network.
- **`string-pancancer.txt`**: Contains corresponding 10x5 CV evaluation matrices on the STRING interaction network.

### 2. Cancer-Specific Benchmark (`cancer_specific/`)
Contains per-cancer evaluation values across 15 individual cancer types (BLCA, BRCA, CESC, COAD, ESCA, HNSC, KIRC, KIRP, LIHC, LUAD, LUSC, PRAD, STAD, THCA, UCEC).
Each method subfolder provides:
- `<Method>_<cancer>_AUC_50values.txt`: 50 raw AUROC values.
- `<Method>_<cancer>_AUPRC_50values.txt`: 50 raw AUPRC values.

### 3. Vocabulary Leakage & Inductive Benchmark (`clean_hit/`)
Contains 10 independent experimental runs for:
- `Clean -> Hit`: Model trained on Clean genes (devoid of driver-related vocabulary) and tested on Hit genes.
- `Hit -> Clean`: Model trained on Hit genes and tested on Clean genes.
- Evaluated under both **Transductive** (full graph topology) and **Strict Inductive** (test edges disconnected and test features zeroed during training) protocols.
- Each method provides per-gene prediction probability tables (`*_test_preds.csv`) and run summaries (`*_summary.txt`).
