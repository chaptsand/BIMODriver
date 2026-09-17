# Raw Benchmark Evaluation Results

This directory archives the complete, unaggregated empirical metric outputs for the paper's primary benchmarks (Tables 1 and 2):

> **"BIMODriver: Integrating Large Model-Generated Biological Knowledge and Multi-Omics Features for Cancer Driver Gene Identification"**

---

## Directory Organization

```text
results/
├── pan-cancer/                      # Table 1: Pan-cancer cross-validation metric arrays
│   ├── cpdb-pancancer.txt           # 10x5 CV AUROC and AUPRC arrays across 10 methods on CPDB
│   └── string-pancancer.txt         # 10x5 CV AUROC and AUPRC arrays across 10 methods on STRING
│
└── cancer_specific/                 # Table 2: 15 cancer-specific benchmark metric arrays
    ├── BIMODriver/                  # 50-value AUC and AUPRC arrays for 15 cancer types
    ├── ChebNet/
    ├── DISFusion/
    ├── DISHyper/
    ├── ECD-CDGI/
    ├── EMOGI/
    ├── GAT/
    ├── MNGCL/
    ├── MTGCN/
    └── SAGE/
```

> [!NOTE]
> For the Clean $\leftrightarrow$ Hit vocabulary leakage evaluations, edge-masking ablations (Table A19), phrase masking (Table R5), and statistical significance test deliverables, please refer directly to [`paper_figures_tables/`](../paper_figures_tables/).

---

## Description of Deliverables

### 1. Pan-Cancer Benchmark (`pan-cancer/`)
- **`cpdb-pancancer.txt`**: Contains 10 runs × 5 folds (50 evaluation values) of AUROC and AUPRC for **BIMODriver** and 9 baseline models on the CPDB interaction network.
- **`string-pancancer.txt`**: Contains corresponding 10x5 CV evaluation matrices on the STRING interaction network.

### 2. Cancer-Specific Benchmark (`cancer_specific/`)
Contains per-cancer evaluation values across 15 individual cancer types (BLCA, BRCA, CESC, COAD, ESCA, HNSC, KIRC, KIRP, LIHC, LUAD, LUSC, PRAD, STAD, THCA, UCEC).
Each method subfolder provides:
- `<Method>_<cancer>_AUC_50values.txt`: 50 raw AUROC values.
- `<Method>_<cancer>_AUPRC_50values.txt`: 50 raw AUPRC values.
