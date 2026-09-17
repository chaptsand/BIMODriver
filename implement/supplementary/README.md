# BIMODriver 审稿补充实验套件 (Supplementary Experiments)

本目录汇集了用于审稿回复与消融分析的补充实验套件。所有脚本均已完成路径规范化，支持从仓库根目录直接调用，统一通过相对路径动态检索 `data/` 中的网络与多组学特征，实验结果保存至 `result/` 或 `outputs/`。

---

## 模块结构与实验清单

| 脚本文件 | 说明 | 对应实验 / 作用 |
|---|---|---|
| `main_top4_routing.py` / `model_top4_routing.py` | Top-4 门控路由导出与分析 | 导出 OOF 门控选择概率与每折指标 (Figure 1) |
| `main_expert_deletion.py` / `model_expert_deletion.py` | 专家删除忠实性检验 (Faithfulness) | 验证门控被选中专家的必要性与统计显著性 |
| `main_sparse_dense.py` / `model_sparse_dense.py` | 稀疏与稠密专家融合对比 | 对比 Top-4 稀疏门控与 6 专家稠密全融合的效果 |
| `main_A16_bert_base.py` / `model_A16_bert_base.py` | BERT-base 语义基线消融 | 使用标准 BERT-base-cased 编码替换 LLM 语义表示 (Table A16) |
| `bert_embedding_A16.py` | BERT 语义表示提取脚本 | 从 GO/statement 文本提取 A16 所需张量 |
| `main_A17.py` / `model_A17.py` | 严格归纳与防泄漏实验 | 验证隐藏测试基因端点边与特征置零下的模型表现 (Table A17) |
| `run_A17_all.py` | A17 跨数据集自动化流水线 | 依次运行 CPDB / STRING 下的原始与归纳式对照 |

---

## 运行方式 (均从项目根目录执行)

### 1. Top-4 门控路由分析 (Figure 1)

- **快速检查 (1 exp, 1 fold, 1 epoch)**：
  ```bash
  python implement/supplementary/main_top4_routing.py \
    --protocol transductive \
    --experiments 1 \
    --folds 1 \
    --epochs 1
  ```
- **完整实验 (10 次 5 折, 160 epochs)**：
  ```bash
  # 传导式 (Transductive)
  python implement/supplementary/main_top4_routing.py --protocol transductive --experiments 10 --folds 5 --epochs 160

  # 严格归纳式 (Inductive)
  python implement/supplementary/main_top4_routing.py --protocol inductive --experiments 10 --folds 5 --epochs 160
  ```
- **默认产物**：`cache/routing_top4/CPDB/cpdb/<protocol>/oof_gene_routing.csv` 与 `metrics.json`。

---

### 2. 专家删除忠实性检验 (Expert Deletion Faithfulness)

- **快速检查**：
  ```bash
  python implement/supplementary/main_expert_deletion.py \
    --protocol transductive \
    --experiments 1 \
    --folds 1 \
    --epochs 1 \
    --bootstrap-samples 100
  ```
- **完整实验**：
  ```bash
  python implement/supplementary/main_expert_deletion.py \
    --protocol transductive \
    --experiments 10 \
    --folds 5 \
    --epochs 160 \
    --bootstrap-samples 10000
  ```
- **默认产物**：`outputs/expert_deletion/CPDB/cpdb/<protocol>/`（包含路由明细、逐基因效应与 Bootstrap 统计结果）。

---

### 3. 稀疏与稠密专家融合对比 (Sparse vs. Dense)

- **运行命令**：
  ```bash
  # 快速运行 (1 次重复)
  python implement/supplementary/main_sparse_dense.py --setting transductive --experiments 1

  # 完整 10 次重复
  python implement/supplementary/main_sparse_dense.py --setting transductive --experiments 10
  python implement/supplementary/main_sparse_dense.py --setting inductive --experiments 10
  ```
- **默认产物**：`cache/original_framework_sparse_dense_<setting>/<timestamp>/`。

---

### 4. Table A17 严格归纳与跨数据集防泄漏实验

- **全量流水线运行 (CPDB & STRING)**：
  ```bash
  python implement/supplementary/run_A17_all.py
  ```
- **单项配置运行**：
  ```bash
  # CPDB 原始设置
  A17_DATASET=cpdb A17_STRICT_MASKING=0 python implement/supplementary/main_A17.py

  # CPDB 严格特征与边屏蔽归纳设置
  A17_DATASET=cpdb A17_STRICT_MASKING=1 python implement/supplementary/main_A17.py
  ```
- **默认产物**：`result/A17/`。

---

### 5. Table A16 BERT-base 语义基线

- **特征生成 (可选，已预置则跳过)**：
  ```bash
  python implement/supplementary/bert_embedding_A16.py \
    --input implement/CANCER_go_features3.csv \
    --output data/CPDB/PAN-CANCER_statement_features_bert_base.pt
  ```
- **基线模型训练**：
  ```bash
  python implement/supplementary/main_A16_bert_base.py
  ```
- **默认产物**：`result/A16/`。
