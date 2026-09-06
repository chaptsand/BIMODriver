"""
Script to generate keyword-masked semantic features for BIMODriver.

Rules:
1. Mask explicit label keywords:
   - 'tumor suppressor' (\btumou?rs?[\s-]+suppressors?\b)
   - 'oncogene' (\boncogenes?\b)
   - 'promote(s) tumor' (\bpromote(?:s)?\s+tumou?rs?\b)
   Preserve ordinary 'cancer/cancers'.
2. Replace matched terms with '[MASK]' using BioBERT tokenizer.
3. Re-encode hit genes with BioBERT (dmis-lab/biobert-base-cased-v1.2) CLS tokens (768*3 = 2304 dims).
4. Keep unhit genes exactly identical to original PAN-CANCER_statement_features.pt.
5. Save as data/CPDB/PAN-CANCER_statement_features_masked.pt without overwriting original.
6. Perform full consistency and identity checks.
"""

import os
import re
import time
import torch
import numpy as np
import pandas as pd
from transformers import BertModel, BertTokenizer


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data', 'CPDB')
    src_dir = os.path.join(base_dir, 'src')

    csv_path = os.path.join(src_dir, 'CANCER_go_features3.csv')
    meta_path = os.path.join(src_dir, 'CPDB_gene_index_id_name_label.csv')
    orig_feat_path = os.path.join(data_dir, 'PAN-CANCER_statement_features.pt')
    out_feat_path = os.path.join(data_dir, 'PAN-CANCER_statement_features_masked.pt')

    print("=" * 75)
    print("1. 加载原始数据与校验对齐")
    print(f"  读取文本文件: {csv_path}")
    df_csv = pd.read_csv(csv_path)
    df_meta = pd.read_csv(meta_path, encoding='utf-8-sig')

    assert len(df_csv) == 13627, f"CANCER_go_features3.csv 行数不匹配: {len(df_csv)}"
    assert len(df_meta) == 13627, f"CPDB_gene_index_id_name_label.csv 行数不匹配: {len(df_meta)}"
    assert (df_meta['Code_Index'].values == np.arange(len(df_meta))).all(), "Code_Index 必须连续严格对齐 0..13626"
    assert (df_csv['Gene_Symbol'].values == df_meta['Gene_Name'].values).all(), "Gene_Symbol 与 Gene_Name 顺序必须 100% 一致"
    print("  [OK] 13,627 个基因节点与 Code_Index 严格按序对齐。")

    print(f"  读取原始语义特征: {orig_feat_path}")
    orig_tensor = torch.load(orig_feat_path, map_location='cpu')
    assert orig_tensor.shape == (13627, 2304), f"原始特征形状不匹配: {orig_tensor.shape}"
    print(f"  [OK] 原始特征张量形状: {orig_tensor.shape}")

    print("\n" + "=" * 75)
    print("2. 关键词匹配与统计")
    patterns = {
        'tumor suppressor': re.compile(r'\btumou?rs?[\s-]+suppressors?\b', re.IGNORECASE),
        'oncogene': re.compile(r'\boncogenes?\b', re.IGNORECASE),
        'promote(s) tumor': re.compile(r'\bpromote(?:s)?\s+tumou?rs?\b', re.IGNORECASE)
    }
    combined_pattern = re.compile(
        r'(\btumou?rs?[\s-]+suppressors?\b|\boncogenes?\b|\bpromote(?:s)?\s+tumou?rs?\b)',
        re.IGNORECASE
    )

    fields = [
        ('self_statement', 'Self'),
        ('neighbor_statement', 'Neighbor'),
        ('together_statement', 'Together')
    ]

    stats_text_hits = {}
    stats_word_hits = {}

    for field, name in fields:
        stats_text_hits[name] = 0
        stats_word_hits[name] = 0
        for term, pat in patterns.items():
            t_hits = sum(1 for t in df_csv[field].dropna() if pat.search(str(t)))
            w_hits = sum(len(list(pat.finditer(str(t)))) for t in df_csv[field].dropna())
            print(f"  {name:8s} | {term:18s} -> {t_hits:4d} 条文本命中, {w_hits:4d} 个词命中")
        
        # 整体命中该类别
        c_text = sum(1 for t in df_csv[field].dropna() if combined_pattern.search(str(t)))
        c_word = sum(len(list(combined_pattern.finditer(str(t)))) for t in df_csv[field].dropna())
        stats_text_hits[name] = c_text
        stats_word_hits[name] = c_word
        print(f"  >> {name:8s} 合计: {c_text} 条文本命中, {c_word} 个关键词将被替换为 [MASK]")

    # 判定每个基因是否命中关键词
    hit_mask = np.zeros(len(df_csv), dtype=bool)
    for field, name in fields:
        field_hit = df_csv[field].dropna().apply(lambda s: bool(combined_pattern.search(str(s)))).values
        hit_mask = hit_mask | field_hit

    n_hit_genes = int(hit_mask.sum())
    n_unhit_genes = int((~hit_mask).sum())
    print(f"\n  >> 总计被修改的基因数量: {n_hit_genes} (未命中保持不变的基因数量: {n_unhit_genes})")
    assert n_hit_genes == 2626, f"期望 2626 个基因命中，实际 {n_hit_genes}"

    print("\n" + "=" * 75)
    print("3. 加载 BioBERT 模型并重新编码命中基因")
    device = torch.device('cuda' if torch.cuda.is_available() and os.environ.get('DEVICE') != 'cpu' else 'cpu')
    print(f"  使用设备: {device}")

    model_id = 'dmis-lab/biobert-base-cased-v1.2'
    print(f"  加载模型与分词器: {model_id} ...")
    tokenizer = BertTokenizer.from_pretrained(model_id)
    model = BertModel.from_pretrained(model_id).to(device)
    model.eval()

    new_tensor = orig_tensor.clone()

    start_time = time.time()
    hit_indices = np.where(hit_mask)[0]

    print(f"  开始对 {len(hit_indices)} 个命中基因进行 Masked 文本重新编码 ...")
    
    with torch.no_grad():
        for count, idx in enumerate(hit_indices):
            row = df_csv.iloc[idx]
            masked_texts = []
            for field, _ in fields:
                orig_text = str(row[field]) if pd.notna(row[field]) else ""
                masked_text = combined_pattern.sub('[MASK]', orig_text)
                masked_texts.append(masked_text)

            inputs = tokenizer(masked_texts, return_tensors='pt', padding=True, truncation=True, max_length=1024)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            cls_tokens = outputs.last_hidden_state[:, 0, :]  # shape: [3, 768]
            combined_emb = torch.cat([cls_tokens[0], cls_tokens[1], cls_tokens[2]], dim=0).cpu()

            new_tensor[idx] = combined_emb

            if (count + 1) % 500 == 0 or (count + 1) == len(hit_indices):
                elapsed = time.time() - start_time
                print(f"  进度: [{count + 1:4d}/{len(hit_indices):4d}] ({((count + 1)/len(hit_indices))*100:.1f}%) | 耗时: {elapsed:.1f}s")

    print(f"  编码完成，总耗时: {time.time() - start_time:.1f}s")

    print("\n" + "=" * 75)
    print("4. 全面一致性与正确性严格校验")
    # 校验 1：张量形状
    assert new_tensor.shape == (13627, 2304), f"新特征形状错误: {new_tensor.shape}"
    print(f"  [Check 1 Passed] 新特征张量形状: {new_tensor.shape} (严格等于 [13627, 2304])")

    # 校验 2：未命中基因完全一致（无任何浮点偏差）
    unhit_indices = np.where(~hit_mask)[0]
    unhit_equal = torch.equal(new_tensor[unhit_indices], orig_tensor[unhit_indices])
    assert unhit_equal, "未命中基因的特征与原特征不完全一致！"
    print(f"  [Check 2 Passed] {len(unhit_indices)} 个未命中基因新旧特征完全相同 (torch.equal = True)")

    # 校验 3：命中基因确实发生改变，且具有高度生物语义连续性
    hit_diff = (new_tensor[hit_indices] - orig_tensor[hit_indices]).abs().max().item()
    assert hit_diff > 0.01, f"命中基因的特征未发生明显变化: max_diff={hit_diff}"
    cos_sims = torch.cosine_similarity(new_tensor[hit_indices], orig_tensor[hit_indices], dim=1)
    mean_cos_sim = cos_sims.mean().item()
    min_cos_sim = cos_sims.min().item()
    print(f"  [Check 3 Passed] {len(hit_indices)} 个命中基因已更新 (最大绝对差: {hit_diff:.4f}, 平均余弦相似度: {mean_cos_sim:.4f}, 最小余弦相似度: {min_cos_sim:.4f})")

    # 校验 4：保存新特征文件
    print(f"\n5. 保存新语义特征文件至: {out_feat_path}")
    os.makedirs(os.path.dirname(out_feat_path), exist_ok=True)
    torch.save(new_tensor, out_feat_path)
    print("  [OK] 保存成功！")

    # 重新加载确认文件完整性
    reload_tensor = torch.load(out_feat_path, map_location='cpu')
    assert torch.equal(reload_tensor, new_tensor), "重新加载的张量与内存张量不一致"
    print("  [OK] 重新读取校验一致！")
    print("=" * 75)

    # 打印最终验收摘要
    print("\n【特征生成与一致性检查最终汇总】")
    print(f"- 基因节点总数: 13,627 (Code_Index 0..13626 严格对齐)")
    print(f"- 被 Mask 修改的基因数: {n_hit_genes}")
    print(f"- 完全未被修改的基因数: {n_unhit_genes}")
    print(f"- Self Statement:     {stats_text_hits['Self']} 篇文本, {stats_word_hits['Self']} 处关键词被替换为 [MASK]")
    print(f"- Neighbor Statement: {stats_text_hits['Neighbor']} 篇文本, {stats_word_hits['Neighbor']} 处关键词被替换为 [MASK]")
    print(f"- Together Statement: {stats_text_hits['Together']} 篇文本, {stats_word_hits['Together']} 处关键词被替换为 [MASK]")
    print(f"- 关键词替换总次数:   {sum(stats_word_hits.values())} 处")
    print(f"- 原始语义特征路径:   {orig_feat_path}")
    print(f"- 新 Masked 特征路径: {out_feat_path}")


if __name__ == '__main__':
    main()
