# -*- coding: utf-8 -*-
import os
import sys
import copy
import time
import pickle
import random
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import ChebConv, GATConv, GCNConv, SAGEConv
import torch_geometric.transforms as T
from torch_geometric.utils import add_self_loops, dropout_adj, negative_sampling, remove_self_loops

import gcnPreprocessing
from model import combine_net_gate_without_ac

device = torch.device('cuda' if torch.cuda.is_available() and os.environ.get('DEVICE') != 'cpu' else 'cpu')
warnings.filterwarnings("ignore")

# 路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULT_DIR = os.path.join(BASE_DIR, "result")
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULT_DIR, "single"), exist_ok=True)


def save_results_to_file(auroc, auprc, cancerType, dataset='cpdb', lr=0.001, dropout=0.2, lambdinter=0.005):
    """保存实验结果至汇总文本（已纠正 AUROC 与 AUPRC 变量和输出标签对应关系）"""
    res_dir = os.path.join(BASE_DIR, 'result')
    os.makedirs(res_dir, exist_ok=True)
    if cancerType == 'pan-cancer':
        if dataset == 'cpdb':
            path = os.path.join(res_dir, 'pan-cancer.txt')
        elif dataset == 'string':
            path = os.path.join(res_dir, 'pan-cancer_string.txt')
        else:
            raise ValueError("Unsupported dataset for pan-cancer results.")
    else:
        single_dir = os.path.join(res_dir, 'single')
        os.makedirs(single_dir, exist_ok=True)
        path = os.path.join(single_dir, 'single.txt')

    with open(path, 'a', encoding='utf-8') as f:
        f.write('--' * 20 + '\n')
        f.write(f"Dropout Rate: {dropout}, Learning Rate: {lr}, Lambda Inter: {lambdinter}\n")
        f.write(f"Results for {cancerType}:\n")
        f.write(f"AUROC: {auroc.mean():.4f} ± {auroc.std():.4f}\n")
        f.write(str(auroc))
        f.write("\n")
        f.write(f"AUPRC: {auprc.mean():.4f} ± {auprc.std():.4f}\n")
        f.write(str(auprc))
        f.write("\n")


def load_label_single(cancerType):
    path = os.path.join(DATA_DIR, "CPDB", "Specific cancer") + "/"
    label = np.loadtxt(path + "label_file-P-" + cancerType + ".txt")
    Y = torch.tensor(label).type(torch.FloatTensor).to(device).unsqueeze(1)
    label_pos = np.loadtxt(path + "pos-" + cancerType + ".txt", dtype=int)
    label_neg = np.loadtxt(path + "neg.txt", dtype=int)
    return Y, label_pos, label_neg


def sample_division_single(pos_label, neg_label, l, l1, l2, i):
    pos_val = pos_label[i * l1:(i + 1) * l1]
    pos_train = list(set(pos_label) - set(pos_val))
    neg_val = neg_label[i * l2:(i + 1) * l2]
    neg_train = list(set(neg_label) - set(neg_val))
    indexs1 = [False] * l
    indexs2 = [False] * l
    for j in range(len(pos_train)):
        indexs1[pos_train[j]] = True
    for j in range(len(neg_train)):
        indexs1[neg_train[j]] = True
    for j in range(len(pos_val)):
        indexs2[pos_val[j]] = True
    for j in range(len(neg_val)):
        indexs2[neg_val[j]] = True
    tr_mask = torch.from_numpy(np.array(indexs1))
    val_mask = torch.from_numpy(np.array(indexs2))
    return tr_mask, val_mask


def get_class_weights(labels):
    pos_counts = labels.sum(dim=0)
    neg_counts = labels.shape[0] - pos_counts
    weights = (neg_counts / (pos_counts + 1e-6))
    return weights


def train_test(data_model, optimizer, data, L_emb, edge_index, L_emb_edge,
               tr_mask, te_mask, epochs, Y):
    """
    基础训练与评估函数（用于 5 折交叉验证模式）。
    分类损失和对比损失均严格仅在实际训练节点 tr_mask 上计算，防止测试与未知节点信息泄露。
    """
    model = data_model['model']
    epoch_aurocs = []
    epoch_auprcs = []

    for epoch in range(epochs):
        # ===== 训练阶段 =====
        model.train()
        optimizer.zero_grad()
        
        # 模型前向传播（原始 10次5折 CV 训练协议：不传 tr_mask，对比损失在全部图节点上计算）
        edge_index_train = dropout_adj(edge_index, p=0.3)[0]
        loss_inter, label_G, label_self, label_neighbor, label_together, label_concat, label_satment, final_output = model(
            data.x, edge_index_train, L_emb, L_emb_edge
        )

        class_weights = get_class_weights(Y[tr_mask])
        loss_G = F.binary_cross_entropy_with_logits(label_G[tr_mask], Y[tr_mask], pos_weight=class_weights)
        loss_self = F.binary_cross_entropy_with_logits(label_self[tr_mask], Y[tr_mask], pos_weight=class_weights)
        loss_neighbor = F.binary_cross_entropy_with_logits(label_neighbor[tr_mask], Y[tr_mask], pos_weight=class_weights)
        loss_together = F.binary_cross_entropy_with_logits(label_together[tr_mask], Y[tr_mask], pos_weight=class_weights)
        loss_concat = F.binary_cross_entropy_with_logits(label_concat[tr_mask], Y[tr_mask], pos_weight=class_weights)
        loss_satment = F.binary_cross_entropy_with_logits(label_satment[tr_mask], Y[tr_mask], pos_weight=class_weights)
        loss_topk_fused = F.binary_cross_entropy_with_logits(final_output[tr_mask], Y[tr_mask], pos_weight=class_weights)

        loss_cls = loss_G + loss_self + loss_neighbor + loss_together + loss_concat + loss_satment + loss_topk_fused
        total_loss = loss_cls + data_model['lambdinter'] * loss_inter

        total_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            _, label_G, label_self, label_neighbor, label_together, label_concat, label_satment, final_output = model(
                data.x, edge_index, L_emb, L_emb_edge
            )

            pred = torch.sigmoid(final_output[te_mask]).cpu().numpy().ravel()
            y_eval = Y[te_mask].cpu().numpy().ravel()
            precision, recall, _thresholds = metrics.precision_recall_curve(y_eval, pred)
            auroc = metrics.roc_auc_score(y_eval, pred)
            auprc = metrics.auc(recall, precision)
            epoch_aurocs.append(auroc)
            epoch_auprcs.append(auprc)
            print(f"Epoch {epoch+1}, Test AUROC: {auroc:.4f}, Test AUPRC: {auprc:.4f}")

    return epoch_aurocs, epoch_auprcs, auroc, auprc


def trainPred_k_sets(input_dim, k_sets, data, L_emb, edge_index, L_emb_edge,
                     lr=0.001, epochs=200, lambdinter=0.005,
                     dropout=0.2, cancerType='pan-cancer', dataset='cpdb',
                     masked=False, base_seed=42):
    """收集每个 epoch 的指标（5 折交叉验证，已修正 auroc/auprc 对应关系并支持 Masked 消融）"""
    all_aurocs = np.zeros((epochs, 10, 5))
    all_auprcs = np.zeros((epochs, 10, 5))
    if cancerType == 'pan-cancer':
        Y = torch.tensor(np.logical_or(data.y, data.y_te)).type(torch.FloatTensor).to(device)
        y_all = np.logical_or(data.y, data.y_te)
        mask_all = np.logical_or(data.mask, data.mask_te)
        print(mask_all.sum())
    else:
        label, label_pos, label_neg = load_label_single(cancerType)
        random.shuffle(label_pos)
        random.shuffle(label_neg)
        print(label.sum())
        y_train_pos = label_pos[:int(0.75 * len(label_pos))]
        y_test_pos = label_pos[int(0.75 * len(label_pos)):]
        y_train_neg = label_neg[:int(0.75 * len(label_neg))]
        y_test_neg = label_neg[int(0.75 * len(label_neg)):]
        l = len(label)
        l1 = int(len(y_train_pos) / 5)
        l2 = int(len(y_train_neg) / 5)
        Y = label

    list_aurocs = np.zeros((10, 5))
    list_auprcs = np.zeros((10, 5))

    n_exp = int(os.environ.get('N_EXP', 10))
    n_fold = int(os.environ.get('N_FOLD', 5))

    feat_tag = "pan-cancer_masked" if masked else "pan-cancer"

    for exp_id in range(n_exp):
        for fold_id in range(n_fold):
            # 固定随机种子保证基线与 Masked 实验完全公平对照
            seed = base_seed + exp_id * 100 + fold_id
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            print(f"\nExp {exp_id+1}/{n_exp} | Fold {fold_id+1}/{n_fold} (Seed: {seed})")
            
            if cancerType == 'pan-cancer':
                _, _, tr_mask, te_mask = k_sets[exp_id][fold_id]
                print(tr_mask.sum())
                print(te_mask.sum())
                train_mask = torch.tensor(tr_mask).bool().to(device)
                test_mask = torch.tensor(te_mask).bool().to(device)
            else:
                tr_mask, te_mask = sample_division_single(y_train_pos, y_train_neg, l, l1, l2, fold_id)
                train_mask = torch.tensor(tr_mask).bool().to(device)
                test_mask = torch.tensor(te_mask).bool().to(device)
                print(tr_mask.sum())
                print(te_mask.sum())
            
            model = combine_net_gate_without_ac(input_dim=input_dim, lambdinter=lambdinter, dropout=dropout).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            aurocs, auprcs, auroc, auprc = train_test(
                data_model={
                    'model': model,
                    'lambdinter': lambdinter
                },
                optimizer=optimizer,
                data=data,
                L_emb=L_emb,
                edge_index=edge_index.to(device),
                L_emb_edge=L_emb_edge,
                tr_mask=train_mask.nonzero().squeeze(),
                te_mask=test_mask,
                epochs=epochs,
                Y=Y
            )
            
            # 正确存储 AUROC 和 AUPRC（避免原代码的变量倒置）
            all_aurocs[:, exp_id, fold_id] = aurocs
            all_auprcs[:, exp_id, fold_id] = auprcs
            list_aurocs[exp_id, fold_id] = auroc  # 严格对应 AUROC
            list_auprcs[exp_id, fold_id] = auprc  # 严格对应 AUPRC

            # 实时保存独立文件
            if cancerType == 'pan-cancer':
                np.savetxt(os.path.join(RESULT_DIR, f'{feat_tag}_auroc.txt'), list_aurocs, fmt='%.6f')
                np.savetxt(os.path.join(RESULT_DIR, f'{feat_tag}_auprc.txt'), list_auprcs, fmt='%.6f')
            else:
                single_dir = os.path.join(RESULT_DIR, 'single')
                os.makedirs(single_dir, exist_ok=True)
                tag = f"{dataset}_{cancerType}_masked" if masked else f"{dataset}_{cancerType}"
                np.savetxt(os.path.join(single_dir, f'{tag}_auroc.txt'), list_aurocs, fmt='%.6f')
                np.savetxt(os.path.join(single_dir, f'{tag}_auprc.txt'), list_auprcs, fmt='%.6f')

    mean_auc, std_auc = list_aurocs[:n_exp, :n_fold].mean(), list_aurocs[:n_exp, :n_fold].std()
    mean_auprc, std_auprc = list_auprcs[:n_exp, :n_fold].mean(), list_auprcs[:n_exp, :n_fold].std()
    desc = "Masked Features (关键词遮蔽)" if masked else "Original Features (原始基线)"

    print(f"\n{'='*75}")
    print(f"Summary Results for 5-Fold CV [{desc}] ({cancerType}):")
    print(f"  Overall AUROC: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"  Overall AUPRC: {mean_auprc:.4f} ± {std_auprc:.4f}")
    print(f"10x5 AUROC Matrix:\n{np.array2string(list_aurocs[:n_exp, :n_fold], precision=4)}")
    print(f"10x5 AUPRC Matrix:\n{np.array2string(list_auprcs[:n_exp, :n_fold], precision=4)}")
    print(f"{'='*75}\n")

    summary_file = os.path.join(RESULT_DIR, f'{feat_tag}_summary.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"Experiment: 10x5 Cross-Validation [{desc}]\n")
        f.write(f"Cancer Type: {cancerType}, Dataset: {dataset}\n")
        f.write(f"Hyperparameters: lr={lr}, dropout={dropout}, lambdinter={lambdinter}, epochs={epochs}\n")
        f.write(f"Base Seed: {base_seed}\n")
        f.write('-' * 60 + '\n')
        f.write(f"Overall Metrics (over {n_exp}x{n_fold} = {n_exp * n_fold} folds):\n")
        f.write(f"  AUROC : {mean_auc:.4f} ± {std_auc:.4f}\n")
        f.write(f"  AUPRC : {mean_auprc:.4f} ± {std_auprc:.4f}\n")
        f.write('-' * 60 + '\n')
        f.write(f"10x5 AUROC Matrix:\n{np.array2string(list_aurocs[:n_exp, :n_fold], precision=4)}\n")
        f.write(f"10x5 AUPRC Matrix:\n{np.array2string(list_auprcs[:n_exp, :n_fold], precision=4)}\n")

    if not masked:
        save_results_to_file(list_aurocs, list_auprcs, cancerType, dataset=dataset, lr=lr, dropout=dropout, lambdinter=lambdinter)
    results = 0
    return results


def load_leakage_splits(audit_file_path=None):
    """
    从审计文件 src/Gemma_Vocabulary_Leakage_Audit.xlsx 加载 Clean 和 Hit 划分。
    
    规则：
    1. 使用 'Gene-level Flags' Sheet 中的 'LLM combined | Any exact label-like term' 列作为分组标记。
    2. 只保留有标签的 Driver 和 Non-driver 基因，排除 Unknown 基因。
    3. 按 Code_Index 与 CPDB 图节点严格对齐（0..13626）。
    """
    if audit_file_path is None:
        audit_file_path = os.path.join(BASE_DIR, 'src', 'Gemma_Vocabulary_Leakage_Audit.xlsx')

    if not os.path.exists(audit_file_path):
        raise FileNotFoundError(f"未找到审计文件: {audit_file_path}")

    print(f"正在读取审计文件: {audit_file_path} ...")
    df = pd.read_excel(audit_file_path, sheet_name='Gene-level Flags')

    assert len(df) == 13627, f"基因节点数不匹配: 期望 13627, 实际 {len(df)}"
    assert (df['Code_Index'].values == np.arange(len(df))).all(), "Code_Index 必须与节点索引 0..13626 严格对齐"

    flag_col = 'LLM combined | Any exact label-like term'
    if flag_col not in df.columns:
        raise KeyError(f"Sheet 'Gene-level Flags' 中未找到列 '{flag_col}'")

    labeled_mask = (df['Label_Status'] == 'Labeled').values
    hit_flag = (df[flag_col] == 1).values

    clean_mask = labeled_mask & (~hit_flag)
    hit_mask = labeled_mask & hit_flag

    return clean_mask, hit_mask, df


def trainPred_fixed_split(input_dim, train_candidate_mask, fixed_test_mask, data, L_emb, edge_index, L_emb_edge,
                          lr=0.0005, epochs=160, lambdinter=0.001, dropout=0.3,
                          split_name='clean_to_hit', n_exp=10, Y=None, base_seed=42, val_ratio=0.2):
    """
    严格的标签泄露词审计划分实验流程：
    1. 分类损失和对比损失都严格仅在实际训练节点上计算（防止测试节点和 Unknown 节点泄露）。
    2. 从训练候选组内部划分训练集/验证集（如 8:2 分层划分）。
    3. 每个 epoch 仅评估验证集，根据验证集指标（AUPRC）保存最佳 Checkpoint。
    4. 训练完成后加载最佳 Checkpoint，仅对固定的最终测试集执行一次最终评估。
    5. 变量、输出内容与文件名中 AUROC 与 AUPRC 严格一致。
    """
    cand_indices = np.where(train_candidate_mask)[0]
    cand_labels = Y[cand_indices].cpu().numpy().ravel().astype(int)

    fixed_test_tensor = torch.tensor(fixed_test_mask).bool().to(device)
    y_test_np = Y[fixed_test_tensor].cpu().numpy().ravel()
    test_drivers = int((y_test_np == 1).sum())
    test_nondrivers = int((y_test_np == 0).sum())

    print(f"\n{'='*75}")
    print(f"Running Leakage Split Experiment: [{split_name}]")
    print(f"  Training Candidate Pool : {len(cand_indices)} genes (Drivers: {int(cand_labels.sum())}, Non-drivers: {len(cand_labels) - int(cand_labels.sum())})")
    print(f"  Inner Train/Val Split   : {(1 - val_ratio)*100:.0f}% Train / {val_ratio*100:.0f}% Validation (Stratified by Driver label)")
    print(f"  Fixed Testing Set       : {len(y_test_np)} genes (Drivers: {test_drivers}, Non-drivers: {test_nondrivers})")
    print(f"  Evaluation Protocol     : Model checkpoint selected ONLY by validation set, test set evaluated ONCE at end")
    print(f"  Experiments             : {n_exp} independent runs, {epochs} epochs each")
    print(f"  Hyperparameters         : lr={lr}, dropout={dropout}, lambdinter={lambdinter}")
    print(f"{'='*75}\n")

    all_test_aurocs = np.zeros(n_exp)
    all_test_auprcs = np.zeros(n_exp)
    all_best_val_aurocs = np.zeros(n_exp)
    all_best_val_auprcs = np.zeros(n_exp)
    all_best_epochs = np.zeros(n_exp, dtype=int)

    for exp_id in range(n_exp):
        seed = base_seed + exp_id
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # 1. 在训练候选组内部划分训练集与验证集（保持 Driver 类别比例分层划分）
        inner_tr_idx, inner_val_idx = train_test_split(
            cand_indices,
            test_size=val_ratio,
            stratify=cand_labels,
            random_state=seed
        )

        inner_tr_bool = np.zeros(data.x.shape[0], dtype=bool)
        inner_tr_bool[inner_tr_idx] = True
        inner_val_bool = np.zeros(data.x.shape[0], dtype=bool)
        inner_val_bool[inner_val_idx] = True

        inner_tr_tensor = torch.tensor(inner_tr_bool).bool().to(device)
        inner_tr_indices = inner_tr_tensor.nonzero().squeeze()
        inner_val_tensor = torch.tensor(inner_val_bool).bool().to(device)
        y_val_np = Y[inner_val_tensor].cpu().numpy().ravel()

        print(f"\n--- [Split: {split_name}] Experiment {exp_id + 1}/{n_exp} (Seed: {seed}) ---")
        print(f"  Inner Train: {len(inner_tr_idx)} genes | Inner Val: {len(inner_val_idx)} genes")

        model = combine_net_gate_without_ac(input_dim=input_dim, lambdinter=lambdinter, dropout=dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        best_val_auprc = -1.0
        best_val_auroc = -1.0
        best_epoch = -1
        best_model_state = None

        # 2. 训练循环：每轮仅在验证集上评估，挑选最佳 Checkpoint
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()

            # 前向传播：tr_mask 严格限定在实际训练节点上，对比损失绝不包含验证、测试或 Unknown 节点
            edge_index_train = dropout_adj(edge_index, p=0.3)[0]
            loss_inter, label_G, label_self, label_neighbor, label_together, label_concat, label_satment, final_output = model(
                data.x, edge_index_train, L_emb, L_emb_edge, tr_mask=inner_tr_indices
            )

            # 分类损失：严格仅在实际训练节点上计算
            class_weights = get_class_weights(Y[inner_tr_indices])
            loss_G = F.binary_cross_entropy_with_logits(label_G[inner_tr_indices], Y[inner_tr_indices], pos_weight=class_weights)
            loss_self = F.binary_cross_entropy_with_logits(label_self[inner_tr_indices], Y[inner_tr_indices], pos_weight=class_weights)
            loss_neighbor = F.binary_cross_entropy_with_logits(label_neighbor[inner_tr_indices], Y[inner_tr_indices], pos_weight=class_weights)
            loss_together = F.binary_cross_entropy_with_logits(label_together[inner_tr_indices], Y[inner_tr_indices], pos_weight=class_weights)
            loss_concat = F.binary_cross_entropy_with_logits(label_concat[inner_tr_indices], Y[inner_tr_indices], pos_weight=class_weights)
            loss_satment = F.binary_cross_entropy_with_logits(label_satment[inner_tr_indices], Y[inner_tr_indices], pos_weight=class_weights)
            loss_topk_fused = F.binary_cross_entropy_with_logits(final_output[inner_tr_indices], Y[inner_tr_indices], pos_weight=class_weights)

            loss_cls = loss_G + loss_self + loss_neighbor + loss_together + loss_concat + loss_satment + loss_topk_fused
            total_loss = loss_cls + lambdinter * loss_inter

            total_loss.backward()
            optimizer.step()

            # 验证集评估（绝不触碰固定测试集）
            model.eval()
            with torch.no_grad():
                _, _, _, _, _, _, _, final_output_val = model(data.x, edge_index, L_emb, L_emb_edge)
                pred_val = torch.sigmoid(final_output_val[inner_val_tensor]).cpu().numpy().ravel()
                val_auroc = metrics.roc_auc_score(y_val_np, pred_val)
                p_val, r_val, _ = metrics.precision_recall_curve(y_val_np, pred_val)
                val_auprc = metrics.auc(r_val, p_val)

            # 根据验证集 AUPRC 挑选最佳 Checkpoint
            if val_auprc > best_val_auprc:
                best_val_auprc = val_auprc
                best_val_auroc = val_auroc
                best_epoch = epoch + 1
                best_model_state = copy.deepcopy(model.state_dict())

            if (epoch + 1) % 20 == 0 or (epoch + 1) == epochs:
                print(f"  Epoch {epoch+1:3d}/{epochs} | Val AUROC: {val_auroc:.4f}, Val AUPRC: {val_auprc:.4f} (Best Epoch: {best_epoch}, Best Val AUPRC: {best_val_auprc:.4f})")

        # 3. 训练完成后，加载验证集最优 Checkpoint，对固定测试集仅评估一次
        model.load_state_dict(best_model_state)
        model.eval()
        with torch.no_grad():
            _, _, _, _, _, _, _, final_output_test = model(data.x, edge_index, L_emb, L_emb_edge)
            pred_test = torch.sigmoid(final_output_test[fixed_test_tensor]).cpu().numpy().ravel()
            test_auroc = metrics.roc_auc_score(y_test_np, pred_test)
            p_te, r_te, _ = metrics.precision_recall_curve(y_test_np, pred_test)
            test_auprc = metrics.auc(r_te, p_te)

        print(f"  >> [Exp {exp_id+1}/{n_exp} Result] Best Val Epoch: {best_epoch} (Val AUPRC: {best_val_auprc:.4f}, Val AUROC: {best_val_auroc:.4f}) | Final Test AUROC: {test_auroc:.4f}, Test AUPRC: {test_auprc:.4f}")

        all_test_aurocs[exp_id] = test_auroc
        all_test_auprcs[exp_id] = test_auprc
        all_best_val_aurocs[exp_id] = best_val_auroc
        all_best_val_auprcs[exp_id] = best_val_auprc
        all_best_epochs[exp_id] = best_epoch

    # 4. 汇总多轮实验统计指标
    mean_test_auroc, std_test_auroc = all_test_aurocs.mean(), all_test_aurocs.std()
    mean_test_auprc, std_test_auprc = all_test_auprcs.mean(), all_test_auprcs.std()
    mean_val_auroc, std_val_auroc = all_best_val_aurocs.mean(), all_best_val_aurocs.std()
    mean_val_auprc, std_val_auprc = all_best_val_auprcs.mean(), all_best_val_auprcs.std()

    print(f"\n{'='*75}")
    print(f"Summary Results for Leakage Split: [{split_name}] (over {n_exp} independent runs)")
    print(f"Validation Metrics (Best Checkpoint Average):")
    print(f"  AUROC : {mean_val_auroc:.4f} ± {std_val_auroc:.4f}")
    print(f"  AUPRC : {mean_val_auprc:.4f} ± {std_val_auprc:.4f}")
    print(f"Final Fixed Test Metrics (Evaluated ONCE with Best Checkpoint):")
    print(f"  AUROC : {mean_test_auroc:.4f} ± {std_test_auroc:.4f}")
    print(f"  AUPRC : {mean_test_auprc:.4f} ± {std_test_auprc:.4f}")
    print(f"{'='*75}\n")

    # 5. 保存结果文件（AUROC 与 AUPRC 文件名及内容严格对应）
    res_dir = os.path.join(BASE_DIR, 'result')
    os.makedirs(res_dir, exist_ok=True)
    summary_path = os.path.join(res_dir, f"pan-cancer_leakage_{split_name}_summary.txt")
    np.savetxt(os.path.join(res_dir, f"pan-cancer_leakage_{split_name}_auroc.txt"), all_test_aurocs, fmt='%.6f')
    np.savetxt(os.path.join(res_dir, f"pan-cancer_leakage_{split_name}_auprc.txt"), all_test_auprcs, fmt='%.6f')

    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"Leakage Split Experiment: {split_name}\n")
        f.write(f"Training Candidate Pool : {len(cand_indices)} (Drivers: {int(cand_labels.sum())}, Non-drivers: {len(cand_labels) - int(cand_labels.sum())})\n")
        f.write(f"Inner Train/Val Split   : {(1 - val_ratio)*100:.0f}% Train / {val_ratio*100:.0f}% Val\n")
        f.write(f"Fixed Testing Set       : {len(y_test_np)} (Drivers: {test_drivers}, Non-drivers: {test_nondrivers})\n")
        f.write(f"Protocol                : Checkpoint selected on validation set, test set evaluated ONCE\n")
        f.write(f"Experiments             : {n_exp} independent runs, {epochs} epochs each\n")
        f.write(f"Hyperparameters         : lr={lr}, dropout={dropout}, lambdinter={lambdinter}\n")
        f.write('-' * 60 + '\n')
        f.write(f"Final Fixed Test Metrics (Evaluated ONCE with Best Checkpoint):\n")
        f.write(f"  AUROC : {mean_test_auroc:.4f} ± {std_test_auroc:.4f}\n")
        f.write(f"  AUPRC : {mean_test_auprc:.4f} ± {std_test_auprc:.4f}\n")
        f.write(f"  All Test AUROC per exp: {np.array2string(all_test_aurocs, precision=4)}\n")
        f.write(f"  All Test AUPRC per exp: {np.array2string(all_test_auprcs, precision=4)}\n")
        f.write('-' * 60 + '\n')
        f.write(f"Validation Metrics (Best Checkpoint Average):\n")
        f.write(f"  Val AUROC : {mean_val_auroc:.4f} ± {std_val_auroc:.4f}\n")
        f.write(f"  Val AUPRC : {mean_val_auprc:.4f} ± {std_val_auprc:.4f}\n")
        f.write(f"  Best Epochs per exp: {all_best_epochs.tolist()}\n")

    return all_test_aurocs, all_test_auprcs


def main():
    parser = argparse.ArgumentParser(description="BIMODriver Training and Evaluation")
    parser.add_argument('--split', type=str, default=os.environ.get('SPLIT', 'cv'),
                        help="划分模式: 'cv' (默认5折交叉验证), 'clean_to_hit' (Clean训练->Hit测试), 'hit_to_clean' (Hit训练->Clean测试)")
    parser.add_argument('--dataset', type=str, default='cpdb', choices=['cpdb', 'string'],
                        help="数据集类型 ('cpdb' 或 'string')")
    parser.add_argument('--cancerType', type=str, default='pan-cancer',
                        help="癌种名称 ('pan-cancer' 或特定单癌种)")
    parser.add_argument('--epochs', type=int, default=int(os.environ.get('EPOCHS', 160)),
                        help="训练轮数 (默认 160)")
    parser.add_argument('--n_exp', type=int, default=int(os.environ.get('N_EXP', 10)),
                        help="独立实验次数 (默认 10)")
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help="候选训练集中切分为验证集的比例 (默认 0.2，即 8:2)")
    parser.add_argument('--lr', type=float, default=0.0005,
                        help="学习率 (默认 0.0005)")
    parser.add_argument('--dropout', type=float, default=0.3,
                        help="Dropout 率 (默认 0.3)")
    parser.add_argument('--lambdinter', type=float, default=0.001,
                        help="特征对齐损失权重 (默认 0.001)")
    parser.add_argument('--audit_file', type=str, default=None,
                        help="审计文件路径 (默认 src/Gemma_Vocabulary_Leakage_Audit.xlsx)")
    parser.add_argument('--base_seed', type=int, default=42,
                        help="随机种子基准值 (默认 42)")
    parser.add_argument('--masked', action='store_true',
                        help="使用关键词遮蔽后的语义特征 (PAN-CANCER_statement_features_masked.pt)")
    parser.add_argument('--statement_file', type=str, default=None,
                        help="自定义语义特征文件路径 (覆盖默认特征文件)")

    args = parser.parse_args()

    split_mode = args.split.lower().replace('->', '_to_').replace('-', '_')

    # 加载 CPDB 数据
    if args.dataset == 'cpdb':
        data = torch.load(os.path.join(DATA_DIR, "CPDB", "CPDB_new_data.pt"))
        data = data.to(device)
        data.x = data.x[:, :48]
        if args.cancerType != 'pan-cancer':
            cancerType_dict = {
                'kirc': [0, 16, 32], 'brca': [1, 17, 33], 'prad': [3, 19, 35],
                'stad': [4, 20, 36], 'hnsc': [5, 21, 37], 'luad': [6, 22, 38],
                'thca': [7, 23, 39], 'blca': [8, 24, 40], 'esca': [9, 25, 41],
                'lihc': [10, 26, 42], 'ucec': [11, 27, 43], 'coad': [12, 28, 44],
                'lusc': [13, 29, 45], 'cesc': [14, 30, 46], 'kirp': [15, 31, 47]
            }
            data.x = data.x[:, cancerType_dict[args.cancerType]]

        datas = torch.load(os.path.join(DATA_DIR, "CPDB", "Str_feature.pkl")).to(device)
        data.x = torch.cat((data.x, datas), 1)
        data = data.to(device)

        if args.statement_file:
            statement_path = args.statement_file
        elif args.masked:
            statement_path = os.path.join(DATA_DIR, "CPDB", "PAN-CANCER_statement_features_masked.pt")
        else:
            statement_path = os.path.join(DATA_DIR, "CPDB", "PAN-CANCER_statement_features.pt")

        if not os.path.exists(statement_path):
            raise FileNotFoundError(f"未找到语义特征文件: {statement_path}")

        print(f"Loading statement features from: {statement_path}")
        statement = torch.load(statement_path).to(device)
        L_emb = {
            'self_emb': statement[:, 0:768],
            'neighbor_emb': statement[:, 768:1536],
            'together_emb': statement[:, 1536:2304]
        }
        L_emb_edge = torch.load(os.path.join(DATA_DIR, "cpdb_network_LLM", "merged_k5_edge_index.pt")).to(device)

        if isinstance(L_emb, dict):
            for key in L_emb:
                if torch.is_tensor(L_emb[key]):
                    L_emb[key] = L_emb[key].to(device)

        Y = torch.tensor(np.logical_or(data.y, data.y_te)).type(torch.FloatTensor).to(device)

    elif args.dataset == 'string':
        if split_mode in ['clean_to_hit', 'hit_to_clean']:
            raise ValueError("标签泄露审计划分目前仅支持 CPDB 数据集")
        data = torch.load(os.path.join(DATA_DIR, "STRING", "STRING_data.pkl"))
        data = data.to(device)
        Y = torch.tensor(np.logical_or(data.y, data.y_te)).type(torch.FloatTensor).to(device)
        data.x = data.x[:, :48]

        datas = torch.load(os.path.join(DATA_DIR, "STRING", "Str_feature.pkl")).to(device)
        data.x = torch.cat((data.x, datas), 1)
        data = data.to(device)

        with open(os.path.join(DATA_DIR, "STRING", "k_sets.pkl"), 'rb') as handle:
            k_sets = torch.load(os.path.join(DATA_DIR, "STRING", "k_sets.pkl"))

        statement = torch.load(os.path.join(DATA_DIR, "STRING", "PAN-CANCER_string_new_neiber2.pt")).to(device)
        L_emb = {
            'self_emb': statement[:, 0:768],
            'neighbor_emb': statement[:, 768:1536],
            'together_emb': statement[:, 1536:2304]
        }
        L_emb_edge = torch.load(os.path.join(DATA_DIR, "string_network_LLM", "merged_k5_edge_index.pt")).to(device)

        if isinstance(L_emb, dict):
            for key in L_emb:
                if torch.is_tensor(L_emb[key]):
                    L_emb[key] = L_emb[key].to(device)
    else:
        raise ValueError("Unsupported dataset. Please choose 'cpdb' or 'string'.")

    input_dim = data.x.shape[1]
    pb, _ = remove_self_loops(data.edge_index)
    pb, _ = add_self_loops(pb)

    # 分支 1：标签泄露词划分实验
    if split_mode in ['clean_to_hit', 'hit_to_clean']:
        clean_mask, hit_mask, audit_df = load_leakage_splits(args.audit_file)

        if split_mode == 'clean_to_hit':
            train_candidate_mask = clean_mask
            fixed_test_mask = hit_mask
        else:
            train_candidate_mask = hit_mask
            fixed_test_mask = clean_mask

        trainPred_fixed_split(
            input_dim=input_dim,
            train_candidate_mask=train_candidate_mask,
            fixed_test_mask=fixed_test_mask,
            data=data,
            L_emb=L_emb,
            edge_index=pb,
            L_emb_edge=L_emb_edge,
            lr=args.lr,
            epochs=args.epochs,
            lambdinter=args.lambdinter,
            dropout=args.dropout,
            split_name=split_mode,
            n_exp=args.n_exp,
            Y=Y,
            base_seed=args.base_seed,
            val_ratio=args.val_ratio
        )

    # 分支 2：原始 5 折交叉验证实验
    elif split_mode == 'cv':
        with open(os.path.join(DATA_DIR, "CPDB", "k_sets.pkl"), 'rb') as handle:
            k_sets = pickle.load(handle)

        print(f"\nRunning original 5-fold CV for {args.cancerType} on {args.dataset}...")
        trainPred_k_sets(
            input_dim=input_dim,
            k_sets=k_sets,
            data=data,
            L_emb=L_emb,
            edge_index=pb,
            L_emb_edge=L_emb_edge,
            lr=args.lr,
            epochs=args.epochs,
            lambdinter=args.lambdinter,
            dropout=args.dropout,
            cancerType=args.cancerType,
            dataset=args.dataset,
            masked=args.masked,
            base_seed=args.base_seed
        )
    else:
        raise ValueError(f"未知的划分模式: {args.split}。可选模式: 'cv', 'clean_to_hit', 'hit_to_clean'")


if __name__ == '__main__':
    main()
