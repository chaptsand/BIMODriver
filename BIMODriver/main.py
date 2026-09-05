# -*- coding: utf-8 -*-
import os
import sys
import copy
import time
import pickle
import random
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
warnings.filterwarnings("ignore")

# 路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULT_DIR = os.path.join(BASE_DIR, "result")
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULT_DIR, "single"), exist_ok=True)


def save_results_to_file(auroc, auprc, cancerType, dataset='cpdb', lr=0.001, dropout=0.2, lambdinter=0.005):
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
        f.write(f"AUPR: {auroc.mean():.4f} ± {auroc.std():.4f}\n")
        f.write(str(auroc))
        f.write("\n")
        f.write(f"AUC: {auprc.mean():.4f} ± {auprc.std():.4f}\n")
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
    """返回每个epoch的指标"""
    model = data_model['model']
    epoch_aurocs = []
    epoch_auprcs = []

    for epoch in range(epochs):
        # ===== 训练阶段 =====
        model.train()
        optimizer.zero_grad()
        
        # 模型前向传播
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
            precision, recall, _thresholds = metrics.precision_recall_curve(Y[te_mask].cpu().numpy(), pred)
            auc = metrics.roc_auc_score(Y[te_mask].cpu().numpy(), pred)
            auprc = metrics.auc(recall, precision)
            epoch_aurocs.append(auc)
            epoch_auprcs.append(auprc)
            print(f"Epoch {epoch+1}, Test AUC: {auc:.4f}, Test AUPRC: {auprc:.4f}")

    return epoch_aurocs, epoch_auprcs, auc, auprc


def trainPred_k_sets(input_dim, k_sets, data, L_emb, edge_index, L_emb_edge,
                     lr=0.001, epochs=200, lambdinter=0.005,
                     dropout=0.2, cancerType='pan-cancer', dataset='cpdb'):
    """收集每个epoch的指标"""
    # 初始化存储结构 [epoch][experiment][fold]
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

    # 遍历独立实验
    for exp_id in range(n_exp):
        for fold_id in range(n_fold):
            print(f"\nExp {exp_id+1}/{n_exp} | Fold {fold_id+1}/{n_fold}")
            
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
            
            # 初始化模型
            model = combine_net_gate_without_ac(input_dim=input_dim, lambdinter=lambdinter, dropout=dropout).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            aurocs, auprcs, auc, auprc = train_test(
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
            
            # 存储结果
            all_aurocs[:, exp_id, fold_id] = aurocs
            all_auprcs[:, exp_id, fold_id] = auprcs
            list_aurocs[exp_id, fold_id] = auprc
            list_auprcs[exp_id, fold_id] = auc
            if cancerType == 'pan-cancer':
                np.savetxt(os.path.join(RESULT_DIR, 'pan-cancer_auroc.txt'), list_aurocs, fmt='%.6f')
                np.savetxt(os.path.join(RESULT_DIR, 'pan-cancer_auprc.txt'), list_auprcs, fmt='%.6f')
            else:
                single_dir = os.path.join(RESULT_DIR, 'single')
                os.makedirs(single_dir, exist_ok=True)
                np.savetxt(os.path.join(single_dir, dataset + '_' + cancerType + '_auroc.txt'), list_aurocs, fmt='%.6f')
                np.savetxt(os.path.join(single_dir, dataset + '_' + cancerType + '_auprc.txt'), list_auprcs, fmt='%.6f')

    save_results_to_file(list_aurocs, list_auprcs, cancerType, dataset=dataset, lr=lr, dropout=dropout, lambdinter=lambdinter)
    results = 0
    return results


def main():
    cancers = ['pan-cancer']
    dataset = 'cpdb'  # 'cpdb' or 'string'
    for cancerType in cancers:
        if dataset == 'cpdb':
            data = torch.load(os.path.join(DATA_DIR, "CPDB", "CPDB_new_data.pt"))
            data = data.to(device)
            data.x = data.x[:, :48]
            if cancerType == 'pan-cancer':
                data.x = data.x[:, :48]
            else:
                cancerType_dict = {
                    'kirc': [0, 16, 32],
                    'brca': [1, 17, 33],
                    'prad': [3, 19, 35],
                    'stad': [4, 20, 36],
                    'hnsc': [5, 21, 37],
                    'luad': [6, 22, 38],
                    'thca': [7, 23, 39],
                    'blca': [8, 24, 40],
                    'esca': [9, 25, 41],
                    'lihc': [10, 26, 42],
                    'ucec': [11, 27, 43],
                    'coad': [12, 28, 44],
                    'lusc': [13, 29, 45],
                    'cesc': [14, 30, 46],
                    'kirp': [15, 31, 47]
                }
                data.x = data.x[:, cancerType_dict[cancerType]]

            datas = torch.load(os.path.join(DATA_DIR, "CPDB", "Str_feature.pkl"))
            data.x = torch.cat((data.x, datas), 1)
            data = data.to(device)

            with open(os.path.join(DATA_DIR, "CPDB", "k_sets.pkl"), 'rb') as handle:
                k_sets = pickle.load(handle)

            statement = torch.load(os.path.join(DATA_DIR, "CPDB", "PAN-CANCER_statement_features.pt")).to(device)
            L_emb = {}
            L_emb['self_emb'] = statement[:, 0:768]
            L_emb['neighbor_emb'] = statement[:, 768:1536]
            L_emb['together_emb'] = statement[:, 1536:2304]

            L_emb_edge = torch.load(os.path.join(DATA_DIR, "cpdb_network_LLM", "merged_k5_edge_index.pt")).to(device)

            if isinstance(L_emb, dict):
                for key in L_emb:
                    if torch.is_tensor(L_emb[key]):
                        L_emb[key] = L_emb[key].to(device)
        elif dataset == 'string':
            data = torch.load(os.path.join(DATA_DIR, "STRING", "STRING_data.pkl"))
            data = data.to(device)
            Y = torch.tensor(np.logical_or(data.y, data.y_te)).type(torch.FloatTensor).to(device)
            y_all = np.logical_or(data.y, data.y_te)
            mask_all = np.logical_or(data.mask, data.mask_te)
            data.x = data.x[:, :48]

            datas = torch.load(os.path.join(DATA_DIR, "STRING", "Str_feature.pkl")).to(device)
            data.x = torch.cat((data.x, datas), 1)

            data = data.to(device)
            k_sets = torch.load(os.path.join(DATA_DIR, "STRING", "k_sets.pkl"))

            statement = torch.load(os.path.join(DATA_DIR, "STRING", "PAN-CANCER_string_new_neiber2.pt")).to(device)
            L_emb = {}
            L_emb['self_emb'] = statement[:, 0:768]
            L_emb['neighbor_emb'] = statement[:, 768:1536]
            L_emb['together_emb'] = statement[:, 1536:2304]
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
        E = data.edge_index
        EPOCH = int(os.environ.get('EPOCHS', 160))

        dropout_rates = [0.3]
        lrs = [0.0005]
        lambdinters = [0.001]

        for dropoutrate in dropout_rates:
            for lr in lrs:
                for lambdinter in lambdinters:
                    # 训练模型
                    print(f"\nTraining for cancer type: {cancerType}, dropout rate: {dropoutrate}, learning rate: {lr}, lambda inter: {lambdinter}")

                    results = trainPred_k_sets(
                        input_dim=input_dim,      # 输入特征维度
                        k_sets=k_sets,            # 加载的交叉验证划分数据
                        data=data,                # 图数据对象
                        L_emb=L_emb,              # 文本特征
                        edge_index=pb,            # 处理后的边索引（带自环）
                        L_emb_edge=L_emb_edge,
                        lr=lr,                    # 学习率
                        epochs=EPOCH,             # 总训练轮次
                        lambdinter=lambdinter,    # 特征对齐系数
                        dropout=dropoutrate,      # 丢弃率
                        cancerType=cancerType,
                        dataset=dataset
                    )


if __name__ == '__main__':
    main()
