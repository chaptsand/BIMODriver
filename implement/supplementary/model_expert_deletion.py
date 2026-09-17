# encoding: utf-8
import numpy as np
import pandas as pd
import time
import pickle
import random
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / 'BIMODriver') not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / 'BIMODriver'))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from torch.nn import Linear
import gcnPreprocessing
import torch_geometric.transforms as T
from torch_geometric.nn import ChebConv, GATConv, GCNConv, SAGEConv
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import dropout_adj, negative_sampling, remove_self_loops, add_self_loops
import copy
from sklearn import metrics
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split as sk_train_test_split
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn import linear_model
import numpy as np
from torch_geometric.utils import dropout_adj

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def sim(z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())
def diagonal_contrastive_loss(h1, h2, tau=0.1):

    sim_matrix = sim(h1, h2)  # [N, N]
    pos_mask = torch.eye(h1.size(0), device=h1.device)  # �Խ���Ϊ1�ľ���
    numerator = torch.exp(sim_matrix.diag() / tau)  # �Խ���Ԫ��
    denominator = torch.exp(sim_matrix / tau).sum(dim=1)  # �����
    loss = -torch.log(numerator / denominator).mean()
    return loss

def save_best_epoch_results(
    list_aurocs, list_auprcs,
    all_aurocs, all_auprcs,
    cancerType, dataset='cpdb',
    lr=0.001, dropout=0.2, lambdinter=0.005,
    epochs=150,txt='none'
):
    res_dir = REPO_ROOT / 'result'
    res_dir.mkdir(parents=True, exist_ok=True)
    if cancerType == 'pan-cancer':
        if dataset == 'cpdb':
            path = str(res_dir / 'pan-cancer.txt')
        elif dataset == 'string':
            path = str(res_dir / 'pan-cancer_string-epoch.txt')
        else:
            raise ValueError("Unsupported dataset for pan-cancer results.")
    else:
        single_dir = res_dir / 'single'
        single_dir.mkdir(parents=True, exist_ok=True)
        path = str(single_dir / 'single_result_new.txt')

    selected_epochs = [i for i in range(epochs) if (i + 1) % 10 == 0]
    selected_aurocs = all_aurocs[selected_epochs]  # shape: [E, n_exp, n_fold]
    selected_auprcs = all_auprcs[selected_epochs]

    mean_selected_aurocs = selected_aurocs.mean(axis=(1, 2))
    mean_selected_auprcs = selected_auprcs.mean(axis=(1, 2))
    
    # combined_metric = mean_selected_aurocs + mean_selected_auprcs
    combined_metric = mean_selected_auprcs
    best_idx = combined_metric.argmax()
    
    best_epoch = selected_epochs[best_idx]
    best_auroc_matrix = selected_aurocs[best_idx]  # shape: [n_exp, n_fold]
    best_auprc_matrix = selected_auprcs[best_idx]

    # ===== �������� =====
    with open(path, 'a') as f:
        f.write('--' * 20 + '\n')
        if txt != 'none':
            f.write(f"{txt}\n")
        f.write(f"Dropout Rate: {dropout}, Learning Rate: {lr}, Lambda Inter: {lambdinter}\n")
        f.write(f"Results for {cancerType}:\n")
        f.write(f"AUPR: {list_aurocs.mean():.4f} �� {list_aurocs.std():.4f}\n")
        f.write(str(list_aurocs) + '\n')
        f.write(f"AUC: {list_auprcs.mean():.4f} �� {list_auprcs.std():.4f}\n")
        f.write(str(list_auprcs) + '\n')

        f.write(f"# Best Epoch: {best_epoch + 1} | Mean AUROC: {best_auroc_matrix .mean():.4f} +- {best_auroc_matrix.std():.4f} | Mean AUPRC: {best_auprc_matrix.mean():.4f} +- {best_auprc_matrix.std():.4f}\n")
        f.write("Best AUROC matrix:\n")
        np.savetxt(f, best_auroc_matrix, fmt='%.6f')
        f.write("Best AUPRC matrix:\n")
        np.savetxt(f, best_auprc_matrix, fmt='%.6f')
class combine_net_gate_without_ac(torch.nn.Module):
    def __init__(self,input_dim=64 , lambdinter=0.005,dropout=0.1):
        super(combine_net_gate_without_ac, self).__init__()
        self.lambdinter = lambdinter    # ����������ʧϵ��
        self.dropout = dropout
        self.g_net = G_Net(in_channels=input_dim, hidden_channels=256, out_channels=1, dropout=dropout)
        self.l_net = L_net(in_channels=768, hidden_channels=256, out_channels=1, dropout=dropout)
        self.top_k = 4


        self.gating_layer = torch.nn.Sequential(
            torch.nn.Linear(6, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 6) 
        )

    def forward(self, x, ppi_edge, L_emb, L_emb_edge, return_routing=False):
        """Run SMoE and optionally expose tensors required for deletion tests.

        ``gate_probability`` is the sparse Top-K distribution: the selected
        experts sum to one and all unselected experts have weight zero.
        """

        input_G = self.g_net(x, ppi_edge)
        input_self      = self.l_net(L_emb['self_emb']    , L_emb_edge)
        input_neighbor  = self.l_net(L_emb['neighbor_emb'], L_emb_edge)
        input_together  = self.l_net(L_emb['together_emb'], L_emb_edge)

        # ����Ԥ��
        label_G = self.g_net.classfy(input_G, ppi_edge)
        label_self = self.l_net.classfy(input_self,         L_emb_edge)
        label_neighbor = self.l_net.classfy(input_neighbor, L_emb_edge)
        label_together = self.l_net.classfy(input_together, L_emb_edge)

        label_concat = torch.einsum('ij,ij->i', input_G, input_self).unsqueeze(1)
        label_satment = torch.einsum('ij,ij->i', input_self, input_together).unsqueeze(1)

        if self.training:
            # In strict-inductive runs x/L_emb already contain only nodes from
            # the training subgraph, so a separate contrastive mask is neither
            # needed nor allowed here.
            loss_inter = diagonal_contrastive_loss(input_G, input_self, tau=0.04) + diagonal_contrastive_loss(input_self, input_together, tau=0.04) + diagonal_contrastive_loss(input_G, input_together, tau=0.04)
        else:
            loss_inter = torch.tensor(0.0).to(x.device)
        logits_all = torch.cat((
            label_G, 
            label_self, label_neighbor, label_together,
            label_concat, label_satment
        ), dim=1)

        gating_score = self.gating_layer(logits_all)
        topk_gate_logits, topk_indices = torch.topk(
            gating_score, k=self.top_k, dim=1
        )

        # ȡ�� top-k ��ר�����
        # batch_size = logits_all.size(0)
        topk_outputs = torch.gather(logits_all, 1, topk_indices)
        topk_weights = F.softmax(topk_gate_logits, dim=1)

        sparse_gate_probability = torch.zeros_like(gating_score)
        sparse_gate_probability.scatter_(1, topk_indices, topk_weights)
        topk_selected = torch.zeros_like(gating_score, dtype=torch.bool)
        topk_selected.scatter_(1, topk_indices, True)

        # ��Ȩƽ���õ��������
        final_output = torch.sum(topk_outputs * topk_weights, dim=1, keepdim=True)  # shape: [N, 1]

        outputs = (
            loss_inter, label_G, label_self, label_neighbor, label_together,
            label_concat, label_satment, final_output,
        )
        if return_routing:
            return outputs, {
                'expert_logits': logits_all,
                'gate_logits': gating_score,
                'gate_probability': sparse_gate_probability,
                'topk_indices': topk_indices,
                'topk_selected': topk_selected,
            }
        return outputs

    @staticmethod
    def predict_after_deletion(expert_logits, gate_probability, remove_index):
        """Delete one expert per gene, renormalize, and return fused logits.

        ``remove_index`` has shape ``[N]``.  A value of ``-1`` means that no
        selected expert is removed (used by fixed-expert deletion when that
        expert was not in a gene's Top-K set).
        """
        if expert_logits.shape != gate_probability.shape:
            raise ValueError("expert_logits and gate_probability must match")
        remove_index = torch.as_tensor(
            remove_index, device=expert_logits.device, dtype=torch.long
        ).reshape(-1)
        if remove_index.numel() != expert_logits.size(0):
            raise ValueError("remove_index must contain one entry per gene")

        weights = gate_probability.clone()
        rows = torch.arange(weights.size(0), device=weights.device)
        valid = remove_index >= 0
        if valid.any():
            weights[rows[valid], remove_index[valid]] = 0.0
        denominator = weights.sum(dim=1, keepdim=True)
        if (denominator <= 0).any():
            raise ValueError("deletion removed all non-zero expert weights")
        weights = weights / denominator
        return torch.sum(expert_logits * weights, dim=1, keepdim=True)



class G_Net(torch.nn.Module):
    def __init__(self, in_channels=64, hidden_channels=256, out_channels=1, dropout=0.1):
        super(G_Net, self).__init__()
        self.dropout = dropout
        self.in_proj = torch.nn.Linear(in_channels, hidden_channels) if in_channels != hidden_channels else None
        self.conv1 = ChebConv(hidden_channels, hidden_channels, K=2, normalization="sym")
        self.conv2 = ChebConv(hidden_channels, hidden_channels, K=2, normalization="sym")
        self.conv3 = ChebConv(hidden_channels, hidden_channels, K=2, normalization="sym")
        self.classfy = ChebConv(hidden_channels, out_channels, K=2, normalization="sym")

    def forward(self, x, edge):
        edge_index = edge
        x = F.dropout(x, self.dropout, training=self.training)
        if self.in_proj:
            x = self.in_proj(x)

        x1 = F.relu(self.conv1(x, edge_index))
        x = x + x1

        x = F.dropout(x/2, self.dropout, training=self.training)
        x2 = F.relu(self.conv2(x, edge_index))
        x = x + x2

        # x = F.dropout(x/2, self.dropout, training=self.training)
        x3 = self.conv3(x/2, edge_index)
        x = x + x3

        return x
class L_net(torch.nn.Module):
    def __init__(self, in_channels=768, hidden_channels=256, out_channels=1, dropout=0.1):
        super(L_net, self).__init__()
        self.dropout = dropout
        self.in_proj = torch.nn.Linear(in_channels, hidden_channels) if in_channels != hidden_channels else None
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.classfy = SAGEConv(hidden_channels, out_channels)

    def forward(self, L_emb, edge):
        edge_index = edge

        L_emb = F.dropout(L_emb, self.dropout, training=self.training)

        if self.in_proj:
            L_emb = self.in_proj(L_emb)

        h1 = F.elu(self.conv1(L_emb, edge_index))
        L_emb = L_emb + h1

        L_emb = F.dropout(L_emb/2, self.dropout, training=self.training)
        h2 = F.elu(self.conv2(L_emb, edge_index))
        L_emb = L_emb + h2

        # L_emb = F.dropout(L_emb/2, self.dropout, training=self.training)
        h3 = self.conv3(L_emb, edge_index)
        L_emb = L_emb + h3

        return L_emb
