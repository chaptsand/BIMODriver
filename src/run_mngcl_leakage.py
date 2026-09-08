import os
import sys
import copy
import time
import pickle
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.utils import dropout_adj
from sklearn import linear_model, metrics

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, 'src'))
sys.path.append(os.path.join(BASE_DIR, 'src', 'baselines', 'mngcl'))

from alignment_check import assert_data_alignment
from gcn import GCN
from mngcl import MNGCL, contrastive_loss


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class SlicedMNGCL(MNGCL):
    def forward(self, aug_adj_1, aug_adj_2, aug_adj_3, aug_feat_1, aug_feat_2, aug_feat_3, train_idx=None):
        encoder_one = self.encoder(aug_adj_1, aug_feat_1)
        encoder_two = self.encoder(aug_adj_2, aug_feat_2)
        encoder_three = self.encoder(aug_adj_3, aug_feat_3)
        
        proj_one = self.projector(encoder_one)
        proj_two = self.projector(encoder_two)
        proj_three = self.projector(encoder_three)

        if train_idx is not None:
            p1 = proj_one[train_idx]
            p2 = proj_two[train_idx]
            p3 = proj_three[train_idx]
            pos0 = self.posList[0]
            pos1 = self.posList[1]
            pos2 = self.posList[2]

            lab = contrastive_loss(p1, p2, pos0, self.tau)
            lac = contrastive_loss(p1, p3, pos0, self.tau)
            lba = contrastive_loss(p2, p1, pos1, self.tau)
            lca = contrastive_loss(p3, p1, pos2, self.tau)

            Conloss = lab + lac + lba + lca
        else:
            Conloss = torch.tensor(0.0, device=aug_feat_1.device)


        emb1 = self.conv1(encoder_one, aug_adj_1)
        emb2 = self.conv2(encoder_two, aug_adj_2)
        emb3 = self.conv3(encoder_three, aug_adj_3)

        emb = torch.cat((emb1, emb2, emb3), 1)
        return emb1, emb2, emb3, emb, Conloss

def run_mngcl_for_split(split_name, splits_data, gene_df, device, args):
    runs_info = splits_data[split_name]
    n_runs = 1 if args.smoke_test else min(args.n_runs, len(runs_info))
    epochs = args.epochs if not args.smoke_test else min(args.epochs, 5)
    
    print(f"\n{'='*75}")
    print(f"Running MNGCL on Split: [{split_name}]")
    print(f"  Total Runs     : {n_runs} (Smoke test: {args.smoke_test})")
    print(f"  Epochs per Run : {epochs}")
    print(f"  Device         : {device}")
    print(f"{'='*75}")

    data_dir = os.path.join(BASE_DIR, 'data', 'CPDB')
    
    # 1. 严格检查与加载特征 (64-dim = 48 multi-omics + 16 str)
    pt_data = torch.load(os.path.join(data_dir, 'CPDB_new_data.pt'), map_location='cpu')
    str_feat = torch.load(os.path.join(data_dir, 'Str_feature.pkl'), map_location='cpu')
    
    # 断言无文本/大模型特征
    assert pt_data.x.shape == (13627, 64), f"Unexpected pt_data.x shape: {pt_data.x.shape}"
    assert str_feat.shape == (13627, 16), f"Unexpected str_feat shape: {str_feat.shape}"
    
    x = torch.cat((pt_data.x[:, :48], str_feat), 1).to(device)
    assert x.shape == (13627, 64), f"Expected 64-dim input, got {x.shape}"
    
    # 2. 加载网络结构
    ppiAdj = torch.load(os.path.join(data_dir, 'ppi.pkl'), map_location='cpu')
    ppiAdj_self = torch.load(os.path.join(data_dir, 'ppi_selfloop.pkl'), map_location='cpu')
    pathAdj = torch.load(os.path.join(data_dir, 'pathway_SimMatrix.pkl'), map_location='cpu')
    goAdj = torch.load(os.path.join(data_dir, 'GO_SimMatrix.pkl'), map_location='cpu')
    
    # CPU 端稠密正样本矩阵（准备切片）
    print("Preparing dense similarity matrices on CPU...")
    pos1_cpu = ppiAdj_self.to_dense().cpu()
    pos2_cpu = pathAdj.to_dense().cpu() + torch.eye(13627)
    pos3_cpu = goAdj.to_dense().cpu() + torch.eye(13627)
    
    ppiAdj_index = ppiAdj.coalesce().indices().to(device)
    pathAdj_index = pathAdj.coalesce().indices().to(device)
    goAdj_index = goAdj.coalesce().indices().to(device)

    # 标签
    all_labels = (gene_df['Gene_Label'] == 'Driver').values.astype(float)
    Y = torch.tensor(all_labels).float().to(device).unsqueeze(1)

    # 超参数（官方推荐）
    LR = args.lr
    drop_edge_rate_1 = 0.2
    drop_edge_rate_2 = 0.1
    drop_edge_rate_3 = 0.1
    drop_feature_rate_1 = 0.5
    drop_feature_rate_2 = 0.3
    drop_feature_rate_3 = 0.4
    tau = 0.3
    LAMBDA = 0.1
    gnn_outsize = 100
    projection_hidden_size = 300
    projection_size = 100

    test_aurocs = []
    test_auprcs = []
    val_aurocs = []
    val_auprcs = []
    best_epochs = []
    
    # 存储每个测试基因在每轮的预测概率
    fixed_test_idx = runs_info[0]['test_idx']
    test_gene_names = gene_df.iloc[fixed_test_idx]['Gene_Name'].values
    test_true_labels = all_labels[fixed_test_idx].astype(int)
    
    preds_table = {
        'Code_Index': fixed_test_idx,
        'Gene_Name': test_gene_names,
        'True_Label': test_true_labels
    }
    pred_runs_matrix = np.zeros((len(fixed_test_idx), n_runs))

    for run_i in range(n_runs):
        run_info = runs_info[run_i]
        exp_id = run_info['exp_id']
        seed = run_info['seed']
        train_idx = run_info['train_idx']
        val_idx = run_info['val_idx']
        test_idx = run_info['test_idx']

        # 严格断言
        assert np.array_equal(test_idx, fixed_test_idx), "Fixed test indices must be identical across runs!"
        assert len(set(train_idx) & set(val_idx)) == 0, "Train & Val overlap!"
        assert len(set(train_idx) & set(test_idx)) == 0, "Train & Test overlap!"
        assert len(set(val_idx) & set(test_idx)) == 0, "Val & Test overlap!"

        fix_seed(seed)
        train_idx_tensor = torch.tensor(train_idx)
        train_idx_cuda = train_idx_tensor.to(device)

        # 切片训练集正样本矩阵（仅训练节点相互对比）
        pos1_tr = pos1_cpu[train_idx_tensor][:, train_idx_tensor].to(device)
        pos2_tr = pos2_cpu[train_idx_tensor][:, train_idx_tensor].to(device)
        pos3_tr = pos3_cpu[train_idx_tensor][:, train_idx_tensor].to(device)
        posList_tr = [pos1_tr, pos2_tr, pos3_tr]

        # 初始化模型
        gcn = GCN(x.shape[1], 300, gnn_outsize).to(device)
        model = SlicedMNGCL(
            gnn=gcn,
            posList=posList_tr,
            tau=tau,
            gnn_outsize=gnn_outsize,
            projection_hidden_size=projection_hidden_size,
            projection_size=projection_size
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        best_val_auprc = -1.0
        best_val_auroc = -1.0
        best_epoch = -1
        best_model_state = None

        t_start = time.time()
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()

            x_1 = F.dropout(x, drop_feature_rate_1)
            x_2 = F.dropout(x, drop_feature_rate_2)
            x_3 = F.dropout(x, drop_feature_rate_3)
            ppi_drop = dropout_adj(ppiAdj_index, p=drop_edge_rate_1, force_undirected=True)[0]
            path_drop = dropout_adj(pathAdj_index, p=drop_edge_rate_2, force_undirected=True)[0]
            go_drop = dropout_adj(goAdj_index, p=drop_edge_rate_3, force_undirected=True)[0]

            p1, p2, p3, _, conloss = model(
                ppi_drop, path_drop, go_drop, x_1, x_2, x_3, train_idx=train_idx_cuda
            )

            loss1 = F.binary_cross_entropy_with_logits(p1[train_idx_cuda], Y[train_idx_cuda])
            loss2 = F.binary_cross_entropy_with_logits(p2[train_idx_cuda], Y[train_idx_cuda])
            loss3 = F.binary_cross_entropy_with_logits(p3[train_idx_cuda], Y[train_idx_cuda])
            crloss = LAMBDA * (loss1 + loss2 + loss3)
            loss = (1 - 3 * LAMBDA) * conloss + crloss
            loss.backward()
            optimizer.step()

            # 验证集评估（严格不触碰测试集）
            model.eval()
            with torch.no_grad():
                _, _, _, emb_eval, _ = model(ppiAdj_index, pathAdj_index, goAdj_index, x, x, x)
                tr_x_eval = torch.sigmoid(emb_eval[train_idx]).cpu().numpy()
                tr_y_eval = Y[train_idx].cpu().numpy().ravel()
                val_x_eval = torch.sigmoid(emb_eval[val_idx]).cpu().numpy()
                val_y_eval = Y[val_idx].cpu().numpy().ravel()

                regr = linear_model.LogisticRegression(max_iter=10000)
                regr.fit(tr_x_eval, tr_y_eval)
                pred_val = regr.predict_proba(val_x_eval)[:, 1]

                val_auc = metrics.roc_auc_score(val_y_eval, pred_val)
                p_v, r_v, _ = metrics.precision_recall_curve(val_y_eval, pred_val)
                val_prc = metrics.auc(r_v, p_v)

            if val_prc > best_val_auprc:
                best_val_auprc = val_prc
                best_val_auroc = val_auc
                best_epoch = epoch + 1
                best_model_state = copy.deepcopy(model.state_dict())

            if (epoch + 1) % 100 == 0 or (epoch + 1) == epochs:
                print(f"  [Run {run_i+1}/{n_runs}] Epoch {epoch+1:4d}/{epochs} | Val AUROC: {val_auc:.4f}, Val AUPRC: {val_prc:.4f} (Best Ep: {best_epoch}, Best Val AUPRC: {best_val_auprc:.4f})")

        # 训练结束后加载最佳 checkpoint，评估固定测试集一次
        model.load_state_dict(best_model_state)
        model.eval()
        with torch.no_grad():
            _, _, _, emb_eval, _ = model(ppiAdj_index, pathAdj_index, goAdj_index, x, x, x)
            tr_x_eval = torch.sigmoid(emb_eval[train_idx]).cpu().numpy()
            tr_y_eval = Y[train_idx].cpu().numpy().ravel()
            te_x_eval = torch.sigmoid(emb_eval[test_idx]).cpu().numpy()
            te_y_eval = Y[test_idx].cpu().numpy().ravel()

            regr = linear_model.LogisticRegression(max_iter=10000)
            regr.fit(tr_x_eval, tr_y_eval)
            pred_test = regr.predict_proba(te_x_eval)[:, 1]

            test_auc = metrics.roc_auc_score(te_y_eval, pred_test)
            p_t, r_t, _ = metrics.precision_recall_curve(te_y_eval, pred_test)
            test_prc = metrics.auc(r_t, p_t)

        elapsed = time.time() - t_start
        print(f"  >> Run {run_i+1} Done in {elapsed:.1f}s | Best Ep: {best_epoch} | Test AUROC: {test_auc:.4f}, Test AUPRC: {test_prc:.4f}")

        test_aurocs.append(test_auc)
        test_auprcs.append(test_prc)
        val_aurocs.append(best_val_auroc)
        val_auprcs.append(best_val_auprc)
        best_epochs.append(best_epoch)
        pred_runs_matrix[:, run_i] = pred_test
        preds_table[f'Pred_Prob_Run{run_i}'] = pred_test

    test_aurocs = np.array(test_aurocs)
    test_auprcs = np.array(test_auprcs)
    val_aurocs = np.array(val_aurocs)
    val_auprcs = np.array(val_auprcs)

    preds_table['Pred_Prob_Mean'] = pred_runs_matrix.mean(axis=1)
    preds_table['Pred_Prob_Std'] = pred_runs_matrix.std(axis=1)
    pred_df = pd.DataFrame(preds_table)

    prefix = "smoke_mngcl" if args.smoke_test else "mngcl"
    res_dir = os.path.join(BASE_DIR, 'result')
    os.makedirs(res_dir, exist_ok=True)

    auroc_path = os.path.join(res_dir, f"{prefix}_leakage_{split_name}_auroc.txt")
    auprc_path = os.path.join(res_dir, f"{prefix}_leakage_{split_name}_auprc.txt")
    summary_path = os.path.join(res_dir, f"{prefix}_leakage_{split_name}_summary.txt")
    preds_path = os.path.join(res_dir, f"{prefix}_leakage_{split_name}_test_preds.csv")

    np.savetxt(auroc_path, test_aurocs, fmt='%.6f')
    np.savetxt(auprc_path, test_auprcs, fmt='%.6f')
    pred_df.to_csv(preds_path, index=False)

    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"Method: MNGCL\n")
        f.write(f"Task: {split_name}\n")
        f.write(f"Runs: {n_runs} (Smoke test: {args.smoke_test})\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"LR: {LR}, Tau: {tau}, Lambda: {LAMBDA}\n")
        f.write(f"{'-'*60}\n")
        f.write(f"Validation Metrics (Best Checkpoint Average):\n")
        f.write(f"  Val AUROC : {val_aurocs.mean():.4f} ± {val_aurocs.std():.4f}\n")
        f.write(f"  Val AUPRC : {val_auprcs.mean():.4f} ± {val_auprcs.std():.4f}\n")
        f.write(f"  Best Epochs: {best_epochs}\n")
        f.write(f"{'-'*60}\n")
        f.write(f"Final Fixed Test Metrics:\n")
        f.write(f"  Test AUROC : {test_aurocs.mean():.4f} ± {test_aurocs.std():.4f}\n")
        f.write(f"  Test AUPRC : {test_auprcs.mean():.4f} ± {test_auprcs.std():.4f}\n")
        f.write(f"  Per-run AUROC: {np.array2string(test_aurocs, precision=4)}\n")
        f.write(f"  Per-run AUPRC: {np.array2string(test_auprcs, precision=4)}\n")

    print(f"\n--- MNGCL Summary for [{split_name}] ---")
    print(f"  Test AUROC: {test_aurocs.mean():.4f} ± {test_aurocs.std():.4f}")
    print(f"  Test AUPRC: {test_auprcs.mean():.4f} ± {test_auprcs.std():.4f}")
    print(f"  Saved AUROC to: {auroc_path}")
    print(f"  Saved AUPRC to: {auprc_path}")
    print(f"  Saved Preds to: {preds_path} (shape: {pred_df.shape})")
    print(f"  Saved Summary: {summary_path}")

    return test_aurocs, test_auprcs, pred_df

def main():
    parser = argparse.ArgumentParser(description="Run MNGCL Baseline on Leakage Splits or 10x5 CV")
    parser.add_argument('--split', type=str, default='both', choices=['clean_to_hit', 'hit_to_clean', 'both', 'cv'])
    parser.add_argument('--smoke_test', action='store_true', help='Run quick test with 5 epochs')
    parser.add_argument('--n_runs', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    if args.split == 'cv':
        from run_mngcl_cv import run_mngcl_cv
        run_mngcl_cv(args)
        return

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # 0. 先行数据对齐严格断言检查
    assert_data_alignment(BASE_DIR)

    # 读取公共划分与基因注释
    splits_path = os.path.join(BASE_DIR, 'data', 'CPDB', 'leakage_splits_10runs.pkl')
    assert os.path.exists(splits_path), f"Shared split file not found at: {splits_path}"
    with open(splits_path, 'rb') as f:
        splits_data = pickle.load(f)

    audit_path = os.path.join(BASE_DIR, 'src', 'Gemma_Vocabulary_Leakage_Audit.xlsx')
    gene_df = pd.read_excel(audit_path, sheet_name='Gene-level Flags').sort_values('Code_Index').reset_index(drop=True)

    splits_to_run = ['clean_to_hit', 'hit_to_clean'] if args.split == 'both' else [args.split]
    for s in splits_to_run:
        run_mngcl_for_split(s, splits_data, gene_df, device, args)

if __name__ == '__main__':
    main()
