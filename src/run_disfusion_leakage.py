import os
import sys
import copy
import time
import pickle
import random
import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn import metrics

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, 'src', 'baselines', 'disfusion'))

from models import hypergrph_HGNN, graph_ChebNet, DISFusion

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def fast_G_from_H_weight(H, W):
    """
    Mathematically identical to DISFusion's _generate_G_from_H_weight,
    optimized to avoid allocating multiple 13627x13627 diagonal matrices.
    """
    DV = np.sum(H * W, axis=1)
    DE = np.sum(H, axis=0)
    invDE_W = W / DE
    H_scaled = H * invDE_W
    A = H_scaled @ H.T
    dv_inv_sqrt = np.power(DV, -0.5)[:, None]
    G = dv_inv_sqrt * A * dv_inv_sqrt.T
    return G

def get_incidence_matrix(data_dir, gene_list):
    """
    Precomputes the incidence matrix from c2 and c5, filtering out cancer/tumor terms.
    """
    ids = ['c2', 'c5']
    incidenceMatrix = pd.DataFrame(index=gene_list)
    for id_name in ids:
        geneSetNameList = pd.read_csv(os.path.join(data_dir, f'{id_name}Name.txt'), sep='\t', header=None)
        geneSetNameList = list(geneSetNameList[0].values)
        idList = []
        for z, name in enumerate(geneSetNameList):
            if id_name == 'c2':
                q = name.split('_')
                if not ('CANCER' in q or 'TUMOR' in q or 'NEOPLASM' in q):
                    idList.append(z)
            elif name[:2] == 'HP':
                q = name.split('_')
                if not ('CANCER' in q or 'TUMOR' in q or 'NEOPLASM' in q):
                    idList.append(z)
            else:
                idList.append(z)
        genesetData = sp.load_npz(os.path.join(data_dir, f'{id_name}_GenesetsMatrix.npz'))
        temp = pd.DataFrame(data=genesetData.A, index=gene_list)
        temp = temp.iloc[:, idList]
        incidenceMatrix = pd.concat([incidenceMatrix, temp], axis=1)
    incidenceMatrix.columns = np.arange(incidenceMatrix.shape[1])
    return incidenceMatrix

def run_disfusion_for_split(split_name, splits_data, gene_df, device, args):
    runs_info = splits_data[split_name]
    n_runs = 1 if args.smoke_test else min(args.n_runs, len(runs_info))
    epochs = args.epochs if not args.smoke_test else min(args.epochs, 5)

    print(f"\n{'='*75}")
    print(f"Running DISFusion on Split: [{split_name}]")
    print(f"  Total Runs     : {n_runs} (Smoke test: {args.smoke_test})")
    print(f"  Epochs per Run : {epochs}")
    print(f"  Device         : {device}")
    print(f"{'='*75}")

    data_dir = os.path.join(BASE_DIR, 'data', 'DISFusion_data')
    gene_list = list(gene_df['Gene_Name'].values)
    assert len(gene_list) == 13627

    # 1. 严格检查与加载特征 (48-dim biological features)
    feature_df = pd.read_csv(os.path.join(data_dir, 'biological features.csv'), sep=',')
    assert feature_df.shape == (13627, 48), f"Expected (13627, 48), got {feature_df.shape}"
    feature_tensor = torch.Tensor(feature_df.values).to(device)

    # 2. 加载 PPI 邻接
    edge_np = np.array(np.loadtxt(os.path.join(data_dir, 'PPI_edge_index.txt')).transpose())
    PPI_graph = torch.from_numpy(edge_np).long().to(device)

    # 3. 预计算超图关联网
    t_inc = time.time()
    print("Loading and filtering incidence matrix...")
    incidence_matrix = get_incidence_matrix(data_dir, gene_list)
    print(f"Incidence matrix ready in {time.time()-t_inc:.2f}s, shape: {incidence_matrix.shape}")

    # 4. 标签与恒等基底
    N = 13627
    fh = torch.eye(N).float().to(device)
    all_labels = (gene_df['Gene_Label'] == 'Driver').values.astype(int)
    labels_tensor = torch.from_numpy(all_labels).to(device)

    label_frame = pd.DataFrame(data=all_labels, index=gene_list)

    # 超参数（官方推荐）
    lr = args.lr
    weight_decay = 5e-5
    n_hid = 256
    lambdinter = 1e-4
    w_self = 0.005
    dropout = 0.5

    test_aurocs = []
    test_auprcs = []
    val_aurocs = []
    val_auprcs = []
    best_epochs = []

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

        # 疾病特异超边加权：仅使用当前内部训练集中的阳性基因
        t_graph = time.time()
        train_frame = label_frame.iloc[train_idx]
        train_pos_genes = list(train_frame.where(train_frame == 1).dropna().index)
        pos_matrix_sum = incidence_matrix.loc[train_pos_genes].sum()

        sel_hyperedge_idx = np.where(pos_matrix_sum >= 3)[0]
        sel_hyperedge = incidence_matrix.iloc[:, sel_hyperedge_idx]
        hyperedge_weight = pos_matrix_sum[sel_hyperedge_idx].values
        sel_weight_sum = sel_hyperedge.values.sum(0)
        hyperedge_weight = hyperedge_weight / sel_weight_sum

        H = np.array(sel_hyperedge).astype('float')
        DV = np.sum(H * hyperedge_weight, axis=1)
        for i in range(DV.shape[0]):
            if DV[i] == 0:
                t_rand = random.randint(0, H.shape[1] - 1)
                H[i][t_rand] = 0.0001

        G = fast_G_from_H_weight(H, hyperedge_weight)
        adj_hyperGraph = torch.Tensor(G).float().to(device)
        # print(f"  Hypergraph G constructed in {time.time()-t_graph:.2f}s")

        # 初始化模型
        model_hypergrph = hypergrph_HGNN(in_ch=N, n_hid=n_hid, dropout=0.2).to(device)
        model_graph = graph_ChebNet(hdim=n_hid, dropout=0.5).to(device)
        model_fusion = DISFusion(n_hid, 2, lambdinter, attention=0, nb_classes=2, dropout=dropout).to(device)

        optimizer_hypergrph = optim.Adam(model_hypergrph.parameters(), lr=0.005, weight_decay=0.000005)
        optimizer_graph = optim.Adam(model_graph.parameters(), lr=0.001, weight_decay=0)
        schedular_hypergrph = optim.lr_scheduler.MultiStepLR(optimizer_hypergrph, milestones=[100, 200, 300, 400], gamma=0.5)
        optimizer_fusion = optim.Adam(model_fusion.parameters(), lr=lr, weight_decay=weight_decay)

        best_val_auprc = -1.0
        best_val_auroc = -1.0
        best_epoch = -1
        best_states = None

        t_start = time.time()
        for epoch in range(epochs):
            model_hypergrph.train()
            model_graph.train()
            model_fusion.train()
            optimizer_hypergrph.zero_grad()
            optimizer_graph.zero_grad()
            optimizer_fusion.zero_grad()

            h1 = model_hypergrph(fh, adj_hyperGraph)
            h2 = model_graph(feature_tensor, PPI_graph)
            # 内部训练节点对比/自监督损失，分类损失也仅在训练节点上计算
            loss_self, output_fusion = model_fusion(h1, h2, train_idx=train_idx_cuda)
            loss_cls = F.nll_loss(output_fusion[train_idx_cuda], labels_tensor[train_idx_cuda].long())
            loss = loss_cls + w_self * loss_self

            loss.backward()
            optimizer_fusion.step()
            optimizer_hypergrph.step()
            optimizer_graph.step()
            schedular_hypergrph.step()

            # 验证集评估（绝不访问测试集）
            model_hypergrph.eval()
            model_graph.eval()
            model_fusion.eval()
            with torch.no_grad():
                h1_val = model_hypergrph(fh, adj_hyperGraph)
                h2_val = model_graph(feature_tensor, PPI_graph)
                _, output_eval = model_fusion(h1_val, h2_val)
                pred_val = output_eval[val_idx, 1].exp().cpu().numpy()
                val_y = all_labels[val_idx]

                val_auc = metrics.roc_auc_score(val_y, pred_val)
                p_v, r_v, _ = metrics.precision_recall_curve(val_y, pred_val)
                val_prc = metrics.auc(r_v, p_v)

            if val_prc > best_val_auprc:
                best_val_auprc = val_prc
                best_val_auroc = val_auc
                best_epoch = epoch + 1
                best_states = {
                    'hypergrph': copy.deepcopy(model_hypergrph.state_dict()),
                    'graph': copy.deepcopy(model_graph.state_dict()),
                    'fusion': copy.deepcopy(model_fusion.state_dict())
                }

            if (epoch + 1) % 50 == 0 or (epoch + 1) == epochs:
                print(f"  [Run {run_i+1}/{n_runs}] Epoch {epoch+1:3d}/{epochs} | Val AUROC: {val_auc:.4f}, Val AUPRC: {val_prc:.4f} (Best Ep: {best_epoch}, Best Val AUPRC: {best_val_auprc:.4f})")

        # 训练结束后加载最佳 checkpoint，评估固定测试集一次
        model_hypergrph.load_state_dict(best_states['hypergrph'])
        model_graph.load_state_dict(best_states['graph'])
        model_fusion.load_state_dict(best_states['fusion'])

        model_hypergrph.eval()
        model_graph.eval()
        model_fusion.eval()
        with torch.no_grad():
            h1_te = model_hypergrph(fh, adj_hyperGraph)
            h2_te = model_graph(feature_tensor, PPI_graph)
            _, output_eval = model_fusion(h1_te, h2_te)
            pred_test = output_eval[test_idx, 1].exp().cpu().numpy()
            te_y = all_labels[test_idx]

            test_auc = metrics.roc_auc_score(te_y, pred_test)
            p_t, r_t, _ = metrics.precision_recall_curve(te_y, pred_test)
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

    prefix = "smoke_disfusion" if args.smoke_test else "disfusion"
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
        f.write(f"Method: DISFusion\n")
        f.write(f"Task: {split_name}\n")
        f.write(f"Runs: {n_runs} (Smoke test: {args.smoke_test})\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"LR: {lr}, w_self: {w_self}, lambdinter: {lambdinter}\n")
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

    print(f"\n--- DISFusion Summary for [{split_name}] ---")
    print(f"  Test AUROC: {test_aurocs.mean():.4f} ± {test_aurocs.std():.4f}")
    print(f"  Test AUPRC: {test_auprcs.mean():.4f} ± {test_auprcs.std():.4f}")
    print(f"  Saved AUROC to: {auroc_path}")
    print(f"  Saved AUPRC to: {auprc_path}")
    print(f"  Saved Preds to: {preds_path} (shape: {pred_df.shape})")
    print(f"  Saved Summary: {summary_path}")

    return test_aurocs, test_auprcs, pred_df

def main():
    parser = argparse.ArgumentParser(description="Run DISFusion Baseline on Leakage Splits")
    parser.add_argument('--split', type=str, default='both', choices=['clean_to_hit', 'hit_to_clean', 'both'])
    parser.add_argument('--smoke_test', action='store_true', help='Run 1 run with 5 epochs for quick verification')
    parser.add_argument('--n_runs', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    splits_path = os.path.join(BASE_DIR, 'data', 'CPDB', 'leakage_splits_10runs.pkl')
    assert os.path.exists(splits_path), f"Shared split file not found at: {splits_path}"
    with open(splits_path, 'rb') as f:
        splits_data = pickle.load(f)

    audit_path = os.path.join(BASE_DIR, 'src', 'Gemma_Vocabulary_Leakage_Audit.xlsx')
    gene_df = pd.read_excel(audit_path, sheet_name='Gene-level Flags').sort_values('Code_Index').reset_index(drop=True)

    splits_to_run = ['clean_to_hit', 'hit_to_clean'] if args.split == 'both' else [args.split]
    for s in splits_to_run:
        run_disfusion_for_split(s, splits_data, gene_df, device, args)

if __name__ == '__main__':
    main()
