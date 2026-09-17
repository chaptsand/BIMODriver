"""Train BIMODriver and export the routing data used by Figure 1.

This copied-and-adapted entry point is based on main.py. Gate probabilities are
sparse Top-4 probabilities: softmax is applied only to the four selected gate
logits and the other two expert probabilities are exactly zero.
"""

import argparse
import csv
import json
import pickle
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / 'BIMODriver') not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / 'BIMODriver'))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
from torch_geometric.utils import add_self_loops, dropout_adj, remove_self_loops

from model_top4_routing import combine_net_gate_without_ac


EXPERTS = ("Omics", "Self", "Neighbor", "Together", "Interact_OS", "Interact_ST")
CSV_FIELDS = (
    "protocol", "model", "seed", "fold", "split", "gene_index", "gene_id",
    "class", "expert", "expert_logit", "gate_logit", "gate_probability",
    "topk_selected", "top_k",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export train/OOF sparse Top-4 routing data for CPDB Figure 1"
    )
    parser.add_argument("--protocol", choices=("transductive", "inductive"),
                        default="transductive")
    parser.add_argument("--experiments", type=int, default=10)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lambda-inter", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def remove_held_out_edges(edge_index, held_out_mask):
    """Return the induced graph after removing edges incident to OOF genes."""
    held_out_mask = held_out_mask.to(edge_index.device).bool()
    source, target = edge_index
    keep = (~held_out_mask[source]) & (~held_out_mask[target])
    return edge_index[:, keep]


def assert_no_held_out_edges(edge_index, held_out_mask, graph_name):
    """Fail fast if a strict-inductive training graph touches a test gene."""
    held_out_mask = held_out_mask.to(edge_index.device).bool()
    if edge_index.numel() and (
        held_out_mask[edge_index[0]].any() or held_out_mask[edge_index[1]].any()
    ):
        raise RuntimeError(
            f"{graph_name} still contains an edge incident to a held-out gene"
        )


def get_class_weight(labels):
    positive = labels.sum(dim=0)
    negative = labels.shape[0] - positive
    return negative / (positive + 1e-6)


def compute_loss(outputs, indices, labels, lambda_inter):
    loss_inter, *expert_and_fused = outputs
    class_weight = get_class_weight(labels[indices])
    classification = sum(
        F.binary_cross_entropy_with_logits(
            logits[indices], labels[indices], pos_weight=class_weight
        )
        for logits in expert_and_fused
    )
    return classification + lambda_inter * loss_inter


def evaluate(outputs, test_mask, labels):
    truth = labels[test_mask].detach().cpu().numpy().ravel()
    prediction = torch.sigmoid(outputs[-1][test_mask]).detach().cpu().numpy().ravel()
    precision, recall, _ = metrics.precision_recall_curve(truth, prediction)
    return {
        "auroc": float(metrics.roc_auc_score(truth, prediction)),
        "auprc": float(metrics.auc(recall, precision)),
    }


def load_gene_ids(path, number_of_nodes):
    if path.exists():
        values = [line.strip().split()[0]
                  for line in path.read_text(encoding="utf-8").splitlines()
                  if line.strip()]
        if len(values) == number_of_nodes:
            return values
        print(f"Warning: {path} contains {len(values)} names; using node indices.")
    return [str(index) for index in range(number_of_nodes)]


def export_split(writer, routing, mask, labels, gene_ids, protocol, seed, fold, split):
    indices = mask.nonzero(as_tuple=True)[0].detach().cpu().numpy()
    expert_logits = routing["expert_logits"].detach().cpu().numpy()
    gate_logits = routing["gate_logits"].detach().cpu().numpy()
    probabilities = routing["gate_probability"].detach().cpu().numpy()
    selected = routing["topk_selected"].detach().cpu().numpy()
    targets = labels.detach().cpu().numpy().reshape(-1)

    for gene_index in indices:
        class_name = "Driver" if targets[gene_index] >= 0.5 else "Non-driver"
        for expert_index, expert_name in enumerate(EXPERTS):
            writer.writerow({
                "protocol": protocol, "model": "sparse_top4", "seed": seed,
                "fold": fold, "split": split, "gene_index": int(gene_index),
                "gene_id": gene_ids[gene_index], "class": class_name,
                "expert": expert_name,
                "expert_logit": float(expert_logits[gene_index, expert_index]),
                "gate_logit": float(gate_logits[gene_index, expert_index]),
                "gate_probability": float(probabilities[gene_index, expert_index]),
                "topk_selected": int(selected[gene_index, expert_index]),
                "top_k": 4,
            })


def load_cpdb(repo_dir, device):
    data_dir = repo_dir / "data" / "CPDB"
    data = torch.load(data_dir / "CPDB_new_data.pt", map_location=device).to(device)
    data.x = data.x[:, :48]
    structural = torch.load(data_dir / "Str_feature.pkl", map_location=device).to(device)
    data.x = torch.cat((data.x, structural), dim=1)
    with (data_dir / "k_sets.pkl").open("rb") as handle:
        split_sets = pickle.load(handle)

    statement = torch.load(
        data_dir / "PAN-CANCER_statement_features.pt", map_location=device
    ).to(device)
    language_embeddings = {
        "self_emb": statement[:, 0:768],
        "neighbor_emb": statement[:, 768:1536],
        "together_emb": statement[:, 1536:2304],
    }
    semantic_edges = torch.load(
        repo_dir / "data" / "cpdb_network_LLM" / "merged_k5_edge_index.pt",
        map_location=device,
    ).to(device)
    ppi_edges, _ = remove_self_loops(data.edge_index)
    ppi_edges, _ = add_self_loops(ppi_edges)
    labels = torch.logical_or(
        torch.as_tensor(data.y, device=device).bool(),
        torch.as_tensor(data.y_te, device=device).bool(),
    ).to(dtype=torch.float32)
    if labels.ndim == 1:
        labels = labels.unsqueeze(1)
    gene_ids = load_gene_ids(data_dir / "node_names.txt", data.num_nodes)
    return data, split_sets, language_embeddings, semantic_edges, ppi_edges, labels, gene_ids


def main():
    args = parse_args()
    if not 1 <= args.experiments <= 10 or not 1 <= args.folds <= 5:
        raise ValueError("--experiments must be 1..10 and --folds must be 1..5")

    output = args.output or (
        REPO_ROOT / "cache" / "routing_top4" / "CPDB" / "cpdb" /
        args.protocol / "oof_gene_routing.csv"
    )
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_path = output.with_name("metrics.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = load_cpdb(REPO_ROOT, device)
    data, split_sets, language_embeddings, semantic_edges, ppi_edges, labels, gene_ids = loaded

    metric_rows = []
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for experiment in range(args.experiments):
            run_seed = args.seed + experiment
            for fold_index in range(args.folds):
                set_seed(run_seed * 100 + fold_index)
                _, _, train_raw, test_raw = split_sets[experiment][fold_index]
                train_mask = torch.as_tensor(train_raw, dtype=torch.bool, device=device)
                test_mask = torch.as_tensor(test_raw, dtype=torch.bool, device=device)
                train_indices = train_mask.nonzero(as_tuple=True)[0]

                if args.protocol == "inductive":
                    ppi_train = remove_held_out_edges(ppi_edges, test_mask)
                    semantic_train = remove_held_out_edges(semantic_edges, test_mask)
                    contrastive_mask = ~test_mask
                    # The same mask is applied inside the model to every input
                    # feature tensor.  Consequently, test genes are isolated in
                    # both graphs, have all-zero features, and contribute to
                    # neither the supervised nor the contrastive objective.
                    training_node_mask = ~test_mask
                    assert_no_held_out_edges(ppi_train, test_mask, "CPDB graph")
                    assert_no_held_out_edges(
                        semantic_train, test_mask, "semantic KNN graph"
                    )
                else:
                    ppi_train, semantic_train = ppi_edges, semantic_edges
                    contrastive_mask = None
                    training_node_mask = None

                model = combine_net_gate_without_ac(
                    input_dim=data.x.shape[1], lambdinter=args.lambda_inter,
                    dropout=args.dropout,
                ).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

                for epoch in range(args.epochs):
                    model.train()
                    optimizer.zero_grad()
                    dropped_ppi = dropout_adj(ppi_train, p=0.3)[0]
                    outputs = model(
                        data.x, dropped_ppi, language_embeddings, semantic_train,
                        contrastive_mask=contrastive_mask,
                        input_node_mask=training_node_mask,
                    )
                    loss = compute_loss(outputs, train_indices, labels, args.lambda_inter)
                    loss.backward()
                    optimizer.step()
                    if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == args.epochs:
                        print(
                            f"protocol={args.protocol} seed={experiment + 1}/{args.experiments} "
                            f"fold={fold_index + 1}/{args.folds} epoch={epoch + 1}/{args.epochs} "
                            f"loss={loss.item():.6f}"
                        )

                model.eval()
                with torch.no_grad():
                    _, train_routing = model(
                        data.x, ppi_train, language_embeddings, semantic_train,
                        return_routing=True,
                    )
                    test_outputs, test_routing = model(
                        data.x, ppi_edges, language_embeddings, semantic_edges,
                        return_routing=True,
                    )

                export_split(writer, train_routing, train_mask, labels, gene_ids,
                             args.protocol, experiment + 1, fold_index + 1, "train")
                export_split(writer, test_routing, test_mask, labels, gene_ids,
                             args.protocol, experiment + 1, fold_index + 1, "test")
                row = evaluate(test_outputs, test_mask, labels)
                row.update({"seed": experiment + 1, "fold": fold_index + 1})
                metric_rows.append(row)
                handle.flush()

    summary = {
        "protocol": args.protocol, "top_k": 4,
        "evaluation_epoch": args.epochs,
        "epoch_selection": "final_epoch",
        "probability_definition": (
            "softmax over selected Top-4 gate logits; other two experts are zero"
        ),
        "experiments": args.experiments, "folds": args.folds,
        "epochs": args.epochs, "metrics": metric_rows,
        "mean_auroc": float(np.mean([row["auroc"] for row in metric_rows])),
        "mean_auprc": float(np.mean([row["auprc"] for row in metric_rows])),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Routing CSV: {output}")
    print(f"Metrics JSON: {summary_path}")


if __name__ == "__main__":
    main()
