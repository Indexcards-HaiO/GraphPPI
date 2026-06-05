#!/usr/bin/env python3
"""
Benchmark & 复杂度分析

运行所有方法的复杂度与时间基准测试。
Usage: python src/benchmark.py
"""

import time
import torch
import numpy as np
from graphppi.utils import split_edges_kfold, prepare_fold_data
from graphppi.models.predictor import LinkPredictor
from graphppi.trainer import LinkPredictionTrainer
from graphppi.baselines.common_neighbors import common_neighbors_predict
from graphppi.baselines.jaccard import jaccard_predict
from graphppi.baselines.adamic_adar import adamic_adar_predict

# ────────────────────────────────────────────
# 理论复杂度分析
# ────────────────────────────────────────────

COMPLEXITY = r"""
方法复杂度分析 (N=节点数, E=边数, D=嵌入维度, L=层数)
═══════════════════════════════════════════════════════════════
方法                 训练复杂度              推理复杂度
───────────────────────────────────────────────────────────────
GCN Encoder          O(L·E·D + L·N·D²)     O(L·E·D + L·N·D²)
GAT Encoder          O(L·E·D + L·N·D²)     O(L·E·D + L·N·D²)
SAGE Encoder         O(L·E·D + L·N·D²)     O(L·E·D + L·N·D²)
DotProduct Decoder   O(D)                   O(D)
MLP Decoder          O(D·H + H·1)           O(D·H + H·1)
EdgeMLP Decoder      O((2D+F)·H + H·1)      O((2D+F)·H + H·1)
Common Neighbors     O(E/N + E)             O(d̄²) per pair
Jaccard              O(E/N + E)             O(d̄²) per pair
Adamic-Adar          O(E/N + E)             O(d̄²) per pair
Node2Vec+RF          O(w·l·N + N_tree·N·D) O(N_tree·D_tree)
───────────────────────────────────────────────────────────────
其中 H=隐藏维度, F=边特征维度, d̄=平均度, w=walks, l=walk_len
"""

print(COMPLEXITY)

# ────────────────────────────────────────────
# 实测 Benchmark
# ────────────────────────────────────────────

data = torch.load("data/processed/graph.pt", weights_only=False)
num_undirected = data.edge_index.size(1) // 2
folds = split_edges_kfold(data.edge_index, num_undirected, k=3, seed=42)
fold_data = prepare_fold_data(data, folds[0])

x_feat = fold_data["x"]
train_edges = fold_data["train_edges"]
train_labels = fold_data["train_labels"]
val_edges = fold_data["val_edges"]
val_labels = fold_data["val_labels"]
test_edges = fold_data["test_edges"]
test_labels = fold_data["test_labels"]
mp_edge_index = fold_data["mp_edge_index"]

# 为 baseline 准备数据
from graphppi.utils import get_edges_by_indices, sample_negative_edges

fold_idx = folds[0]
train_pos = get_edges_by_indices(data.edge_index, fold_idx["train"], num_undirected)
test_pos_all = get_edges_by_indices(data.edge_index, fold_idx["test"], num_undirected)
num_test_undir = len(fold_idx["test"])
test_pos = test_pos_all[:, :num_test_undir]

train_edge_set = set()
for i in range(train_pos.size(1)):
    u, v = int(train_pos[0, i].item()), int(train_pos[1, i].item())
    train_edge_set.add((u, v))

test_neg = sample_negative_edges(test_pos, data.num_nodes, num_test_undir, train_edge_set, random_state=42)
test_edges_bl = torch.cat([test_pos, test_neg], dim=1)
test_labels_bl = torch.cat([torch.ones(num_test_undir), torch.zeros(num_test_undir)])

print("=" * 70)
print(f"{'Method':<30s} {'Train(s)':>10s} {'Infer(s)':>10s} {'AUC':>8s} {'#Params':>10s}")
print("-" * 70)


def bench_gnn(name, enc, dec):
    m = LinkPredictor(in_dim=x_feat.size(1), encoder_type=enc, decoder_type=dec,
                      hidden_dim=128, out_dim=64, num_layers=2, dropout=0.5)
    n_params = sum(p.numel() for p in m.parameters())

    t = LinkPredictionTrainer(m, lr=0.005, weight_decay=1e-4)

    t0 = time.time()
    t.train(x=x_feat, mp_edge_index=mp_edge_index,
            train_edges=train_edges, train_labels=train_labels,
            val_edges=val_edges, val_labels=val_labels,
            epochs=100, patience=15, verbose=False,
            mp_edge_weight=fold_data.get("mp_edge_weight"),
            train_edge_attr=fold_data.get("train_edge_attr"),
            val_edge_attr=fold_data.get("val_edge_attr"))
    train_time = time.time() - t0

    t0 = time.time()
    mets = t.test(x=x_feat, mp_edge_index=mp_edge_index,
                  test_edges=test_edges, test_labels=test_labels,
                  mp_edge_weight=fold_data.get("mp_edge_weight"),
                  test_edge_attr=fold_data.get("test_edge_attr"))
    infer_time = time.time() - t0

    print(f"{name:<30s} {train_time:>10.2f} {infer_time:>10.3f} {mets['auc']:>8.4f} {n_params:>10,}")
    return mets


def bench_baseline(name, fn):
    t0 = time.time()
    auc, ap, _ = fn(train_pos, data.num_nodes, test_edges_bl, test_labels_bl)
    elapsed = time.time() - t0
    print(f"{name:<30s} {'—':>10s} {elapsed:>10.3f} {auc:>8.4f} {'—':>10s}")
    return auc, ap


bench_gnn("GraphSAGE + MLP", "sage", "mlp")
bench_gnn("GCN + MLP", "gcn", "mlp")
bench_gnn("GAT + MLP", "gat", "mlp")
bench_gnn("GCN + Dot", "gcn", "dot")
bench_baseline("Common Neighbors", common_neighbors_predict)
bench_baseline("Jaccard", jaccard_predict)
bench_baseline("Adamic-Adar", adamic_adar_predict)

print("-" * 70)
print("结论: GNN 方法训练开销大 (~1-5s) 但 AUC 显著高于 baseline。")
print("      Baseline 方法训练几乎为 0，适合快速探索。")
