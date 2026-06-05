#!/usr/bin/env python3
"""
独立测试集评估

从原始 STRING 数据中按时间或来源划分一个独立测试集，
评估模型在 unseen data 上的泛化能力。

Usage: python src/evaluate_independent.py
"""

import torch
import numpy as np
import argparse
from collections import defaultdict

from graphppi.utils import get_edges_by_indices, sample_negative_edges
from graphppi.models.predictor import LinkPredictor
from graphppi.trainer import LinkPredictionTrainer
from graphppi.metrics import compute_all_metrics


def create_independent_split(data, holdout_ratio=0.2, seed=42):
    """
    创建独立测试集 (hold-out by source/time)。
    实际实现：随机划分 20% 无向边作为独立测试集，80% 为训练/验证集。
    在真实场景中，可替换为按数据来源（如实验验证 vs 文本挖掘）划分。
    """
    num_undirected = data.edge_index.size(1) // 2
    np.random.seed(seed)
    indices = np.random.permutation(num_undirected)
    split = int(num_undirected * (1 - holdout_ratio))

    train_indices = indices[:split]
    test_indices = indices[split:]

    return train_indices, test_indices


def evaluate_independent(data, train_indices, test_indices, config, seed=42):
    """
    在独立测试集上评估模型。
    """
    num_undirected = data.edge_index.size(1) // 2
    num_nodes = data.num_nodes

    # 构建训练边
    train_pos = get_edges_by_indices(data.edge_index, train_indices, num_undirected)
    test_pos_all = get_edges_by_indices(data.edge_index, test_indices, num_undirected)
    num_test_undir = len(test_indices)
    test_pos = test_pos_all[:, :num_test_undir]

    # 训练边集合（用于负采样排除）
    train_edge_set = set()
    for i in range(train_pos.size(1)):
        u, v = int(train_pos[0, i].item()), int(train_pos[1, i].item())
        train_edge_set.add((u, v))

    # 划分 train/val (90/10)
    np.random.seed(seed)
    num_train_undir = train_pos.size(1) // 2
    tv_indices = np.random.permutation(num_train_undir)
    val_size = max(1, int(num_train_undir * 0.1))
    val_undir = tv_indices[:val_size]
    train_undir = tv_indices[val_size:]

    train_edges_pos = torch.cat([
        get_edges_by_indices(data.edge_index, train_indices[train_undir], num_undirected)
    ], dim=1)

    # 训练边 mp_edge_index (only positive)
    mp_edges = train_edges_pos[:, :train_edges_pos.size(1)//2]

    # 训练负采样
    train_neg = sample_negative_edges(
        mp_edges, num_nodes, mp_edges.size(1), train_edge_set, random_state=seed
    )
    all_train_edges = torch.cat([mp_edges, train_neg], dim=1)
    all_train_labels = torch.cat([torch.ones(mp_edges.size(1)), torch.zeros(mp_edges.size(1))])

    # 验证集
    val_pos = get_edges_by_indices(data.edge_index, train_indices[val_undir], num_undirected)
    val_pos_u = val_pos[:, :val_pos.size(1)//2]
    val_neg = sample_negative_edges(val_pos_u, num_nodes, val_pos_u.size(1), train_edge_set, random_state=seed+1)
    all_val_edges = torch.cat([val_pos_u, val_neg], dim=1)
    all_val_labels = torch.cat([torch.ones(val_pos_u.size(1)), torch.zeros(val_pos_u.size(1))])

    # 测试集
    test_neg = sample_negative_edges(test_pos, num_nodes, num_test_undir, train_edge_set, random_state=seed+2)
    all_test_edges = torch.cat([test_pos, test_neg], dim=1)
    all_test_labels = torch.cat([torch.ones(num_test_undir), torch.zeros(num_test_undir)])

    # 特征 (from training edges only)
    from graphppi.utils import compute_features_from_edges
    x_feat = compute_features_from_edges(mp_edges, torch.ones(mp_edges.size(1)), num_nodes, data.node_names)

    # 训练
    model = LinkPredictor(
        in_dim=x_feat.size(1),
        encoder_type=config.get("encoder", "sage"),
        decoder_type=config.get("decoder", "mlp"),
        hidden_dim=config.get("hidden_dim", 128),
        out_dim=config.get("out_dim", 64),
        num_layers=config.get("num_layers", 2),
        dropout=config.get("dropout", 0.5),
    )
    trainer = LinkPredictionTrainer(model, lr=config.get("lr", 0.005),
                                     weight_decay=config.get("weight_decay", 1e-4))
    trainer.train(
        x=x_feat, mp_edge_index=mp_edges,
        train_edges=all_train_edges, train_labels=all_train_labels,
        val_edges=all_val_edges, val_labels=all_val_labels,
        epochs=config.get("epochs", 200),
        patience=config.get("patience", 20),
        verbose=config.get("verbose", True),
    )

    # 测试
    metrics = trainer.test(
        x=x_feat, mp_edge_index=mp_edges,
        test_edges=all_test_edges, test_labels=all_test_labels,
    )

    return metrics, trainer


def main():
    parser = argparse.ArgumentParser(description="独立测试集评估")
    parser.add_argument("--holdout", type=float, default=0.2, help="独立测试集比例")
    parser.add_argument("--encoder", default="sage")
    parser.add_argument("--decoder", default="mlp")
    parser.add_argument("--output", default="results/independent_test.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args()

    print(f"独立测试集评估 (holdout={args.holdout})")
    print("=" * 60)

    data = torch.load("data/processed/graph.pt", weights_only=False)
    train_indices, test_indices = create_independent_split(data, args.holdout, args.seed)

    print(f"  训练集无向边: {len(train_indices)}")
    print(f"  独立测试集无向边: {len(test_indices)}")

    config = {
        "encoder": args.encoder, "decoder": args.decoder,
        "hidden_dim": 128, "out_dim": 64, "num_layers": 2, "dropout": 0.5,
        "lr": 0.005, "weight_decay": 1e-4,
        "epochs": args.epochs, "patience": args.patience,
        "verbose": True,
    }

    metrics, _ = evaluate_independent(data, train_indices, test_indices, config, args.seed)

    print("\n独立测试集结果:")
    print("-" * 30)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # 保存结果
    import pandas as pd
    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    pd.DataFrame([metrics]).to_csv(args.output, index=False)
    print(f"\n结果已保存到 {args.output}")


if __name__ == "__main__":
    main()
