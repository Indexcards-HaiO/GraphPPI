#!/usr/bin/env python3
"""
GNN 链路预测评估脚本
- k-fold 交叉验证
- 输出 AUC / AP / Hits@K 均值 ± 标准差
- 与 baseline 对比
"""

import torch
import numpy as np
import os
import argparse
from collections import defaultdict

from graphppi.utils import split_edges_kfold, prepare_fold_data
from graphppi.models.predictor import LinkPredictor
from graphppi.trainer import LinkPredictionTrainer
from graphppi.metrics import format_metrics


def evaluate_gnn_kfold(data, config, k=5, verbose=True):
    """
    对 GNN 模型运行 k-fold 交叉验证
    
    参数:
        data: PyG Data 对象
        config: dict 模型/训练配置
        k: fold 数
        verbose: 是否打印详细信息
    
    返回:
        results: dict 汇总指标（均值+标准差）
        all_fold_results: list 每个 fold 的指标
    """
    num_undirected = data.edge_index.size(1) // 2
    folds = split_edges_kfold(data.edge_index, num_undirected, k=k, seed=config.get('seed', 42))

    all_metrics = []
    all_fold_results = []

    if verbose:
        print(f"\n{'='*60}")
        print(f"GNN k-Fold CV (k={k})")
        print(f"  编码器: {config.get('encoder', 'gcn')}")
        print(f"  解码器: {config.get('decoder', 'dot')}")
        print(f"  节点特征: {'identity' if config.get('use_identity') else 'topology (5)'}")
        print(f"{'='*60}")

    for fold_idx, fold_indices in enumerate(folds):
        if verbose:
            print(f"\n--- Fold {fold_idx + 1}/{k} ---")
            print(f"  Train: {len(fold_indices['train'])} undirected edges")
            print(f"  Val:   {len(fold_indices['val'])} undirected edges")
            print(f"  Test:  {len(fold_indices['test'])} undirected edges")

        # 准备数据
        fold_data = prepare_fold_data(
            data, fold_indices,
            neg_ratio=config.get('neg_ratio', 1),
            hard_negative=config.get('hard_negative', False)
        )

        x = fold_data['x']
        num_nodes = x.size(0)
        in_dim = num_nodes if config.get('use_identity', False) else x.size(1)

        if config.get('use_identity', False):
            x = torch.eye(num_nodes)

        # 创建模型
        model = LinkPredictor(
            in_dim=in_dim,
            encoder_type=config.get('encoder', 'gcn'),
            decoder_type=config.get('decoder', 'dot'),
            hidden_dim=config.get('hidden_dim', 128),
            out_dim=config.get('out_dim', 64),
            num_layers=config.get('num_layers', 2),
            dropout=config.get('dropout', 0.5),
        )

        # 训练
        trainer = LinkPredictionTrainer(
            model,
            lr=config.get('lr', 0.005),
            weight_decay=config.get('weight_decay', 1e-4),
        )

        trainer.train(
            x=x,
            mp_edge_index=fold_data['mp_edge_index'],
            train_edges=fold_data['train_edges'],
            train_labels=fold_data['train_labels'],
            val_edges=fold_data['val_edges'],
            val_labels=fold_data['val_labels'],
            epochs=config.get('epochs', 500),
            patience=config.get('patience', 50),
            mp_edge_weight=fold_data['mp_edge_weight'],
            train_edge_attr=fold_data['train_edge_attr'],
            val_edge_attr=fold_data['val_edge_attr'],
            verbose=verbose,
        )

        # 测试
        test_metrics = trainer.test(
            x=x,
            mp_edge_index=fold_data['mp_edge_index'],
            test_edges=fold_data['test_edges'],
            test_labels=fold_data['test_labels'],
            mp_edge_weight=fold_data['mp_edge_weight'],
            test_edge_attr=fold_data['test_edge_attr'],
        )

        if verbose:
            print(f"  Test:  {format_metrics(test_metrics)}")

        all_metrics.append(test_metrics)
        all_fold_results.append(test_metrics)

    # 汇总
    summary = {}
    metric_keys = list(all_metrics[0].keys())
    for key in metric_keys:
        values = [m[key] for m in all_metrics]
        summary[f'{key}_mean'] = np.mean(values)
        summary[f'{key}_std'] = np.std(values)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Summary ({k}-fold CV):")
        for key in metric_keys:
            mean_val = summary[f'{key}_mean']
            std_val = summary[f'{key}_std']
            print(f"  {key:12s}: {mean_val:.4f} ± {std_val:.4f}")
        print(f"{'='*60}")

    return summary, all_fold_results


def run_full_evaluation(data, k=5):
    """运行完整的 GNN 评估 + baseline 对照"""
    print("=" * 60)
    print("GraphPPI 完整评估")
    print(f"节点数: {data.num_nodes}, 有向边数: {data.num_edges}")
    print(f"无向边数: {data.num_edges // 2}")
    print("=" * 60)

    # === GNN 配置 ===
    configs = [
        {
            'name': 'GCN-identity-dot',
            'encoder': 'gcn', 'decoder': 'dot',
            'use_identity': True, 'hidden_dim': 128, 'out_dim': 64,
            'num_layers': 2, 'dropout': 0.5,
            'lr': 0.005, 'weight_decay': 1e-4,
            'epochs': 500, 'patience': 50,
            'seed': 42,
        },
        {
            'name': 'GCN-topology-dot',
            'encoder': 'gcn', 'decoder': 'dot',
            'use_identity': False, 'hidden_dim': 128, 'out_dim': 64,
            'num_layers': 2, 'dropout': 0.5,
            'lr': 0.005, 'weight_decay': 1e-4,
            'epochs': 500, 'patience': 50,
            'seed': 42,
        },
        {
            'name': 'GCN-topology-mlp',
            'encoder': 'gcn', 'decoder': 'mlp',
            'use_identity': False, 'hidden_dim': 128, 'out_dim': 64,
            'num_layers': 2, 'dropout': 0.5,
            'lr': 0.005, 'weight_decay': 1e-4,
            'epochs': 500, 'patience': 50,
            'seed': 42,
        },
        {
            'name': 'GCN-topology-edge_mlp',
            'encoder': 'gcn', 'decoder': 'edge_mlp',
            'use_identity': False, 'hidden_dim': 128, 'out_dim': 64,
            'num_layers': 2, 'dropout': 0.5,
            'lr': 0.005, 'weight_decay': 1e-4,
            'epochs': 500, 'patience': 50,
            'seed': 42,
        },
        {
            'name': 'GAT-topology-mlp',
            'encoder': 'gat', 'decoder': 'mlp',
            'use_identity': False, 'hidden_dim': 128, 'out_dim': 64,
            'num_layers': 2, 'dropout': 0.5,
            'lr': 0.005, 'weight_decay': 1e-4,
            'epochs': 500, 'patience': 50,
            'seed': 42,
        },
        {
            'name': 'SAGE-topology-mlp',
            'encoder': 'sage', 'decoder': 'mlp',
            'use_identity': False, 'hidden_dim': 128, 'out_dim': 64,
            'num_layers': 2, 'dropout': 0.5,
            'lr': 0.005, 'weight_decay': 1e-4,
            'epochs': 500, 'patience': 50,
            'seed': 42,
        },
    ]

    all_results = {}
    for cfg in configs:
        name = cfg.pop('name')
        print(f"\n{'#'*60}")
        print(f"# {name}")
        print(f"{'#'*60}")
        summary, fold_results = evaluate_gnn_kfold(data, cfg, k=k, verbose=False)
        cfg['name'] = name
        all_results[name] = {
            'summary': summary,
            'fold_results': fold_results,
            'config': cfg.copy(),
        }

        # 简洁输出
        print(f"  AUC: {summary['auc_mean']:.4f} ± {summary['auc_std']:.4f}")
        print(f"  AP:  {summary['ap_mean']:.4f} ± {summary['ap_std']:.4f}")
        hits_keys = [k for k in summary if 'hits@' in k and 'mean' in k]
        for hk in hits_keys:
            sk = hk.replace('_mean', '_std')
            print(f"  {hk.replace('_mean', '')}: {summary[hk]:.4f} ± {summary.get(sk, 0):.4f}")

    # === 汇总对比 ===
    print(f"\n{'='*60}")
    print("最终汇总")
    print(f"{'='*60}")
    header = f"{'Model':<25s} {'AUC':>10s} {'AP':>10s} {'Hits@10':>10s}"
    print(header)
    print("-" * 55)
    for name, res in all_results.items():
        s = res['summary']
        print(f"{name:<25s} {s['auc_mean']:>8.4f}±{s['auc_std']:.2f} "
              f"{s['ap_mean']:>8.4f}±{s['ap_std']:.2f} "
              f"{s.get('hits@10_mean', 0):>8.4f}±{s.get('hits@10_std', 0):.2f}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=5, help='Number of folds')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data', 'processed', 'graph.pt'
    )
    data = torch.load(data_path, weights_only=False)
    run_full_evaluation(data, k=args.k)
