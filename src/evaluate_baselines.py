# src/evaluate_baselines.py

import torch
import numpy as np
import sys
import os
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baselines.common_neighbors import common_neighbors_predict
from baselines.jaccard import jaccard_predict
from baselines.adamic_adar import adamic_adar_predict
from baselines.node2vec_rf import Node2VecRF
from utils import split_edges_kfold, get_edges_by_indices, sample_negative_edges
from metrics import compute_auc, compute_ap


def evaluate_baselines_kfold(data, k=5, seed=42, verbose=True):
    """
    k-fold CV 评估所有 baseline 方法
    
    参数:
        data: PyG Data对象
        k: fold 数
        seed: 随机种子
        verbose: 是否打印详细信息
    
    返回:
        all_results: dict {method: {'auc_mean', 'auc_std', 'ap_mean', 'ap_std'}}
    """
    num_undirected = data.edge_index.size(1) // 2
    num_nodes = data.num_nodes
    folds = split_edges_kfold(data.edge_index, num_undirected, k=k, seed=seed)

    methods = {
        'Common Neighbors': common_neighbors_predict,
        'Jaccard': jaccard_predict,
        'Adamic-Adar': adamic_adar_predict,
    }

    # 初始化结果收集
    all_fold_metrics = defaultdict(lambda: defaultdict(list))

    for fold_idx, fold_indices in enumerate(folds):
        if verbose:
            print(f"\n--- Fold {fold_idx + 1}/{k} ---")
            print(f"  Train: {len(fold_indices['train'])} edges, "
                  f"Test: {len(fold_indices['test'])} edges")

        # 获取训练边和测试边（有向）
        train_edges = get_edges_by_indices(data.edge_index, fold_indices['train'], num_undirected)
        test_pos = get_edges_by_indices(data.edge_index, fold_indices['test'], num_undirected)

        # 训练边集用于负采样排除
        train_edge_set = set()
        for i in range(train_edges.size(1)):
            u, v = int(train_edges[0, i].item()), int(train_edges[1, i].item())
            train_edge_set.add((u, v))

        # 负采样（与 GNN 评估对齐：只用无向测试边数）
        num_test_undirected = len(fold_indices['test'])
        test_neg = sample_negative_edges(test_pos, num_nodes, num_test_undirected, train_edge_set, random_state=seed + fold_idx)

        # 合并正负样本（都用无向边）
        test_pos_undir = test_pos[:, :num_test_undirected]
        all_test_edges = torch.cat([test_pos_undir, test_neg], dim=1)
        all_test_labels = torch.cat([torch.ones(num_test_undirected), torch.zeros(num_test_undirected)])

        # --- 评估启发式方法 ---
        for method_name, predict_fn in methods.items():
            auc, ap, _ = predict_fn(train_edges, num_nodes, all_test_edges, all_test_labels)
            all_fold_metrics[method_name]['auc'].append(auc)
            all_fold_metrics[method_name]['ap'].append(ap)

        # --- Node2Vec + RF ---
        # 训练正负样本
        train_undir = train_edges[:, :train_edges.size(1)//2]
        num_train_undir = train_undir.size(1)

        # 生成训练负样本
        exclude_for_train = train_edge_set.copy()
        train_neg = sample_negative_edges(train_undir, num_nodes, num_train_undir, exclude_for_train, random_state=seed + fold_idx)

        train_pairs = torch.cat([train_undir, train_neg], dim=1)
        train_pair_labels = torch.cat([torch.ones(num_train_undir), torch.zeros(num_train_undir)])

        model = Node2VecRF(dimensions=64, n_estimators=100, random_state=seed)
        model.fit(train_undir, num_nodes, train_pairs, train_pair_labels)
        auc, ap, _ = model.evaluate(all_test_edges, all_test_labels)
        all_fold_metrics['Node2Vec+RF']['auc'].append(auc)
        all_fold_metrics['Node2Vec+RF']['ap'].append(ap)

        if verbose:
            for method_name in ['Common Neighbors', 'Jaccard', 'Adamic-Adar', 'Node2Vec+RF']:
                m = all_fold_metrics[method_name]
                print(f"  {method_name:20s}: AUC={m['auc'][-1]:.4f}, AP={m['ap'][-1]:.4f}")

    # 汇总
    results = {}
    for method_name in ['Common Neighbors', 'Jaccard', 'Adamic-Adar', 'Node2Vec+RF']:
        aucs = all_fold_metrics[method_name]['auc']
        aps = all_fold_metrics[method_name]['ap']
        results[method_name] = {
            'auc_mean': np.mean(aucs),
            'auc_std': np.std(aucs),
            'ap_mean': np.mean(aps),
            'ap_std': np.std(aps),
        }

    if verbose:
        print(f"\n{'='*60}")
        print(f"Baseline {k}-Fold CV 汇总")
        print(f"{'='*60}")
        print(f"{'Method':<20s} {'AUC':>12s} {'AP':>12s}")
        print(f"{'-'*44}")
        for method_name in ['Common Neighbors', 'Jaccard', 'Adamic-Adar', 'Node2Vec+RF']:
            r = results[method_name]
            print(f"{method_name:<20s} {r['auc_mean']:>6.4f}±{r['auc_std']:.4f}  "
                  f"{r['ap_mean']:>6.4f}±{r['ap_std']:.4f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=5, help='Number of folds')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data', 'processed', 'graph.pt'
    )
    data = torch.load(data_path, weights_only=False)
    print(f"节点数: {data.num_nodes}, 有向边数: {data.num_edges}")
    print(f"无向边数: {data.num_edges // 2}")

    results = evaluate_baselines_kfold(data, k=args.k, seed=args.seed)

    # 保存
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'results'
    )
    os.makedirs(output_dir, exist_ok=True)
    import pandas as pd
    rows = []
    for method, r in results.items():
        rows.append({
            'Method': method,
            'AUC': f"{r['auc_mean']:.4f}±{r['auc_std']:.4f}",
            'AP': f"{r['ap_mean']:.4f}±{r['ap_std']:.4f}",
            'auc_raw': r['auc_mean'],
        })
    df = pd.DataFrame(rows).sort_values('auc_raw', ascending=False)
    csv_path = os.path.join(output_dir, 'baseline_kfold_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n结果已保存到: {csv_path}")