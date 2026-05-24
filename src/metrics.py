#!/usr/bin/env python3
"""
评估指标
- AUROC (ROC-AUC)
- AP (Average Precision)
- Hits@K (K=1,3,5,10,20)
"""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def compute_auc(pred, labels):
    """计算 ROC-AUC"""
    pred_np = pred.cpu().numpy() if hasattr(pred, 'cpu') else np.array(pred)
    labels_np = labels.cpu().numpy() if hasattr(labels, 'cpu') else np.array(labels)
    return roc_auc_score(labels_np, pred_np)


def compute_ap(pred, labels):
    """计算 Average Precision"""
    pred_np = pred.cpu().numpy() if hasattr(pred, 'cpu') else np.array(pred)
    labels_np = labels.cpu().numpy() if hasattr(labels, 'cpu') else np.array(labels)
    return average_precision_score(labels_np, pred_np)


def compute_hits_at_k(pred, labels, ks=None):
    """
    计算 Hits@K 指标（链接预测标准定义）
    
    对每个正样本边，计算它在所有候选中的排名。
    Hits@K = 排名 ≤ K 的正样本比例 (recall-based)。
    
    例如 Hits@10 = 0.8 意味着 80% 的正样本排在 Top-10。
    
    参数:
        pred: (N,) 预测分数
        labels: (N,) 真实标签 (0/1)
        ks: list of int, 例如 [1,3,5,10,20]
    
    返回:
        dict: {k: hits_score}
    """
    if ks is None:
        ks = [1, 3, 5, 10, 20]

    pred_np = pred.cpu().numpy() if hasattr(pred, 'cpu') else np.array(pred)
    labels_np = labels.cpu().numpy() if hasattr(labels, 'cpu') else np.array(labels)

    total_positives = int(labels_np.sum())
    if total_positives == 0:
        return {k: 0.0 for k in ks}

    # 按预测分数降序排列
    sorted_indices = np.argsort(-pred_np)
    sorted_labels = labels_np[sorted_indices]

    # 正样本的排名（0-indexed）
    positive_ranks = np.where(sorted_labels == 1)[0]  # 所有正样本在排序后的位置

    results = {}
    for k in ks:
        # 统计排名 ≤ K-1 的正样本数 (0-indexed)
        hits = int((positive_ranks < k).sum())
        results[k] = hits / total_positives

    return results


def compute_all_metrics(pred, labels, ks=None):
    """计算所有指标"""
    if ks is None:
        ks = [1, 3, 5, 10, 20]

    auc = compute_auc(pred, labels)
    ap = compute_ap(pred, labels)
    hits = compute_hits_at_k(pred, labels, ks)

    metrics = {
        'auc': auc,
        'ap': ap,
    }
    for k, v in hits.items():
        metrics[f'hits@{k}'] = v

    return metrics


def format_metrics(metrics, prefix=''):
    """格式化打印指标"""
    lines = []
    for key in ['auc', 'ap'] + [f'hits@{k}' for k in [1, 3, 5, 10, 20]]:
        if key in metrics:
            lines.append(f"{prefix}{key}={metrics[key]:.4f}")
    return ' | '.join(lines)
