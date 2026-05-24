"""测试 AUC / AP / Hits@K 指标"""
import torch
import numpy as np
from graphppi.metrics import compute_auc, compute_ap, compute_hits_at_k, compute_all_metrics


def test_auc_perfect():
    """完美预测 AUC 应为 1.0"""
    pred = torch.tensor([0.9, 0.8, 0.3, 0.2])
    labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
    auc = compute_auc(pred, labels)
    assert auc == 1.0


def test_auc_random():
    """随机预测 AUC 约 0.5"""
    np.random.seed(42)
    pred = torch.randn(100)
    labels = torch.randint(0, 2, (100,)).float()
    auc = compute_auc(pred, labels)
    assert 0.3 < auc < 0.7


def test_ap_perfect():
    """完美预测 AP 应为 1.0"""
    pred = torch.tensor([0.9, 0.8, 0.3, 0.2])
    labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
    ap = compute_ap(pred, labels)
    assert ap == 1.0


def test_hits_at_k():
    """Hits@K 基本测试"""
    # 正样本分数高 → 排前面
    pred = torch.tensor([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    labels = torch.tensor([1.0, 1.0, 0.0, 1.0, 0.0, 0.0])
    hits = compute_hits_at_k(pred, labels, ks=[1, 2, 3])
    # 3个正样本，排序后: 0.9(pos), 0.8(pos), 0.7(neg), 0.3(pos), 0.2(neg), 0.1(neg)
    # hits@1: 第1个是正 → 1/3
    # hits@2: 前2个都是正 → 2/3
    # hits@3: 前3个有2正 → 2/3
    assert hits[1] == 1 / 3
    assert hits[2] == 2 / 3
    assert hits[3] == 2 / 3


def test_all_metrics():
    """compute_all_metrics 返回所有指标"""
    pred = torch.tensor([0.9, 0.3])
    labels = torch.tensor([1.0, 0.0])
    metrics = compute_all_metrics(pred, labels, ks=[1])
    assert 'auc' in metrics
    assert 'ap' in metrics
    assert 'hits@1' in metrics
