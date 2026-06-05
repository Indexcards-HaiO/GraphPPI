# 评估指标

## `compute_auc(scores, labels)`

计算 AUROC (Area Under ROC Curve)。

## `compute_ap(scores, labels)`

计算 Average Precision。

## `compute_hits_at_k(scores, labels, k)`

计算标准 Recall@K：正样本中排名前 K 的比例。

## `compute_all_metrics(scores, labels)`

一次性计算所有指标，返回 dict:
```python
{
    'auc': float,
    'ap': float,
    'hits@1': float,
    'hits@3': float,
    'hits@5': float,
    'hits@10': float,
    'hits@20': float,
}
```

## `format_metrics(metrics_dict)`

格式化指标为可打印字符串。
