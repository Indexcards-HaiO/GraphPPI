# 基因排序

基于训练好的 GNN 模型对候选基因按与种子基因的预测互作强度排序。

## 核心函数

### `train_ranking_model(data, config)`

训练排序模型，返回包含模型、配置、最佳 AUC 的 bundle。

### `rank_candidates(bundle, seed_genes, candidate_genes=None, top_k=20)`

对候选基因排序，返回排序后的 DataFrame 和缺失的种子基因列表。

### `aggregate_scores(scores, method='mean')`

聚合多个种子基因对同一候选基因的预测分数。

支持: `mean`, `max`, `min`

### `load_name_list(items)`

从逗号分隔的字符串列表加载基因名。

## 默认种子基因

```python
DEFAULT_SEEDS = ['TP53', 'BRCA1', 'ERBB2', 'PIK3CA', 'ESR1']
```
