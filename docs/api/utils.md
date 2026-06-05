# 核心工具函数

## `split_edges_kfold(edge_index, num_edges_undirected, k=5, seed=42)`

对无向边进行 k-fold 划分，每个 fold 返回 `{'train', 'val', 'test'}` 边索引。使用 `KFold` 确保每个样本仅在一个测试 fold 中出现。

## `prepare_fold_data(data, fold_indices)`

为单个 fold 准备训练/验证/测试数据。包括：
- 动态计算 5 个拓扑特征（仅基于训练边）
- 构建消息传递边索引
- 负采样

返回 dict 含 `x, mp_edge_index, train_edges, train_labels, val_edges, val_labels, test_edges, test_labels` 等。

## `get_edges_by_indices(edge_index, undirected_indices, num_undirected)`

从无向边索引获取对应的有向边。

## `sample_negative_edges(pos_edges, num_nodes, num_samples, exclude_set=None, random_state=None)`

从未出现的边对中随机负采样。

## `compute_features_from_edges(edge_index, edge_weight, num_nodes, node_names, seed_genes=None)`

基于给定边集动态计算 5 个拓扑特征：度、加权度、聚类系数、邻居权重均值、种子基因邻居数。这是消除数据泄露的关键设计。
