# 训练器

## LinkPredictionTrainer

GNN 链路预测训练器，支持：
- Binary cross-entropy loss
- Early stopping (patience-based)
- 训练/验证/测试三阶段评估

### `__init__(model, lr=0.005, weight_decay=1e-4)`

初始化训练器。

### `train(x, mp_edge_index, train_edges, train_labels, val_edges, val_labels, epochs=300, patience=30, ...)`

训练模型。关键参数：
- `mp_edge_index`: 仅包含训练正边的消息传递边（防泄漏）
- `train_edge_attr` / `val_edge_attr`: 边特征（EdgeMLP 解码器需要）

### `test(x, mp_edge_index, test_edges, test_labels, ...)`

在测试集上评估，返回 `{'auc', 'ap', 'hits@1', 'hits@3', 'hits@5', 'hits@10', 'hits@20'}`。
