# 快速开始

## 环境要求

- Python >= 3.10
- PyTorch >= 2.0.0
- PyTorch Geometric >= 2.5.0

## 安装

```bash
pip install -e .
```

## 数据准备

```bash
graphppi preprocess
```

该命令从 `data/raw/edges.tsv` 读取 STRING PPI 数据，提取 8 通道边特征，生成 `data/processed/graph.pt`。

## 模型评估

### GNN 方法

```bash
# 5-fold 交叉验证
graphppi evaluate --k 5 --encoder sage --decoder mlp
```

支持参数：
- `--encoder`: gcn | gat | sage
- `--decoder`: dot | mlp | edge_mlp
- `--k`: fold 数 (默认 5)

### Baseline 方法

```bash
graphppi baselines --k 5
```

### 消融实验

```bash
graphppi ablation
```

## 基因排序

```bash
graphppi rank --seeds TP53 BRCA1 ESR1 --top-k 20
```

## Python API

```python
import torch
from graphppi.utils import split_edges_kfold, prepare_fold_data
from graphppi.models.predictor import LinkPredictor
from graphppi.trainer import LinkPredictionTrainer

# 加载数据
data = torch.load("data/processed/graph.pt", weights_only=False)

# 数据划分
folds = split_edges_kfold(data.edge_index, data.edge_index.size(1)//2, k=5)
fold_data = prepare_fold_data(data, folds[0])

# 训练
model = LinkPredictor(in_dim=5, encoder_type="sage", decoder_type="mlp")
trainer = LinkPredictionTrainer(model, lr=0.005)
trainer.train(
    x=fold_data["x"],
    mp_edge_index=fold_data["mp_edge_index"],
    train_edges=fold_data["train_edges"],
    train_labels=fold_data["train_labels"],
    val_edges=fold_data["val_edges"],
    val_labels=fold_data["val_labels"],
    epochs=200, patience=20
)

# 测试
metrics = trainer.test(
    x=fold_data["x"],
    mp_edge_index=fold_data["mp_edge_index"],
    test_edges=fold_data["test_edges"],
    test_labels=fold_data["test_labels"]
)
print(f"AUC: {metrics['auc']:.4f}, AP: {metrics['ap']:.4f}")
```
