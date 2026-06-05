# CLI 命令行接口

GraphPPI 提供 `graphppi` 命令行工具，支持以下子命令：

| 命令 | 说明 |
|------|------|
| `graphppi preprocess` | 提取 STRING 8 通道边特征 |
| `graphppi evaluate` | GNN k-fold 交叉验证评估 |
| `graphppi baselines` | Baseline 方法 k-fold CV |
| `graphppi ablation` | 消融实验 |
| `graphppi rank` | 候选基因排序 |
| `graphppi download-string` | 从 STRING API 下载 PPI 数据 |

## 通用参数

所有评估命令支持以下参数：

- `--k`: fold 数（默认 5）
- `--seed`: 随机种子（默认 42）

## rank 命令参数

- `--seeds`: 种子基因列表
- `--seeds-file`: 种子基因文件
- `--candidates`: 候选基因列表
- `--top-k`: 输出前 K 行（默认 20）
- `--encoder`: 编码器类型 (gcn/gat/sage)
- `--decoder`: 解码器类型 (dot/mlp/edge_mlp)
- `--epochs`: 训练轮数（默认 300）
- `--patience`: 早停耐心值（默认 30）
