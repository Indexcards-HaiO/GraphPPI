# 数据集说明

## 数据来源

本项目使用 [STRING 数据库](https://string-db.org/) (v12.0) 的蛋白质互作数据。

- **物种**: Homo sapiens
- **基因集**: 146 个乳腺癌相关基因
- **边数**: 3,412 条无向 PPI 边
- **边特征**: 8 个 STRING 证据通道

## STRING 证据通道

| 通道 | 说明 |
|------|------|
| neighborhood | 基因邻接 (conserved genomic neighborhood) |
| fusion | 基因融合 |
| cooccurence | 共出现 (phylogenetic co-occurrence) |
| coexpression | 共表达 |
| experimental | 实验验证 |
| database | 数据库注释 |
| textmining | 文本挖掘 |
| combined_score | STRING 综合评分 |

## 数据格式

### `data/raw/edges.tsv`

TSV 格式，每行一条 PPI 边：

```
protein1  protein2  neighborhood  fusion  cooccurence  coexpression  experimental  database  textmining  combined_score
TP53      BRCA1     0            0       0            0.123         0.456         0.789     0.321       0.992
...
```

### `data/processed/graph.pt`

PyTorch Geometric `Data` 对象：

```python
data.edge_index    # (2, 6824) 有向边
data.edge_attr     # (6824, 8) STRING 8 通道特征
data.num_nodes     # 146
data.node_names    # 基因名列表
```

## 扩展数据

可从 STRING API 直接下载更多 PPI 数据：

```bash
graphppi download-string --species 9606 --min-score 700 --output data/raw/edges.tsv
```
