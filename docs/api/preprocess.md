# 数据预处理

预处理模块将 STRING TSV 格式的 PPI 数据转换为 PyTorch Geometric 图数据。

## 核心函数

### `load_edges_with_attrs(path)`

读取 `data/raw/edges.tsv`，提取 8 个 STRING 证据通道作为边特征。

返回:
- `edge_index`: (2, 2E) 有向边索引
- `edge_attr`: (2E, 8) 8 通道边特征
- `node_names`: 基因名列表
- `gene_to_idx`: 基因名→索引映射

### `build_pyg_graph(edge_index, edge_attr, node_names, gene_to_idx)`

构建 PyG Data 对象。

返回: `torch_geometric.data.Data` 对象
