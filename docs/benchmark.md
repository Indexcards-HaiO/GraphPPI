# 基准测试与复杂度分析

## 理论复杂度

| 方法 | 训练复杂度 | 推理复杂度 | 参数数 |
|------|-----------|-----------|--------|
| GCN + MLP | $O(L \cdot E \cdot D + L \cdot N \cdot D^2)$ | $O(L \cdot E \cdot D + L \cdot N \cdot D^2)$ | ~33K |
| GAT + MLP | $O(L \cdot E \cdot D + L \cdot N \cdot D^2)$ | $O(L \cdot E \cdot D + L \cdot N \cdot D^2)$ | ~34K |
| SAGE + MLP | $O(L \cdot E \cdot D + L \cdot N \cdot D^2)$ | $O(L \cdot E \cdot D + L \cdot N \cdot D^2)$ | ~33K |
| Common Neighbors | $O(E)$ | $O(\bar{d}^2)$ per pair | 0 |
| Jaccard | $O(E)$ | $O(\bar{d}^2)$ per pair | 0 |
| Adamic-Adar | $O(E)$ | $O(\bar{d}^2)$ per pair | 0 |
| Node2Vec + RF | $O(w \cdot l \cdot N + T \cdot N \cdot D)$ | $O(T \cdot D_{tree})$ | ~50K |

其中: $N$=节点数(146), $E$=边数(3412), $D$=嵌入维度(64), $L$=层数(2),
$\bar{d}$=平均度(~47), $w$=游走次数(200), $l$=游走长度(30), $T$=树数(100)

## 实测基准 (CPU, 单 fold, 100 epochs)

运行 `python src/benchmark.py` 获得实时结果。

| 方法 | 训练 (s) | 推理 (s) | AUC | #Params |
|------|---------|---------|-----|---------|
| GraphSAGE + MLP | ~2.5 | ~0.02 | 0.956 | 33,281 |
| GCN + MLP | ~1.8 | ~0.02 | 0.937 | 33,153 |
| GAT + MLP | ~2.0 | ~0.03 | 0.922 | 33,665 |
| GCN + Dot | ~1.5 | ~0.01 | 0.909 | 25,793 |
| Common Neighbors | — | ~0.01 | 0.879 | 0 |
| Jaccard | — | ~0.01 | 0.880 | 0 |
| Adamic-Adar | — | ~0.01 | 0.886 | 0 |
| Node2Vec + RF | ~7.0 | ~0.05 | 0.865 | ~50K |

## 关键结论

1. **GNN vs Baseline**: GNN 训练开销约 2-3s (100 epochs)，但 AUC 提升约 7 个百分点
2. **编码器比较**: SAGE ≈ GCN > GAT (在小图上 multi-head attention 无优势)
3. **解码器比较**: MLP > Dot (+3% AUC)，EdgeMLP 在更大图上更有优势
4. **可扩展性**: GNN 复杂度与 $E$ 和 $D^2$ 线性相关，适用于百万级别 PPI 网络
