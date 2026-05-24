# GraphPPI: 基于图神经网络的蛋白质互作预测

> 使用 GNN 从蛋白质互作网络中预测潜在的蛋白质-蛋白质互作关系，并评估多种基线方法与 GNN 变体的性能。

---

## 📖 项目概述

### 问题定义

给定一个蛋白质互作网络（146 个蛋白质/基因，3412 条已知互作边），预测任意两个蛋白质之间是否存在互作关系。

把每个蛋白质看作**图的节点**，每条已知互作看作**图的边**，这就变成了一个经典的**链接预测（Link Prediction）**问题：图中还有哪些"缺失的边"（未被发现的互作）？

```
蛋白质 A ──── 蛋白质 B      ← 已知互作（训练边）
蛋白质 C ──── 蛋白质 D      ← 已知互作（训练边）
蛋白质 A ──?── 蛋白质 C      ← 未知，需要预测！
```

### 数据来源

- **STRING 数据库**：提供蛋白质互作的多维度证据
  - 8 个证据通道：基因邻接、基因融合、系统发生共现、同源性、共表达、实验验证、数据库标注、文本挖掘
  - `combined_score`：综合置信度（0~1）
- **146 个乳腺癌候选基因**的注释信息

---

## 🧠 技术栈

| 类别 | 技术 | 用途 |
|------|------|------|
| **图神经网络** | PyTorch Geometric | GCN / GAT / SAGE 编码器 |
| **深度学习** | PyTorch | 模型训练与推理 |
| **图分析** | NetworkX | 拓扑特征计算（聚类系数等） |
| **传统方法** | Node2Vec + Random Forest | 嵌入学习基线 |
| **启发式方法** | Common Neighbors / Jaccard / Adamic-Adar | 拓扑基线 |
| **评估** | scikit-learn | AUC、AP 指标计算 |
| **数据处理** | pandas, NumPy | 数据加载与预处理 |
| **交叉验证** | k-fold CV | 5-fold 稳定评估 |

---

## 🏗️ 技术实现

### 整体流程

```
┌──────────────┐    ┌────────────────┐    ┌─────────────────┐
│ edges.tsv    │───▶│ preprocess.py  │───▶│ graph.pt        │
│ (STRING原始) │    │ 8通道边特征提取 │    │ 含 edge_attr 8维 │
└──────────────┘    └────────────────┘    └────────┬────────┘
                                                   │
                    ┌──────────────────────────────┘
                    ▼
┌──────────────────────────────────────────────────────┐
│              k-fold 数据划分                          │
│  每条边只属于 train / val / test 之一                  │
│  ⚠️ 关键：消息传递图只用 train_pos 边                  │
└──────────────────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────────────┐
│              GNN 编码器 (Encoder)                     │
│  ┌──────┐  ┌──────┐  ┌──────┐                       │
│  │ GCN  │  │ GAT  │  │ SAGE │  ← 3 种编码器可选      │
│  └──────┘  └──────┘  └──────┘                       │
│  输入: 节点特征 (5维拓扑 / 146维one-hot)              │
│  输出: 每个节点的 64维嵌入向量 z                       │
└──────────────────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────────────┐
│              解码器 (Decoder)                         │
│  ┌──────┐  ┌──────┐  ┌──────────┐                   │
│  │ Dot  │  │ MLP  │  │ EdgeMLP  │  ← 3 种解码器      │
│  └──────┘  └──────┘  └──────────┘                   │
│  输入: [z_u, z_v] 或 [z_u, z_v, edge_attr]           │
│  输出: 链接存在概率 (0~1)                             │
└──────────────────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────────────┐
│              评估指标                                 │
│  AUC (ROC曲线下面积) | AP (平均精度)                   │
│  Hits@K (Top-K命中率): K = 1, 3, 5, 10, 20           │
└──────────────────────────────────────────────────────┘
```

### 节点特征（5 维拓扑特征，训练时动态计算）

| 特征 | 含义 | 示例 |
|------|------|------|
| `degree` | 该蛋白质在训练图中连接多少其他蛋白质 | AKT1 连接了 105 个蛋白 |
| `weighted_degree` | 连接的边的置信度之和 | 高置信度连接的蛋白总分更高 |
| `clustering_coefficient` | 邻居之间是否也互相连接 | 0.45（45% 的邻居对也互作） |
| `neighbor_weight_mean` | 邻居边的平均置信度 | 该蛋白互作的平均可信度 |
| `seed_neighbor_count` | 与 5 个关键基因的重叠邻居数 | 与 TP53/BRCA1 等的关联 |

### 边特征（8 维 STRING 证据通道，预处理提取）

| 通道 | 含义 | 数据中非零比例 |
|------|------|:---:|
| `coexpression` | 基因共表达证据 | 58.9% |
| `experimental` | 实验验证证据 | 60.1% |
| `database` | 数据库标注证据 | 48.3% |
| `textmining` | 文献文本挖掘 | 99.9% |
| `homology` | 同源性证据 | 15.3% |
| `phylo_cooccur` | 系统发生共现 | 6.3% |
| `gene_fusion` | 基因融合证据 | 0.1% |
| `neighborhood` | 基因邻接 | 0.0% |

### 数据泄露修复

本项目的关键是**严格消除数据泄露**：

| 泄露来源 | 修复方式 |
|----------|---------|
| 消息传递图包含验证/测试边 | GNN 编码时**只用 train_pos 边**传播消息 |
| 节点特征在全图上预计算 | 改为**每 fold 基于训练边动态计算**特征 |
| 边特征从全图查找 | 使用**哈希表 O(1) 查找**而非遍历全图 |

---

## 🔬 实验复现

### 环境准备

```bash
# 1. 创建 Python 虚拟环境（Python >= 3.10）
conda create -n graphppi python=3.11 -y
conda activate graphppi

# 2. 安装 PyTorch（根据你的 CUDA 版本选择，CPU 版如下）
pip install torch torch-geometric

# 3. 安装其余依赖
pip install -r requirements.txt
```

> 如果 `torch-geometric` 安装遇到问题，参考 [PyG 官方安装指南](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)。

### 1. 数据预处理

```bash
# 提取 8 通道 STRING 边特征，生成 graph.pt
python src/preprocess.py
```

### 2. GNN 评估（5-fold CV）

```bash
# 5-fold 交叉验证，300 epochs
python src/evaluate_gnn.py --k 5 --epochs 300 --patience 30
```

### 3. Baseline 评估（5-fold CV）

```bash
# 4 种基线方法的 5-fold CV
python src/evaluate_baselines.py --k 5
```

### 4. 消融实验

```bash
# 完整消融实验（特征 + 架构）
python src/ablation.py --k 3 --epochs 300
```

### 5. 单次快速验证

```bash
# 2-fold 快速检查
python src/evaluate_gnn.py --k 2 --epochs 100 --patience 20
```

---

## 🏆 实验结果排名

### 边预测（Link Prediction）— 3-fold CV

> 排名按 AUC 从高到低。AUC 越接近 1 表示预测越准确。
> 以下所有方法均同等条件下仅使用图拓扑结构。

| 排名 | 模型 | AUC | AP | 一句话解释 |
|:---:|------|:---:|:---:|------|
| 🥇 | **GraphSAGE + MLP** | **0.9399** | 0.9425 | SAGE 聚合邻居 + 神经网络打分，击败所有传统方法 |
| 🥈 | GCN + MLP | 0.9257 | 0.9246 | 图卷积编码 + 神经网络打分 |
| 🥉 | GAT + MLP | 0.9150 | 0.9098 | 带注意力机制的图卷积 + 神经网络打分 |
| 4 | Adamic-Adar | 0.9035 | 0.8798 | 给"冷门共同邻居"更高权重（纯拓扑统计） |
| 5 | GCN + Dot | 0.8978 | 0.9048 | 图卷积编码 + 向量内积打分（最简 GNN） |
| 6 | Common Neighbors | 0.8969 | 0.8684 | 数两个蛋白有多少共同邻居 |
| 7 | Jaccard | 0.8884 | 0.8598 | 共同邻居数 ÷ 总邻居数（归一化版 CN） |
| 8 | GCN-Topology-Dot | 0.8884 | 0.8917 | 同上但用 5 个手工拓扑特征代替节点 ID |
| 9 | Node2Vec + RF | 0.8801 | 0.8565 | 随机游走嵌入 + 随机森林分类 |

> ⚠️ **公平性说明**：GraphSAGE + EdgeMLP + STRING（AUC 0.9994, AP 0.9997）未纳入主榜单。该方法在解码时额外输入了 STRING 数据库的 8 维外部证据（共表达、实验验证、数据库标注等），这些特征来自外部知识库而非图拓扑本身，其他方法无法获取，因此不参与公平对比，仅作参考。

**术语速查：**
- **GCN** = 图卷积网络，每个节点聚合邻居信息来更新自己
- **GAT** = 图注意力网络，给不同邻居分配不同权重（注意力）
- **GraphSAGE** = 对邻居做均值采样聚合，适合大图
- **SAGE-MLP** = GraphSAGE 编码器 + 多层感知机（MLP）解码器，用神经网络打分替代简单内积
- **+ STRING** = 解码时额外输入 STRING 数据库的 8 维外部证据
- **Dot** = 最简单的解码方式：两个节点嵌入做内积，越大越可能互作

### 消融实验关键发现

```
┌─────────────────────────────────────────────────┐
│ 各组件增量贡献（AUC）                              │
│                                                 │
│ Baseline (Adamic-Adar)    ████████ 0.904        │
│ + GCN + Dot Decoder      ████████ 0.886 (-2%)   │
│ + MLP Decoder            ████████████ 0.925     │
│ + SAGE Encoder           █████████████ 0.940 🥇 │
│                                                 │
│ (参考) + STRING Edge      ████████████████ 0.999 │
└─────────────────────────────────────────────────┘
```

---

---

## 📂 项目结构

```
GraphPPI/
├── README.md                          # 本文件
├── requirements.txt                   # Python 依赖
│
├── data/
│   ├── raw/                           # 原始数据
│   │   ├── edges.tsv                  # STRING PPI 边 (13列含8证据通道)
│   │   ├── annotations.tsv            # 146个基因注释
│   │   └── degrees.tsv                # 节点度数
│   └── processed/
│       └── graph.pt                   # 预处理后的 PyG Data 对象
│
├── results/                           # 实验结果 CSV
│   ├── ablation_results.csv
│   └── baseline_kfold_results.csv
│
└── src/
    ├── preprocess.py                  # [新] 8通道边特征提取 + 重建 graph.pt
    ├── data_loader.py                 # [旧] 原始数据加载器（保留）
    ├── features.py                    # [旧] 原始特征计算（保留）
    ├── utils.py                       # [新] k-fold划分 + 特征动态计算 + 负采样
    ├── metrics.py                     # [新] AUC / AP / Hits@K
    ├── trainer.py                     # [新] 训练循环 + 早停（消息传递隔离）
    │
    ├── models/
    │   ├── encoder.py                 # [新] GCN / GAT / SAGE 编码器
    │   ├── decoder.py                 # [新] Dot / MLP / EdgeMLP 解码器
    │   ├── predictor.py               # [新] 统一 LinkPredictor
    │   └── gcn.py                     # [旧] 原始 GCN 模型（保留）
    │
    ├── evaluate_gnn.py                # [新] GNN k-fold CV 评估
    ├── evaluate_baselines.py          # [改] Baseline k-fold CV 评估
    ├── ablation.py                    # [新] 消融实验编排
    │
    ├── train_link_prediction.py       # [旧] 原始训练脚本（保留，含泄露bug）
    ├── train_link_prediction_tuned.py # [旧] 原始调优脚本（保留）
    │
    └── baselines/                     # Baseline 方法（不变）
        ├── common_neighbors.py
        ├── jaccard.py
        ├── adamic_adar.py
        └── node2vec_rf.py
```

> **[新]** = 本次重构新增；**[改]** = 重构修改；**[旧]** = 原始版本保留作参考

---

## 📝 未来的节点排序模块（乳腺癌候选基因排序）

> 以下为预留位置，待后续实现。

### 概述

基于训练好的 GNN 模型，对 146 个乳腺癌候选基因按其"与已知关键基因的潜在互作强度"进行排序，识别最可能的新互作靶点。

### 计划流程

1. 使用最佳模型（SAGE-MLP）在全图上训练，获得所有节点的嵌入表示
2. 对每个候选基因，计算其与已知乳腺癌关键基因（如 TP53, BRCA1, ERBB2, PIK3CA, ESR1）的嵌入相似度
3. 按预测分数降序排列，输出排名列表
4. 对 Top-N 候选进行文献验证

### 命令行（计划）

```bash
# 节点排序（待实现）
python src/rank_genes.py --model sage --decoder mlp --top_k 20
```

---

## 📄 许可证

本项目仅用于学术研究。
