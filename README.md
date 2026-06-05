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

## 🚀 快速开始

```bash
git clone https://github.com/Totheinfinity66y/GraphPPI_test.git
cd GraphPPI_test

# 安装（推荐使用 conda 环境）
conda create -n graphppi python=3.11 -y && conda activate graphppi
pip install -e .

# 数据预处理
graphppi preprocess

# 5-fold GNN 评估
graphppi evaluate --k 5

# Baseline 评估
graphppi baselines --k 5

# 候选基因排序
graphppi rank --top-k 20

# 查看所有命令
graphppi --help
```

> 📓 完整演示见 [`demo.ipynb`](demo.ipynb)：边预测 + 节点排序 + 可视化，一键运行。

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
# 安装 graphppi（开发模式）
pip install -e .

# 或含文档/开发依赖
pip install -e ".[dev,docs]"
```

> 依赖项会自动安装（PyTorch, PyG, pandas, scikit-learn 等）。
> 如果 `torch-geometric` 安装遇到问题，参考 [PyG 官方安装指南](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)。

### 1. 数据预处理

```bash
# 提取 8 通道 STRING 边特征，生成 graph.pt
graphppi preprocess
```

### 2. GNN 评估（k-fold CV）

```bash
# 5-fold 交叉验证
graphppi evaluate --k 5

# 指定编码器和解码器
graphppi evaluate --k 5 --encoder sage --decoder mlp
```

### 3. Baseline 评估（k-fold CV）

```bash
graphppi baselines --k 5
```

### 4. 消融实验

```bash
graphppi ablation --k 3
```

### 5. 基因排序

```bash
# 默认乳腺癌关键基因，输出 Top-20
graphppi rank --top-k 20

# 自定义种子基因
graphppi rank --seeds TP53 BRCA1 ERBB2 --top-k 50
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

### 节点排序（Novel Interaction Prediction）

`graphppi rank` 对候选基因按与 5 个乳腺癌关键基因（TP53, BRCA1, ERBB2, PIK3CA, ESR1）的预测互作强度排序，已存在互作被排除：

| 排名 | gene | score | best_seed | 说明 |
|:---:|------|:-----:|:---------:|------|
| 1 | **MAPK3** | 0.985 | ERBB2 | MAPK/ERK 通路关键激酶 |
| 2 | **MAPK1** | 0.975 | ERBB2 | MAPK 通路核心成员 |
| 3 | **KIT** | 0.966 | ERBB2 | 受体酪氨酸激酶 |
| 4 | AKT2 | 0.869 | ERBB2 | PI3K/AKT 通路 |
| 5 | PIK3CB | 0.844 | ESR1 | PI3K 催化亚基 |

> `score` = 聚合预测概率（mean）；`best_seed` = 最强预测的种子基因

---

## 📂 项目结构

```
GraphPPI/
├── README.md                          # 本文件
├── setup.py                           # pip install -e .
├── pyproject.toml                     # 构建系统
├── requirements.txt / environment.yml # 依赖
├── mkdocs.yml                         # 文档配置
├── Dockerfile / graphppi.def          # 容器化
├── Snakefile                          # 工作流编排
├── demo.ipynb                         # 演示 notebook
├── .github/workflows/ci.yml           # CI/CD
├── docs/                              # MkDocs 文档源
├── data/  raw/  processed/            # 数据
├── tests/                             # 24 个单元测试
├── results/                           # 输出结果
└── src/
    ├── cli.py                         # graphppi 命令行
    ├── utils / metrics / trainer      # 核心模块
    ├── models/ (encoder, decoder, predictor)
    ├── baselines/ (CN, Jaccard, AA, N2V+RF)
    ├── evaluate_gnn / evaluate_baselines / ablation
    ├── rank_genes / benchmark
    ├── evaluate_independent / download_string
    └── plot_results.py
```

---

## 📝 CLI 命令参考

```bash
graphppi --help                    # 查看所有命令

# 核心命令
graphppi preprocess                # 提取 STRING 8 通道边特征
graphppi evaluate --k 5            # GNN k-fold CV（支持 --encoder/--decoder）
graphppi baselines --k 5           # Baseline k-fold CV
graphppi ablation --k 3            # 消融实验
graphppi rank --top-k 20           # 候选基因排序

# 进阶命令
graphppi download-string \         # 从 STRING API 下载 PPI 数据
    --genes TP53 BRCA1 ESR1

# 脚本工具
python src/benchmark.py            # 复杂度分析 + 实测基准
python src/evaluate_independent.py # 独立测试集评估
python src/plot_results.py         # 基准对比可视化
```

---

## 进阶功能

### 🐳 Docker / Apptainer

```bash
docker build -t graphppi .
docker run --rm graphppi evaluate --k 3

# Apptainer
apptainer build graphppi.sif graphppi.def
apptainer run graphppi.sif rank --top-k 20
```

### ⚙️ Snakemake 工作流

```bash
snakemake -j1 all             # 运行完整流程
snakemake -j1 evaluate_gnn    # 仅 GNN 评估
```

### 📊 基准测试

```bash
python src/benchmark.py       # 复杂度分析 + 所有方法实测对比
```

### 🔬 独立测试集

```bash
python src/evaluate_independent.py --holdout 0.2 --epochs 200
```

### 🌐 公开数据对接

```bash
graphppi download-string --genes TP53 BRCA1 ESR1 PIK3CA
```

### 📖 文档

```bash
pip install -e ".[docs]"
mkdocs serve                    # http://localhost:8000
```

### 🧪 CI/CD

推送到 GitHub 后自动运行：pytest + CLI smoke test（`.github/workflows/ci.yml`）

---

## 📄 许可证

本项目仅用于学术研究。
