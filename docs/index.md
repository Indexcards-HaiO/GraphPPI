# GraphPPI

**Graph Neural Network for Protein-Protein Interaction Link Prediction**

GraphPPI 使用图神经网络（GCN / GAT / GraphSAGE）预测蛋白质-蛋白质互作（PPI）网络中缺失的边。项目基于 STRING 数据库的 146 个乳腺癌相关基因构建 PPI 网络，支持 k-fold 交叉验证、多基线对比、消融实验和候选基因排序。

## 核心特性

- 🧬 **3 种 GNN 编码器**: GCN, GAT (multi-head), GraphSAGE
- 🔗 **3 种解码器**: Dot Product, MLP, Edge Feature MLP（支持 STRING 8 通道边特征）
- 📊 **严格 k-fold CV**: 消息传递仅使用训练边，无数据泄露
- 🎯 **多指标**: AUROC, Average Precision, Hits@K (1/3/5/10/20)
- 🏆 **4 种 Baseline**: Common Neighbors, Jaccard, Adamic-Adar, Node2Vec+RF
- 🔬 **消融实验**: 特征消融 + 架构消融
- 🏷️ **基因排序**: 基于种子基因预测新互作候选

## 安装

```bash
git clone https://github.com/Totheinfinity66y/GraphPPI_test.git
cd GraphPPI_test
pip install -e .
```

## 快速开始

```bash
# 预处理 STRING 数据
graphppi preprocess

# 评估 GNN 模型 (3-fold CV)
graphppi evaluate --k 3

# 评估 Baseline
graphppi baselines --k 3

# 消融实验
graphppi ablation

# 基因排序
graphppi rank --output results/gene_rankings.csv
```

## 项目结构

```
GraphPPI_test/
├── src/
│   ├── graphppi/              # 核心包
│   │   ├── models/            # GNN 编码器 & 解码器
│   │   ├── baselines/         # 传统链路预测方法
│   │   ├── utils.py           # 数据划分 & 特征工程
│   │   ├── trainer.py         # 训练循环
│   │   ├── metrics.py         # 评估指标
│   │   └── rank_genes.py      # 基因排序
│   ├── preprocess.py          # STRING 数据预处理
│   ├── cli.py                 # 命令行入口
│   └── plot_results.py        # 可视化
├── tests/                     # 单元测试
├── data/
│   ├── raw/edges.tsv          # STRING 原始数据
│   └── processed/graph.pt     # PyG 图数据
├── demo.ipynb                 # 演示 notebook
├── Dockerfile                 # 容器化
├── Snakefile                  # Snakemake 工作流
└── .github/workflows/ci.yml  # CI/CD
```
