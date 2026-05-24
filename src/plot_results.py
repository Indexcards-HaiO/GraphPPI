#!/usr/bin/env python3
"""生成结果对比柱状图"""
import matplotlib.pyplot as plt
import numpy as np
import os

# 最终结果（3-fold CV）
methods = [
    'GraphSAGE\n+ MLP',
    'GCN\n+ MLP',
    'GAT\n+ MLP',
    'Adamic-Adar',
    'GCN\n+ Dot',
    'Common\nNeighbors',
    'Jaccard',
    'GCN-Topology\n+ Dot',
    'Node2Vec\n+ RF',
]

auc_values = [0.9399, 0.9257, 0.9150, 0.9035, 0.8978, 0.8969, 0.8884, 0.8884, 0.8801]
ap_values  = [0.9425, 0.9246, 0.9098, 0.8798, 0.9048, 0.8684, 0.8598, 0.8917, 0.8565]
baseline_auc = 0.9035  # Adamic-Adar

colors = ['#2196F3' if 'GNN' in m or 'Graph' in m or 'GCN' in m or 'GAT' in m
          else '#FF9800' for m in ['GNN'] * 3 + ['Heuristic'] * 6]
colors = ['#2196F3'] * 3 + ['#FF9800'] * 6

fig, ax = plt.subplots(figsize=(12, 5))

x = np.arange(len(methods))
width = 0.35

bars1 = ax.bar(x - width/2, auc_values, width, label='AUC', color='#2196F3', alpha=0.85)
bars2 = ax.bar(x + width/2, ap_values, width, label='AP', color='#4CAF50', alpha=0.85)

# 标记 baseline
ax.axhline(y=baseline_auc, color='red', linestyle='--', linewidth=1.5, label=f'Adamic-Adar baseline (AUC={baseline_auc:.3f})')

# 标注数值
for bar, val in zip(bars1, auc_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{val:.4f}', ha='center', va='bottom', fontsize=7, fontweight='bold')
for bar, val in zip(bars2, ap_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{val:.4f}', ha='center', va='bottom', fontsize=7)

ax.set_ylabel('Score', fontsize=12)
ax.set_title('GraphPPI Link Prediction: GNN vs Baselines (3-fold CV)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=9)
ax.legend(loc='lower right', fontsize=10)
ax.set_ylim(0.82, 0.98)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()

output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'benchmark_comparison.png'), dpi=150, bbox_inches='tight')
print(f"图表已保存到: {output_dir}/benchmark_comparison.png")
plt.close()
