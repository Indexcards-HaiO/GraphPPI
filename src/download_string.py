#!/usr/bin/env python3
"""
从 STRING API 下载 PPI 数据

Usage:
    python src/download_string.py --species 9606 --min-score 700 --genes TP53 BRCA1 ESR1
    python src/download_string.py --species 9606 --min-score 700 --gene-file gene_list.txt
"""

import argparse
import requests
import sys
import time
import os

STRING_API = "https://string-db.org/api"


def fetch_ppi_network(genes, species=9606, min_score=700, limit=500):
    """
    从 STRING API v12 获取 PPI 网络。

    参数:
        genes: list[str] 基因名列表
        species: int NCBI taxonomy ID (9606 = human)
        min_score: int 最低 combined_score (0-1000)
        limit: int 最多返回边数

    返回:
        list[dict] 每条边一个字典，含所有 channel scores
    """
    params = {
        "identifiers": "%0d".join(genes),
        "species": species,
        "required_score": min_score,
        "limit": limit,
        "caller_identity": "GraphPPI",
    }

    url = f"{STRING_API}/tsv/network"
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()

    lines = resp.text.strip().split("\n")
    if len(lines) < 2:
        return []

    header = lines[0].split("\t")
    rows = []
    for line in lines[1:]:
        values = line.split("\t")
        rows.append(dict(zip(header, values)))

    return rows


def save_as_edges_tsv(rows, output_path):
    """保存为 GraphPPI 兼容的 edges.tsv 格式"""
    columns = [
        "preferredName_A", "preferredName_B",
        "neighborhood", "fusion", "cooccurence",
        "coexpression", "experimental", "database",
        "textmining", "combined_score"
    ]

    with open(output_path, "w") as f:
        f.write("\t".join(columns) + "\n")
        for row in rows:
            f.write("\t".join(row.get(c, "") for c in columns) + "\n")

    return len(rows)


def load_gene_list(path):
    """从文件加载基因列表"""
    genes = []
    with open(path) as f:
        for line in f:
            gene = line.strip()
            if gene and not gene.startswith("#"):
                genes.append(gene)
    return genes


def main():
    parser = argparse.ArgumentParser(description="从 STRING API 下载 PPI 数据")
    parser.add_argument("--species", type=int, default=9606, help="NCBI taxonomy ID")
    parser.add_argument("--min-score", type=int, default=700, help="最低 combined_score")
    parser.add_argument("--genes", nargs="*", default=None, help="基因列表")
    parser.add_argument("--gene-file", default=None, help="基因列表文件（每行一个）")
    parser.add_argument("--output", default="data/raw/edges_from_string.tsv")
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()

    # 获取基因列表
    if args.genes:
        genes = args.genes
    elif args.gene_file:
        genes = load_gene_list(args.gene_file)
    else:
        print("错误: 需要 --genes 或 --gene-file")
        sys.exit(1)

    print(f"从 STRING API 下载 PPI 数据")
    print(f"  物种: {args.species}")
    print(f"  最低分数: {args.min_score}")
    print(f"  基因数: {len(genes)}")
    print(f"  基因: {', '.join(genes[:10])}{'...' if len(genes)>10 else ''}")

    rows = fetch_ppi_network(genes, args.species, args.min_score, args.limit)

    if not rows:
        print("  未找到互作数据。")
        sys.exit(0)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    n = save_as_edges_tsv(rows, args.output)
    print(f"  保存 {n} 条边到 {args.output}")

    # 统计
    genes_found = set()
    for row in rows:
        genes_found.add(row["preferredName_A"])
        genes_found.add(row["preferredName_B"])
    print(f"  覆盖基因: {len(genes_found)}/{len(genes)}")


if __name__ == "__main__":
    main()
