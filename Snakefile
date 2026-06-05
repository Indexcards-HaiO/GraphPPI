"""
Snakemake workflow for GraphPPI — full pipeline from raw data to results.
Usage:
    snakemake -j1              # run default pipeline
    snakemake -j1 all          # run everything
    snakemake -j1 evaluate_gnn baselines plot
"""

# All rules produce these final outputs
rule all:
    input:
        "data/processed/graph.pt",
        "results/gnn_3fold_sage_mlp.csv",
        "results/baselines_3fold.csv",
        "results/ablation.csv",
        "results/benchmark_plot.png",
        "results/gene_rankings.csv",

# --- Preprocess ---
rule preprocess:
    """Extract STRING 8-channel edge features → graph.pt"""
    input:
        "data/raw/edges.tsv",
    output:
        "data/processed/graph.pt",
    shell:
        "graphppi preprocess"

# --- GNN Evaluation ---
rule evaluate_gnn:
    """3-fold CV for GNN models"""
    input:
        "data/processed/graph.pt",
    output:
        "results/gnn_3fold_sage_mlp.csv",
    params:
        k=3,
        encoder="sage",
        decoder="mlp",
    shell:
        "graphppi evaluate --k {params.k} --encoder {params.encoder} "
        "--decoder {params.decoder} > {output}"

# --- Baselines ---
rule baselines:
    """3-fold CV for traditional baselines"""
    input:
        "data/processed/graph.pt",
    output:
        "results/baselines_3fold.csv",
    params:
        k=3,
    shell:
        "graphppi baselines --k {params.k} > {output}"

# --- Ablation ---
rule ablation:
    """Feature & architecture ablation"""
    input:
        "data/processed/graph.pt",
    output:
        "results/ablation.csv",
    shell:
        "graphppi ablation > {output}"

# --- Plot ---
rule plot:
    """Generate benchmark comparison plot"""
    input:
        "results/gnn_3fold_sage_mlp.csv",
        "results/baselines_3fold.csv",
    output:
        "results/benchmark_plot.png",
    shell:
        "python src/plot_results.py --output {output}"

# --- Gene Ranking ---
rule rank:
    """Rank candidate genes against seed genes"""
    input:
        "data/processed/graph.pt",
    output:
        "results/gene_rankings.csv",
    params:
        top_k=20,
    shell:
        "graphppi rank --output {output} --top-k {params.top_k}"

# --- Independent Test Set Evaluation ---
rule independent_test:
    """Evaluate on held-out independent test set"""
    input:
        "data/processed/graph.pt",
    output:
        "results/independent_test.csv",
    shell:
        "python src/evaluate_independent.py --output {output}"
