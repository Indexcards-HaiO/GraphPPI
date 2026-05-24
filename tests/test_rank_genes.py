"""Tests for node-ranking helpers."""

import torch

from graphppi.rank_genes import (
    aggregate_scores,
    build_observed_edge_set,
    load_name_list,
    make_bidirectional,
)


def test_load_name_list_deduplicates_comma_values():
    names = load_name_list(["TP53,BRCA1", "TP53", "ESR1"])
    assert names == ["TP53", "BRCA1", "ESR1"]


def test_build_observed_edge_set_is_undirected():
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    observed = build_observed_edge_set(edge_index)
    assert (0, 1) in observed
    assert (1, 0) in observed
    assert (1, 2) in observed
    assert (2, 1) in observed


def test_make_bidirectional_duplicates_weights():
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_weight = torch.tensor([0.4, 0.8])
    bi_edges, bi_weights = make_bidirectional(edge_index, edge_weight)
    assert bi_edges.tolist() == [[0, 1, 1, 2], [1, 2, 0, 1]]
    assert torch.allclose(bi_weights, torch.tensor([0.4, 0.8, 0.4, 0.8]))


def test_aggregate_scores_supports_mean_max_min():
    values = aggregate_scores([0.2, 0.8, 0.5], "max")
    assert values["score"] == 0.8
    assert values["mean_score"] == 0.5
    assert values["max_score"] == 0.8
    assert values["min_score"] == 0.2
    assert values["num_scored_seeds"] == 3
