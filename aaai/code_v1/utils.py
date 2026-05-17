from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch


@dataclass
class SyntheticGraph:
    x: torch.Tensor
    edge_index: torch.Tensor
    labels: torch.Tensor


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def edge_index_to_dense_adj(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weight: torch.Tensor | None = None,
    *,
    make_symmetric: bool = True,
    add_self_loops: bool = False,
) -> torch.Tensor:
    device = edge_index.device
    dtype = edge_weight.dtype if edge_weight is not None else torch.float32
    adj = torch.zeros((num_nodes, num_nodes), device=device, dtype=dtype)
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=device, dtype=dtype)
    row, col = edge_index
    adj.index_put_((row, col), edge_weight, accumulate=True)
    if make_symmetric:
        adj.index_put_((col, row), edge_weight, accumulate=True)
        adj = 0.5 * (adj + adj.t())
    if add_self_loops:
        adj = adj + torch.eye(num_nodes, device=device, dtype=dtype)
    return adj


def normalize_adjacency(adj: torch.Tensor, mode: str = "sym", eps: float = 1e-12) -> torch.Tensor:
    if mode == "sym":
        degree = adj.sum(dim=1).clamp_min(eps)
        inv_sqrt = degree.rsqrt()
        return inv_sqrt[:, None] * adj * inv_sqrt[None, :]
    if mode == "row":
        degree = adj.sum(dim=1).clamp_min(eps)
        return adj / degree[:, None]
    raise ValueError(f"Unknown normalization mode: {mode}")


def normalized_laplacian_from_adj(adj: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    adj_norm = normalize_adjacency(adj, mode="sym", eps=eps)
    eye = torch.eye(adj.size(0), device=adj.device, dtype=adj.dtype)
    return eye - adj_norm


@torch.no_grad()
def pagerank_scores(
    edge_index: torch.Tensor,
    num_nodes: int,
    *,
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> torch.Tensor:
    adj = edge_index_to_dense_adj(
        edge_index,
        num_nodes,
        make_symmetric=True,
        add_self_loops=False,
    )
    transition = normalize_adjacency(adj, mode="row")
    pr = torch.full((num_nodes,), 1.0 / num_nodes, device=edge_index.device)
    teleport = torch.full_like(pr, 1.0 / num_nodes)
    for _ in range(max_iter):
        prev = pr
        pr = (1.0 - damping) * teleport + damping * transition.t().matmul(prev)
        if torch.norm(pr - prev, p=1) < tol:
            break
    return pr.clamp_min(0)


@torch.no_grad()
def sample_pagerank_anchors(
    scores: torch.Tensor,
    num_anchors: int,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    num_nodes = scores.numel()
    if num_anchors > num_nodes:
        raise ValueError("num_anchors cannot exceed number of nodes")
    probs = scores / scores.sum().clamp_min(1e-12)
    return torch.multinomial(probs, num_anchors, replacement=False, generator=generator)


def random_node_pairs(
    num_nodes: int,
    num_pairs: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    src = torch.randint(num_nodes, (num_pairs,), device=device)
    dst = torch.randint(num_nodes, (num_pairs,), device=device)
    same = src == dst
    while same.any():
        dst[same] = torch.randint(num_nodes, (same.sum().item(),), device=device)
        same = src == dst
    return torch.stack([src, dst], dim=0)


def make_synthetic_graph(
    num_nodes: int = 90,
    num_features: int = 16,
    num_clusters: int = 3,
    p_in: float = 0.18,
    p_out: float = 0.035,
    feature_noise: float = 0.25,
    seed: int = 7,
) -> SyntheticGraph:
    set_seed(seed)
    labels = torch.arange(num_nodes) % num_clusters
    centers = torch.randn(num_clusters, num_features)
    x = centers[labels] + feature_noise * torch.randn(num_nodes, num_features)
    rows, cols = [], []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            p = p_in if labels[i] == labels[j] else p_out
            if random.random() < p:
                rows.extend([i, j])
                cols.extend([j, i])
    if not rows:
        raise RuntimeError("Synthetic graph sampled no edges; increase p_in or p_out")
    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    scale = math.sqrt(float(num_features))
    x = x / scale
    return SyntheticGraph(x=x, edge_index=edge_index, labels=labels)
