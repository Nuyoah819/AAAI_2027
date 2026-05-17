from __future__ import annotations

import torch
import torch.nn.functional as F

from utils import random_node_pairs


def subspace_smoothness_loss(u: torch.Tensor, laplacian: torch.Tensor) -> torch.Tensor:
    """Trace(U^T L U), normalized for stable scale across graph sizes."""
    energy = torch.sum(u * laplacian.matmul(u))
    return energy / max(1, u.size(0))


def pivot_anchored_ranking_loss(
    embeddings: torch.Tensor,
    edge_index: torch.Tensor,
    weights_homo: torch.Tensor,
    weights_hetero: torch.Tensor,
    *,
    margin_homo: float = 0.5,
    margin_hetero: float = 0.5,
    eps: float = 1e-12,
) -> torch.Tensor:
    num_edges = edge_index.size(1)
    pivots = random_node_pairs(embeddings.size(0), num_edges, device=embeddings.device)

    edge_sim = F.cosine_similarity(
        embeddings[edge_index[0]],
        embeddings[edge_index[1]],
        dim=-1,
        eps=eps,
    )
    pivot_sim = F.cosine_similarity(
        embeddings[pivots[0]],
        embeddings[pivots[1]],
        dim=-1,
        eps=eps,
    )

    homo_loss = F.relu(pivot_sim - edge_sim + margin_homo)
    hetero_loss = F.relu(edge_sim - pivot_sim + margin_hetero)

    homo_loss = (weights_homo * homo_loss).sum() / weights_homo.sum().clamp_min(eps)
    hetero_loss = (weights_hetero * hetero_loss).sum() / weights_hetero.sum().clamp_min(eps)
    return homo_loss + hetero_loss
