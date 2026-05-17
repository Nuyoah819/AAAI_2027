from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses import pivot_anchored_ranking_loss, subspace_smoothness_loss
from utils import edge_index_to_dense_adj, normalize_adjacency, normalized_laplacian_from_adj


@dataclass
class UnifiedOutput:
    loss: torch.Tensor
    subspace_loss: torch.Tensor
    ranking_loss: torch.Tensor
    embeddings: torch.Tensor
    basis: torch.Tensor
    adj_homo: torch.Tensor
    adj_hetero: torch.Tensor
    weights_homo: torch.Tensor
    weights_hetero: torch.Tensor


class EdgeDiscriminator(nn.Module):
    """GREET-style edge discriminator with differentiable Gumbel masks."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        temperature: float = 1.0,
        hard: bool = False,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.hard = hard
        self.node_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.edge_mlp = nn.Linear(2 * hidden_dim, 1)

    def _symmetric_logits(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        logits_forward = self.edge_mlp(torch.cat([h[src], h[dst]], dim=-1)).squeeze(-1)
        logits_backward = self.edge_mlp(torch.cat([h[dst], h[src]], dim=-1)).squeeze(-1)
        return 0.5 * (logits_forward + logits_backward)

    def forward(self, node_inputs: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.node_mlp(node_inputs)
        logits = self._symmetric_logits(h, edge_index)
        if self.training:
            weights_homo = F.gumbel_softmax(
                torch.stack([torch.zeros_like(logits), logits], dim=-1),
                tau=self.temperature,
                hard=self.hard,
                dim=-1,
            )[:, 1]
        else:
            weights_homo = torch.sigmoid(logits)
        weights_hetero = 1.0 - weights_homo
        return weights_homo, weights_hetero


class GraphPropagationEncoder(nn.Module):
    """Simple dense low-pass/high-pass SGC encoder."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.2,
        high_pass_alpha: float = 0.5,
    ) -> None:
        super().__init__()
        self.low_linear = nn.Linear(input_dim, hidden_dim)
        self.high_linear = nn.Linear(input_dim, hidden_dim)
        self.num_layers = num_layers
        self.dropout = dropout
        self.high_pass_alpha = high_pass_alpha

    def forward(self, x: torch.Tensor, adj_homo: torch.Tensor, adj_hetero: torch.Tensor) -> torch.Tensor:
        low = F.relu(self.low_linear(x))
        high = F.relu(self.high_linear(x))
        low = F.dropout(low, p=self.dropout, training=self.training)
        high = F.dropout(high, p=self.dropout, training=self.training)

        eye = torch.eye(adj_hetero.size(0), device=x.device, dtype=x.dtype)
        high_filter = eye - self.high_pass_alpha * adj_hetero

        for _ in range(self.num_layers):
            low = adj_homo.matmul(low)
            high = high_filter.matmul(high)
        return torch.cat([low, high], dim=-1)


class NystromLowRank(nn.Module):
    """Differentiable Nyström low-rank module with stabilized decompositions."""

    def __init__(
        self,
        rank: int,
        ridge: float = 1e-4,
        svd_eps: float = 1e-7,
        diag_jitter: float = 1e-8,
        symmetrize: bool = True,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.ridge = ridge
        self.svd_eps = svd_eps
        self.diag_jitter = diag_jitter
        self.symmetrize = symmetrize

    def _safe_eigh(self, matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.symmetrize:
            matrix = 0.5 * (matrix + matrix.t())
        eye = torch.eye(matrix.size(0), device=matrix.device, dtype=matrix.dtype)
        jitter = torch.linspace(
            0.0,
            self.diag_jitter,
            matrix.size(0),
            device=matrix.device,
            dtype=matrix.dtype,
        )
        matrix = matrix + self.ridge * eye + torch.diag(jitter)
        evals, evecs = torch.linalg.eigh(matrix)
        evals = evals.clamp_min(self.svd_eps)
        return evals, evecs

    def _safe_left_svd_basis(self, matrix: torch.Tensor, rank: int) -> torch.Tensor:
        gram = matrix.t().matmul(matrix)
        evals, right_vecs = self._safe_eigh(gram)
        order = torch.argsort(evals, descending=True)
        order = order[:rank]
        singular = evals[order].sqrt().clamp_min(self.svd_eps)
        right_vecs = right_vecs[:, order]
        basis = matrix.matmul(right_vecs) / singular.unsqueeze(0)
        basis, _ = torch.linalg.qr(basis, mode="reduced")
        return basis[:, :rank]

    def forward(self, h: torch.Tensor, anchor_indices: torch.Tensor) -> torch.Tensor:
        if anchor_indices.dim() != 1:
            raise ValueError("anchor_indices must be a 1D tensor")
        if anchor_indices.numel() < self.rank:
            raise ValueError("number of anchors must be >= rank")

        anchors = h.index_select(0, anchor_indices.to(device=h.device))
        w = anchors.matmul(anchors.t())
        c = h.matmul(anchors.t())

        evals, evecs = self._safe_eigh(w)
        inv_sqrt = torch.diag(evals.rsqrt())
        b = c.matmul(evecs).matmul(inv_sqrt)
        return self._safe_left_svd_basis(b, self.rank)


class EndToEndUnifiedClustering(nn.Module):
    """GREET frontend + ELSS differentiable Nyström backend."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        rank: int,
        *,
        discriminator_hidden_dim: int = 128,
        gnn_layers: int = 2,
        dropout: float = 0.2,
        high_pass_alpha: float = 0.5,
        gumbel_temperature: float = 1.0,
        hard_gumbel: bool = False,
        nystrom_ridge: float = 1e-4,
        ranking_weight: float = 1.0,
        margin_homo: float = 0.5,
        margin_hetero: float = 0.5,
    ) -> None:
        super().__init__()
        self.edge_discriminator = EdgeDiscriminator(
            input_dim=input_dim,
            hidden_dim=discriminator_hidden_dim,
            temperature=gumbel_temperature,
            hard=hard_gumbel,
        )
        self.encoder = GraphPropagationEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=gnn_layers,
            dropout=dropout,
            high_pass_alpha=high_pass_alpha,
        )
        self.nystrom = NystromLowRank(rank=rank, ridge=nystrom_ridge)
        self.ranking_weight = ranking_weight
        self.margin_homo = margin_homo
        self.margin_hetero = margin_hetero

    def build_views(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        weights_homo: torch.Tensor,
        weights_hetero: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_nodes = x.size(0)
        adj_raw = edge_index_to_dense_adj(
            edge_index,
            num_nodes,
            make_symmetric=True,
            add_self_loops=False,
        ).to(dtype=x.dtype)
        adj_homo = edge_index_to_dense_adj(
            edge_index,
            num_nodes,
            weights_homo.to(dtype=x.dtype),
            make_symmetric=True,
            add_self_loops=True,
        )
        adj_hetero = edge_index_to_dense_adj(
            edge_index,
            num_nodes,
            weights_hetero.to(dtype=x.dtype),
            make_symmetric=True,
            add_self_loops=True,
        )
        adj_homo = normalize_adjacency(adj_homo, mode="sym")
        adj_hetero = normalize_adjacency(adj_hetero, mode="sym")
        return adj_homo, adj_hetero, adj_raw

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        anchor_indices: torch.Tensor,
        *,
        laplacian: torch.Tensor | None = None,
    ) -> UnifiedOutput:
        weights_homo, weights_hetero = self.edge_discriminator(x, edge_index)
        adj_homo, adj_hetero, adj_raw = self.build_views(x, edge_index, weights_homo, weights_hetero)
        embeddings = self.encoder(x, adj_homo, adj_hetero)
        basis = self.nystrom(embeddings, anchor_indices)

        if laplacian is None:
            laplacian = normalized_laplacian_from_adj(adj_raw.to(dtype=x.dtype))
        subspace_loss = subspace_smoothness_loss(basis, laplacian)
        ranking_loss = pivot_anchored_ranking_loss(
            embeddings,
            edge_index,
            weights_homo,
            weights_hetero,
            margin_homo=self.margin_homo,
            margin_hetero=self.margin_hetero,
        )
        loss = subspace_loss + self.ranking_weight * ranking_loss

        return UnifiedOutput(
            loss=loss,
            subspace_loss=subspace_loss,
            ranking_loss=ranking_loss,
            embeddings=embeddings,
            basis=basis,
            adj_homo=adj_homo,
            adj_hetero=adj_hetero,
            weights_homo=weights_homo,
            weights_hetero=weights_hetero,
        )
