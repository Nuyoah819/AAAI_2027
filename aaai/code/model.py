from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

from losses import pivot_anchored_ranking_loss, subspace_smoothness_loss
from utils import build_weighted_sparse_view, sparse_spmm


@dataclass
class UnifiedOutput:
    loss: torch.Tensor
    subspace_loss: torch.Tensor
    ranking_loss: torch.Tensor
    embeddings: torch.Tensor
    basis: torch.Tensor
    weights_homo: torch.Tensor
    weights_hetero: torch.Tensor
    adj_homo: torch.Tensor
    adj_hetero: torch.Tensor


class EdgeDiscriminator(nn.Module):
    """GREET-style edge discriminator on feature + structural encoding inputs."""

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

    def forward(self, node_inputs: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.node_mlp(node_inputs)
        src, dst = edge_index
        logits_forward = self.edge_mlp(torch.cat([h[src], h[dst]], dim=-1)).squeeze(-1)
        logits_backward = self.edge_mlp(torch.cat([h[dst], h[src]], dim=-1)).squeeze(-1)
        logits = 0.5 * (logits_forward + logits_backward)
        if self.training:
            # Keep the edge split differentiable during training so the final
            # clustering loss can still update the discriminator.
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
    """Sparse low-pass/high-pass propagation for large attributed graphs."""

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

        for _ in range(self.num_layers):
            # Homophilic edges act as the low-pass channel.
            low = sparse_spmm(adj_homo, low)
            # Heterophilic edges act as a high-pass correction. This keeps the
            # GREET intuition while remaining sparse enough for ELSS-scale data.
            high = high - self.high_pass_alpha * sparse_spmm(adj_hetero, high)
        return torch.cat([low, high], dim=-1)


class NystromLowRank(nn.Module):
    """Differentiable Nyström low-rank module with stabilized eigendecompositions."""

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
        # Ridge + tiny diagonal jitter reduce the chance of repeated or
        # near-zero eigenvalues breaking the differentiable Nyström path.
        matrix = matrix + self.ridge * eye + torch.diag(jitter)
        evals, evecs = torch.linalg.eigh(matrix)
        evals = evals.clamp_min(self.svd_eps)
        return evals, evecs

    def _safe_left_svd_basis(self, matrix: torch.Tensor, rank: int) -> torch.Tensor:
        gram = matrix.t().matmul(matrix)
        evals, right_vecs = self._safe_eigh(gram)
        order = torch.argsort(evals, descending=True)[:rank]
        singular = evals[order].sqrt().clamp_min(self.svd_eps)
        basis = matrix.matmul(right_vecs[:, order]) / singular.unsqueeze(0)
        basis, _ = torch.linalg.qr(basis, mode="reduced")
        return basis[:, :rank]

    def forward(self, h: torch.Tensor, anchor_indices: torch.Tensor) -> torch.Tensor:
        anchors = h.index_select(0, anchor_indices.to(device=h.device))
        w = anchors.matmul(anchors.t())
        c = h.matmul(anchors.t())
        evals, evecs = self._safe_eigh(w)
        # Gradients still flow through h -> c/w -> basis, so this low-rank
        # module can supervise the frontend end-to-end.
        b = c.matmul(evecs).matmul(torch.diag(evals.rsqrt()))
        return self._safe_left_svd_basis(b, self.rank)


class EndToEndUnifiedClustering(nn.Module):
    """ELSS-oriented attributed graph clustering with GREET frontend."""

    def __init__(
        self,
        feature_dim: int,
        discriminator_input_dim: int,
        rank: int,
        *,
        hidden_dim: int = 64,
        discriminator_hidden_dim: int = 128,
        gnn_layers: int = 2,
        dropout: float = 0.2,
        high_pass_alpha: float = 0.5,
        gumbel_temperature: float = 1.0,
        hard_gumbel: bool = False,
        nystrom_ridge: float = 1e-4,
        ranking_weight: float = 0.5,
        margin_homo: float = 0.5,
        margin_hetero: float = 0.5,
    ) -> None:
        super().__init__()
        self.edge_discriminator = EdgeDiscriminator(
            input_dim=discriminator_input_dim,
            hidden_dim=discriminator_hidden_dim,
            temperature=gumbel_temperature,
            hard=hard_gumbel,
        )
        self.encoder = GraphPropagationEncoder(
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_layers=gnn_layers,
            dropout=dropout,
            high_pass_alpha=high_pass_alpha,
        )
        self.nystrom = NystromLowRank(rank=rank, ridge=nystrom_ridge)
        self.ranking_weight = ranking_weight
        self.margin_homo = margin_homo
        self.margin_hetero = margin_hetero

    def forward(
        self,
        x: torch.Tensor,
        discriminator_inputs: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        anchor_indices: torch.Tensor,
        norm_adj: torch.Tensor,
    ) -> UnifiedOutput:
        weights_homo, weights_hetero = self.edge_discriminator(discriminator_inputs, edge_index)
        # ELSS remains the main body: GREET only provides edge-aware graph views
        # for the downstream low-rank clustering pipeline.
        adj_homo = build_weighted_sparse_view(
            edge_index,
            edge_weight,
            weights_homo,
            x.size(0),
            dtype=x.dtype,
            device=x.device,
        )
        adj_hetero = build_weighted_sparse_view(
            edge_index,
            edge_weight,
            weights_hetero,
            x.size(0),
            dtype=x.dtype,
            device=x.device,
        )
        embeddings = self.encoder(x, adj_homo, adj_hetero)
        basis = self.nystrom(embeddings, anchor_indices)
        subspace_loss = subspace_smoothness_loss(basis, norm_adj)
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
            weights_homo=weights_homo,
            weights_hetero=weights_hetero,
            adj_homo=adj_homo,
            adj_hetero=adj_hetero,
        )


def square_feat_map(z: torch.Tensor, c: float = 2 ** -0.5) -> torch.Tensor:
    n, d = z.shape
    bias = torch.ones(n, 1, device=z.device, dtype=z.dtype)
    linear = z
    quadratic = []
    for i in range(d):
        quadratic.append(z[:, i : i + 1] ** 2)
        for j in range(i + 1, d):
            quadratic.append(z[:, i : i + 1] * z[:, j : j + 1])
    quadratic = torch.cat(quadratic, dim=1)
    mapped = torch.cat([bias, linear, quadratic], dim=1)

    coeffs = torch.ones(mapped.size(1), device=z.device, dtype=z.dtype)
    coeffs[0] = c
    coeffs[1 : d + 1] = np.sqrt(2 * c)
    coeffs[d + 1 :] = np.sqrt(2.0)
    return mapped * coeffs.unsqueeze(0)


@torch.no_grad()
def elss_cluster_assignments(
    basis: torch.Tensor,
    num_clusters: int,
    *,
    random_state: int = 42,
) -> np.ndarray:
    # Keep the ELSS-style postprocessing spirit, but replace the fragile direct
    # SVD with a stabilized Gram eigendecomposition on PubMed.
    mapped = square_feat_map(basis)
    gram = mapped.t().matmul(mapped)
    gram = 0.5 * (gram + gram.t())
    ridge = 1e-6 * torch.eye(gram.size(0), device=gram.device, dtype=gram.dtype)
    evals, evecs = torch.linalg.eigh(gram + ridge)
    order = torch.argsort(evals, descending=True)
    rank = min(num_clusters + 1, evecs.size(1))
    top_vecs = evecs[:, order[:rank]]
    singular = evals[order[:rank]].clamp_min(1e-8).sqrt()
    u_full = mapped.matmul(top_vecs) / singular.unsqueeze(0)
    q = u_full[:, 1:num_clusters + 1].cpu().numpy()
    kmeans = KMeans(n_clusters=num_clusters, random_state=random_state, n_init=10)
    return kmeans.fit_predict(q)
