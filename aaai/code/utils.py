from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import scipy.io as io
import scipy.sparse as sp
import torch
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize


@dataclass
class AttributedGraphData:
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    norm_adj: torch.Tensor
    labels: np.ndarray
    num_clusters: int
    pagerank_scores: torch.Tensor
    structural_encoding: torch.Tensor


@dataclass
class SyntheticGraph:
    x: torch.Tensor
    edge_index: torch.Tensor
    labels: torch.Tensor


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def row_normalize_sparse(mx: sp.spmatrix, add_loops: bool = True, alpha: float = 1.0) -> sp.csr_matrix:
    if add_loops:
        mx = mx + alpha * sp.eye(mx.shape[0], format="csr")
    rowsum = np.asarray(mx.sum(1)).ravel()
    inv = np.power(rowsum, -1.0, where=rowsum > 0)
    inv[~np.isfinite(inv)] = 0.0
    return sp.diags(inv).dot(mx).tocsr()


def symmetric_normalize_sparse(mx: sp.spmatrix, add_loops: bool = True, alpha: float = 1.0) -> sp.coo_matrix:
    if add_loops:
        mx = mx + alpha * sp.eye(mx.shape[0], format=mx.format)
    mx = mx.tocoo()
    rowsum = np.asarray(mx.sum(1)).ravel()
    inv_sqrt = np.power(rowsum + 1e-8, -0.5)
    inv_sqrt[~np.isfinite(inv_sqrt)] = 0.0
    d_inv_sqrt = sp.diags(inv_sqrt)
    return d_inv_sqrt.dot(mx).dot(d_inv_sqrt).tocoo()


def scipy_sparse_to_torch_sparse(matrix: sp.spmatrix, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    matrix = matrix.tocoo()
    indices = np.vstack((matrix.row, matrix.col)).astype(np.int64)
    values = matrix.data.astype(np.float32 if dtype == torch.float32 else np.float64)
    return torch.sparse_coo_tensor(
        torch.from_numpy(indices),
        torch.from_numpy(values).to(dtype=dtype),
        torch.Size(matrix.shape),
    ).coalesce()


def scipy_sparse_to_edge_index(matrix: sp.spmatrix) -> tuple[torch.Tensor, torch.Tensor]:
    matrix = matrix.tocoo()
    edge_index = torch.from_numpy(np.vstack((matrix.row, matrix.col)).astype(np.int64))
    edge_weight = torch.from_numpy(matrix.data.astype(np.float32))
    return edge_index, edge_weight


def load_mat_attributed_graph(
    dataset: str,
    *,
    root: str,
    tf_idf: bool = True,
    feature_norm: str = "l2",
    structure_dim: int = 8,
    pagerank_damping: float = 0.85,
) -> AttributedGraphData:
    path = os.path.join(root, "ELSS", "data", f"{dataset}.mat")
    data = io.loadmat(path)

    x = data["fea"].astype(np.float32)
    adj = data["W"]
    if not sp.issparse(adj):
        adj = sp.csr_matrix(adj)
    adj = adj.astype(np.float32).tocsr()
    labels = data["gnd"].reshape(-1).astype(np.int64)
    num_clusters = int(np.unique(labels).size)

    if tf_idf:
        x = TfidfTransformer(norm=feature_norm).fit_transform(x).astype(np.float32)
        if sp.issparse(x):
            x = x.toarray()
    else:
        x = normalize(x, norm=feature_norm).astype(np.float32)

    norm_adj = symmetric_normalize_sparse(adj, add_loops=True, alpha=1.0)
    edge_index, edge_weight = scipy_sparse_to_edge_index(adj)
    pr_scores = pagerank_scores_from_scipy(adj, damping=pagerank_damping)
    structural_encoding = structural_encoding_from_rw(adj, dim=structure_dim)

    return AttributedGraphData(
        x=torch.from_numpy(x),
        edge_index=edge_index.long(),
        edge_weight=edge_weight.float(),
        norm_adj=scipy_sparse_to_torch_sparse(norm_adj, dtype=torch.float32),
        labels=labels,
        num_clusters=num_clusters,
        pagerank_scores=pr_scores.float(),
        structural_encoding=structural_encoding.float(),
    )


def structural_encoding_from_rw(adj: sp.spmatrix, dim: int = 8) -> torch.Tensor:
    adj = adj.tocsr()
    degree = np.asarray(adj.sum(1)).ravel()
    inv_degree = np.power(degree, -1.0, where=degree > 0)
    inv_degree[~np.isfinite(inv_degree)] = 0.0
    rw = adj.dot(sp.diags(inv_degree)).tocsr()

    encodings = []
    power = rw.copy()
    encodings.append(torch.from_numpy(power.diagonal().astype(np.float32)))
    for _ in range(dim - 1):
        power = power.dot(rw).tocsr()
        encodings.append(torch.from_numpy(power.diagonal().astype(np.float32)))
    return torch.stack(encodings, dim=-1)


def pagerank_scores_from_scipy(
    adj: sp.spmatrix,
    *,
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> torch.Tensor:
    transition = row_normalize_sparse(adj, add_loops=False, alpha=0.0).tocsr()
    num_nodes = transition.shape[0]
    pr = np.full(num_nodes, 1.0 / num_nodes, dtype=np.float64)
    teleport = np.full(num_nodes, 1.0 / num_nodes, dtype=np.float64)
    for _ in range(max_iter):
        prev = pr.copy()
        pr = (1.0 - damping) * teleport + damping * transition.T.dot(prev)
        if np.abs(pr - prev).sum() < tol:
            break
    return torch.from_numpy(pr.astype(np.float32))


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


def normalize_torch_sparse_adj(adj: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    adj = adj.coalesce()
    row, col = adj.indices()
    value = adj.values()
    degree = torch.zeros(adj.size(0), device=value.device, dtype=value.dtype)
    degree.index_add_(0, row, value)
    inv_sqrt = degree.clamp_min(eps).pow(-0.5)
    norm_value = value * inv_sqrt[row] * inv_sqrt[col]
    return torch.sparse_coo_tensor(adj.indices(), norm_value, adj.size(), device=value.device).coalesce()


def build_weighted_sparse_view(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    learned_weight: torch.Tensor,
    num_nodes: int,
    *,
    add_self_loops: bool = True,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    if device is None:
        device = edge_index.device
    base_weight = edge_weight.to(device=device, dtype=dtype) * learned_weight.to(device=device, dtype=dtype)
    row, col = edge_index.to(device=device)
    indices = torch.cat(
        [
            torch.stack([row, col], dim=0),
            torch.stack([col, row], dim=0),
        ],
        dim=1,
    )
    values = torch.cat([base_weight, base_weight], dim=0)
    if add_self_loops:
        diag = torch.arange(num_nodes, device=device)
        indices = torch.cat([indices, torch.stack([diag, diag], dim=0)], dim=1)
        values = torch.cat([values, torch.ones(num_nodes, device=device, dtype=dtype)], dim=0)
    adj = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes), device=device).coalesce()
    return normalize_torch_sparse_adj(adj)


def sparse_spmm(adj: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.sparse.mm(adj, x)


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
    return SyntheticGraph(x=x, edge_index=edge_index, labels=labels)
