from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

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
    adj: sp.csr_matrix
    labels: np.ndarray
    num_clusters: int


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


def load_mat_attributed_graph(
    dataset: str,
    *,
    root: str,
    data_path: str | None = None,
    data_root: str | None = None,
    tf_idf: bool = True,
    feature_norm: str = "l2",
) -> AttributedGraphData:
    path = resolve_mat_dataset_path(dataset, root=root, data_path=data_path, data_root=data_root)
    data = io.loadmat(path)

    x = data["fea"].astype(np.float32)
    adj = data["W"]
    if not sp.issparse(adj):
        adj = sp.csr_matrix(adj)
    adj = adj.astype(np.float32).tocsr()
    labels = data["gnd"].reshape(-1).astype(np.int64)
    if labels.min() == 1:
        labels = labels - 1
    num_clusters = int(np.unique(labels).size)

    if tf_idf:
        x = TfidfTransformer(norm=feature_norm).fit_transform(x).astype(np.float32)
        if sp.issparse(x):
            x = x.toarray()
    else:
        x = normalize(x, norm=feature_norm).astype(np.float32)

    edge_index = scipy_sparse_to_edge_index(adj)
    return AttributedGraphData(
        x=torch.from_numpy(x),
        edge_index=edge_index.long(),
        adj=adj,
        labels=labels,
        num_clusters=num_clusters,
    )


def resolve_mat_dataset_path(dataset: str, *, root: str, data_path: str | None = None, data_root: str | None = None) -> Path:
    filename = f"{dataset}.mat"
    if data_path:
        path = Path(data_path).expanduser().resolve()
        if path.is_file():
            return path
        raise FileNotFoundError(f"Dataset file does not exist: {path}")

    candidates: list[Path] = []
    if data_root:
        candidates.append(Path(data_root).expanduser() / filename)

    repo_root = Path(root).expanduser().resolve()
    candidates.extend(
        [
            repo_root / "data" / filename,
            repo_root / "ELSS" / "data" / filename,
            repo_root.parent / "data" / filename,
            repo_root.parent / "ELSS" / "data" / filename,
        ]
    )

    project_root = Path.home() / "Project"
    if project_root.is_dir():
        candidates.extend(sorted(project_root.glob(f"*/data/{filename}")))

    seen: set[Path] = set()
    checked: list[Path] = []
    for candidate in candidates:
        path = candidate.expanduser().resolve()
        if path in seen:
            continue
        seen.add(path)
        checked.append(path)
        if path.is_file():
            return path

    checked_text = "\n  - ".join(str(path) for path in checked)
    raise FileNotFoundError(
        f"Could not find {filename}. Put it under <repo>/data/ or pass "
        f"--data-path /path/to/{filename}.\nChecked:\n  - {checked_text}"
    )


def scipy_sparse_to_edge_index(matrix: sp.spmatrix) -> torch.Tensor:
    matrix = matrix.tocoo()
    return torch.from_numpy(np.vstack((matrix.row, matrix.col)).astype(np.int64))


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
