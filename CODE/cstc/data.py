from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize


DATA_ROOT = Path("/mnt/data/users/liusong/data")


@dataclass(frozen=True)
class GraphDataset:
    name: str
    adj: sp.csr_matrix
    features: sp.spmatrix | np.ndarray
    labels: np.ndarray
    n_clusters: int


def load_dataset(name: str, data_root: str | Path = DATA_ROOT) -> GraphDataset:
    key = name.lower()
    root = Path(data_root)
    if key in {"acm", "dblp", "pubmed", "wiki", "blogcatalog"}:
        mat_path = root / f"{key}.mat"
        if mat_path.exists():
            data = sio.loadmat(mat_path)
            adj = _first_present(data, ["W", "A", "adj", "network"])
            features = _first_present(data, ["fea", "X", "features", "attrb"])
            labels = _first_present(data, ["gnd", "label", "labels", "Y"]).reshape(-1)
            return _build_dataset(key, adj, features, labels, preserve_graph=True)
        return _load_raw_npz(key, root / key / "raw")
    if key == "flickr":
        return _load_raw_npz(key, root / "flickr" / "raw")
    if key in {"texas", "squirrel", "chameleon"}:
        return _load_geom_gcn(key, root / key)
    raise ValueError(f"Unsupported dataset: {name}")


def preprocess_features(features, *, tfidf: bool = True, norm: str = "l2", dtype=np.float32):
    if sp.issparse(features):
        out = TfidfTransformer(norm=norm).fit_transform(features) if tfidf else normalize(features, norm=norm)
        return out.astype(dtype).tocsr()
    arr = np.asarray(features, dtype=dtype)
    if tfidf:
        return TfidfTransformer(norm=norm).fit_transform(arr).toarray().astype(dtype)
    return normalize(arr, norm=norm).astype(dtype)


def scipy_to_torch_dense(features, device):
    import torch

    if sp.issparse(features):
        return torch.from_numpy(features.toarray().astype(np.float32)).to(device)
    return torch.from_numpy(np.asarray(features, dtype=np.float32)).to(device)


def build_candidate_edges(
    adj: sp.csr_matrix,
    features,
    *,
    feature_knn: int,
    max_edges: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    graph_edges = _edge_array(adj)
    edge_parts = [graph_edges]
    if feature_knn > 0:
        from sklearn.neighbors import NearestNeighbors

        feat = features
        if sp.issparse(feat):
            feat = feat.astype(np.float32).tocsr()
        else:
            feat = np.asarray(feat, dtype=np.float32)
        nn = NearestNeighbors(n_neighbors=feature_knn + 1, metric="cosine", algorithm="brute")
        nn.fit(feat)
        indices = nn.kneighbors(feat, return_distance=False)[:, 1:]
        rows = np.repeat(np.arange(adj.shape[0], dtype=np.int64), feature_knn)
        cols = indices.reshape(-1).astype(np.int64)
        knn_edges = np.stack([rows, cols], axis=1)
        edge_parts.append(knn_edges)
    edges = np.concatenate(edge_parts, axis=0)
    edges = edges[edges[:, 0] != edges[:, 1]]
    lo = np.minimum(edges[:, 0], edges[:, 1])
    hi = np.maximum(edges[:, 0], edges[:, 1])
    packed = lo.astype(np.int64) * adj.shape[0] + hi.astype(np.int64)
    _, unique_idx = np.unique(packed, return_index=True)
    edges = edges[np.sort(unique_idx)]
    graph_lo = np.minimum(graph_edges[:, 0], graph_edges[:, 1])
    graph_hi = np.maximum(graph_edges[:, 0], graph_edges[:, 1])
    graph_packed = graph_lo.astype(np.int64) * adj.shape[0] + graph_hi.astype(np.int64)
    edge_packed = np.minimum(edges[:, 0], edges[:, 1]).astype(np.int64) * adj.shape[0] + np.maximum(edges[:, 0], edges[:, 1]).astype(np.int64)
    source = np.isin(edge_packed, graph_packed, assume_unique=False).astype(np.float32)
    if edges.shape[0] > max_edges:
        rng = np.random.default_rng(seed)
        keep = rng.choice(edges.shape[0], size=max_edges, replace=False)
        keep.sort()
        edges = edges[keep]
        source = source[keep]
    return edges.astype(np.int64), source.astype(np.float32)


def _first_present(data: dict, keys: list[str]):
    for key in keys:
        if key in data:
            return data[key]
    available = ", ".join(sorted(k for k in data if not k.startswith("__")))
    raise KeyError(f"None of {keys} found. Available keys: {available}")


def _load_raw_npz(name: str, raw_dir: Path) -> GraphDataset:
    features = sp.load_npz(raw_dir / "attrs.npz").astype(np.float32).tocsr()
    n = features.shape[0]
    labels = np.full(n, -1, dtype=np.int64)
    with (raw_dir / "labels.txt").open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                labels[int(parts[0])] = int(parts[1])
    rows: list[int] = []
    cols: list[int] = []
    with (raw_dir / "edgelist.txt").open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                rows.append(int(parts[0]))
                cols.append(int(parts[1]))
    adj = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, n))
    return _build_dataset(name, adj, features, labels, preserve_graph=False)


def _load_geom_gcn(name: str, root: Path) -> GraphDataset:
    node_path = root / "out1_node_feature_label.txt"
    edge_path = root / "out1_graph_edges.txt"
    node_ids: list[int] = []
    labels: list[int] = []
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    with node_path.open("r", encoding="utf-8") as f:
        next(f, "")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            node_id = int(parts[0])
            pos = len(node_ids)
            node_ids.append(node_id)
            labels.append(int(parts[2]))
            for j, raw in enumerate(parts[1].split(",")):
                if raw:
                    value = float(raw)
                    if value != 0.0:
                        rows.append(pos)
                        cols.append(j)
                        vals.append(value)
    n = len(node_ids)
    features = sp.csr_matrix((vals, (rows, cols)), shape=(n, max(cols) + 1 if cols else 0), dtype=np.float32)
    id_to_pos = {node_id: i for i, node_id in enumerate(node_ids)}
    erows: list[int] = []
    ecols: list[int] = []
    with edge_path.open("r", encoding="utf-8") as f:
        next(f, "")
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2 and int(parts[0]) in id_to_pos and int(parts[1]) in id_to_pos:
                i = id_to_pos[int(parts[0])]
                j = id_to_pos[int(parts[1])]
                if i != j:
                    erows.append(i)
                    ecols.append(j)
    adj = sp.csr_matrix((np.ones(len(erows), dtype=np.float32), (erows, ecols)), shape=(n, n))
    return _build_dataset(name, adj, features, np.asarray(labels, dtype=np.int64), preserve_graph=False)


def _build_dataset(name: str, adj, features, labels: np.ndarray, *, preserve_graph: bool) -> GraphDataset:
    adj = adj.tocsr() if sp.issparse(adj) else sp.csr_matrix(adj)
    adj = adj.astype(np.float32)
    adj.setdiag(0)
    adj.eliminate_zeros()
    if not preserve_graph:
        adj = ((adj + adj.T) > 0).astype(np.float32).tocsr()
    features = features.astype(np.float32).tocsr() if sp.issparse(features) else np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels).reshape(-1).astype(np.int64)
    return GraphDataset(name=name, adj=adj, features=features, labels=labels, n_clusters=int(np.unique(labels).size))


def _edge_array(adj: sp.csr_matrix) -> np.ndarray:
    coo = sp.triu(adj, k=1).tocoo()
    return np.stack([coo.row, coo.col], axis=1).astype(np.int64)
