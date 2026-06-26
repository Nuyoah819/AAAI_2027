from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize


DATA_ROOT = Path(r"D:\study\graduate_student\papers\AAAI2027\data")


@dataclass(frozen=True)
class GraphDataset:
    name: str
    adj: sp.csr_matrix
    features: sp.spmatrix | np.ndarray
    labels: np.ndarray
    n_clusters: int
    directed_adj: sp.csr_matrix | None = None


def load_dataset(name: str, data_root: str | Path = DATA_ROOT) -> GraphDataset:
    key = name.lower()
    data_root = Path(data_root)
    if key in {"acm", "dblp", "pubmed", "wiki", "blogcatalog"}:
        mat_path = data_root / f"{key}.mat"
        if mat_path.exists():
            data = sio.loadmat(mat_path)
            adj = _first_present(data, ["W", "A", "adj", "network"])
            features = _first_present(data, ["fea", "X", "features", "attrb"])
            labels = _first_present(data, ["gnd", "label", "labels", "Y"]).reshape(-1)
            return _build_dataset(key, adj, features, labels, preserve_graph=True)
        return _load_raw_npz(key, data_root / key / "raw")
    if key == "flickr":
        return _load_raw_npz(key, data_root / "flickr" / "raw")
    if key in {"squirrel", "texas", "chameleon"}:
        return _load_geom_gcn_dataset(key, data_root / key)
    raise ValueError(
        f"Unsupported dataset {name!r}. Expected ACM, DBLP, PubMed, Wiki, flickr, "
        "blogcatalog, squirrel, texas, or chameleon."
    )


def _first_present(data: dict, keys: list[str]):
    for key in keys:
        if key in data:
            return data[key]
    available = ", ".join(sorted(k for k in data if not k.startswith("__")))
    raise KeyError(f"None of {keys} found. Available keys: {available}")


def _load_raw_npz(name: str, raw_dir: Path) -> GraphDataset:
    features = sp.load_npz(raw_dir / "attrs.npz").astype(np.float32).tocsr()
    n_nodes = features.shape[0]
    labels = np.full(n_nodes, -1, dtype=np.int64)
    with (raw_dir / "labels.txt").open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                node, label = int(parts[0]), int(parts[1])
                if 0 <= node < n_nodes:
                    labels[node] = label
    if np.any(labels < 0):
        raise ValueError(f"{name} has {int(np.sum(labels < 0))} nodes without labels.")

    rows: list[int] = []
    cols: list[int] = []
    with (raw_dir / "edgelist.txt").open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                i, j = int(parts[0]), int(parts[1])
                if 0 <= i < n_nodes and 0 <= j < n_nodes:
                    rows.append(i)
                    cols.append(j)
    adj = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n_nodes, n_nodes))
    return _build_dataset(name, adj, features, labels, preserve_graph=False)


def _load_geom_gcn_dataset(name: str, root: Path) -> GraphDataset:
    node_path = root / "out1_node_feature_label.txt"
    edge_path = root / "out1_graph_edges.txt"
    if not node_path.exists() or not edge_path.exists():
        raise FileNotFoundError(f"Expected Geom-GCN files under {root}")

    node_ids: list[int] = []
    labels: list[int] = []
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    with node_path.open("r", encoding="utf-8") as f:
        header = next(f, "")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            node_id = int(parts[0])
            values = [float(v) for v in parts[1].split(",") if v != ""]
            row_idx = len(node_ids)
            node_ids.append(node_id)
            labels.append(int(parts[2]))
            for col_idx, value in enumerate(values):
                if value != 0.0:
                    rows.append(row_idx)
                    cols.append(col_idx)
                    data.append(value)

    if not node_ids:
        raise ValueError(f"{name} has no nodes in {node_path}")
    id_to_pos = {node_id: pos for pos, node_id in enumerate(node_ids)}
    n_nodes = len(node_ids)
    n_features = (max(cols) + 1) if cols else 0
    features = sp.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_features), dtype=np.float32)

    edge_rows: list[int] = []
    edge_cols: list[int] = []
    with edge_path.open("r", encoding="utf-8") as f:
        header = next(f, "")
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            src_raw, dst_raw = int(parts[0]), int(parts[1])
            src = id_to_pos.get(src_raw)
            dst = id_to_pos.get(dst_raw)
            if src is None or dst is None or src == dst:
                continue
            edge_rows.append(src)
            edge_cols.append(dst)
    adj = sp.csr_matrix(
        (np.ones(len(edge_rows), dtype=np.float32), (edge_rows, edge_cols)),
        shape=(n_nodes, n_nodes),
    )
    directed = adj.astype(np.float32).tocsr()
    directed.setdiag(0)
    directed.eliminate_zeros()
    built = _build_dataset(name, adj, features, np.asarray(labels, dtype=np.int64), preserve_graph=False)
    return GraphDataset(
        name=built.name,
        adj=built.adj,
        features=built.features,
        labels=built.labels,
        n_clusters=built.n_clusters,
        directed_adj=directed,
    )


def _build_dataset(name: str, adj, features, labels: np.ndarray, *, preserve_graph: bool) -> GraphDataset:
    adj = adj.tocsr() if sp.issparse(adj) else sp.csr_matrix(adj)
    adj = adj.astype(np.float32)
    if not preserve_graph:
        adj.setdiag(0)
        adj.eliminate_zeros()
        adj = ((adj + adj.T) > 0).astype(np.float32).tocsr()
    else:
        adj = adj.tocsr()
        adj.setdiag(0)
        adj.eliminate_zeros()

    if sp.issparse(features):
        features = features.astype(np.float64 if preserve_graph else np.float32).tocsr()
    else:
        features = np.asarray(features, dtype=np.float64 if preserve_graph else np.float32)
    labels = np.asarray(labels).reshape(-1).astype(np.int64)
    return GraphDataset(
        name=name,
        adj=adj,
        features=features,
        labels=labels,
        n_clusters=int(np.unique(labels).size),
    )


def preprocess_features(features, *, tfidf: bool = True, norm: str = "l2", dtype=np.float32):
    if sp.issparse(features):
        if tfidf:
            out = TfidfTransformer(norm=norm).fit_transform(features)
        else:
            out = normalize(features, norm=norm)
        return out.astype(dtype).tocsr()
    arr = np.asarray(features, dtype=dtype)
    if tfidf:
        return TfidfTransformer(norm=norm).fit_transform(arr).toarray().astype(dtype)
    return normalize(arr, norm=norm).astype(dtype)


def row_normalize(mx: sp.spmatrix, *, add_self_loops: bool = False, alpha: float = 1.0) -> sp.csr_matrix:
    mx = mx.astype(np.float32).tocsr()
    if add_self_loops:
        mx = mx + alpha * sp.eye(mx.shape[0], dtype=np.float32, format="csr")
    rowsum = np.asarray(mx.sum(axis=1)).reshape(-1)
    inv = np.divide(1.0, rowsum, out=np.zeros_like(rowsum, dtype=np.float32), where=rowsum > 0)
    return sp.diags(inv, dtype=np.float32).dot(mx).tocsr()


def sym_normalize(mx: sp.spmatrix, *, add_self_loops: bool = True, alpha: float = 1.0) -> sp.csr_matrix:
    mx = mx.astype(np.float32).tocsr()
    if add_self_loops:
        mx = mx + alpha * sp.eye(mx.shape[0], dtype=np.float32, format="csr")
    rowsum = np.asarray(mx.sum(axis=1)).reshape(-1)
    inv_sqrt = np.divide(1.0, np.sqrt(rowsum), out=np.zeros_like(rowsum, dtype=np.float32), where=rowsum > 0)
    d = sp.diags(inv_sqrt, dtype=np.float32)
    return d.dot(mx).dot(d).tocsr()
