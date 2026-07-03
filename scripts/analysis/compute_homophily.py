from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import scipy.sparse as sp

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.data.data_utils import DATA_ROOT, load_dataset


DATASETS = ["acm", "dblp", "pubmed", "wiki", "flickr", "blogcatalog"]


def edge_homophily(name: str) -> dict[str, float | int | str]:
    dataset = load_dataset(name, DATA_ROOT)
    adj = dataset.adj.astype(np.float32).tocsr()
    adj.setdiag(0)
    adj.eliminate_zeros()
    undirected = adj.maximum(adj.T).tocsr()
    rows, cols = sp.triu(undirected, k=1).tocoo().row, sp.triu(undirected, k=1).tocoo().col
    labels = np.asarray(dataset.labels).reshape(-1)
    total = int(rows.size)
    same = int(np.sum(labels[rows] == labels[cols]))
    return {
        "dataset": dataset.name,
        "nodes": int(adj.shape[0]),
        "edges_undirected": total,
        "classes": int(dataset.n_clusters),
        "same_label_edges": same,
        "edge_homophily": float(same / total) if total else 0.0,
    }


def main() -> None:
    results = [edge_homophily(name) for name in DATASETS]
    out_path = PROJECT_ROOT / "results" / "edge_homophily.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
