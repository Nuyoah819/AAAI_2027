from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.data.data_utils import DATA_ROOT, load_dataset
from core.eval.metrics import evaluate_clustering


ELSS_DIR = Path(r"D:\study\graduate_student\papers\AAAI2027\AAAI0610\ELSS_code")
PARAMS = {
    "flickr": {"k_rank": 9, "n_anchors": 300, "power": 1, "d": 0.8, "gamma": 0.005},
    "blogcatalog": {"k_rank": 7, "n_anchors": 300, "power": 3, "d": 0.8, "gamma": 0.005},
}


def preprocess(adj: sp.spmatrix, x, *, tfidf: bool):
    if str(ELSS_DIR) not in sys.path:
        sys.path.insert(0, str(ELSS_DIR))
    from data_load import convert_sparse_matrix_to_sparse_tensor, preprocess_dataset

    if tfidf:
        norm_adj, x_out = preprocess_dataset(adj, x, row_norm=True, sym_norm=True, tf_idf=True, sparse=True)
    else:
        norm_adj, x_out = preprocess_dataset(adj, x, row_norm=True, sym_norm=True, tf_idf=False, sparse=True)
    if sp.issparse(x_out):
        x_out = x_out.toarray()
    return convert_sparse_matrix_to_sparse_tensor(norm_adj.astype(np.float64)), torch.tensor(x_out.astype("float64"), dtype=torch.float64)


def run_one(dataset_name: str, seed: int, device: str, params: dict[str, float | int], tfidf: bool):
    if str(ELSS_DIR) not in sys.path:
        sys.path.insert(0, str(ELSS_DIR))
    from model import FastGraphSubClustering

    data = load_dataset(dataset_name, DATA_ROOT)
    norm_adj_t, x_t = preprocess(data.adj, data.features, tfidf=tfidf)
    model = FastGraphSubClustering(
        n_clusters=data.n_clusters,
        alpha1=0.0,
        k_rank=int(params["k_rank"]),
        n_anchors=int(params["n_anchors"]),
        power=int(params["power"]),
        d=float(params["d"]),
        alpha2=0.00005,
        gamma=float(params["gamma"]),
        device=device,
        random_state=seed,
        using_pgrank=True,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        model.fit(x_t, norm_adj_t)
    pred = model.predict(x_t)
    return evaluate_clustering(data.labels, pred)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["flickr", "blogcatalog"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tfidf", action="store_true")
    args = parser.parse_args()

    detail_path = PROJECT_ROOT / "results" / "elss_local_missing_detail.jsonl"
    detail_path.write_text("", encoding="utf-8")
    rows = []
    for dataset_name in args.datasets:
        params = PARAMS[dataset_name]
        for seed in args.seeds:
            start = time.time()
            try:
                metrics = run_one(dataset_name, seed, args.device, params, args.tfidf)
                row = {
                    "dataset": dataset_name,
                    "method": "ELSS-local",
                    "seed": seed,
                    "seconds": round(time.time() - start, 3),
                    "params": params,
                    **metrics,
                }
            except Exception as exc:  # keep the failed attempt auditable
                row = {
                    "dataset": dataset_name,
                    "method": "ELSS-local",
                    "seed": seed,
                    "seconds": round(time.time() - start, 3),
                    "params": params,
                    "error": repr(exc),
                }
            rows.append(row)
            with detail_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(json.dumps(row, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
