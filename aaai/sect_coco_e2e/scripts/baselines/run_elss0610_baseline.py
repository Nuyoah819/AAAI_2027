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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.eval.metrics import evaluate_clustering


ELSS_DIR = Path(r"D:\study\graduate_student\papers\AAAI2027\AAAI0610\ELSS_code")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AAAI0610 ELSS_code with clean stdout.")
    parser.add_argument("--dataset", default="pubmed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--power", type=int, default=None)
    parser.add_argument("--anchors", type=int, default=None)
    parser.add_argument("--k-rank", type=int, default=None)
    parser.add_argument("--d", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--alpha2", type=float, default=0.00005)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = time.time()
    if str(ELSS_DIR) not in sys.path:
        sys.path.insert(0, str(ELSS_DIR))
    cwd = os.getcwd()
    os.chdir(ELSS_DIR)
    try:
        from data_load import convert_sparse_matrix_to_sparse_tensor, datagen, preprocess_dataset
        from demo import DATASET_PARAMS
        from model import FastGraphSubClustering

        params = dict(DATASET_PARAMS[args.dataset])
        if args.power is not None:
            params["power"] = args.power
        if args.anchors is not None:
            params["n_anchors"] = args.anchors
        if args.k_rank is not None:
            params["k_rank"] = args.k_rank
        if args.d is not None:
            params["d"] = args.d
        if args.gamma is not None:
            params["gamma"] = args.gamma

        adj, x, labels, n_classes = datagen(args.dataset)
        norm_adj, x = preprocess_dataset(adj, x, row_norm=True, sym_norm=True, tf_idf=True, sparse=True)
        if sp.issparse(x):
            x = x.toarray()
        norm_adj_t = convert_sparse_matrix_to_sparse_tensor(norm_adj.astype(np.float64))
        x_t = torch.tensor(x.astype("float64"), dtype=torch.float64)
        model = FastGraphSubClustering(
            n_clusters=n_classes,
            alpha1=0.0,
            k_rank=params["k_rank"],
            n_anchors=params["n_anchors"],
            power=params["power"],
            d=params["d"],
            alpha2=args.alpha2,
            gamma=params["gamma"],
            device=args.device,
            random_state=args.seed,
            using_pgrank=True,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            model.fit(x_t, norm_adj_t)
        pred = model.predict(x_t)
        payload = {
            "method": "ELSS_code.FastGraphSubClustering",
            "dataset": args.dataset,
            "metrics": evaluate_clustering(labels, pred),
            "seconds": round(time.time() - start, 3),
            "params": params,
            "seed": args.seed,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    finally:
        os.chdir(cwd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
