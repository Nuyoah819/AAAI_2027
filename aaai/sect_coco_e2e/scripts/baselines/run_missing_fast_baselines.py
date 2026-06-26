from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

import numpy as np
import scipy.sparse as sp
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.data.data_utils import DATA_ROOT, load_dataset, preprocess_features, sym_normalize
from core.eval.metrics import evaluate_clustering


DATASETS = ["flickr", "blogcatalog"]
SEEDS = [0, 1, 2]


def cluster(z, n_clusters: int, seed: int) -> np.ndarray:
    z = z.astype(np.float32)
    if sp.issparse(z) and z.shape[1] > 128:
        z = TruncatedSVD(n_components=64, random_state=seed).fit_transform(z)
    z = normalize(z, norm="l2") if sp.issparse(z) else normalize(np.asarray(z), norm="l2")
    model = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=seed,
        n_init=5,
        max_iter=80,
        batch_size=4096,
        reassignment_ratio=0.01,
    )
    return model.fit_predict(z)


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for dataset in sorted({str(r["dataset"]) for r in rows}):
        for method in sorted({str(r["method"]) for r in rows if r["dataset"] == dataset}):
            group = [r for r in rows if r["dataset"] == dataset and r["method"] == method]
            item: dict[str, object] = {"dataset": dataset, "method": method, "source": "local-rerun"}
            for metric in ["acc", "nmi", "ari", "f1"]:
                vals = np.asarray([float(r[metric]) for r in group]) * 100.0
                item[f"{metric}_mean"] = round(float(vals.mean()), 2)
                item[f"{metric}_std"] = round(float(vals.std(ddof=0)), 2)
            out.append(item)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--methods", nargs="+", default=["KMeans-X", "SGC-1", "SGC-2"])
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()

    result_dir = PROJECT_ROOT / "results"
    detail_path = result_dir / "missing_fast_baselines_detail.jsonl"
    summary_path = result_dir / "missing_fast_baselines_summary.json"
    if not args.append:
        detail_path.write_text("", encoding="utf-8")

    all_rows: list[dict[str, object]] = []
    for dataset_name in args.datasets:
        data = load_dataset(dataset_name, DATA_ROOT)
        x = preprocess_features(data.features, tfidf=False, norm="l2", dtype=np.float32)
        p = sym_normalize(data.adj, add_self_loops=True)
        reps = {
            "KMeans-X": x,
            "SGC-1": p @ x,
        }
        reps["SGC-2"] = p @ reps["SGC-1"]

        for method, z in reps.items():
            if method not in set(args.methods):
                continue
            for seed in SEEDS:
                start = time.time()
                pred = cluster(z, data.n_clusters, seed)
                metrics = evaluate_clustering(data.labels, pred)
                row = {
                    "dataset": dataset_name,
                    "method": method,
                    "seed": seed,
                    "seconds": round(time.time() - start, 3),
                    **metrics,
                }
                all_rows.append(row)
                with detail_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(
                    f"[{dataset_name}] {method} seed={seed} "
                    f"ACC={metrics['acc']:.4f} NMI={metrics['nmi']:.4f} "
                    f"ARI={metrics['ari']:.4f} F1={metrics['f1']:.4f} "
                    f"time={row['seconds']:.3f}s",
                    flush=True,
                )

    summary = summarize(all_rows)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
