from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

import numpy as np
import scipy.sparse as sp
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.data.data_utils import DATA_ROOT, load_dataset, preprocess_features, row_normalize, sym_normalize
from core.eval.metrics import evaluate_clustering


DATASETS = ["flickr", "blogcatalog"]
SEEDS = [0, 1, 2]
OUT_DETAIL = PROJECT_ROOT / "results" / "lightweight_baselines_detail.jsonl"


def _cluster(z: np.ndarray, k: int, seed: int) -> np.ndarray:
    z = normalize(z.astype(np.float32), norm="l2") if sp.issparse(z) else normalize(np.asarray(z, dtype=np.float32), norm="l2")
    return MiniBatchKMeans(
        n_clusters=k,
        random_state=seed,
        n_init=5,
        batch_size=4096,
        max_iter=80,
        reassignment_ratio=0.01,
    ).fit_predict(z)


def _powers_mean(p: sp.csr_matrix, x, steps: int):
    h = x
    acc = x.copy()
    for _ in range(steps):
        h = p @ h
        acc = acc + h
    return acc * (1.0 / (steps + 1))


def _ppr_diffusion(p: sp.csr_matrix, x, steps: int, alpha: float):
    h = x
    out = alpha * x
    coef = 1.0
    for _ in range(steps):
        coef *= 1.0 - alpha
        h = p @ h
        out = out + alpha * coef * h
    return out


def representations(dataset_name: str):
    data = load_dataset(dataset_name, DATA_ROOT)
    x = preprocess_features(data.features, tfidf=True, norm="l2", dtype=np.float32)
    p_sym = sym_normalize(data.adj, add_self_loops=True)
    p_row = row_normalize(data.adj, add_self_loops=True)

    reps = {
        "KMeans-X": x,
        "SGC": p_sym @ (p_sym @ x),
        "S2GC": _powers_mean(p_sym, x, steps=4),
        "PPR-GCC": _ppr_diffusion(p_row, x, steps=10, alpha=0.2),
    }
    # A simple high-pass residual is useful on low-homophily graphs.
    low = p_sym @ (p_sym @ x)
    reps["LowHigh-Fuse"] = sp.hstack([low, x - low], format="csr") if sp.issparse(x) else np.hstack([low, x - low])
    return reps


def append_row(row: dict[str, object]) -> None:
    OUT_DETAIL.parent.mkdir(parents=True, exist_ok=True)
    with OUT_DETAIL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_dataset(dataset_name: str) -> list[dict[str, object]]:
    data = load_dataset(dataset_name, DATA_ROOT)
    reps = representations(dataset_name)
    rows: list[dict[str, object]] = []
    for seed in SEEDS:
        start = time.time()
        for method, z in reps.items():
            pred = _cluster(z, data.n_clusters, seed)
            metrics = evaluate_clustering(data.labels, pred)
            row = {
                "dataset": dataset_name,
                "method": method,
                "seed": seed,
                "seconds": round(time.time() - start, 3),
                **metrics,
            }
            rows.append(row)
            append_row(row)
            print(
                f"[{dataset_name}] {method} seed={seed} "
                f"ACC={metrics['acc']:.4f} NMI={metrics['nmi']:.4f} "
                f"ARI={metrics['ari']:.4f} F1={metrics['f1']:.4f}",
                flush=True,
            )
    return rows


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    keys = sorted({(str(r["dataset"]), str(r["method"])) for r in rows})
    for dataset_name, method in keys:
        group = [r for r in rows if r["dataset"] == dataset_name and r["method"] == method]
        item: dict[str, object] = {"dataset": dataset_name, "method": method, "source": "local-rerun"}
        for metric in ["acc", "nmi", "ari", "f1"]:
            values = np.asarray([float(r[metric]) for r in group], dtype=np.float64) * 100.0
            item[f"{metric}_mean"] = round(float(values.mean()), 2)
            item[f"{metric}_std"] = round(float(values.std(ddof=0)), 2)
        summary.append(item)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    args = parser.parse_args()

    OUT_DETAIL.parent.mkdir(parents=True, exist_ok=True)
    OUT_DETAIL.write_text("", encoding="utf-8")

    all_rows: list[dict[str, object]] = []
    for dataset_name in args.datasets:
        all_rows.extend(run_dataset(dataset_name))

    result_dir = PROJECT_ROOT / "results"
    summary_path = result_dir / "lightweight_baselines_summary.json"
    summary = summarize(all_rows)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
