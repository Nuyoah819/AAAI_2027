from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
import sys
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = Path(__file__).resolve().parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cstc import DATA_ROOT, CSTCConfig, evaluate_clustering, load_dataset, preprocess_features
from cstc.model import fit_predict


DATASETS = ["acm", "dblp", "pubmed", "wiki", "flickr", "blogcatalog", "squirrel", "texas", "chameleon"]


SOTA_TARGETS = {
    "acm": {"acc": 0.9362, "nmi": 0.7588, "ari": 0.8189},
    "dblp": {"acc": 0.9369, "nmi": 0.7974, "ari": 0.8483},
    "pubmed": {"acc": 0.7617, "nmi": 0.3771, "ari": 0.4266},
    "wiki": {"acc": 0.6482, "nmi": 0.5979, "ari": 0.4851},
    "flickr": {"acc": 0.8389, "nmi": 0.7125, "ari": 0.6752},
    "blogcatalog": {"acc": 0.9172, "nmi": 0.7860, "ari": 0.8163},
    "texas": {"acc": 0.7508, "nmi": 0.5149, "ari": 0.6086},
    "squirrel": {"acc": 0.3443, "nmi": 0.1224, "ari": 0.0932},
    "chameleon": {"acc": 0.4202, "nmi": 0.2199, "ari": 0.1562},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run unified CSTC graph clustering experiments.")
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--config", type=Path, default=CODE_ROOT / "configs" / "cstc_default.json")
    parser.add_argument("--output", type=Path, default=CODE_ROOT / "results" / "cstc_main_results.csv")
    parser.add_argument("--diagnostics-dir", type=Path, default=CODE_ROOT / "results" / "diagnostics")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-edges", type=int, default=None)
    parser.add_argument("--feature-knn", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def load_config(path: Path, args: argparse.Namespace) -> CSTCConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload = {k: v for k, v in payload.items() if k in CSTCConfig.__dataclass_fields__}
    cfg = CSTCConfig(**payload)
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.max_edges is not None:
        cfg.max_edges = args.max_edges
    if args.feature_knn is not None:
        cfg.feature_knn = args.feature_knn
    if args.device is not None:
        cfg.device = args.device
    if args.smoke:
        cfg.epochs = min(cfg.epochs, 5)
        cfg.max_edges = min(cfg.max_edges, 35_000)
        cfg.feature_knn = min(cfg.feature_knn, 4)
        cfg.kmeans_n_init = 2
    return cfg


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.diagnostics_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset_name in args.datasets:
        start = time.time()
        row = {"dataset": dataset_name, "method": "CSTC", "seed": cfg.seed, "status": "ok"}
        try:
            dataset = load_dataset(dataset_name, args.data_root)
            features = preprocess_features(dataset.features, tfidf=cfg.use_tfidf)
            pred, diagnostics = fit_predict(dataset, features, cfg)
            metrics = evaluate_clustering(dataset.labels, pred)
            row.update(metrics)
            target = SOTA_TARGETS.get(dataset_name, {})
            for metric in ["acc", "nmi", "ari"]:
                row[f"sota_{metric}"] = target.get(metric, "")
                row[f"gap_{metric}"] = metrics[metric] - target[metric] if metric in target else ""
            diagnostics["metrics"] = metrics
            diagnostics["dataset"] = dataset_name
            diagnostics["runtime_sec"] = time.time() - start
            (args.diagnostics_dir / f"{dataset_name}_diagnostics.json").write_text(
                json.dumps(diagnostics, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            row["status"] = "error"
            row["error"] = repr(exc)
            row["traceback"] = traceback.format_exc()
        row["runtime_sec"] = round(time.time() - start, 3)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))
    fieldnames = [
        "dataset",
        "method",
        "seed",
        "status",
        "acc",
        "nmi",
        "ari",
        "f1",
        "sota_acc",
        "sota_nmi",
        "sota_ari",
        "gap_acc",
        "gap_nmi",
        "gap_ari",
        "runtime_sec",
        "error",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
