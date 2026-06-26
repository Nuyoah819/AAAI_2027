from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from pathlib import Path
import sys
import time
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.data.data_utils import DATA_ROOT, load_dataset, preprocess_features
from core.eval.metrics import evaluate_clustering
from core.e2e.sect_coco_e2e import E2ESECTCoCo, E2ESECTCoCoConfig


ROOT = PROJECT_ROOT
DATASETS = ["acm", "dblp", "pubmed", "wiki", "flickr", "blogcatalog", "squirrel", "texas", "chameleon"]

TARGETS = {
    "acm": {"acc": 0.9362, "nmi": 0.7588, "ari": 0.8189},
    "dblp": {"acc": 0.9369, "nmi": 0.7971, "ari": 0.8483},
    "pubmed": {"acc": 0.7617, "nmi": 0.3771, "ari": 0.4266},
    "wiki": {"acc": 0.6440, "nmi": 0.5920, "ari": 0.4490},
    "flickr": {"acc": 0.8159, "nmi": 0.6636, "ari": 0.6425},
    "blogcatalog": {"acc": 0.9172, "nmi": 0.7860, "ari": 0.8163},
    "squirrel": {"acc": 0.3443, "nmi": 0.1224, "ari": 0.0932},
    "texas": {"acc": 0.7508, "nmi": 0.4619, "ari": 0.5324},
    "chameleon": {"acc": 0.4202, "nmi": 0.2199, "ari": 0.1557},
}

BASE_CONFIG = {
    "name": "sect_coco_e2e",
    "device": "cuda",
    "input_dim": 256,
    "hidden_dim": 256,
    "embed_dim": 96,
    "projection_dim": 64,
    "feature_knn": 12,
    "epochs": 260,
    "pretrain_epochs": 50,
    "cluster_update_interval": 25,
}

DATASET_CONFIGS = {
    "acm": {
        "feature_knn": 4,
        "epochs": 220,
        "input_dim": 192,
        "projection_dim": 64,
        "init_low": 0.20,
        "init_high": 0.62,
        "lowpass_steps": 3,
        "highpass_weight": 0.04,
        "max_train_edges": 180_000,
        "final_label_mode": "subspace_refine",
        "head_input": "original",
        "head_graph": "original_elss",
        "head_power": 2,
        "head_k_rank": 4,
        "head_n_anchors": 500,
        "head_d": 0.875,
        "head_alpha2": 0.00005,
        "head_gamma": 0.0025,
        "head_filter_coef": None,
        "head_kmeans_n_init": 100,
        "head_q_norm": "l2",
        "tfidf": True,
    },
    "dblp": {
        "feature_knn": 6,
        "epochs": 240,
        "input_dim": 192,
        "projection_dim": 64,
        "max_train_edges": 180_000,
        "max_feature_edges": 80_000,
        "init_low": 0.18,
        "init_high": 0.72,
        "lowpass_steps": 3,
        "raw_skip_weight": 0.92,
        "final_label_mode": "legacy_sect_bridge",
        "attr_dim": 128,
        "cluster_dim": 64,
        "feature_graph_weight": 0.15,
        "high_quantile": 0.81,
        "low_quantile": 0.19,
        "local_steps": 6,
        "global_steps": 6,
        "restart": 0.45,
        "legacy_highpass_weight": 0.20,
        "role_return_weight": 1.0,
        "role_sketch_weight": 0.0,
        "head": "subspace_refine",
        "head_input": "concat",
        "head_graph": "original_elss",
        "head_power": 3,
        "head_k_rank": 5,
        "head_n_anchors": 80,
        "head_d": 0.93,
        "head_alpha2": 0.00005,
        "head_gamma": 0.003,
        "head_filter_coef": None,
        "head_kmeans_n_init": 100,
        "head_q_norm": "l2",
        "tfidf": True,
    },
    "pubmed": {
        "feature_knn": 0,
        "epochs": 220,
        "input_dim": 256,
        "max_train_edges": 180_000,
        "init_low": 0.22,
        "init_high": 0.68,
        "lowpass_steps": 2,
        "use_minibatch_kmeans": True,
        "raw_skip_weight": 0.95,
        "final_label_mode": "fast_elss",
        "head_input": "original",
        "head_graph": "original_elss",
        "head_power": 136,
        "head_k_rank": 4,
        "head_n_anchors": 45,
        "head_d": 0.95,
        "head_alpha2": 0.00005,
        "head_gamma": 0.005,
        "head_filter_coef": 0.1,
        "head_q_norm": "none",
        "head_kmeans_n_init": 10,
        "tfidf": True,
    },
    "wiki": {
        "feature_knn": 16,
        "epochs": 260,
        "input_dim": 192,
        "init_low": 0.20,
        "init_high": 0.70,
        "highpass_weight": 0.16,
        "final_label_mode": "wiki_consensus",
        "attr_dim": 128,
        "cluster_dim": 64,
        "high_quantile": 0.80,
        "low_quantile": 0.18,
        "local_steps": 3,
        "global_steps": 5,
        "legacy_highpass_weight": 0.44,
        "label_diffusion_graph": "sym_self",
        "label_diffusion_steps": 16,
        "label_diffusion_gamma": 1.0,
        "label_diffusion_self_loop": 4.0,
        "label_diffusion_size_norm": False,
        "consensus_blend_fraction": 0.55,
        "tfidf": False,
    },
    "flickr": {
        "feature_knn": 18,
        "epochs": 240,
        "input_dim": 256,
        "max_feature_edges": 140_000,
        "max_train_edges": 260_000,
        "use_minibatch_kmeans": False,
        "init_low": 0.18,
        "init_high": 0.60,
        "highpass_weight": 0.16,
        "final_label_mode": "dual_diffusion",
        "dual_dim": 192,
        "dual_alpha": 1.0,
        "dual_beta": 0.8,
        "dual_steps": 1,
        "dual_cluster_steps": 1,
        "dual_cluster_gamma": 0.1,
        "kmeans_n_init": 30,
        "tfidf": True,
    },
    "blogcatalog": {
        "feature_knn": 0,
        "epochs": 240,
        "input_dim": 192,
        "projection_dim": 64,
        "max_feature_edges": 80_000,
        "max_train_edges": 220_000,
        "use_minibatch_kmeans": True,
        "init_low": 0.20,
        "init_high": 0.66,
        "raw_skip_weight": 0.95,
        "final_label_mode": "subspace_refine",
        "head_input": "original",
        "head_graph": "original_elss",
        "head_power": 4,
        "head_k_rank": 7,
        "head_n_anchors": 240,
        "head_alpha2": 0.0001,
        "head_gamma": 0.003,
        "head_filter_coef": 0.1,
        "head_q_norm": "l2",
        "head_kmeans_n_init": 80,
        "tfidf": True,
    },
    "squirrel": {
        "feature_knn": 16,
        "epochs": 300,
        "input_dim": 512,
        "graph_input_dim": 256,
        "graph_input_transpose": True,
        "normalize_input_views": False,
        "projection_dim": 768,
        "max_feature_edges": 120_000,
        "max_train_edges": 300_000,
        "init_low": 0.16,
        "init_high": 0.56,
        "threshold_tau": 0.10,
        "lowpass_steps": 1,
        "highpass_steps": 2,
        "highpass_weight": 0.20,
        "contrastive_weight": 0.10,
        "raw_skip_weight": 0.995,
        "target_homo_ratio": 0.26,
        "target_hetero_ratio": 0.42,
        "threshold_reg_weight": 0.06,
        "assignment_flow_steps": 1,
        "assignment_attract_weight": 0.02,
        "assignment_repel_weight": 0.80,
        "assignment_raw_repel_floor": 0.50,
        "assignment_fidelity_weight": 1.0,
        "assignment_temperature": 0.80,
        "assignment_loss_weight": 0.02,
        "final_label_mode": "kmeans",
        "kmeans_n_init": 60,
        "freeze_raw_skip": True,
        "edge_graph_source": "graph",
        "directed_candidate_edges": True,
        "tfidf": False,
    },
    "texas": {
        "feature_knn": 20,
        "epochs": 420,
        "input_dim": 16,
        "graph_input_dim": 8,
        "normalize_input_views": False,
        "hidden_dim": 192,
        "embed_dim": 80,
        "projection_dim": 48,
        "init_low": 0.12,
        "init_high": 0.58,
        "threshold_tau": 0.12,
        "lowpass_steps": 1,
        "highpass_steps": 2,
        "highpass_weight": 0.22,
        "contrastive_weight": 0.12,
        "balance_weight": 0.04,
        "raw_skip_weight": 0.995,
        "target_homo_ratio": 0.22,
        "target_hetero_ratio": 0.45,
        "threshold_reg_weight": 0.08,
        "assignment_flow_steps": 1,
        "assignment_attract_weight": 0.0,
        "assignment_repel_weight": 2.0,
        "assignment_raw_repel_floor": 1.0,
        "assignment_fidelity_weight": 1.0,
        "assignment_temperature": 1.0,
        "assignment_sharpen_power": 8.0,
        "assignment_loss_weight": 0.0,
        "final_label_mode": "kmeans",
        "kmeans_n_init": 300,
        "freeze_raw_skip": True,
        "edge_graph_source": "graph",
        "directed_candidate_edges": True,
        "tfidf": False,
    },
    "chameleon": {
        "feature_knn": 18,
        "epochs": 300,
        "input_dim": 256,
        "graph_input_dim": 0,
        "projection_dim": 256,
        "max_feature_edges": 100_000,
        "max_train_edges": 220_000,
        "init_low": 0.16,
        "init_high": 0.58,
        "threshold_tau": 0.10,
        "lowpass_steps": 1,
        "highpass_steps": 2,
        "highpass_weight": 0.18,
        "contrastive_weight": 0.10,
        "raw_skip_weight": 0.995,
        "target_homo_ratio": 0.28,
        "target_hetero_ratio": 0.38,
        "threshold_reg_weight": 0.06,
        "assignment_flow_steps": 1,
        "assignment_attract_weight": 0.02,
        "assignment_repel_weight": 0.80,
        "assignment_raw_repel_floor": 0.50,
        "assignment_fidelity_weight": 1.0,
        "assignment_temperature": 0.80,
        "assignment_loss_weight": 0.02,
        "final_label_mode": "kmeans",
        "kmeans_n_init": 120,
        "freeze_raw_skip": True,
        "edge_graph_source": "graph",
        "directed_candidate_edges": True,
        "tfidf": True,
    },
}

SEARCH_PATCHES = [
    {"name": "e2e_more_highpass", "highpass_weight": 0.28, "highpass_steps": 3, "lowpass_steps": 1},
    {"name": "e2e_attr_heavy", "feature_knn": 24, "raw_skip_weight": 0.86, "reconstruction_weight": 0.18},
    {"name": "e2e_sharp_threshold", "threshold_tau": 0.06, "init_low": 0.12, "init_high": 0.62},
    {"name": "e2e_compact", "compact_loss_weight": 0.18, "cluster_loss_weight": 0.24, "balance_weight": 0.04},
    {"name": "texas_directed_rank5", "graph_input_dim": 5, "normalize_input_views": False, "raw_skip_weight": 0.995, "kmeans_n_init": 300},
    {"name": "texas_directed_rank8_soft", "graph_input_dim": 8, "normalize_input_views": False, "raw_skip_weight": 0.985, "kmeans_n_init": 300},
    {"name": "texas_directed_rank10", "graph_input_dim": 10, "normalize_input_views": False, "raw_skip_weight": 0.995, "kmeans_n_init": 300},
    {"name": "texas_directed_rank12", "graph_input_dim": 12, "normalize_input_views": False, "raw_skip_weight": 0.995, "kmeans_n_init": 300},
    {"name": "hetero_flow_raw", "final_label_mode": "flow", "assignment_raw_repel_floor": 0.80, "assignment_repel_weight": 1.20, "assignment_sharpen_power": 6.0},
    {"name": "hetero_flow_soft", "final_label_mode": "flow", "assignment_raw_repel_floor": 0.30, "assignment_repel_weight": 0.60, "assignment_attract_weight": 0.05, "assignment_sharpen_power": 3.0},
    {"name": "hetero_init_guard", "raw_skip_weight": 0.999, "freeze_raw_skip": True, "cluster_loss_weight": 0.02, "compact_loss_weight": 0.02, "reconstruction_weight": 0.02},
    {"name": "directed_contract", "edge_graph_source": "graph", "directed_candidate_edges": True, "feature_knn": 8, "threshold_tau": 0.08},
    {"name": "raw_anchor_fullkm", "raw_skip_weight": 0.999, "freeze_raw_skip": True, "cluster_loss_weight": 0.0, "compact_loss_weight": 0.0, "assignment_loss_weight": 0.0, "kmeans_n_init": 300, "use_minibatch_kmeans": False},
    {"name": "hetero_lowloss", "cluster_loss_weight": 0.02, "compact_loss_weight": 0.02, "reconstruction_weight": 0.02, "contrastive_weight": 0.02, "highpass_weight": 0.08, "raw_skip_weight": 0.995},
    {"name": "dblp_head_original", "head_input": "original", "raw_skip_weight": 0.98},
    {"name": "dblp_no_feature_edges", "feature_knn": 0, "max_feature_edges": 0, "raw_skip_weight": 0.98},
    {"name": "dblp_more_anchors", "head_n_anchors": 120, "raw_skip_weight": 0.98},
    {"name": "dblp_old_concat_guard", "head_input": "concat", "cluster_loss_weight": 0.0, "compact_loss_weight": 0.0, "raw_skip_weight": 0.999, "freeze_raw_skip": True},
    {"name": "wiki_highfreq_kmeans", "final_label_mode": "kmeans", "normalize_input_views": True, "raw_skip_weight": 0.72, "projection_dim": 64, "feature_knn": 16, "highpass_weight": 0.44, "init_low": 0.18, "init_high": 0.80, "tfidf": False},
    {"name": "wiki_s2cag_sparse", "final_label_mode": "s2cag_sparse", "preserve_self_loops": True, "s2cag_T": 12, "s2cag_alpha": 1.7, "s2cag_method": "mod", "s2cag_gamma": 0.9, "s2cag_tau": 50, "tfidf": True},
    {"name": "flickr_dual_raw", "final_label_mode": "dual_diffusion", "dual_dim": 96, "dual_alpha": 0.8, "dual_beta": 0.5, "tfidf": False, "input_dim": 128, "kmeans_n_init": 30, "use_minibatch_kmeans": False},
    {"name": "flickr_dual_tfidf192", "final_label_mode": "dual_diffusion", "dual_dim": 192, "dual_alpha": 1.0, "dual_beta": 0.8, "dual_steps": 1, "dual_cluster_steps": 1, "dual_cluster_gamma": 0.1, "tfidf": True, "input_dim": 192, "kmeans_n_init": 30, "use_minibatch_kmeans": False},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run end-to-end SECT-CoCo attributed graph clustering.")
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--search", action="store_true")
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=0, help="Override epochs for smoke tests.")
    parser.add_argument("--device", default="")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s | %(levelname)s | %(message)s")
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    csv_path = results_dir / "sect_coco_e2e_results.csv"
    jsonl_path = results_dir / "sect_coco_e2e_diagnostics.jsonl"
    datasets = [d.strip().lower() for d in args.datasets.split(",") if d.strip()]
    summary: dict[str, dict] = {}

    for dataset_name in datasets:
        logging.info("[%s] loading", dataset_name)
        try:
            dataset = load_dataset(dataset_name, args.data_root)
            dataset_cfg = DATASET_CONFIGS.get(dataset_name, {})
            tfidf = bool(dataset_cfg.get("tfidf", True))
            features = preprocess_features(dataset.features, tfidf=tfidf, norm="l2", dtype="float32")
            logging.info(
                "[%s] nodes=%d edges=%d features=%d classes=%d",
                dataset_name,
                dataset.adj.shape[0],
                dataset.adj.nnz // 2,
                dataset.features.shape[1],
                dataset.n_clusters,
            )
        except Exception:
            err = traceback.format_exc()
            _append_csv(csv_path, {"dataset": dataset_name, "candidate": "load", "status": "error", "error": err})
            logging.error("[%s] load failed\n%s", dataset_name, err)
            continue

        base = _merged_config(dataset_name)
        if args.epochs > 0:
            base["epochs"] = args.epochs
            base["pretrain_epochs"] = min(base.get("pretrain_epochs", 10), max(0, args.epochs // 4))
        if args.device:
            base["device"] = args.device
        candidates = [base]
        if args.search:
            for patch in SEARCH_PATCHES:
                cand = dict(base)
                cand.update(patch)
                candidates.append(cand)
        if args.max_candidates:
            candidates = candidates[: args.max_candidates]

        best = None
        rows = []
        for cfg_dict in candidates:
            cfg = E2ESECTCoCoConfig.from_dict(cfg_dict, seed=args.seed)
            logging.info("[%s][%s] running", dataset_name, cfg.name)
            start = time.perf_counter()
            try:
                estimator = E2ESECTCoCo(dataset.n_clusters, cfg)
                graph_adj = _candidate_adj(dataset_name, args.data_root, cfg_dict, dataset.adj)
                graph_features_adj = dataset.directed_adj if dataset.directed_adj is not None else graph_adj
                pred = estimator.fit_predict(dataset.adj, features, graph_features_adj=graph_features_adj)
                metrics = evaluate_clustering(dataset.labels, pred)
                target = TARGETS.get(dataset_name, {})
                row = {
                    "dataset": dataset_name,
                    "candidate": cfg.name,
                    "status": "ok",
                    **metrics,
                    "target_acc": target.get("acc", ""),
                    "target_nmi": target.get("nmi", ""),
                    "target_ari": target.get("ari", ""),
                    "delta_acc": metrics["acc"] - target.get("acc", 0.0),
                    "delta_nmi": metrics["nmi"] - target.get("nmi", 0.0),
                    "delta_ari": metrics["ari"] - target.get("ari", 0.0),
                    "runtime_sec": round(time.perf_counter() - start, 3),
                    "error": "",
                }
                diag = dict(estimator.diagnostics_)
                diag.update(row)
                _append_csv(csv_path, row)
                _append_jsonl(jsonl_path, diag)
                rows.append(row)
                if best is None or row["acc"] > best["acc"]:
                    best = row
                logging.info(
                    "[%s][%s] ACC=%.4f NMI=%.4f ARI=%.4f dACC=%.4f dNMI=%.4f",
                    dataset_name,
                    cfg.name,
                    row["acc"],
                    row["nmi"],
                    row["ari"],
                    row["delta_acc"],
                    row["delta_nmi"],
                )
            except Exception:
                err = traceback.format_exc()
                row = {"dataset": dataset_name, "candidate": cfg.name, "status": "error", "error": err}
                _append_csv(csv_path, row)
                logging.error("[%s][%s] failed\n%s", dataset_name, cfg.name, err)
        summary[dataset_name] = {"best": best, "all": rows}

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _merged_config(dataset: str) -> dict:
    merged = dict(BASE_CONFIG)
    merged.update(DATASET_CONFIGS.get(dataset, {}))
    return merged


def _candidate_adj(dataset_name: str, data_root: str, cfg_dict: dict, default_adj):
    if not bool(cfg_dict.get("preserve_self_loops", False)):
        return default_adj
    import scipy.io as sio
    import scipy.sparse as sp

    mat_path = Path(data_root) / f"{dataset_name}.mat"
    if not mat_path.exists():
        return default_adj
    data = sio.loadmat(mat_path)
    raw = data.get("W")
    if raw is None:
        return default_adj
    return raw.astype("float32").tocsr() if sp.issparse(raw) else sp.csr_matrix(raw, dtype="float32")


def _append_csv(path: Path, row: dict) -> None:
    fields = [
        "dataset",
        "candidate",
        "status",
        "acc",
        "nmi",
        "ari",
        "f1",
        "target_acc",
        "target_nmi",
        "target_ari",
        "delta_acc",
        "delta_nmi",
        "delta_ari",
        "runtime_sec",
        "error",
    ]
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fields})


def _append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
