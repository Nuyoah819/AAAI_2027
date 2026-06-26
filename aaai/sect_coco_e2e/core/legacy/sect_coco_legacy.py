from __future__ import annotations

from dataclasses import dataclass, field
import contextlib
import io
import logging
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


LOGGER = logging.getLogger(__name__)


@dataclass
class SECTCoCoConfig:
    seed: int = 42
    device: str = "cuda"
    attr_dim: int = 128
    cluster_dim: int = 64
    feature_knn: int = 18
    feature_graph_weight: float = 0.55
    raw_graph_weight: float = 0.03
    hard_graph_weight: float = 0.06
    self_loop_weight: float = 1.0
    sect_rounds: int = 4
    high_quantile: float = 0.72
    low_quantile: float = 0.28
    threshold_eta: float = 0.50
    min_threshold_gap: float = 0.045
    homo_attr_quantile: float = 0.50
    homo_struct_quantile: float = 0.50
    hete_attr_quantile: float = 0.35
    mismatch_quantile: float = 0.72
    role_steps: int = 4
    role_samples: int = 16
    role_sketch_dim: int = 32
    role_sketch_steps: int = 2
    role_return_weight: float = 1.0
    role_sketch_weight: float = 0.35
    role_degree_weight: float = 0.20
    local_steps: int = 4
    global_steps: int = 6
    restart: float = 0.20
    ppr_alpha: float = 0.15
    highpass_weight: float = 0.35
    raw_lowpass_weight: float = 0.25
    compact_rank: int = 32
    compact_residual: float = 0.75
    compact_iters: int = 6
    compact_sigma: float = 1.0
    anchors: int = 256
    consistency_tau: float = 0.30
    consistency_weight: float = 0.45
    use_minibatch_kmeans: bool = False
    kmeans_n_init: int = 50
    kmeans_max_iter: int = 200
    name: str = "sect_coco_v1"
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, values: dict[str, Any], *, seed: int) -> "SECTCoCoConfig":
        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        payload = {k: v for k, v in values.items() if k in field_names}
        payload.setdefault("seed", seed)
        payload["extras"] = {k: v for k, v in values.items() if k not in field_names}
        return cls(**payload)


class SECTCoCo:
    """Self-evolving edge contraction plus CoCo-style compact consistency clustering."""

    def __init__(self, n_clusters: int, config: SECTCoCoConfig):
        self.n_clusters = int(n_clusters)
        self.config = config
        self.embedding_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.diagnostics_: dict[str, Any] = {}
        self.base_features_: np.ndarray | None = None
        self.input_features_: sp.csr_matrix | None = None
        self.raw_adj_: sp.csr_matrix | None = None
        self.denoised_adj_: sp.csr_matrix | None = None
        self.homo_graph_: sp.csr_matrix | None = None

    def fit_predict(self, adj: sp.spmatrix, features: sp.spmatrix | np.ndarray) -> np.ndarray:
        _set_seed(self.config.seed)
        t0 = time.perf_counter()
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)
        device = _resolve_device(cfg.device)
        adj = _as_csr(adj)
        features = _as_csr(features, dtype=np.float64 if cfg.extras.get("head") == "fast_elss" else np.float32)
        self.input_features_ = features
        self.raw_adj_ = adj
        x = _svd_features(features, cfg.attr_dim, cfg.seed, device)
        self.base_features_ = x
        base_norm = _sym_normalize(adj, add_self_loops=True)
        role = _role_features(adj, cfg, rng, device)
        rows, cols = _upper_edges(adj)

        h = float(cfg.high_quantile)
        l = float(cfg.low_quantile)
        homo = adj.copy()
        hetero = sp.csr_matrix(adj.shape, dtype=np.float32)
        hard = sp.csr_matrix(adj.shape, dtype=np.float32)
        round_diag: list[dict[str, float]] = []
        score_payload: dict[str, np.ndarray] | None = None
        for r in range(max(1, int(cfg.sect_rounds))):
            scores = _score_edges(x, role, rows, cols, h, l, cfg, rng)
            homo, hetero, hard, diag = _split_edges(adj.shape[0], rows, cols, scores, cfg)
            score_payload = scores
            round_diag.append({"round": float(r), **diag})
            x = _update_edge_teacher_features(adj, homo, hetero, x, cfg, device)
            h, l = _evolve_thresholds(scores["score"], h, l, cfg)
            if h - l <= cfg.min_threshold_gap:
                break

        feature_graph = _feature_knn_graph(x, cfg.feature_knn, cfg, device)
        denoised = (
            homo
            + cfg.feature_graph_weight * feature_graph
            + cfg.hard_graph_weight * hard
            + cfg.raw_graph_weight * adj
            + cfg.self_loop_weight * sp.eye(adj.shape[0], dtype=np.float32, format="csr")
        ).tocsr()
        denoised.eliminate_zeros()
        self.denoised_adj_ = denoised
        self.homo_graph_ = homo

        local = _personalized_filter(_sym_normalize(denoised, add_self_loops=False), x, cfg.local_steps, cfg.restart, device)
        global_view = _ppr_feature_diffusion(
            _sym_normalize(denoised, add_self_loops=False),
            x,
            cfg.global_steps,
            cfg.ppr_alpha,
            device,
        )
        raw_low = _personalized_filter(base_norm, x, max(1, cfg.local_steps // 2), cfg.restart, device)
        high = _hetero_highpass(hetero, x, device)

        compact_l, compact_g = _compact_reconstruct_pair(local, global_view, cfg)
        consistent = _anchor_consistency_fusion(compact_l, compact_g, cfg, rng)
        fused = np.concatenate(
            [
                0.5 * (compact_l + compact_g),
                cfg.consistency_weight * consistent,
                cfg.raw_lowpass_weight * raw_low,
                cfg.highpass_weight * high,
            ],
            axis=1,
        )
        fused = normalize(np.nan_to_num(fused), norm="l2", axis=1)
        if cfg.cluster_dim > 0 and fused.shape[1] > cfg.cluster_dim:
            fused = _dense_svd(fused, cfg.cluster_dim, cfg.seed, device)
            fused = normalize(np.nan_to_num(fused), norm="l2", axis=1)

        self.embedding_ = fused
        if cfg.extras.get("head") == "subspace_refine":
            self.labels_ = _subspace_refine(self, cfg, device)
        elif cfg.extras.get("head") == "sgc_lowpass":
            self.labels_ = _sgc_lowpass_head(self, cfg, device)
        elif cfg.extras.get("head") == "fast_elss":
            self.labels_ = _fast_elss_head(self, cfg, device)
        else:
            self.labels_ = _cluster(fused, self.n_clusters, cfg)
        if bool(cfg.extras.get("assignment_smooth", False)):
            self.labels_ = _assignment_smooth(self.labels_, self, cfg)
        if bool(cfg.extras.get("assignment_sinkhorn", False)):
            self.labels_ = _assignment_sinkhorn(self.labels_, self, cfg)
        if bool(cfg.extras.get("assignment_spillover", False)):
            self.labels_ = _assignment_spillover(self.labels_, self, cfg)
        if bool(cfg.extras.get("rep_spillover", False)):
            rep_name = str(cfg.extras.get("rep_spillover_representation", "embedding"))
            rep = self.base_features_ if rep_name == "base" and self.base_features_ is not None else self.embedding_
            self.labels_ = _rep_spillover(self.labels_, rep, self.n_clusters, cfg)
        t1 = time.perf_counter()
        last = round_diag[-1] if round_diag else {}
        self.diagnostics_ = {
            "candidate": cfg.name,
            "nodes": int(adj.shape[0]),
            "edges": int(adj.nnz // 2),
            "attr_dim": int(x.shape[1]),
            "cluster_dim": int(fused.shape[1]),
            "sect_rounds_used": len(round_diag),
            "homo_ratio": float(last.get("homo_ratio", 0.0)),
            "hetero_ratio": float(last.get("hetero_ratio", 0.0)),
            "hard_ratio": float(last.get("hard_ratio", 0.0)),
            "score_mean": float(np.mean(score_payload["score"])) if score_payload is not None else 0.0,
            "score_std": float(np.std(score_payload["score"])) if score_payload is not None else 0.0,
            "final_high_quantile": float(h),
            "final_low_quantile": float(l),
            "runtime_sec": round(t1 - t0, 3),
        }
        return self.labels_


def _set_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _resolve_device(name: str) -> torch.device:
    if str(name).startswith("cuda") and torch.cuda.is_available():
        return torch.device(name)
    return torch.device("cpu")


def _as_csr(x, dtype=np.float32) -> sp.csr_matrix:
    if sp.issparse(x):
        return x.astype(dtype).tocsr()
    return sp.csr_matrix(np.asarray(x, dtype=dtype))


def _upper_edges(adj: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    coo = sp.triu(adj, k=1).tocoo()
    return coo.row.astype(np.int64), coo.col.astype(np.int64)


def _svd_features(features: sp.csr_matrix, dim: int, seed: int, device: torch.device) -> np.ndarray:
    dim = int(min(max(2, dim), features.shape[0] - 1, features.shape[1] - 1))
    if dim < 2:
        arr = features.toarray()
    elif device.type == "cuda" and features.shape[0] * features.shape[1] < 90_000_000:
        arr = _dense_svd(features.toarray(), dim, seed, device)
    else:
        arr = TruncatedSVD(n_components=dim, random_state=seed).fit_transform(features)
    return normalize(np.nan_to_num(arr), norm="l2", axis=1).astype(np.float32)


def _dense_svd(x: np.ndarray, dim: int, seed: int, device: torch.device) -> np.ndarray:
    dim = int(min(dim, x.shape[0] - 1, x.shape[1] - 1))
    if dim < 2:
        return np.asarray(x, dtype=np.float32)
    if device.type == "cuda":
        try:
            with torch.no_grad():
                xt = torch.as_tensor(x, dtype=torch.float32, device=device)
                xt = xt - xt.mean(dim=0, keepdim=True)
                _, _, vh = torch.pca_lowrank(xt, q=dim, center=False, niter=4)
                return (xt @ vh[:, :dim]).cpu().numpy().astype(np.float32)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
            else:
                raise
    return TruncatedSVD(n_components=dim, random_state=seed).fit_transform(x).astype(np.float32)


def _sym_normalize(mx: sp.spmatrix, *, add_self_loops: bool) -> sp.csr_matrix:
    mx = mx.astype(np.float32).tocsr()
    if add_self_loops:
        mx = mx + sp.eye(mx.shape[0], dtype=np.float32, format="csr")
    rowsum = np.asarray(mx.sum(axis=1)).reshape(-1)
    inv = np.divide(1.0, np.sqrt(rowsum), out=np.zeros_like(rowsum, dtype=np.float32), where=rowsum > 0)
    d = sp.diags(inv, dtype=np.float32)
    return d.dot(mx).dot(d).tocsr()


def _row_normalize(mx: sp.spmatrix) -> sp.csr_matrix:
    mx = mx.astype(np.float32).tocsr()
    rowsum = np.asarray(mx.sum(axis=1)).reshape(-1)
    inv = np.divide(1.0, rowsum, out=np.zeros_like(rowsum, dtype=np.float32), where=rowsum > 0)
    return sp.diags(inv, dtype=np.float32).dot(mx).tocsr()


def _scipy_to_torch(mx: sp.spmatrix, device: torch.device) -> torch.Tensor:
    coo = mx.tocoo()
    idx = np.vstack([coo.row, coo.col]).astype(np.int64)
    return torch.sparse_coo_tensor(
        torch.from_numpy(idx),
        torch.from_numpy(coo.data.astype(np.float32)),
        size=coo.shape,
        dtype=torch.float32,
        device=device,
    ).coalesce()


def _spmm(mx: sp.spmatrix, x: np.ndarray, device: torch.device) -> np.ndarray:
    if device.type == "cuda":
        try:
            with torch.no_grad():
                out = torch.sparse.mm(_scipy_to_torch(mx, device), torch.as_tensor(x, dtype=torch.float32, device=device))
                return out.cpu().numpy().astype(np.float32)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
            else:
                raise
    return (mx @ x).astype(np.float32)


def _personalized_filter(mx: sp.spmatrix, x0: np.ndarray, steps: int, restart: float, device: torch.device) -> np.ndarray:
    h = np.asarray(x0, dtype=np.float32).copy()
    x0 = h.copy()
    restart = float(np.clip(restart, 0.0, 1.0))
    for _ in range(max(1, int(steps))):
        h = (1.0 - restart) * _spmm(mx, h, device) + restart * x0
    return normalize(np.nan_to_num(h), norm="l2", axis=1).astype(np.float32)


def _diffusion_approx(mx: sp.spmatrix, steps: int, alpha: float) -> sp.csr_matrix:
    out = alpha * sp.eye(mx.shape[0], dtype=np.float32, format="csr")
    power = sp.eye(mx.shape[0], dtype=np.float32, format="csr")
    coeff = 1.0
    for _ in range(max(1, int(steps))):
        power = power.dot(mx).tocsr()
        coeff *= 1.0 - float(alpha)
        out = out + float(alpha * coeff) * power
    out.eliminate_zeros()
    return _sym_normalize(out.maximum(out.T), add_self_loops=False)


def _ppr_feature_diffusion(mx: sp.spmatrix, x0: np.ndarray, steps: int, alpha: float, device: torch.device) -> np.ndarray:
    """Sparse PPR view without materializing the dense-ish diffusion matrix."""
    h = np.asarray(x0, dtype=np.float32).copy()
    acc = float(alpha) * h
    coeff = 1.0
    for _ in range(max(1, int(steps))):
        h = _spmm(mx, h, device)
        coeff *= 1.0 - float(alpha)
        acc = acc + float(alpha * coeff) * h
    return normalize(np.nan_to_num(acc), norm="l2", axis=1).astype(np.float32)


def _role_features(adj: sp.csr_matrix, cfg: SECTCoCoConfig, rng: np.random.Generator, device: torch.device) -> np.ndarray:
    pieces: list[np.ndarray] = []
    if cfg.role_return_weight > 0:
        pieces.append(cfg.role_return_weight * _random_walk_returns(adj, cfg.role_steps, cfg.role_samples, rng))
    if cfg.role_sketch_weight > 0:
        pieces.append(cfg.role_sketch_weight * _diffusion_sketch(adj, cfg.role_sketch_dim, cfg.role_sketch_steps, rng))
    if cfg.role_degree_weight > 0:
        deg = np.asarray(adj.sum(axis=1)).reshape(-1).astype(np.float32)
        deg = np.log1p(deg)[:, None]
        pieces.append(cfg.role_degree_weight * normalize(deg, norm="l2", axis=0))
    if not pieces:
        return np.zeros((adj.shape[0], 1), dtype=np.float32)
    return normalize(np.nan_to_num(np.concatenate(pieces, axis=1)), norm="l2", axis=1).astype(np.float32)


def _random_walk_returns(adj: sp.csr_matrix, steps: int, samples: int, rng: np.random.Generator) -> np.ndarray:
    indptr, indices = adj.indptr, adj.indices
    n = adj.shape[0]
    out = np.zeros((n, max(1, steps)), dtype=np.float32)
    active = np.flatnonzero(np.diff(indptr) > 0)
    if active.size == 0:
        return out
    for node in active:
        for _ in range(max(1, samples)):
            cur = int(node)
            for s in range(max(1, steps)):
                neigh = indices[indptr[cur] : indptr[cur + 1]]
                if neigh.size == 0:
                    break
                cur = int(neigh[rng.integers(0, neigh.size)])
                if cur == node:
                    out[node, s] += 1.0
    out /= float(max(1, samples))
    return out


def _diffusion_sketch(adj: sp.csr_matrix, dim: int, steps: int, rng: np.random.Generator) -> np.ndarray:
    n = adj.shape[0]
    dim = max(2, int(dim))
    probe = rng.normal(0.0, 1.0 / math.sqrt(dim), size=(n, dim)).astype(np.float32)
    norm = _row_normalize(adj + sp.eye(n, dtype=np.float32, format="csr"))
    h = probe.copy()
    outs = [h]
    for _ in range(max(1, int(steps))):
        h = norm @ h
        outs.append(h.astype(np.float32))
    return normalize(np.nan_to_num(np.mean(outs, axis=0)), norm="l2", axis=1).astype(np.float32)


def _score_edges(
    x: np.ndarray,
    role: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    high_q: float,
    low_q: float,
    cfg: SECTCoCoConfig,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    attr = np.sum(x[rows] * x[cols], axis=1)
    struct = np.sum(role[rows] * role[cols], axis=1)
    attr01 = 0.5 * (attr + 1.0)
    struct01 = 0.5 * (struct + 1.0)
    mismatch = np.abs(attr01 - struct01)
    score = 0.55 * attr01 + 0.35 * struct01 + 0.10 * (1.0 - mismatch)
    score = np.clip(score, 1e-6, 1.0 - 1e-6).astype(np.float32)
    return {
        "score": score,
        "attr": attr01.astype(np.float32),
        "struct": struct01.astype(np.float32),
        "mismatch": mismatch.astype(np.float32),
        "high_thr": np.asarray(np.quantile(score, np.clip(high_q, 0.0, 1.0)), dtype=np.float32),
        "low_thr": np.asarray(np.quantile(score, np.clip(low_q, 0.0, 1.0)), dtype=np.float32),
    }


def _split_edges(
    n: int,
    rows: np.ndarray,
    cols: np.ndarray,
    scores: dict[str, np.ndarray],
    cfg: SECTCoCoConfig,
) -> tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix, dict[str, float]]:
    score, attr, struct, mismatch = scores["score"], scores["attr"], scores["struct"], scores["mismatch"]
    hi, lo = float(scores["high_thr"]), float(scores["low_thr"])
    attr_hi = float(np.quantile(attr, cfg.homo_attr_quantile))
    struct_hi = float(np.quantile(struct, cfg.homo_struct_quantile))
    attr_lo = float(np.quantile(attr, cfg.hete_attr_quantile))
    mismatch_hi = float(np.quantile(mismatch, cfg.mismatch_quantile))
    homo_mask = (score >= hi) & (attr >= attr_hi) & (struct >= struct_hi)
    hetero_mask = ((score <= lo) | ((attr <= attr_lo) & (mismatch >= mismatch_hi))) & ~homo_mask
    hard_mask = ~(homo_mask | hetero_mask)
    homo = _graph_from_mask(n, rows, cols, np.clip(score, 0.0, 1.0), homo_mask)
    hetero = _graph_from_mask(n, rows, cols, np.ones_like(score), hetero_mask)
    hard = _graph_from_mask(n, rows, cols, np.clip(score, 0.0, 1.0), hard_mask)
    m = max(1, score.size)
    return homo, hetero, hard, {
        "high_thr": hi,
        "low_thr": lo,
        "homo_ratio": float(homo_mask.sum() / m),
        "hetero_ratio": float(hetero_mask.sum() / m),
        "hard_ratio": float(hard_mask.sum() / m),
    }


def _graph_from_mask(n: int, rows: np.ndarray, cols: np.ndarray, values: np.ndarray, mask: np.ndarray) -> sp.csr_matrix:
    g = sp.coo_matrix((values[mask], (rows[mask], cols[mask])), shape=(n, n), dtype=np.float32)
    g = (g + g.T).tocsr()
    g.eliminate_zeros()
    return g


def _update_edge_teacher_features(adj: sp.csr_matrix, homo: sp.csr_matrix, hetero: sp.csr_matrix, x: np.ndarray, cfg: SECTCoCoConfig, device: torch.device) -> np.ndarray:
    clean = homo + cfg.hard_graph_weight * (adj - homo - hetero).maximum(0)
    clean = clean + sp.eye(adj.shape[0], dtype=np.float32, format="csr")
    low = _personalized_filter(_sym_normalize(clean, add_self_loops=False), x, 2, cfg.restart, device)
    high = _hetero_highpass(hetero, x, device)
    return normalize(np.nan_to_num(low + 0.10 * high), norm="l2", axis=1).astype(np.float32)


def _evolve_thresholds(score: np.ndarray, high_q: float, low_q: float, cfg: SECTCoCoConfig) -> tuple[float, float]:
    spread = float(np.std(score))
    eta = float(np.clip(cfg.threshold_eta * (0.5 + spread), 0.05, 0.90))
    next_h = high_q - eta * (high_q - 0.52)
    next_l = low_q + eta * (0.48 - low_q)
    if next_h - next_l < cfg.min_threshold_gap:
        mid = 0.5 * (next_h + next_l)
        next_h = mid + 0.5 * cfg.min_threshold_gap
        next_l = mid - 0.5 * cfg.min_threshold_gap
    return float(np.clip(next_h, 0.0, 0.98)), float(np.clip(next_l, 0.02, 1.0))


def _hetero_highpass(hetero: sp.csr_matrix, x: np.ndarray, device: torch.device) -> np.ndarray:
    if hetero.nnz == 0:
        return np.zeros_like(x)
    smooth = _spmm(_sym_normalize(hetero, add_self_loops=False), x, device)
    return normalize(np.nan_to_num(x - smooth), norm="l2", axis=1).astype(np.float32)


def _feature_knn_graph(x: np.ndarray, k: int, cfg: SECTCoCoConfig, device: torch.device) -> sp.csr_matrix:
    n = x.shape[0]
    k = int(min(max(1, k), n - 1))
    mode = str(cfg.extras.get("feature_graph_mode", "exact"))
    if mode == "anchor" or n > int(cfg.extras.get("exact_knn_max_nodes", 15000)):
        return _anchor_feature_graph(x, k, int(cfg.extras.get("feature_graph_anchors", 4096)), cfg.seed, device)
    return _exact_knn_graph(x, k, device)


def _exact_knn_graph(x: np.ndarray, k: int, device: torch.device) -> sp.csr_matrix:
    n = x.shape[0]
    block = max(128, min(2048, int(160_000_000 / max(1, n * 4))))
    target_device = device if device.type == "cuda" else torch.device("cpu")
    try:
        xt = F.normalize(torch.as_tensor(x, dtype=torch.float32, device=target_device), p=2, dim=1)
        rows_all: list[np.ndarray] = []
        cols_all: list[np.ndarray] = []
        vals_all: list[np.ndarray] = []
        with torch.no_grad():
            xt_t = xt.t().contiguous()
            for s in range(0, n, block):
                e = min(n, s + block)
                sim = xt[s:e] @ xt_t
                sim[torch.arange(e - s, device=target_device), torch.arange(s, e, device=target_device)] = -1.0
                vals, idx = torch.topk(sim, k=k, dim=1, largest=True, sorted=False)
                rr = torch.arange(s, e, device=target_device).unsqueeze(1).expand(-1, k).reshape(-1)
                vals = torch.clamp(vals.reshape(-1), min=0.0)
                idx = idx.reshape(-1)
                keep = vals > 0
                rows_all.append(rr[keep].cpu().numpy())
                cols_all.append(idx[keep].cpu().numpy())
                vals_all.append(vals[keep].cpu().numpy().astype(np.float32))
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower() or target_device.type == "cpu":
            raise
        torch.cuda.empty_cache()
        return _exact_knn_graph(x, k, torch.device("cpu"))
    return _make_symmetric_knn(n, rows_all, cols_all, vals_all)


def _anchor_feature_graph(x: np.ndarray, k: int, anchors: int, seed: int, device: torch.device) -> sp.csr_matrix:
    n = x.shape[0]
    rng = np.random.default_rng(seed)
    anchors = int(min(max(k + 1, anchors), n))
    anchor_idx = np.sort(rng.choice(n, anchors, replace=False)).astype(np.int64)
    block = 4096

    def build(dev: torch.device) -> sp.csr_matrix:
        xt = F.normalize(torch.as_tensor(x, dtype=torch.float32, device=dev), p=2, dim=1)
        at = xt[torch.as_tensor(anchor_idx, dtype=torch.long, device=dev)].t().contiguous()
        rows_all: list[np.ndarray] = []
        cols_all: list[np.ndarray] = []
        vals_all: list[np.ndarray] = []
        with torch.no_grad():
            for s in range(0, n, block):
                e = min(n, s + block)
                sim = xt[s:e] @ at
                vals, pos = torch.topk(sim, k=min(k, anchors), dim=1, largest=True, sorted=False)
                rr = torch.arange(s, e, device=dev).unsqueeze(1).expand(-1, vals.shape[1]).reshape(-1)
                cc = torch.as_tensor(anchor_idx, dtype=torch.long, device=dev)[pos.reshape(-1)]
                vals = torch.clamp(vals.reshape(-1), min=0.0)
                keep = (vals > 0) & (rr != cc)
                rows_all.append(rr[keep].cpu().numpy())
                cols_all.append(cc[keep].cpu().numpy())
                vals_all.append(vals[keep].cpu().numpy().astype(np.float32))
        return _make_symmetric_knn(n, rows_all, cols_all, vals_all)

    if device.type == "cuda":
        try:
            return build(device)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
            else:
                raise
    return build(torch.device("cpu"))


def _make_symmetric_knn(n: int, rows_all: list[np.ndarray], cols_all: list[np.ndarray], vals_all: list[np.ndarray]) -> sp.csr_matrix:
    if not rows_all:
        return sp.csr_matrix((n, n), dtype=np.float32)
    g = sp.coo_matrix(
        (np.concatenate(vals_all), (np.concatenate(rows_all), np.concatenate(cols_all))),
        shape=(n, n),
        dtype=np.float32,
    )
    g = g.maximum(g.T).tocsr()
    g.eliminate_zeros()
    return g


def _compact_reconstruct_pair(local: np.ndarray, global_view: np.ndarray, cfg: SECTCoCoConfig) -> tuple[np.ndarray, np.ndarray]:
    z = np.vstack([local, global_view]).astype(np.float32)
    z_low = _gmm_column_compact(z, cfg)
    z_tilde = normalize(np.nan_to_num(z_low + cfg.compact_residual * z), norm="l2", axis=1).astype(np.float32)
    n = local.shape[0]
    return z_tilde[:n], z_tilde[n:]


def _gmm_column_compact(z: np.ndarray, cfg: SECTCoCoConfig) -> np.ndarray:
    n2, d = z.shape
    rank = int(min(max(2, cfg.compact_rank), d, n2))
    km = KMeans(n_clusters=rank, n_init=3, random_state=cfg.seed, max_iter=50)
    cols = z.T
    labels = km.fit_predict(cols)
    centers = km.cluster_centers_.T.astype(np.float32)
    sigma = max(float(cfg.compact_sigma), 1e-6)
    for _ in range(max(1, int(cfg.compact_iters))):
        dist = ((cols[:, None, :] - centers.T[None, :, :]) ** 2).mean(axis=2)
        gamma = np.exp(-dist / (2.0 * sigma * sigma))
        gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), 1e-12)
        denom = np.maximum(gamma.sum(axis=0), 1e-12)
        centers = (z @ gamma / denom[None, :]).astype(np.float32)
    recon = centers @ gamma.T
    return recon.astype(np.float32)


def _anchor_consistency_fusion(local: np.ndarray, global_view: np.ndarray, cfg: SECTCoCoConfig, rng: np.random.Generator) -> np.ndarray:
    n = local.shape[0]
    m = int(min(max(cfg.n_clusters if hasattr(cfg, "n_clusters") else 8, cfg.anchors), n))
    anchors = rng.choice(n, size=m, replace=False)
    l = normalize(local, norm="l2", axis=1)
    g = normalize(global_view, norm="l2", axis=1)
    tau = max(float(cfg.consistency_tau), 1e-6)
    p = _softmax((l @ l[anchors].T) / tau)
    q = _softmax((g @ g[anchors].T) / tau)
    shared = 0.5 * (p + q)
    anchor_repr = 0.5 * (l[anchors] + g[anchors])
    return normalize(np.nan_to_num(shared @ anchor_repr), norm="l2", axis=1).astype(np.float32)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=1, keepdims=True)
    ex = np.exp(x)
    return (ex / np.maximum(ex.sum(axis=1, keepdims=True), 1e-12)).astype(np.float32)


def _cluster(z: np.ndarray, n_clusters: int, cfg: SECTCoCoConfig) -> np.ndarray:
    if cfg.use_minibatch_kmeans:
        return MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=int(cfg.extras.get("kmeans_batch_size", 8192)),
            n_init=int(cfg.kmeans_n_init),
            max_iter=int(cfg.kmeans_max_iter),
            random_state=cfg.seed,
            reassignment_ratio=0.0,
        ).fit_predict(z)
    return KMeans(
        n_clusters=n_clusters,
        n_init=int(cfg.kmeans_n_init),
        max_iter=int(cfg.kmeans_max_iter),
        random_state=cfg.seed,
    ).fit_predict(z)


def _subspace_refine(model: SECTCoCo, cfg: SECTCoCoConfig, device: torch.device) -> np.ndarray:
    if model.embedding_ is None or model.base_features_ is None or model.raw_adj_ is None or model.denoised_adj_ is None:
        raise RuntimeError("SECT-CoCo artifacts are incomplete; cannot run subspace head.")
    head_input = str(cfg.extras.get("head_input", "embedding"))
    if head_input == "original":
        x = model.input_features_.toarray() if model.input_features_ is not None else model.embedding_
    elif head_input == "base":
        x = model.base_features_
    elif head_input == "concat":
        x = np.concatenate([model.base_features_, model.embedding_], axis=1)
    else:
        x = model.embedding_
    x = np.asarray(x, dtype=np.float64)
    svd_dim = int(cfg.extras.get("head_input_svd_dim", 0))
    if svd_dim > 0 and x.shape[1] > svd_dim:
        x = _dense_svd(x.astype(np.float32), svd_dim, cfg.seed, device).astype(np.float64)
        x = normalize(np.nan_to_num(x), norm="l2", axis=1)

    graph_name = str(cfg.extras.get("head_graph", "denoised_elss"))
    if graph_name == "original_elss":
        graph = model.raw_adj_
    elif graph_name == "homo_elss":
        graph = model.homo_graph_ if model.homo_graph_ is not None else model.denoised_adj_
    else:
        graph = model.denoised_adj_
    norm_adj = _elss_row_normalize(_elss_sym_normalize(graph, add_self_loops=True), add_self_loops=True)
    norm_adj_t = _scipy_to_torch64(norm_adj, device)
    x_t = torch.as_tensor(x, dtype=torch.float64, device=device)

    n_anchors = int(cfg.extras.get("head_n_anchors", min(300, x.shape[0] - 1)))
    n_anchors = max(model.n_clusters + 2, min(n_anchors, x.shape[0] - 1))
    head = _AnchorSubspaceHead(
        n_clusters=model.n_clusters,
        k_rank=int(cfg.extras.get("head_k_rank", model.n_clusters + 1)),
        n_anchors=n_anchors,
        power=int(cfg.extras.get("head_power", 2)),
        d=float(cfg.extras.get("head_d", 0.85)),
        alpha2=float(cfg.extras.get("head_alpha2", 5e-5)),
        gamma=float(cfg.extras.get("head_gamma", 0.005)),
        filter_coef=cfg.extras.get("head_filter_coef", 0.1),
        return_k_rank=bool(cfg.extras.get("head_return_k_rank", False)),
        seed=int(cfg.seed),
        device=device,
    )
    try:
        q = head.fit_transform(x_t, norm_adj_t)
    except RuntimeError as exc:
        if device.type == "cuda" and ("out of memory" in str(exc).lower() or "cuda" in str(exc).lower()):
            torch.cuda.empty_cache()
            cpu = torch.device("cpu")
            head.device = cpu
            q = head.fit_transform(torch.as_tensor(x, dtype=torch.float64), _scipy_to_torch64(norm_adj, cpu))
        else:
            raise
    q_np = q.detach().cpu().numpy()
    mode = str(cfg.extras.get("head_q_norm", "none"))
    if mode == "l2":
        q_np = normalize(np.nan_to_num(q_np), norm="l2", axis=1)
    elif mode.startswith("pca"):
        dim = int(mode.replace("pca", ""))
        q_np = normalize(_dense_svd(q_np.astype(np.float32), dim, cfg.seed, device), norm="l2", axis=1)
    model.embedding_ = q_np.astype(np.float32)
    return _cluster_with_options(
        q_np,
        model.n_clusters,
        seed=cfg.seed,
        n_init=int(cfg.extras.get("head_kmeans_n_init", cfg.kmeans_n_init)),
        max_iter=int(cfg.extras.get("head_kmeans_max_iter", cfg.kmeans_max_iter)),
        use_minibatch=bool(cfg.extras.get("head_minibatch_kmeans", False)),
        batch_size=int(cfg.extras.get("head_kmeans_batch_size", 8192)),
    )


def _sgc_lowpass_head(model: SECTCoCo, cfg: SECTCoCoConfig, device: torch.device) -> np.ndarray:
    if model.base_features_ is None or model.raw_adj_ is None or model.denoised_adj_ is None:
        raise RuntimeError("SECT-CoCo artifacts are incomplete; cannot run SGC low-pass head.")
    source = str(cfg.extras.get("sgc_input", "base"))
    if source == "original" and model.input_features_ is not None:
        dim = int(cfg.extras.get("sgc_input_svd_dim", cfg.attr_dim))
        x = _svd_features(model.input_features_, dim, cfg.seed, device)
    elif source == "embedding" and model.embedding_ is not None:
        x = model.embedding_
    else:
        x = model.base_features_
    graph_name = str(cfg.extras.get("sgc_graph", "raw"))
    if graph_name == "denoised":
        graph = model.denoised_adj_
    elif graph_name == "homo" and model.homo_graph_ is not None:
        graph = model.homo_graph_
    else:
        graph = model.raw_adj_
    mode = str(cfg.extras.get("sgc_norm", "row"))
    if mode == "sym":
        norm = _sym_normalize(graph, add_self_loops=True)
    else:
        norm = _row_normalize(graph + sp.eye(graph.shape[0], dtype=np.float32, format="csr"))
    h0 = np.asarray(x, dtype=np.float32)
    h = h0.copy()
    restart = float(np.clip(cfg.extras.get("sgc_restart", cfg.restart), 0.0, 1.0))
    for _ in range(max(0, int(cfg.extras.get("sgc_steps", 8)))):
        h = (1.0 - restart) * (norm @ h) + restart * h0
    residual = float(np.clip(cfg.extras.get("sgc_residual", 0.0), 0.0, 1.0))
    if residual > 0:
        h = (1.0 - residual) * h + residual * h0
    h = normalize(np.nan_to_num(h), norm="l2", axis=1)
    out_dim = int(cfg.extras.get("sgc_output_svd_dim", 0))
    if out_dim > 0 and h.shape[1] > out_dim:
        h = _dense_svd(h.astype(np.float32), out_dim, cfg.seed, device)
        h = normalize(np.nan_to_num(h), norm="l2", axis=1)
    model.embedding_ = h.astype(np.float32)
    return _cluster_with_options(
        h,
        model.n_clusters,
        seed=cfg.seed,
        n_init=int(cfg.extras.get("sgc_kmeans_n_init", cfg.kmeans_n_init)),
        max_iter=int(cfg.extras.get("sgc_kmeans_max_iter", cfg.kmeans_max_iter)),
        use_minibatch=bool(cfg.extras.get("sgc_minibatch_kmeans", False)),
        batch_size=int(cfg.extras.get("sgc_kmeans_batch_size", 8192)),
    )


def _fast_elss_head(model: SECTCoCo, cfg: SECTCoCoConfig, device: torch.device) -> np.ndarray:
    if model.input_features_ is None or model.raw_adj_ is None or model.denoised_adj_ is None:
        raise RuntimeError("SECT-CoCo artifacts are incomplete; cannot run fast ELSS head.")
    x = model.input_features_.toarray().astype(np.float64)
    graph_name = str(cfg.extras.get("fast_elss_graph", "raw"))
    if graph_name == "denoised":
        graph = model.denoised_adj_
    elif graph_name == "homo" and model.homo_graph_ is not None:
        graph = model.homo_graph_
    else:
        graph = model.raw_adj_
    norm_adj = _elss_row_normalize(_elss_sym_normalize(graph, add_self_loops=True), add_self_loops=True)
    if bool(cfg.extras.get("fast_elss_external", True)):
        labels, q_np = _external_fast_elss_fit_predict(x, norm_adj, model.n_clusters, cfg, device)
        model.embedding_ = q_np.astype(np.float32)
        return labels
    norm_adj_t = _scipy_to_torch64(norm_adj, device)
    x_t = torch.as_tensor(x, dtype=torch.float64, device=device)
    head = _FastELSSHead(
        n_clusters=model.n_clusters,
        k_rank=int(cfg.extras.get("fast_elss_k_rank", cfg.extras.get("head_k_rank", model.n_clusters + 1))),
        n_anchors=int(cfg.extras.get("fast_elss_n_anchors", cfg.extras.get("head_n_anchors", 50))),
        power=int(cfg.extras.get("fast_elss_power", cfg.extras.get("head_power", 136))),
        d=float(cfg.extras.get("fast_elss_d", cfg.extras.get("head_d", 0.955))),
        alpha2=float(cfg.extras.get("fast_elss_alpha2", cfg.extras.get("head_alpha2", 5e-5))),
        gamma=float(cfg.extras.get("fast_elss_gamma", cfg.extras.get("head_gamma", 0.005))),
        seed=int(cfg.seed),
        device=device,
    )
    try:
        q_np = head.fit_transform(x_t, norm_adj_t)
    except RuntimeError as exc:
        if device.type == "cuda" and ("out of memory" in str(exc).lower() or "cuda" in str(exc).lower()):
            torch.cuda.empty_cache()
            cpu = torch.device("cpu")
            head.device = cpu
            q_np = head.fit_transform(torch.as_tensor(x, dtype=torch.float64), _scipy_to_torch64(norm_adj, cpu))
        else:
            raise
    mode = str(cfg.extras.get("fast_elss_q_norm", cfg.extras.get("head_q_norm", "none")))
    if mode == "l2":
        q_np = normalize(np.nan_to_num(q_np), norm="l2", axis=1)
    model.embedding_ = q_np.astype(np.float32)
    return _cluster_with_options(
        q_np,
        model.n_clusters,
        seed=cfg.seed,
        n_init=int(cfg.extras.get("fast_elss_kmeans_n_init", cfg.extras.get("head_kmeans_n_init", cfg.kmeans_n_init))),
        max_iter=int(cfg.extras.get("fast_elss_kmeans_max_iter", cfg.kmeans_max_iter)),
        use_minibatch=bool(cfg.extras.get("fast_elss_minibatch_kmeans", False)),
        batch_size=int(cfg.extras.get("fast_elss_kmeans_batch_size", 8192)),
    )


def _external_fast_elss_fit_predict(
    x: np.ndarray,
    norm_adj: sp.spmatrix,
    n_clusters: int,
    cfg: SECTCoCoConfig,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    elss_dir = Path(r"D:\study\graduate_student\papers\AAAI2027\AAAI0610\ELSS_code")
    if not elss_dir.exists():
        raise FileNotFoundError(f"ELSS_code directory not found: {elss_dir}")
    if str(elss_dir) not in sys.path:
        sys.path.insert(0, str(elss_dir))
    from data_load import convert_sparse_matrix_to_sparse_tensor  # type: ignore
    from model import FastGraphSubClustering  # type: ignore

    head = FastGraphSubClustering(
        n_clusters=n_clusters,
        alpha1=0.0,
        k_rank=int(cfg.extras.get("fast_elss_k_rank", cfg.extras.get("head_k_rank", n_clusters + 1))),
        n_anchors=int(cfg.extras.get("fast_elss_n_anchors", cfg.extras.get("head_n_anchors", 50))),
        power=int(cfg.extras.get("fast_elss_power", cfg.extras.get("head_power", 136))),
        d=float(cfg.extras.get("fast_elss_d", cfg.extras.get("head_d", 0.955))),
        alpha2=float(cfg.extras.get("fast_elss_alpha2", cfg.extras.get("head_alpha2", 5e-5))),
        gamma=float(cfg.extras.get("fast_elss_gamma", cfg.extras.get("head_gamma", 0.005))),
        device=str(device),
        random_state=int(cfg.seed),
        using_pgrank=True,
    )
    x_t = torch.tensor(x.astype("float64"), dtype=torch.float64)
    adj_t = convert_sparse_matrix_to_sparse_tensor(norm_adj.astype(np.float64))
    torch.use_deterministic_algorithms(False)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            head.fit(x_t, adj_t)
    finally:
        torch.use_deterministic_algorithms(False)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
    labels = np.asarray(head.predict(x_t), dtype=np.int64)
    q = np.asarray(getattr(head, "Q", np.zeros((x.shape[0], n_clusters), dtype=np.float64)), dtype=np.float64)
    return labels, q


class _FastELSSHead:
    def __init__(
        self,
        *,
        n_clusters: int,
        k_rank: int,
        n_anchors: int,
        power: int,
        d: float,
        alpha2: float,
        gamma: float,
        seed: int,
        device: torch.device,
    ):
        self.n_clusters = int(n_clusters)
        self.k_rank = int(k_rank)
        self.n_anchors = int(n_anchors)
        self.power = int(power)
        self.d = float(d)
        self.alpha2 = float(alpha2)
        self.gamma = float(gamma)
        self.seed = int(seed)
        self.device = device

    def fit_transform(self, x: torch.Tensor, adj: torch.Tensor) -> np.ndarray:
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        x = x.to(self.device)
        adj = adj.to(self.device).coalesce()
        lap = self._eye(adj.shape[0]) - adj
        h = x
        for _ in range(max(0, self.power)):
            h = torch.sparse.mm(adj, h)
        pr = self._pagerank(adj, max_iter=100)
        prob = (pr / pr.sum()).detach().cpu().numpy()
        anchors = np.random.choice(x.shape[0], size=min(self.n_anchors, x.shape[0]), replace=False, p=prob)
        idx = torch.as_tensor(anchors, dtype=torch.long, device=self.device)
        emb = self._solve(h, lap, idx)
        z = self._square_feat_map(emb)
        u, _, _ = torch.linalg.svd(z, full_matrices=False)
        return u[:, 1 : self.n_clusters + 1].detach().cpu().numpy()

    def _eye(self, n: int) -> torch.Tensor:
        idx = torch.arange(n, device=self.device).repeat(2, 1)
        vals = torch.ones(n, dtype=torch.float64, device=self.device)
        return torch.sparse_coo_tensor(idx, vals, (n, n), dtype=torch.float64, device=self.device).coalesce()

    def _pagerank(self, adj: torch.Tensor, max_iter: int = 100) -> torch.Tensor:
        n = adj.shape[0]
        deg = torch.sparse.sum(adj, dim=1).to_dense().view(-1, 1)
        inv = 1.0 / torch.clamp(deg, min=1e-10)
        pr = torch.ones(n, 1, dtype=torch.float64, device=self.device) / n
        teleport = pr.clone()
        for _ in range(max_iter):
            pr = (1.0 - self.d) * teleport + self.d * torch.sparse.mm(adj.t(), pr * inv)
        return pr.squeeze()

    def _subcolumns(self, sparse: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        n = sparse.shape[0]
        col = torch.arange(idx.numel(), device=self.device)
        vals = torch.ones(idx.numel(), dtype=torch.float64, device=self.device)
        p = torch.sparse_coo_tensor(torch.stack([idx, col]), vals, (n, idx.numel()), dtype=torch.float64, device=self.device).coalesce()
        return torch.sparse.mm(sparse, p.to_dense())

    def _rbf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_norm = (x * x).sum(dim=1, keepdim=True)
        y_norm = (y * y).sum(dim=1, keepdim=True).t()
        dist = torch.clamp(x_norm + y_norm - 2.0 * (x @ y.t()), min=0.0)
        return torch.exp(-self.gamma * dist)

    def _solve(self, h: torch.Tensor, lap: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        h_anchor = h[idx]
        k_anchor = self._rbf(h_anchor, h_anchor)
        l_cols = self._subcolumns(lap, idx)
        w = k_anchor - self.alpha2 * l_cols[idx]
        c = self._rbf(h, h_anchor) - self.alpha2 * l_cols
        eigvals, eigvecs = torch.linalg.eigh(w)
        mask = eigvals > 1e-12
        eigvals = eigvals[mask]
        eigvecs = eigvecs[:, mask]
        q = c @ eigvecs @ torch.diag(1.0 / torch.sqrt(eigvals))
        u, _, _ = torch.linalg.svd(q, full_matrices=False)
        return u[:, : self.k_rank]

    def _square_feat_map(self, z: torch.Tensor, c: float = 2 ** -0.5) -> torch.Tensor:
        n, d = z.shape
        pieces = [torch.ones(n, 1, dtype=z.dtype, device=z.device), z]
        quad = []
        for i in range(d):
            quad.append(z[:, i : i + 1] ** 2)
            for j in range(i + 1, d):
                quad.append(z[:, i : i + 1] * z[:, j : j + 1])
        pieces.append(torch.cat(quad, dim=1))
        out = torch.cat(pieces, dim=1)
        coefs = torch.ones(out.shape[1], dtype=z.dtype, device=z.device)
        coefs[0], coefs[1 : d + 1], coefs[d + 1 :] = c, math.sqrt(2 * c), math.sqrt(2.0)
        return out * coefs.unsqueeze(0)


class _AnchorSubspaceHead:
    def __init__(
        self,
        *,
        n_clusters: int,
        k_rank: int,
        n_anchors: int,
        power: int,
        d: float,
        alpha2: float,
        gamma: float,
        filter_coef,
        return_k_rank: bool,
        seed: int,
        device: torch.device,
    ):
        self.n_clusters = int(n_clusters)
        self.k_rank = int(k_rank)
        self.n_anchors = int(n_anchors)
        self.power = int(power)
        self.d = float(d)
        self.alpha2 = float(alpha2)
        self.gamma = float(gamma)
        self.filter_coef = filter_coef
        self.return_k_rank = bool(return_k_rank)
        self.seed = int(seed)
        self.device = device

    def fit_transform(self, x: torch.Tensor, norm_adj: torch.Tensor) -> torch.Tensor:
        torch.manual_seed(self.seed)
        x = x.to(self.device)
        norm_adj = norm_adj.to(self.device).coalesce()
        lap = self._eye(norm_adj.shape[0]) - norm_adj
        h = self._convolve(x, norm_adj)
        pr = self._pagerank(norm_adj)
        prob = (pr / pr.sum()).detach().cpu().numpy()
        rng = np.random.default_rng(self.seed)
        anchors = rng.choice(x.shape[0], size=self.n_anchors, replace=False, p=prob)
        idx = torch.as_tensor(anchors, dtype=torch.long, device=self.device)
        emb = self._solve_irls(h, lap, idx)
        z = self._square_feat_map(emb)
        u, _, _ = torch.linalg.svd(z, full_matrices=False)
        if self.return_k_rank:
            return u[:, 1 : min(self.k_rank + 1, u.shape[1])]
        return u[:, 1 : self.n_clusters + 1]

    def _eye(self, n: int) -> torch.Tensor:
        idx = torch.arange(n, device=self.device).repeat(2, 1)
        vals = torch.ones(n, dtype=torch.float64, device=self.device)
        return torch.sparse_coo_tensor(idx, vals, (n, n), dtype=torch.float64, device=self.device).coalesce()

    def _convolve(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        coef = self.filter_coef
        if coef is None:
            coo = adj.coalesce()
            row, col = coo.indices()
            mask = row != col
            row, col = row[mask], col[mask]
            vals = torch.ones_like(row, dtype=torch.float64)
            binary = torch.sparse_coo_tensor(torch.stack([row, col]), vals, adj.shape, dtype=torch.float64, device=self.device).coalesce()
            neigh = torch.sparse.mm(binary, x)
            deg = torch.sparse.sum(binary, dim=1).to_dense().clamp(min=1.0)
            neigh = neigh / deg.unsqueeze(1)
            s = ((F.normalize(x, dim=1) * F.normalize(neigh, dim=1)).sum(dim=1, keepdim=True) + 1.0) / 2.0
            coef = (1.0 - s) ** 2
        h0 = x.clone()
        h = x
        for _ in range(max(1, self.power)):
            h = (1.0 - coef) * torch.sparse.mm(adj, h) + coef * h0
        return h

    def _pagerank(self, adj: torch.Tensor, max_iter: int = 10) -> torch.Tensor:
        n = adj.shape[0]
        deg = torch.sparse.sum(adj, dim=1).to_dense().view(-1, 1)
        inv = 1.0 / torch.clamp(deg, min=1e-10)
        pr = torch.ones(n, 1, dtype=torch.float64, device=self.device) / n
        teleport = pr.clone()
        for _ in range(max_iter):
            pr = (1.0 - self.d) * teleport + self.d * torch.sparse.mm(adj.t(), pr * inv)
        return pr.squeeze()

    def _subcolumns(self, sparse: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        n = sparse.shape[0]
        col = torch.arange(idx.numel(), device=self.device)
        vals = torch.ones(idx.numel(), dtype=torch.float64, device=self.device)
        p = torch.sparse_coo_tensor(torch.stack([idx, col]), vals, (n, idx.numel()), dtype=torch.float64, device=self.device).coalesce()
        return torch.sparse.mm(sparse, p.to_dense())

    def _rbf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_norm = (x * x).sum(dim=1, keepdim=True)
        y_norm = (y * y).sum(dim=1, keepdim=True).t()
        dist = torch.clamp(x_norm + y_norm - 2.0 * (x @ y.t()), min=0.0)
        return torch.exp(-self.gamma * dist)

    def _solve_irls(self, h: torch.Tensor, lap: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        h_anchor = h[idx]
        l_cols = self._subcolumns(lap, idx)
        l_sub = l_cols[idx]
        k_all = self._rbf(h, h_anchor)
        k_anchor = k_all[idx]
        w = 0.5 * (k_anchor + k_anchor.t()) - self.alpha2 * l_sub
        c = k_all - self.alpha2 * l_cols
        eigvals, eigvecs = torch.linalg.eigh(w)
        mask = eigvals > 1e-12
        eigvals = eigvals[mask]
        eigvecs = eigvecs[:, mask]
        q = c @ eigvecs @ torch.diag(1.0 / torch.sqrt(eigvals))
        u, _, _ = torch.linalg.svd(q, full_matrices=False)
        return u[:, : self.k_rank]

    def _square_feat_map(self, z: torch.Tensor, c: float = 2 ** -0.5) -> torch.Tensor:
        n, d = z.shape
        pieces = [torch.ones(n, 1, dtype=z.dtype, device=z.device), z]
        quad = []
        for i in range(d):
            quad.append(z[:, i : i + 1] ** 2)
            for j in range(i + 1, d):
                quad.append(z[:, i : i + 1] * z[:, j : j + 1])
        pieces.append(torch.cat(quad, dim=1))
        out = torch.cat(pieces, dim=1)
        coefs = torch.ones(out.shape[1], dtype=z.dtype, device=z.device)
        coefs[0], coefs[1 : d + 1], coefs[d + 1 :] = c, math.sqrt(2 * c), math.sqrt(2.0)
        return out * coefs.unsqueeze(0)


def _elss_sym_normalize(mx: sp.spmatrix, *, add_self_loops: bool) -> sp.csr_matrix:
    mx = mx.astype(np.float64).tocsr()
    if add_self_loops:
        mx = mx + sp.eye(mx.shape[0], dtype=np.float64, format="csr")
    rowsum = np.asarray(mx.sum(axis=1)).reshape(-1)
    inv = np.divide(1.0, np.sqrt(rowsum), out=np.zeros_like(rowsum), where=rowsum > 0)
    return sp.diags(inv).dot(mx).dot(sp.diags(inv)).tocsr()


def _elss_row_normalize(mx: sp.spmatrix, *, add_self_loops: bool) -> sp.csr_matrix:
    mx = mx.astype(np.float64).tocsr()
    if add_self_loops:
        mx = mx + sp.eye(mx.shape[0], dtype=np.float64, format="csr")
    rowsum = np.asarray(mx.sum(axis=1)).reshape(-1)
    inv = np.divide(1.0, rowsum, out=np.zeros_like(rowsum), where=rowsum > 0)
    return sp.diags(inv).dot(mx).tocsr()


def _scipy_to_torch64(mx: sp.spmatrix, device: torch.device) -> torch.Tensor:
    coo = mx.tocoo()
    idx = np.vstack([coo.row, coo.col]).astype(np.int64)
    return torch.sparse_coo_tensor(
        torch.from_numpy(idx),
        torch.from_numpy(coo.data.astype(np.float64)),
        size=coo.shape,
        dtype=torch.float64,
        device=device,
    ).coalesce()


def _cluster_with_options(
    x: np.ndarray,
    n_clusters: int,
    *,
    seed: int,
    n_init: int,
    max_iter: int,
    use_minibatch: bool,
    batch_size: int,
) -> np.ndarray:
    if use_minibatch:
        return MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=batch_size,
            n_init=n_init,
            max_iter=max_iter,
            random_state=seed,
            reassignment_ratio=0.0,
        ).fit_predict(x)
    return KMeans(n_clusters=n_clusters, n_init=n_init, max_iter=max_iter, random_state=seed).fit_predict(x)


def _assignment_smooth(labels: np.ndarray, model: SECTCoCo, cfg: SECTCoCoConfig) -> np.ndarray:
    graph = _select_assignment_graph(model, cfg, "assignment_smooth_graph", "raw")
    if graph is None:
        return labels
    n = graph.shape[0]
    c = model.n_clusters
    prob = np.zeros((n, c), dtype=np.float32)
    prob[np.arange(n), np.asarray(labels, dtype=np.int64)] = 1.0
    init = prob.copy()
    norm = _row_normalize(graph + sp.eye(n, dtype=np.float32, format="csr"))
    steps = int(cfg.extras.get("assignment_smooth_steps", 30))
    restart = float(cfg.extras.get("assignment_smooth_restart", 0.0))
    residual = float(cfg.extras.get("assignment_smooth_residual", 0.0))
    for _ in range(max(1, steps)):
        prob = (1.0 - restart) * (norm @ prob) + restart * init
    prob = (1.0 - residual) * prob + residual * init
    return np.argmax(prob, axis=1).astype(np.int64)


def _assignment_sinkhorn(labels: np.ndarray, model: SECTCoCo, cfg: SECTCoCoConfig) -> np.ndarray:
    graph = _select_assignment_graph(model, cfg, "assignment_sinkhorn_graph", str(cfg.extras.get("assignment_smooth_graph", "raw")))
    if graph is None:
        return labels
    labels0 = np.asarray(labels, dtype=np.int64)
    n = labels0.shape[0]
    c = model.n_clusters
    prob0 = np.zeros((n, c), dtype=np.float64)
    prob0[np.arange(n), labels0] = 1.0
    prob = prob0.copy()
    norm = _row_normalize(graph + sp.eye(n, dtype=np.float32, format="csr")).astype(np.float64)
    steps = int(cfg.extras.get("assignment_sinkhorn_steps", cfg.extras.get("assignment_smooth_steps", 20)))
    restart = float(np.clip(cfg.extras.get("assignment_sinkhorn_restart", cfg.extras.get("assignment_smooth_restart", 0.0)), 0.0, 1.0))
    residual = float(np.clip(cfg.extras.get("assignment_sinkhorn_residual", cfg.extras.get("assignment_smooth_residual", 0.0)), 0.0, 1.0))
    for _ in range(max(0, steps)):
        prob = (1.0 - restart) * (norm @ prob) + restart * prob0
    if residual > 0:
        prob = (1.0 - residual) * prob + residual * prob0
    tau = float(max(1e-3, cfg.extras.get("assignment_sinkhorn_temperature", 1.0)))
    if abs(tau - 1.0) > 1e-8:
        prob = np.power(np.clip(prob, 1e-12, None), 1.0 / tau)
        prob = prob / np.clip(prob.sum(axis=1, keepdims=True), 1e-12, None)

    target_mode = str(cfg.extras.get("assignment_sinkhorn_target", "initial"))
    if target_mode == "uniform":
        target = np.full(c, 1.0 / max(1, c), dtype=np.float64)
    else:
        target = np.clip(prob0.mean(axis=0), 1e-8, None)
        target = target / target.sum()
    strength = float(cfg.extras.get("assignment_sinkhorn_balance", 0.25))
    rounds = int(cfg.extras.get("assignment_sinkhorn_rounds", 8))
    for _ in range(max(0, rounds)):
        mass = np.clip(prob.mean(axis=0), 1e-12, None)
        prob *= np.power(target / mass, strength).reshape(1, -1)
        prob = prob / np.clip(prob.sum(axis=1, keepdims=True), 1e-12, None)
    out = np.argmax(prob, axis=1).astype(np.int64, copy=False)
    if len(np.unique(out)) < c and bool(cfg.extras.get("assignment_sinkhorn_require_full_clusters", True)):
        return labels0.copy()
    return out


def _assignment_spillover(labels: np.ndarray, model: SECTCoCo, cfg: SECTCoCoConfig) -> np.ndarray:
    graph = _select_assignment_graph(model, cfg, "assignment_spillover_graph", str(cfg.extras.get("assignment_smooth_graph", "raw")))
    if graph is None:
        return labels
    labels0 = np.asarray(labels, dtype=np.int64)
    c = model.n_clusters
    n = labels0.shape[0]
    prob0 = np.zeros((n, c), dtype=np.float64)
    prob0[np.arange(n), labels0] = 1.0
    norm = _row_normalize(graph + sp.eye(n, dtype=np.float32, format="csr")).astype(np.float64)
    prob = prob0.copy()
    steps = int(cfg.extras.get("assignment_spillover_steps", cfg.extras.get("assignment_smooth_steps", 20)))
    restart = float(np.clip(cfg.extras.get("assignment_spillover_restart", cfg.extras.get("assignment_smooth_restart", 0.0)), 0.0, 1.0))
    residual = float(np.clip(cfg.extras.get("assignment_spillover_residual", cfg.extras.get("assignment_smooth_residual", 0.0)), 0.0, 1.0))
    for _ in range(max(0, steps)):
        prob = (1.0 - restart) * (norm @ prob) + restart * prob0
    if residual > 0:
        prob = (1.0 - residual) * prob + residual * prob0
    labels = np.argmax(prob, axis=1).astype(np.int64, copy=False)
    avg = n / max(1, c)
    max_size = int(np.ceil(float(cfg.extras.get("assignment_spillover_max_ratio", 1.35)) * avg))
    min_size = int(np.floor(float(cfg.extras.get("assignment_spillover_min_ratio", 0.50)) * avg))
    max_moves = int(cfg.extras.get("assignment_spillover_max_moves", n))
    margin_q = float(np.clip(cfg.extras.get("assignment_spillover_margin_quantile", 0.35), 0.0, 1.0))
    min_alt = float(cfg.extras.get("assignment_spillover_min_alt_mass", 0.0))
    total = 0
    for _ in range(max(1, int(cfg.extras.get("assignment_spillover_rounds", 1)))):
        sizes = np.bincount(labels, minlength=c)
        oversized = np.flatnonzero(sizes > max_size)
        if oversized.size == 0 or total >= max_moves:
            break
        proposals: list[tuple[float, int, int]] = []
        for src in oversized:
            idx = np.flatnonzero(labels == src)
            if idx.size == 0:
                continue
            scores = prob[idx].copy()
            src_score = scores[:, src]
            scores[:, src] = -np.inf
            for k in range(c):
                if sizes[k] >= max_size:
                    scores[:, k] = -np.inf
            tgt = np.argmax(scores, axis=1)
            tgt_score = scores[np.arange(idx.size), tgt]
            finite = np.isfinite(tgt_score) & (tgt_score >= min_alt)
            if not np.any(finite):
                continue
            margin = src_score - tgt_score
            threshold = np.quantile(margin[finite], margin_q)
            order = np.argsort(margin)
            added = 0
            for local in order:
                if not finite[local] or margin[local] > threshold:
                    continue
                proposals.append((float(margin[local]), int(idx[local]), int(tgt[local])))
                added += 1
                if added >= int(sizes[src] - max_size):
                    break
        moved = 0
        for _, node, tgt in sorted(proposals, key=lambda item: item[0]):
            src = int(labels[node])
            if src == tgt:
                continue
            if sizes[src] <= max(max_size, min_size) or sizes[tgt] >= max_size:
                continue
            labels[node] = tgt
            sizes[src] -= 1
            sizes[tgt] += 1
            moved += 1
            total += 1
            if total >= max_moves:
                break
        if moved == 0:
            break
    if len(np.unique(labels)) < c and bool(cfg.extras.get("assignment_spillover_require_full_clusters", True)):
        return labels0.copy()
    return labels


def _select_assignment_graph(model: SECTCoCo, cfg: SECTCoCoConfig, key: str, default: str) -> sp.csr_matrix | None:
    graph_name = str(cfg.extras.get(key, default))
    if graph_name == "homo" and model.homo_graph_ is not None:
        return model.homo_graph_
    if graph_name == "denoised" and model.denoised_adj_ is not None:
        return model.denoised_adj_
    return model.raw_adj_


def _rep_spillover(labels: np.ndarray, rep: np.ndarray | None, n_clusters: int, cfg: SECTCoCoConfig) -> np.ndarray:
    if rep is None:
        return labels
    labels = np.asarray(labels, dtype=np.int64).copy()
    labels0 = labels.copy()
    z = normalize(np.nan_to_num(np.asarray(rep, dtype=np.float64)), norm="l2", axis=1)
    max_moves = int(cfg.extras.get("rep_spillover_max_moves", 500))
    min_gain = float(cfg.extras.get("rep_spillover_min_gain", -1.0))
    source_mode = str(cfg.extras.get("rep_spillover_source", "largest"))
    target_mode = str(cfg.extras.get("rep_spillover_target", "smallest"))
    moves = 0
    for _ in range(max(1, int(cfg.extras.get("rep_spillover_rounds", 1)))):
        sizes = np.bincount(labels, minlength=n_clusters)
        if source_mode == "largest":
            src = int(np.argmax(sizes))
        else:
            src = int(source_mode)
        if target_mode == "smallest":
            candidates = [k for k in range(n_clusters) if k != src]
            targets = [int(min(candidates, key=lambda k: sizes[k]))]
        elif target_mode == "second_smallest":
            candidates = sorted([k for k in range(n_clusters) if k != src], key=lambda k: sizes[k])
            targets = [int(candidates[min(1, len(candidates) - 1)])]
        else:
            targets = [int(target_mode)]
        src_idx = np.flatnonzero(labels == src)
        if src_idx.size == 0:
            break
        centers = np.zeros((n_clusters, z.shape[1]), dtype=np.float64)
        for k in range(n_clusters):
            idx = labels == k
            centers[k] = z[idx].mean(axis=0) if np.any(idx) else 0.0
        centers = normalize(np.nan_to_num(centers), norm="l2", axis=1)
        for tgt in targets:
            if tgt == src:
                continue
            gain = z[src_idx] @ centers[tgt] - z[src_idx] @ centers[src]
            eligible = np.flatnonzero(gain >= min_gain)
            if eligible.size == 0:
                continue
            order = eligible[np.argsort(-gain[eligible])]
            count = min(max_moves - moves, int(order.size))
            if count <= 0:
                break
            labels[src_idx[order[:count]]] = tgt
            moves += count
        if moves >= max_moves:
            break
    if len(np.unique(labels)) < n_clusters and bool(cfg.extras.get("rep_spillover_require_full_clusters", True)):
        return labels0
    return labels
