from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import random
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD

from .data import build_candidate_edges, scipy_to_torch_dense


@dataclass
class CSTCConfig:
    seed: int = 42
    device: str = "cuda"
    input_dim: int = 256
    hidden_dim: int = 256
    embed_dim: int = 96
    projection_dim: int = 64
    dropout: float = 0.10
    feature_knn: int = 12
    max_edges: int = 220_000
    epochs: int = 180
    pretrain_epochs: int = 35
    pretrain_mask_rate: float = 0.0
    pretrain_edge_weight: float = 0.0
    pretrain_graph_recon_weight: float = 0.0
    pretrain_edge_high: float = 0.70
    pretrain_edge_low: float = 0.30
    lr: float = 1e-3
    weight_decay: float = 5e-4
    lowpass_steps: int = 2
    diffusion_restart: float = 0.20
    threshold_tau: float = 0.09
    init_low: float = 0.22
    init_high: float = 0.66
    min_threshold_gap: float = 0.05
    target_homo_ratio: float = 0.34
    target_hetero_ratio: float = 0.26
    reconstruction_weight: float = 0.10
    cluster_weight: float = 0.22
    dtc_weight: float = 0.05
    spectral_weight: float = 0.04
    transport_weight: float = 0.06
    view_consistency_weight: float = 0.03
    view_balance_weight: float = 0.05
    assignment_graph_weight: float = 0.08
    assignment_neg_weight: float = 0.25
    edge_contrast_weight: float = 0.02
    edge_contrast_margin: float = 0.65
    edge_contrast_adaptive: bool = False
    edge_contrast_center: float = 0.30
    edge_contrast_scale: float = 0.06
    anchor_weight_start: float = 0.0
    anchor_weight_end: float = 0.0
    entropy_weight: float = 0.10
    ambiguity_weight: float = 0.04
    temperature: float = 0.24
    temperature_start_scale: float = 1.0
    sinkhorn_epsilon: float = 0.08
    sinkhorn_iters: int = 8
    refine_attract: float = 0.20
    refine_repel: float = 0.20
    highpass_scale: float = 0.55
    target_power_start: float = 1.05
    target_power_end: float = 1.80
    entropy_start: float = 0.58
    entropy_end: float = 0.24
    target_ambiguous_ratio: float = 0.34
    prototype_update_start: int = 20
    prototype_update_interval: int = 10
    prototype_momentum: float = 0.88
    prototype_conf_power: float = 2.0
    num_transport_experts: int = 0
    expert_gate_balance_weight: float = 0.01
    expert_consistency_weight: float = 0.005
    expert_diversity_weight: float = 0.005
    expert_temperature_scale: float = 1.15
    expert_mix_weight: float = 0.25
    prototype_separation_weight: float = 0.0
    prototype_separation_margin: float = 0.15
    log_interval: int = 25
    use_tfidf: bool = True
    kmeans_n_init: int = 20
    use_spectral_feature_bank: bool = False
    input_lowpass_hops: int = 2
    input_residual_weight: float = 0.70

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MLPEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, embed_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=1)


class EdgeConcordance(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 6))
        self.residual = nn.Sequential(nn.Linear(16, 20), nn.ReLU(), nn.Linear(20, 1))

    def forward(
        self,
        edge_stats: torch.Tensor,
        evidences: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        alpha = torch.softmax(self.gate(edge_stats), dim=1)
        base = (alpha * evidences).sum(dim=1).clamp(1e-4, 1.0 - 1e-4)
        residual = torch.tanh(self.residual(torch.cat([edge_stats, alpha], dim=1)).squeeze(1))
        logit = torch.logit(base) + 0.75 * residual
        return torch.sigmoid(logit), alpha


class DifferentiableTopologicalContraction(nn.Module):
    def __init__(self, cfg: CSTCConfig):
        super().__init__()
        self.low_raw = nn.Parameter(torch.tensor(_logit(cfg.init_low), dtype=torch.float32))
        upper_frac = (cfg.init_high - cfg.init_low - cfg.min_threshold_gap) / max(1e-6, 1.0 - cfg.init_low - cfg.min_threshold_gap)
        self.high_raw = nn.Parameter(torch.tensor(_logit(float(np.clip(upper_frac, 1e-3, 1 - 1e-3))), dtype=torch.float32))
        self.min_gap = cfg.min_threshold_gap
        self.tau = cfg.threshold_tau

    def forward(self, score: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        low = torch.sigmoid(self.low_raw)
        high = low + self.min_gap + (1.0 - low - self.min_gap) * torch.sigmoid(self.high_raw)
        pos = torch.sigmoid((score - high) / self.tau)
        neg = torch.sigmoid((low - score) / self.tau)
        amb = torch.sigmoid((score - low) / self.tau) * torch.sigmoid((high - score) / self.tau)
        total = pos + neg + amb + 1e-8
        return pos / total, neg / total, amb / total, low, high


class TransportHead(nn.Module):
    def __init__(self, dim: int, n_clusters: int, cfg: CSTCConfig):
        super().__init__()
        self.cfg = cfg
        self.n_experts = max(0, int(cfg.num_transport_experts))
        self.prototypes = nn.Parameter(torch.empty(n_clusters, dim))
        nn.init.xavier_uniform_(self.prototypes)
        if self.n_experts > 0:
            self.expert_prototypes = nn.Parameter(torch.empty(self.n_experts, n_clusters, dim))
            nn.init.xavier_uniform_(self.expert_prototypes)
            self.expert_gate = nn.Sequential(nn.Linear(dim * 3, 32), nn.ReLU(), nn.Linear(32, self.n_experts + 1))
        else:
            self.register_parameter("expert_prototypes", None)
            self.expert_gate = None
        self.prior_logits = nn.Parameter(torch.zeros(n_clusters))
        self.view_gate = nn.Sequential(nn.Linear(dim * 3, 32), nn.ReLU(), nn.Linear(32, 3))
        self.temperature = cfg.temperature
        self.current_temperature_scale = 1.0
        self.epsilon = cfg.sinkhorn_epsilon
        self.iters = cfg.sinkhorn_iters
        self.refine_attract = cfg.refine_attract
        self.refine_repel = cfg.refine_repel

    def forward(
        self,
        z: torch.Tensor,
        z_low: torch.Tensor,
        z_high: torch.Tensor,
        edge_index: torch.Tensor,
        pos_w: torch.Tensor,
        neg_w: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        prot = F.normalize(self.prototypes, dim=1)
        temperature = self.temperature * self.current_temperature_scale
        views = [z, z_low, z_high]
        qs = [torch.softmax(F.normalize(v, dim=1) @ prot.t() / temperature, dim=1) for v in views]
        gate = torch.softmax(self.view_gate(torch.cat(views, dim=1)), dim=1)
        q_mix = sum(gate[:, i : i + 1] * qs[i] for i in range(3))
        q_tr = self._sinkhorn(q_mix)
        pos_msg = row_spmm(edge_index, pos_w, q_tr, q_tr.shape[0])
        neg_msg = row_spmm(edge_index, neg_w, q_tr, q_tr.shape[0])
        logits = torch.log(q_tr.clamp_min(1e-8)) + self.refine_attract * pos_msg - self.refine_repel * neg_msg
        q_base = torch.softmax(logits, dim=1)
        expert_qs: list[torch.Tensor] = []
        expert_gate = None
        if self.n_experts > 0 and self.expert_prototypes is not None and self.expert_gate is not None:
            expert_gate = torch.softmax(self.expert_gate(torch.cat(views, dim=1)), dim=1)
            for expert_idx in range(self.n_experts):
                expert_proto = F.normalize(self.expert_prototypes[expert_idx], dim=1)
                expert_logits = F.normalize(z, dim=1) @ expert_proto.t() / (temperature * self.cfg.expert_temperature_scale)
                expert_q = self._sinkhorn(torch.softmax(expert_logits, dim=1))
                expert_pos = row_spmm(edge_index, pos_w, expert_q, expert_q.shape[0])
                expert_neg = row_spmm(edge_index, neg_w, expert_q, expert_q.shape[0])
                expert_q = torch.softmax(
                    torch.log(expert_q.clamp_min(1e-8))
                    + self.refine_attract * expert_pos
                    - self.refine_repel * expert_neg,
                    dim=1,
                )
                expert_qs.append(expert_q)
            expert_mix = (expert_gate[:, 1:].unsqueeze(2) * torch.stack(expert_qs, dim=1)).sum(dim=1)
            q = (1.0 - self.cfg.expert_mix_weight) * q_base + self.cfg.expert_mix_weight * expert_mix
            q = q / q.sum(dim=1, keepdim=True).clamp_min(1e-8)
        else:
            q = q_base
        return {
            "q": q,
            "q_base": q_base,
            "q_mix": q_mix,
            "q_transport": q_tr,
            "view_gate": gate,
            "expert_gate": expert_gate,
            "view_q": qs,
            "expert_q": expert_qs,
        }

    def _sinkhorn(self, q: torch.Tensor) -> torch.Tensor:
        k = q.shape[1]
        kernel = q.clamp_min(1e-8).pow(1.0 / max(self.epsilon, 1e-4))
        kernel = kernel / kernel.sum(dim=1, keepdim=True).clamp_min(1e-8)
        target_col = torch.softmax(self.prior_logits, dim=0).unsqueeze(0)
        out = kernel
        for _ in range(self.iters):
            out = out / out.sum(dim=1, keepdim=True).clamp_min(1e-8)
            out = out * (target_col / out.mean(dim=0, keepdim=True).clamp_min(1e-8))
        return out / out.sum(dim=1, keepdim=True).clamp_min(1e-8)


class CSTC(nn.Module):
    def __init__(self, input_dim: int, n_clusters: int, cfg: CSTCConfig):
        super().__init__()
        self.cfg = cfg
        self.current_epoch = 0
        self.encoder = MLPEncoder(input_dim, cfg.hidden_dim, cfg.embed_dim, cfg.dropout)
        self.raw_proj = nn.Linear(input_dim, cfg.embed_dim)
        self.decoder = nn.Linear(cfg.embed_dim, input_dim)
        self.edge_concordance = EdgeConcordance()
        self.dtc = DifferentiableTopologicalContraction(cfg)
        self.fusion = nn.Linear(cfg.embed_dim * 3, cfg.projection_dim)
        self.transport = TransportHead(cfg.projection_dim, n_clusters, cfg)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        degree: torch.Tensor,
        graph_source: torch.Tensor,
        edge_aux: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        z0 = self.encoder(x)
        raw = F.normalize(self.raw_proj(x), dim=1)
        edge_stats, evidences = self._edge_features(z0, raw, edge_index, degree, graph_source, edge_aux)
        conf, alpha = self.edge_concordance(edge_stats, evidences)
        m_pos, m_neg, m_amb, low_thr, high_thr = self.dtc(conf)
        z_low = self._lowpass(z0, edge_index, m_pos + 0.20 * m_amb)
        z_high = self._highpass(z0, edge_index, m_neg)
        h = F.normalize(self.fusion(torch.cat([z0, z_low, z_high], dim=1)), dim=1)
        q_pack = self.transport(h, F.normalize(self.fusion(torch.cat([z_low, z_low, z_high], dim=1)), dim=1), F.normalize(self.fusion(torch.cat([z_high, z_low, z_high], dim=1)), dim=1), edge_index, m_pos, m_neg)
        recon = self.decoder(z0)
        out = {
            **q_pack,
            "z": z0,
            "h": h,
            "raw": raw,
            "z_low": z_low,
            "z_high": z_high,
            "recon": recon,
            "confidence": conf,
            "alpha": alpha,
            "m_pos": m_pos,
            "m_neg": m_neg,
            "m_amb": m_amb,
            "low_thr": low_thr,
            "high_thr": high_thr,
            "edge_stats": edge_stats,
        }
        return out

    def loss(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        out: dict[str, torch.Tensor],
        anchor_q: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        q = out["q"]
        progress = min(1.0, max(0.0, self.current_epoch / max(1, self.cfg.epochs - 1)))
        with torch.no_grad():
            target_power = self.cfg.target_power_start + progress * (self.cfg.target_power_end - self.cfg.target_power_start)
            p = sharpen_target(q, target_power)
        cluster = F.kl_div(torch.log(q.clamp_min(1e-8)), p, reduction="batchmean")
        if anchor_q is None:
            anchor = torch.zeros((), device=q.device, dtype=q.dtype)
            anchor_weight = 0.0
        else:
            anchor = F.kl_div(torch.log(q.clamp_min(1e-8)), anchor_q, reduction="batchmean")
            anchor_weight = self.cfg.anchor_weight_start + progress * (self.cfg.anchor_weight_end - self.cfg.anchor_weight_start)
        rec = F.mse_loss(out["recon"], x)
        pos_mean = out["m_pos"].mean()
        neg_mean = out["m_neg"].mean()
        amb_mean = out["m_amb"].mean()
        dtc = (pos_mean - self.cfg.target_homo_ratio).pow(2) + (neg_mean - self.cfg.target_hetero_ratio).pow(2)
        amb = (amb_mean - self.cfg.target_ambiguous_ratio).pow(2)
        i, j = edge_index
        pos_dist = (out["h"][i] - out["h"][j]).pow(2).sum(dim=1)
        neg_dist = (out["z_high"][i] - out["z_high"][j]).pow(2).sum(dim=1)
        spec = weighted_mean(pos_dist, out["m_pos"] + 0.20 * out["m_amb"]) - 0.05 * weighted_mean(neg_dist, out["m_neg"])
        balance = (q.mean(dim=0) * torch.log(q.mean(dim=0).clamp_min(1e-8) * q.shape[1])).sum()
        view_cons = sum(symmetric_kl(q, vq) for vq in out["view_q"]) / len(out["view_q"])
        gate_mean = out["view_gate"].mean(dim=0)
        view_balance = (gate_mean * torch.log(gate_mean.clamp_min(1e-8) * gate_mean.shape[0])).sum()
        if out["expert_gate"] is None:
            expert_balance = torch.zeros((), device=q.device, dtype=q.dtype)
            expert_cons = torch.zeros((), device=q.device, dtype=q.dtype)
            expert_div = torch.zeros((), device=q.device, dtype=q.dtype)
        else:
            expert_gate_mean = out["expert_gate"].mean(dim=0)
            expert_balance = (expert_gate_mean * torch.log(expert_gate_mean.clamp_min(1e-8) * expert_gate_mean.shape[0])).sum()
            expert_cons = sum(symmetric_kl(q, eq) for eq in out["expert_q"]) / max(1, len(out["expert_q"]))
            expert_div = _prototype_diversity(self.transport)
        proto_sep = _prototype_separation(self.transport, self.cfg.prototype_separation_margin)
        q_affinity = (q[i] * q[j]).sum(dim=1)
        assign_graph = weighted_mean(1.0 - q_affinity, out["m_pos"] + 0.15 * out["m_amb"])
        assign_graph = assign_graph + self.cfg.assignment_neg_weight * weighted_mean(q_affinity, out["m_neg"])
        h_dist = (out["h"][i] - out["h"][j]).pow(2).sum(dim=1)
        edge_contrast = weighted_mean(h_dist, out["m_pos"] + 0.10 * out["m_amb"])
        edge_contrast = edge_contrast + weighted_mean(
            F.relu(self.cfg.edge_contrast_margin - torch.sqrt(h_dist.clamp_min(1e-8))).pow(2),
            out["m_neg"],
        )
        if self.cfg.edge_contrast_adaptive:
            edge_contrast_weight = self.cfg.edge_contrast_weight * torch.sigmoid(
                (neg_mean.detach() - self.cfg.edge_contrast_center) / max(self.cfg.edge_contrast_scale, 1e-6)
            )
        else:
            edge_contrast_weight = torch.as_tensor(self.cfg.edge_contrast_weight, device=q.device, dtype=q.dtype)
        entropy_mean = entropy(q).mean()
        entropy_floor = self.cfg.entropy_start + progress * (self.cfg.entropy_end - self.cfg.entropy_start)
        entropy_guard = F.relu(torch.as_tensor(entropy_floor, device=q.device, dtype=q.dtype) - entropy_mean).pow(2)
        loss = (
            self.cfg.cluster_weight * cluster
            + self.cfg.reconstruction_weight * rec
            + self.cfg.dtc_weight * dtc
            + self.cfg.ambiguity_weight * amb
            + self.cfg.spectral_weight * spec
            + self.cfg.transport_weight * balance
            + self.cfg.view_consistency_weight * view_cons
            + self.cfg.view_balance_weight * view_balance
            + self.cfg.expert_gate_balance_weight * expert_balance
            + self.cfg.expert_consistency_weight * expert_cons
            + self.cfg.expert_diversity_weight * expert_div
            + self.cfg.prototype_separation_weight * proto_sep
            + self.cfg.assignment_graph_weight * assign_graph
            + edge_contrast_weight * edge_contrast
            + anchor_weight * anchor
            + self.cfg.entropy_weight * entropy_guard
        )
        logs = {
            "loss": float(loss.detach().cpu()),
            "cluster_loss": float(cluster.detach().cpu()),
            "reconstruction_loss": float(rec.detach().cpu()),
            "dtc_loss": float(dtc.detach().cpu()),
            "ambiguity_loss": float(amb.detach().cpu()),
            "spectral_loss": float(spec.detach().cpu()),
            "balance_loss": float(balance.detach().cpu()),
            "view_consistency_loss": float(view_cons.detach().cpu()),
            "view_balance_loss": float(view_balance.detach().cpu()),
            "expert_balance_loss": float(expert_balance.detach().cpu()),
            "expert_consistency_loss": float(expert_cons.detach().cpu()),
            "expert_diversity_loss": float(expert_div.detach().cpu()),
            "prototype_separation_loss": float(proto_sep.detach().cpu()),
            "assignment_graph_loss": float(assign_graph.detach().cpu()),
            "edge_contrast_loss": float(edge_contrast.detach().cpu()),
            "edge_contrast_weight": float(edge_contrast_weight.detach().cpu()),
            "anchor_loss": float(anchor.detach().cpu()),
            "anchor_weight": float(anchor_weight),
            "entropy_guard_loss": float(entropy_guard.detach().cpu()),
            "homo_mass": float(pos_mean.detach().cpu()),
            "hetero_mass": float(neg_mean.detach().cpu()),
            "ambiguous_mass": float(amb_mean.detach().cpu()),
            "low_threshold": float(out["low_thr"].detach().cpu()),
            "high_threshold": float(out["high_thr"].detach().cpu()),
            "posterior_entropy": float(entropy_mean.detach().cpu()),
            "entropy_floor": float(entropy_floor),
            "target_power": float(target_power),
        }
        return loss, logs

    def _edge_features(
        self,
        z: torch.Tensor,
        raw: torch.Tensor,
        edge_index: torch.Tensor,
        degree: torch.Tensor,
        graph_source: torch.Tensor,
        edge_aux: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        i, j = edge_index
        attr = cosine01(z[i], z[j])
        raw_sim = cosine01(raw[i], raw[j])
        di = torch.log1p(degree[i])
        dj = torch.log1p(degree[j])
        role = 1.0 - (di - dj).abs() / (di + dj + 1.0)
        overlap = edge_aux[:, 0]
        two_hop = edge_aux[:, 1]
        static_attr = edge_aux[:, 2]
        mismatch = (attr - role).abs()
        raw_mismatch = (raw_sim - role).abs()
        structure_match = 0.5 * role + 0.5 * two_hop
        attr_consensus = 0.5 * raw_sim + 0.5 * static_attr
        concordant_structure = 1.0 - (attr_consensus - structure_match).abs()
        edge_prior = graph_source
        stats = torch.stack(
            [
                attr,
                raw_sim,
                static_attr,
                role,
                overlap,
                two_hop,
                1.0 - mismatch,
                1.0 - raw_mismatch,
                concordant_structure,
                edge_prior,
            ],
            dim=1,
        )
        evidences = torch.stack([attr, raw_sim, static_attr, role, two_hop, concordant_structure], dim=1).clamp(1e-4, 1.0 - 1e-4)
        return stats, evidences

    def _lowpass(self, z: torch.Tensor, edge_index: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        h = z
        acc = z
        for _ in range(self.cfg.lowpass_steps):
            h = (1.0 - self.cfg.diffusion_restart) * row_spmm(edge_index, weight, h, z.shape[0]) + self.cfg.diffusion_restart * z
            acc = acc + h
        return F.normalize(acc / float(self.cfg.lowpass_steps + 1), dim=1)

    def _highpass(self, z: torch.Tensor, edge_index: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        neigh = row_spmm(edge_index, weight, z, z.shape[0])
        scale = self.cfg.highpass_scale * weight.mean().detach().clamp(0.05, 0.95)
        return F.normalize(z - scale * neigh, dim=1)


def fit_predict(dataset, features, cfg: CSTCConfig) -> tuple[np.ndarray, dict[str, Any]]:
    set_seed(cfg.seed)
    device = torch.device(cfg.device if cfg.device == "cpu" or torch.cuda.is_available() else "cpu")
    x_np = _prepare_input_features(features, dataset.adj, cfg)
    edges_np, graph_source_np = build_candidate_edges(dataset.adj, features, feature_knn=cfg.feature_knn, max_edges=cfg.max_edges, seed=cfg.seed)
    edge_aux_np = _edge_structural_aux(dataset.adj, edges_np, x_np)
    x = scipy_to_torch_dense(x_np, device)
    edge_index = torch.from_numpy(edges_np.T).long().to(device)
    graph_source = torch.from_numpy(graph_source_np).float().to(device)
    edge_aux = torch.from_numpy(edge_aux_np).float().to(device)
    degree_np = np.asarray(dataset.adj.sum(axis=1)).reshape(-1).astype(np.float32)
    degree = torch.from_numpy(degree_np).to(device)
    model = CSTC(x.shape[1], dataset.n_clusters, cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    pretrain_log = _pretrain_autoencoder(model, x, edge_index, edge_aux, cfg)
    _initialize_prototypes(model, x, dataset.n_clusters, cfg)
    anchor_q = _build_anchor_posterior(model, x, edge_index, degree, graph_source, edge_aux, cfg)
    logs: list[dict[str, float]] = []
    model.train()
    for epoch in range(cfg.epochs):
        model.current_epoch = epoch
        progress = min(1.0, max(0.0, epoch / max(1, cfg.epochs - 1)))
        model.transport.current_temperature_scale = cfg.temperature_start_scale + progress * (1.0 - cfg.temperature_start_scale)
        optimizer.zero_grad(set_to_none=True)
        out = model(x, edge_index, degree, graph_source, edge_aux)
        loss, log = model.loss(x, edge_index, out, anchor_q)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        _refresh_prototypes(model, out, epoch, cfg)
        if epoch % cfg.log_interval == 0 or epoch == cfg.epochs - 1:
            log["epoch"] = epoch
            logs.append(log)
    model.eval()
    with torch.no_grad():
        model.current_epoch = cfg.epochs - 1
        out = model(x, edge_index, degree, graph_source, edge_aux)
        pred = out["q"].argmax(dim=1).cpu().numpy()
        diagnostics = {
            "n_nodes": int(x.shape[0]),
            "n_edges": int(edge_index.shape[1]),
            "n_clusters": int(dataset.n_clusters),
            "config": cfg.to_dict(),
            "pretrain_log": pretrain_log,
            "training_log": logs,
            "final": {
                "homo_mass": float(out["m_pos"].mean().cpu()),
                "hetero_mass": float(out["m_neg"].mean().cpu()),
                "ambiguous_mass": float(out["m_amb"].mean().cpu()),
                "low_threshold": float(out["low_thr"].cpu()),
                "high_threshold": float(out["high_thr"].cpu()),
                "confidence_mean": float(out["confidence"].mean().cpu()),
                "confidence_std": float(out["confidence"].std().cpu()),
                "posterior_entropy": float(entropy(out["q"]).mean().cpu()),
                "cluster_prior": out["q"].mean(dim=0).cpu().numpy().round(6).tolist(),
                "view_gate_mean": out["view_gate"].mean(dim=0).cpu().numpy().round(6).tolist(),
                "expert_gate_mean": (
                    out["expert_gate"].mean(dim=0).cpu().numpy().round(6).tolist()
                    if out["expert_gate"] is not None
                    else []
                ),
            },
        }
    return pred, diagnostics


def row_spmm(edge_index: torch.Tensor, weight: torch.Tensor, x: torch.Tensor, n: int) -> torch.Tensor:
    src, dst = edge_index
    src2 = torch.cat([src, dst], dim=0)
    dst2 = torch.cat([dst, src], dim=0)
    w = torch.cat([weight, weight], dim=0).clamp_min(0.0)
    denom = torch.zeros(n, device=x.device, dtype=x.dtype)
    denom.scatter_add_(0, src2, w.to(x.dtype))
    out = torch.zeros_like(x)
    out.index_add_(0, src2, x[dst2] * w.to(x.dtype).unsqueeze(1))
    return out / denom.clamp_min(1e-8).unsqueeze(1)


def sharpen_target(q: torch.Tensor, power: float = 2.0) -> torch.Tensor:
    p = q.pow(power) / q.sum(dim=0, keepdim=True).clamp_min(1e-8)
    return p / p.sum(dim=1, keepdim=True).clamp_min(1e-8)


def symmetric_kl(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return 0.5 * (
        F.kl_div(torch.log(a.clamp_min(1e-8)), b.detach(), reduction="batchmean")
        + F.kl_div(torch.log(b.clamp_min(1e-8)), a.detach(), reduction="batchmean")
    )


def weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (value * weight).sum() / weight.sum().clamp_min(1e-8)


def entropy(q: torch.Tensor) -> torch.Tensor:
    return -(q.clamp_min(1e-8) * torch.log(q.clamp_min(1e-8))).sum(dim=1) / math.log(q.shape[1])


def cosine01(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return 0.5 * (F.cosine_similarity(a, b, dim=1) + 1.0)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _reduce_features(features, dim: int, seed: int):
    n, d = features.shape
    if d <= dim:
        return features.astype(np.float32) if sp.issparse(features) else np.asarray(features, dtype=np.float32)
    svd = TruncatedSVD(n_components=dim, random_state=seed)
    return svd.fit_transform(features).astype(np.float32)


def _prepare_input_features(features, adj: sp.csr_matrix, cfg: CSTCConfig) -> np.ndarray:
    if not cfg.use_spectral_feature_bank:
        return _reduce_features(features, cfg.input_dim, cfg.seed)
    x = features.astype(np.float32).tocsr() if sp.issparse(features) else sp.csr_matrix(np.asarray(features, dtype=np.float32))
    norm_adj = _normalized_adjacency(adj)
    parts = [x]
    h = x
    for _ in range(max(0, cfg.input_lowpass_hops)):
        h = norm_adj @ h
        parts.append(h.astype(np.float32).tocsr())
    if cfg.input_lowpass_hops > 0:
        residual = (x - h).astype(np.float32).tocsr()
        if cfg.input_residual_weight != 1.0:
            residual = residual * float(cfg.input_residual_weight)
        parts.append(residual)
    bank = sp.hstack(parts, format="csr", dtype=np.float32)
    return _reduce_features(bank, cfg.input_dim, cfg.seed)


def _normalized_adjacency(adj: sp.csr_matrix) -> sp.csr_matrix:
    sym = ((adj + adj.T) > 0).astype(np.float32).tocsr()
    sym = sym + sp.eye(sym.shape[0], dtype=np.float32, format="csr")
    degree = np.asarray(sym.sum(axis=1)).reshape(-1).astype(np.float32)
    inv_sqrt = np.power(np.maximum(degree, 1.0), -0.5)
    return sp.diags(inv_sqrt) @ sym @ sp.diags(inv_sqrt)


def _initialize_prototypes(model: CSTC, x: torch.Tensor, n_clusters: int, cfg: CSTCConfig) -> None:
    with torch.no_grad():
        z = model.encoder(x).detach().cpu().numpy()
        n_init = max(1, cfg.kmeans_n_init)
        if z.shape[0] > 8000:
            km = MiniBatchKMeans(n_clusters=n_clusters, random_state=cfg.seed, n_init=min(n_init, 10), batch_size=2048)
        else:
            km = KMeans(n_clusters=n_clusters, random_state=cfg.seed, n_init=n_init)
        km.fit(z)
        centers = torch.from_numpy(km.cluster_centers_.astype(np.float32)).to(x.device)
        h_centers = F.normalize(model.fusion(torch.cat([centers, centers, centers], dim=1)), dim=1)
        model.transport.prototypes.copy_(h_centers)
        if model.transport.expert_prototypes is not None:
            for expert_idx in range(model.transport.expert_prototypes.shape[0]):
                rolled = torch.roll(h_centers, shifts=expert_idx + 1, dims=0)
                model.transport.expert_prototypes[expert_idx].copy_(F.normalize(0.85 * h_centers + 0.15 * rolled, dim=1))


def _refresh_prototypes(model: CSTC, out: dict[str, torch.Tensor], epoch: int, cfg: CSTCConfig) -> None:
    if cfg.prototype_update_interval <= 0 or epoch < cfg.prototype_update_start:
        return
    if (epoch - cfg.prototype_update_start) % cfg.prototype_update_interval != 0:
        return
    with torch.no_grad():
        q = out["q"].detach()
        h = out["h"].detach()
        reliability = q.max(dim=1, keepdim=True).values.clamp_min(1e-4).pow(cfg.prototype_conf_power)
        weights = q * reliability
        centers = weights.t() @ h
        denom = weights.sum(dim=0, keepdim=True).t().clamp_min(1e-6)
        centers = F.normalize(centers / denom, dim=1)
        old = F.normalize(model.transport.prototypes.data, dim=1)
        model.transport.prototypes.data.copy_(F.normalize(cfg.prototype_momentum * old + (1.0 - cfg.prototype_momentum) * centers, dim=1))
        if model.transport.expert_prototypes is not None and out["expert_q"]:
            for expert_idx, expert_q in enumerate(out["expert_q"]):
                reliability_e = expert_q.max(dim=1, keepdim=True).values.clamp_min(1e-4).pow(cfg.prototype_conf_power)
                weights_e = expert_q * reliability_e
                centers_e = weights_e.t() @ h
                denom_e = weights_e.sum(dim=0, keepdim=True).t().clamp_min(1e-6)
                centers_e = F.normalize(centers_e / denom_e, dim=1)
                old_e = F.normalize(model.transport.expert_prototypes.data[expert_idx], dim=1)
                model.transport.expert_prototypes.data[expert_idx].copy_(
                    F.normalize(cfg.prototype_momentum * old_e + (1.0 - cfg.prototype_momentum) * centers_e, dim=1)
                )


def _build_anchor_posterior(
    model: CSTC,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    degree: torch.Tensor,
    graph_source: torch.Tensor,
    edge_aux: torch.Tensor,
    cfg: CSTCConfig,
) -> torch.Tensor | None:
    if cfg.anchor_weight_start <= 0 and cfg.anchor_weight_end <= 0:
        return None
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model.current_epoch = 0
        out = model(x, edge_index, degree, graph_source, edge_aux)
        anchor = out["q"].detach().clamp_min(1e-8)
        anchor = anchor / anchor.sum(dim=1, keepdim=True).clamp_min(1e-8)
    model.train(was_training)
    return anchor


def _pretrain_autoencoder(
    model: CSTC,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_aux: torch.Tensor,
    cfg: CSTCConfig,
) -> list[dict[str, float]]:
    if cfg.pretrain_epochs <= 0:
        return []
    modules = list(model.encoder.parameters()) + list(model.decoder.parameters()) + list(model.raw_proj.parameters())
    opt = torch.optim.AdamW(modules, lr=cfg.lr, weight_decay=cfg.weight_decay)
    logs: list[dict[str, float]] = []
    model.train()
    graph_weight = torch.ones(edge_index.shape[1], device=x.device, dtype=x.dtype)
    graph_target = row_spmm(edge_index, graph_weight, x, x.shape[0]).detach()
    edge_target = (0.55 * edge_aux[:, 2] + 0.45 * edge_aux[:, 1]).clamp(0.0, 1.0).detach()
    high_w = torch.sigmoid((edge_target - cfg.pretrain_edge_high) / 0.08).detach()
    low_w = torch.sigmoid((cfg.pretrain_edge_low - edge_target) / 0.08).detach()
    src, dst = edge_index
    for epoch in range(cfg.pretrain_epochs):
        opt.zero_grad(set_to_none=True)
        if cfg.pretrain_mask_rate > 0:
            keep = (torch.rand_like(x) > cfg.pretrain_mask_rate).to(x.dtype)
            x_in = x * keep
        else:
            x_in = x
        z = model.encoder(x_in)
        recon = model.decoder(z)
        raw = model.raw_proj(x)
        rec_loss = F.mse_loss(recon, x)
        graph_recon = F.mse_loss(recon, graph_target)
        sim = cosine01(z[src], z[dst])
        edge_loss = weighted_mean(1.0 - sim, high_w) + weighted_mean(sim, low_w)
        align_loss = F.mse_loss(F.normalize(raw, dim=1), z.detach())
        loss = (
            rec_loss
            + cfg.pretrain_graph_recon_weight * graph_recon
            + cfg.pretrain_edge_weight * edge_loss
            + 0.02 * align_loss
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(modules, 5.0)
        opt.step()
        if epoch == 0 or epoch == cfg.pretrain_epochs - 1:
            logs.append(
                {
                    "epoch": epoch,
                    "pretrain_loss": float(loss.detach().cpu()),
                    "pretrain_reconstruction": float(rec_loss.detach().cpu()),
                    "pretrain_graph_reconstruction": float(graph_recon.detach().cpu()),
                    "pretrain_edge_calibration": float(edge_loss.detach().cpu()),
                }
            )
    return logs


def _logit(value: float) -> float:
    value = float(np.clip(value, 1e-5, 1 - 1e-5))
    return math.log(value / (1.0 - value))


def _prototype_diversity(head: TransportHead) -> torch.Tensor:
    if head.expert_prototypes is None:
        return torch.zeros((), device=head.prototypes.device, dtype=head.prototypes.dtype)
    banks = [F.normalize(head.prototypes, dim=1)]
    banks.extend(F.normalize(head.expert_prototypes[i], dim=1) for i in range(head.expert_prototypes.shape[0]))
    losses = []
    for i in range(len(banks)):
        for j in range(i + 1, len(banks)):
            losses.append((banks[i] * banks[j]).sum(dim=1).pow(2).mean())
    if not losses:
        return torch.zeros((), device=head.prototypes.device, dtype=head.prototypes.dtype)
    return torch.stack(losses).mean()


def _prototype_separation(head: TransportHead, margin: float) -> torch.Tensor:
    prot = F.normalize(head.prototypes, dim=1)
    sim = prot @ prot.t()
    off_diag = sim[~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)]
    loss = F.relu(off_diag - margin).pow(2).mean()
    if head.expert_prototypes is not None:
        expert_losses = []
        for i in range(head.expert_prototypes.shape[0]):
            p = F.normalize(head.expert_prototypes[i], dim=1)
            s = p @ p.t()
            expert_losses.append(F.relu(s[~torch.eye(s.shape[0], dtype=torch.bool, device=s.device)] - margin).pow(2).mean())
        if expert_losses:
            loss = loss + torch.stack(expert_losses).mean()
    return loss


def _edge_structural_aux(adj: sp.csr_matrix, edges: np.ndarray, x: np.ndarray) -> np.ndarray:
    sym = ((adj + adj.T) > 0).astype(np.float32).tocsr()
    rows = edges[:, 0]
    cols = edges[:, 1]
    degree = np.asarray(sym.sum(axis=1)).reshape(-1).astype(np.float32)
    common = np.asarray(sym[rows].multiply(sym[cols]).sum(axis=1)).reshape(-1).astype(np.float32)
    deg_i = degree[rows]
    deg_j = degree[cols]
    union = deg_i + deg_j - common
    overlap = common / np.maximum(union, 1.0)
    two_hop = common / np.sqrt(np.maximum(deg_i * deg_j, 1.0))
    x_norm = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    static_attr = 0.5 * ((x_norm[rows] * x_norm[cols]).sum(axis=1) + 1.0)
    aux = np.stack([overlap, two_hop, static_attr.astype(np.float32)], axis=1)
    return np.nan_to_num(aux, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
