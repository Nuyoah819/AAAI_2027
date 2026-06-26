from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
import time
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from core.eval.metrics import evaluate_clustering
from core.legacy.sect_coco_legacy import SECTCoCoConfig, _subspace_refine as legacy_subspace_refine


LOGGER = logging.getLogger(__name__)


@dataclass
class E2ESECTCoCoConfig:
    seed: int = 42
    device: str = "cuda"
    input_dim: int = 256
    graph_input_dim: int = 0
    graph_input_transpose: bool = False
    normalize_input_views: bool = True
    hidden_dim: int = 256
    embed_dim: int = 96
    projection_dim: int = 64
    feature_knn: int = 12
    max_feature_edges: int = 250_000
    max_train_edges: int = 420_000
    epochs: int = 260
    pretrain_epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 5e-4
    dropout: float = 0.10
    threshold_tau: float = 0.08
    init_low: float = 0.35
    init_high: float = 0.55
    min_threshold_gap: float = 0.05
    lowpass_steps: int = 2
    highpass_steps: int = 1
    diffusion_restart: float = 0.20
    cluster_update_interval: int = 25
    target_bootstrap_source: str = "q_refined"
    target_bootstrap_flow_weight: float = 0.0
    target_bootstrap_adaptive_flow: bool = False
    target_bootstrap_flow_min: float = 0.40
    target_bootstrap_flow_max: float = 0.70
    target_bootstrap_entropy_center: float = 0.94
    target_bootstrap_entropy_scale: float = 0.04
    target_bootstrap_amplitude_center: float = 0.30
    target_bootstrap_amplitude_scale: float = 0.08
    target_bootstrap_late_source: str = ""
    target_bootstrap_late_flow_weight: float = 0.0
    target_bootstrap_late_start_epoch: int = -1
    loss_posterior_source: str = "q_refined"
    loss_regularizer_source: str = "same"
    loss_posterior_late_source: str = ""
    loss_regularizer_late_source: str = ""
    loss_posterior_late_start_epoch: int = -1
    loss_posterior_flow_weight: float = 0.0
    prototype_refresh_interval: int = 0
    prototype_refresh_bootstrap_mode: str = ""
    prototype_refresh_momentum: float = 0.0
    prior_refresh_momentum: float = 0.0
    teacher_refresh_source: str = "q_refined"
    teacher_refresh_momentum: float = 0.0
    cluster_loss_weight: float = 0.18
    compact_loss_weight: float = 0.10
    reconstruction_weight: float = 0.12
    contrastive_weight: float = 0.08
    dirichlet_weight: float = 0.05
    highpass_weight: float = 0.10
    highpass_adaptive_scale: bool = False
    balance_weight: float = 0.02
    entropy_weight: float = 0.01
    confidence_entropy_weight: float = 0.0
    confidence_entropy_power: float = 1.0
    confidence_entropy_hetero_weight: float = 1.0
    threshold_reg_weight: float = 0.01
    edge_prior_weight: float = 0.0
    raw_skip_weight: float = 0.65
    target_homo_ratio: float = 0.35
    target_hetero_ratio: float = 0.25
    assignment_flow_steps: int = 1
    assignment_attract_weight: float = 0.10
    assignment_repel_weight: float = 0.50
    assignment_raw_repel_floor: float = 0.0
    assignment_fidelity_weight: float = 1.0
    assignment_temperature: float = 0.75
    assignment_sharpen_power: float = 1.0
    assignment_loss_weight: float = 0.10
    final_label_mode: str = "aptc"
    main_posterior_mode: str = "refined"
    main_posterior_flow_weight: float = 0.5
    main_posterior_flow_center: float = 1.0
    main_posterior_flow_scale: float = 0.35
    main_posterior_prior_center: float = 0.12
    main_posterior_prior_scale: float = 0.06
    main_posterior_entropy_center: float = 0.82
    main_posterior_entropy_scale: float = 0.06
    main_posterior_entropy_rescue_weight: float = 0.0
    main_posterior_align_compact: bool = False
    aptc_temperature: float = 0.20
    aptc_posterior_mode: str = "cosine"
    aptc_student_alpha: float = 1.0
    aptc_logit_std_floor: float = 0.0
    aptc_logit_std_strength: float = 1.0
    aptc_sinkhorn_epsilon: float = 0.08
    aptc_sinkhorn_iters: int = 8
    aptc_refine_steps: int = 1
    aptc_hard_attract_weight: float = 0.25
    aptc_view_consistency_weight: float = 0.05
    aptc_edge_posterior_weight: float = 0.08
    aptc_prior_entropy_weight: float = 0.01
    aptc_transport_weight: float = 0.12
    aptc_flow_anchor_weight: float = 0.0
    aptc_student_posterior_weight: float = 0.0
    aptc_student_posterior_entropy_center: float = 0.92
    aptc_student_posterior_entropy_scale: float = 0.04
    aptc_student_posterior_prior_center: float = 0.12
    aptc_student_posterior_prior_scale: float = 0.06
    aptc_flow_posterior_weight: float = 0.0
    aptc_flow_posterior_flow_center: float = 0.85
    aptc_flow_posterior_flow_scale: float = 0.35
    aptc_flow_posterior_prior_center: float = 0.12
    aptc_flow_posterior_prior_scale: float = 0.06
    aptc_flow_posterior_warmup_epochs: int = 0
    aptc_flow_posterior_ramp_epochs: int = 1
    aptc_flow_mix_weight: float = 0.0
    aptc_flow_mix_flow_center: float = 1.0
    aptc_flow_mix_flow_scale: float = 0.35
    aptc_flow_mix_prior_center: float = 0.12
    aptc_flow_mix_prior_scale: float = 0.06
    aptc_flow_mix_warmup_epochs: int = 0
    aptc_flow_mix_ramp_epochs: int = 1
    aptc_dynamic_proto_weight: float = 0.0
    aptc_proto_momentum: float = 0.90
    aptc_embedding_view: bool = False
    aptc_embedding_conflict_gate: bool = False
    aptc_embedding_residual_fusion: bool = False
    aptc_embedding_node_gate: bool = False
    aptc_embedding_learned_node_gate: bool = False
    aptc_embedding_hybrid_node_gate: bool = False
    aptc_embedding_disagreement_node_gate: bool = False
    aptc_embedding_rank_node_gate: bool = False
    aptc_embedding_residual_delta_gate: bool = False
    aptc_embedding_amplitude_soft_gate: bool = False
    aptc_embedding_amplitude_gate_floor: float = 0.0
    aptc_embedding_amplitude_graph_adaptive_floor: bool = False
    aptc_embedding_amplitude_node_adaptive_floor: bool = False
    aptc_embedding_amplitude_inverse_node_floor: bool = False
    aptc_embedding_amplitude_node_floor_blend: float = 0.0
    aptc_embedding_amplitude_graph_conditioned_blend: bool = False
    aptc_embedding_amplitude_blend_prior_center: float = 0.05
    aptc_embedding_amplitude_blend_prior_scale: float = 0.02
    aptc_embedding_amplitude_floor_warmup_epochs: int = 0
    aptc_embedding_amplitude_floor_ramp_epochs: int = 1
    aptc_embedding_gate_flow_center: float = 1.00
    aptc_embedding_gate_flow_scale: float = 0.35
    aptc_embedding_gate_prior_center: float = 0.18
    aptc_embedding_gate_prior_scale: float = 0.08
    aptc_embedding_gate_std_margin: float = 0.08
    aptc_embedding_gate_std_scale: float = 0.06
    aptc_embedding_gate_entropy_margin: float = 0.02
    aptc_embedding_gate_entropy_scale: float = 0.04
    aptc_embedding_node_entropy_margin: float = 0.03
    aptc_embedding_node_entropy_scale: float = 0.05
    aptc_embedding_node_kl_margin: float = 0.02
    aptc_embedding_node_kl_scale: float = 0.05
    aptc_embedding_node_transport_margin: float = 0.02
    aptc_embedding_node_transport_scale: float = 0.05
    aptc_embedding_node_refine_margin: float = 0.02
    aptc_embedding_node_refine_scale: float = 0.05
    aptc_embedding_node_rank_quantile: float = 0.80
    aptc_embedding_node_rank_scale: float = 0.08
    aptc_embedding_gate_floor: float = 0.0
    aptc_embedding_gate_residual_scale: float = 1.0
    aptc_embedding_delta_temperature: float = 1.0
    aptc_embedding_amplitude_center: float = 0.35
    aptc_embedding_amplitude_scale: float = 0.10
    aptc_embedding_gate_hidden_dim: int = 16
    aptc_embedding_gate_init_bias: float = -1.5
    aptc_embedding_gate_warmup_epochs: int = 0
    aptc_embedding_gate_ramp_epochs: int = 1
    aptc_embed_consistency_graph_gate: bool = False
    aptc_init_teacher_weight: float = 0.10
    aptc_teacher_conf_power: float = 2.0
    aptc_teacher_conf_floor: float = 0.10
    aptc_teacher_conf_center: float = 0.50
    aptc_teacher_conf_center_mode: str = "fixed"
    aptc_teacher_conf_quantile: float = 0.65
    aptc_teacher_conf_min_scale: float = 0.05
    aptc_teacher_reliability_mode: str = "prob"
    aptc_teacher_agreement_k: int = 10
    aptc_teacher_agreement_floor: float = 0.10
    aptc_teacher_agreement_power: float = 1.0
    aptc_proto_readout_weight: float = 0.0
    aptc_proto_readout_temperature: float = 0.20
    aptc_proto_readout_conf_power: float = 1.0
    aptc_proto_readout_entropy_power: float = 1.0
    aptc_proto_readout_graph_gate: bool = False
    aptc_proto_readout_prior_scale: float = 8.0
    aptc_proto_readout_alpha_floor: float = 0.35
    aptc_proto_readout_alpha_span: float = 0.45
    aptc_proto_readout_gate_floor: float = 0.0
    aptc_proto_readout_warmup_epochs: int = 0
    aptc_proto_readout_ramp_epochs: int = 1
    aptc_prototype_anchor_weight: float = 0.0
    aptc_prototype_separation_weight: float = 0.02
    aptc_prototype_separation_margin: float = 0.20
    init_bootstrap_mode: str = "embedding"
    init_bootstrap_dim: int = 0
    mid_init_epoch: int = -1
    mid_init_bootstrap_mode: str = "embedding"
    init_prior_uniform_blend: float = 0.0
    init_prior_adaptive_blend: bool = False
    aptc_repel_mass_balance: bool = True
    aptc_min_repel_scale: float = 0.05
    aptc_max_repel_scale: float = 1.0
    adaptive_threshold_targets: bool = True
    quantile_threshold_anchor: bool = False
    quantile_threshold_weight: float = 0.00
    calib_alpha_weight: float = 0.04
    calib_mask_weight: float = 0.01
    calib_struct_attr_weight: float = 0.005
    calib_alpha_entropy_floor: float = 0.78
    calib_alpha_usage_weight: float = 0.25
    calib_alpha_usage_floor: float = 0.08
    calib_alpha_usage_floor_weight: float = 1.00
    edge_alpha_smoothing: float = 0.02
    calib_mask_floor: float = 0.06
    calib_mask_floor_weight: float = 1.00
    edge_logit_recenter_strength: float = 1.00
    edge_logit_recenter_scale: float = 1.10
    edge_logit_recenter_min_std: float = 0.05
    edge_prior_evidence: bool = False
    edge_rank_weight: float = 0.01
    edge_rank_margin: float = 0.12
    edge_rank_pos_quantile: float = 0.82
    edge_rank_neg_quantile: float = 0.18
    edge_rank_max_pairs: int = 8192
    edge_rank_local_tau: float = 0.12
    edge_rank_raw_teacher_weight: float = 0.0
    edge_rank_raw_gate_margin: float = 0.02
    edge_rank_raw_gate_temperature: float = 0.05
    edge_quantile_anchor_weight: float = 0.12
    edge_quantile_anchor_rho: float = 0.18
    raw_leakage_init: float = -12.0
    subspace_loss_weight: float = 0.0
    subspace_temperature: float = 0.25
    subspace_l1_weight: float = 1e-3
    subspace_max_nodes: int = 2048
    postproc_subspace_margin: float = 1.0
    subspace_sil_threshold: float = 0.15
    rayleigh_routing_weight: float = 0.0
    emb_dirichlet_weight: float = 0.15
    zattr_dirichlet_weight: float = 0.05
    rayleigh_temperature: float = 0.20
    posterior_stitch_weight: float = 0.0
    partition_spread_weight: float = 0.01
    edge_supervision_weight: float = 0.00
    partition_min_spread: float = 0.30
    partition_ambiguous_penalty_weight: float = 1.0
    freq_separation_weight: float = 0.0
    freq_ortho_weight: float = 0.0
    freq_ortho_target: float = 0.50
    balance_to_uniform: bool = False
    freeze_raw_skip: bool = False
    kmeans_n_init: int = 40
    use_minibatch_kmeans: bool = False
    edge_graph_source: str = "adj"
    directed_candidate_edges: bool = False
    name: str = "sect_coco_e2e"
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, values: dict[str, Any], *, seed: int) -> "E2ESECTCoCoConfig":
        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        payload = {k: v for k, v in values.items() if k in field_names}
        payload.setdefault("seed", seed)
        payload["extras"] = {k: v for k, v in values.items() if k not in field_names}
        return cls(**payload)


class SparseInputEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, embed_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), p=2, dim=1)


class AdaptiveEdgeConfidence(nn.Module):
    """Learn edge-wise evidence weights instead of fixed scalar fusion."""

    def __init__(
        self,
        edge_feature_dim: int,
        evidence_dim: int = 3,
        hidden_dim: int = 64,
        recenter_strength: float = 0.75,
        recenter_scale: float = 1.0,
        min_std: float = 0.05,
        alpha_smoothing: float = 0.0,
    ):
        super().__init__()
        self.recenter_strength = float(recenter_strength)
        self.recenter_scale = float(recenter_scale)
        self.min_std = float(min_std)
        self.alpha_smoothing = float(alpha_smoothing)
        self.evidence_dim = int(evidence_dim)
        self.gate = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.evidence_dim),
        )
        self.calibrator = nn.Sequential(
            nn.Linear(edge_feature_dim + self.evidence_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, edge_features: torch.Tensor, evidences: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        alpha = F.softmax(self.gate(edge_features), dim=-1)
        smooth = float(np.clip(self.alpha_smoothing, 0.0, 0.45))
        if smooth > 0.0:
            alpha = (1.0 - smooth) * alpha + smooth / float(alpha.shape[-1])
        fused = torch.sum(alpha * evidences, dim=-1, keepdim=True)
        residual = self.calibrator(torch.cat([edge_features, alpha], dim=-1))
        raw_logit = torch.logit(fused.clamp(1e-4, 1.0 - 1e-4)) + residual
        strength = float(np.clip(self.recenter_strength, 0.0, 1.0))
        if strength > 0.0 and raw_logit.numel() > 1:
            eps = torch.finfo(raw_logit.dtype).eps
            mean = raw_logit.mean().detach()
            std = raw_logit.std(unbiased=False).detach().clamp_min(max(float(self.min_std), float(eps)))
            normalized = (raw_logit - mean) / std
            logit = (1.0 - strength) * raw_logit + strength * float(self.recenter_scale) * normalized
        else:
            logit = raw_logit
        score = torch.sigmoid(logit).squeeze(-1)
        return score, alpha, logit.squeeze(-1)


class DifferentiableTopologyContraction(nn.Module):
    """Ordered learnable thresholds with soft homophily/heterophily masks."""

    def __init__(self, init_low: float, init_high: float, min_gap: float, tau: float):
        super().__init__()
        init_low = float(np.clip(init_low, 1e-4, 1.0 - min_gap - 1e-4))
        init_high = float(np.clip(init_high, init_low + min_gap + 1e-4, 1.0 - 1e-4))
        self.theta_low = nn.Parameter(torch.tensor(_logit(init_low), dtype=torch.float32))
        gap_ratio = (init_high - init_low - min_gap) / max(1e-6, 1.0 - init_low - min_gap)
        self.theta_gap = nn.Parameter(torch.tensor(_logit(float(np.clip(gap_ratio, 1e-4, 1.0 - 1e-4))), dtype=torch.float32))
        self.min_gap = float(min_gap)
        self.tau = float(tau)

    def thresholds(self) -> tuple[torch.Tensor, torch.Tensor]:
        low = torch.sigmoid(self.theta_low)
        high = low + self.min_gap + (1.0 - low - self.min_gap) * torch.sigmoid(self.theta_gap)
        return low, high.clamp(max=1.0 - 1e-4)

    def forward(self, score: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        low, high = self.thresholds()
        if bool(getattr(self, "use_quantile_anchor", False)) and score.numel() > 8:
            with torch.no_grad():
                q_low = torch.quantile(score.detach(), 0.25)
                q_high = torch.quantile(score.detach(), 0.75)
            anchor_w = float(getattr(self, "quantile_anchor_weight", 0.0))
            low = (1.0 - anchor_w) * low + anchor_w * q_low
            high = (1.0 - anchor_w) * high + anchor_w * q_high
            high = torch.maximum(high, low + self.min_gap).clamp(max=1.0 - 1e-4)
        homo_raw = torch.sigmoid((score - high) / self.tau)
        hetero_raw = torch.sigmoid((low - score) / self.tau)
        hard_raw = torch.sigmoid((score - low) / self.tau) * torch.sigmoid((high - score) / self.tau)
        masks = torch.stack([homo_raw, hetero_raw, hard_raw], dim=-1)
        masks = masks / masks.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return masks[:, 0], masks[:, 1], masks[:, 2], low, high


class AdaptivePosteriorTransportHead(nn.Module):
    """Unified differentiable clustering head for all datasets."""

    def __init__(self, embed_dim: int, projection_dim: int, n_clusters: int, cfg: E2ESECTCoCoConfig):
        super().__init__()
        self.cfg = cfg
        self.n_clusters = int(n_clusters)
        self.attr_projector = nn.Linear(embed_dim, projection_dim)
        self.low_projector = nn.Linear(embed_dim, projection_dim)
        self.high_projector = nn.Linear(embed_dim, projection_dim)
        self.prototypes = nn.Parameter(torch.empty(n_clusters, projection_dim))
        self.cluster_prior_logits = nn.Parameter(torch.zeros(n_clusters, dtype=torch.float32))
        self.register_buffer("prototype_memory", torch.empty(n_clusters, projection_dim))
        self.n_views = 4 if bool(cfg.aptc_embedding_view) else 3
        self.base_view_gate = nn.Sequential(
            nn.Linear(projection_dim + 3, max(16, projection_dim // 2)),
            nn.GELU(),
            nn.Linear(max(16, projection_dim // 2), 3),
        )
        self.view_gate = nn.Sequential(
            nn.Linear(projection_dim + self.n_views, max(16, projection_dim // 2)),
            nn.GELU(),
            nn.Linear(max(16, projection_dim // 2), self.n_views),
        )
        gate_hidden = max(8, int(getattr(cfg, "aptc_embedding_gate_hidden_dim", 16)))
        self.embed_node_gate_mlp = nn.Sequential(
            nn.Linear(5, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 1),
        )
        final_linear = self.embed_node_gate_mlp[-1]
        if isinstance(final_linear, nn.Linear):
            nn.init.constant_(final_linear.bias, float(getattr(cfg, "aptc_embedding_gate_init_bias", -1.5)))
        with torch.no_grad():
            protos = F.normalize(torch.randn(n_clusters, projection_dim), p=2, dim=1)
            if n_clusters <= projection_dim:
                _, _, protos = torch.linalg.svd(protos, full_matrices=False)
            self.prototypes.copy_(protos)
        nn.init.xavier_uniform_(self.prototype_memory)

    def forward(
        self,
        *,
        embedding: torch.Tensor,
        z_attr: torch.Tensor,
        low_view: torch.Tensor,
        hetero_view: torch.Tensor,
        edge_index: torch.Tensor,
        edge_prior: torch.Tensor,
        degree: torch.Tensor,
        homo: torch.Tensor,
        hetero: torch.Tensor,
        hard: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        attr = F.normalize(self.attr_projector(z_attr), p=2, dim=1)
        low = F.normalize(self.low_projector(low_view), p=2, dim=1)
        high = F.normalize(self.high_projector(hetero_view), p=2, dim=1)
        embedding = F.normalize(embedding, p=2, dim=1)
        embed_view = embedding
        base_prototypes = F.normalize(self.prototypes, p=2, dim=1)
        memory_prototypes = F.normalize(self.prototype_memory, p=2, dim=1)
        warm_prototypes = F.normalize(
            (1.0 - float(self.cfg.aptc_dynamic_proto_weight)) * base_prototypes
            + float(self.cfg.aptc_dynamic_proto_weight) * memory_prototypes,
            p=2,
            dim=1,
        )

        q_attr, attr_logit_std = self._posterior_from_view(attr, warm_prototypes)
        q_low, low_logit_std = self._posterior_from_view(low, warm_prototypes)
        q_high, high_logit_std = self._posterior_from_view(high, warm_prototypes)
        q_embed, embed_logit_std = self._posterior_from_view(embed_view, warm_prototypes)
        q_student = student_t_distribution(embedding, warm_prototypes)
        q_views = [q_attr, q_low, q_high]
        entropy_views = [posterior_entropy(q_attr), posterior_entropy(q_low), posterior_entropy(q_high)]
        rayleigh_views = [
            posterior_rayleigh(q_attr, edge_index, edge_prior),
            posterior_rayleigh(q_low, edge_index, edge_prior),
            posterior_rayleigh(q_high, edge_index, edge_prior),
        ]
        base_entropies = torch.stack(entropy_views, dim=1)
        base_gate_logits = self.base_view_gate(torch.cat([embedding, base_entropies], dim=1))
        base_gate = F.softmax(base_gate_logits, dim=1)
        q_base_mix = sum(base_gate[:, i : i + 1] * view for i, view in enumerate(q_views))
        q_base_mix = q_base_mix / q_base_mix.sum(dim=1, keepdim=True).clamp_min(1e-8)
        if bool(self.cfg.aptc_embedding_view):
            q_views.append(q_embed)
            entropy_views.append(posterior_entropy(q_embed))
            rayleigh_views.append(posterior_rayleigh(q_embed, edge_index, edge_prior))
        prior = F.softmax(self.cluster_prior_logits, dim=0)
        q_base_transport = sinkhorn_transport(
            q_base_mix,
            prior,
            epsilon=float(self.cfg.aptc_sinkhorn_epsilon),
            iters=int(self.cfg.aptc_sinkhorn_iters),
        )
        q_base_refined = self._refine(
            q_base_transport,
            prior,
            edge_index,
            edge_prior,
            degree,
            homo,
            hetero,
            hard,
        )
        base_flow_kl = F.kl_div(
            q_base_refined.clamp_min(1e-8).log(),
            q_base_mix.detach(),
            reduction="batchmean",
        ).detach().clamp_min(0.0)
        prior_penalty = prior_entropy_regularizer(prior.detach()).clamp_min(0.0)
        base_entropy_mean = base_entropies.mean().detach() / max(1e-8, math.log(float(max(2, self.n_clusters))))
        embed_entropy_mean = posterior_entropy(q_embed).mean().detach() / max(1e-8, math.log(float(max(2, self.n_clusters))))
        base_logit_std = torch.stack([attr_logit_std, low_logit_std, high_logit_std]).mean().detach()
        if bool(self.cfg.aptc_embedding_view) and bool(self.cfg.aptc_embedding_conflict_gate):
            flow_center = float(self.cfg.aptc_embedding_gate_flow_center)
            flow_scale = max(1e-4, float(self.cfg.aptc_embedding_gate_flow_scale))
            prior_center = float(self.cfg.aptc_embedding_gate_prior_center)
            prior_scale = max(1e-4, float(self.cfg.aptc_embedding_gate_prior_scale))
            std_margin = float(self.cfg.aptc_embedding_gate_std_margin)
            std_scale = max(1e-4, float(self.cfg.aptc_embedding_gate_std_scale))
            entropy_margin = float(self.cfg.aptc_embedding_gate_entropy_margin)
            entropy_scale = max(1e-4, float(self.cfg.aptc_embedding_gate_entropy_scale))
            flow_gate = torch.sigmoid((torch.log1p(base_flow_kl) - flow_center) / flow_scale)
            prior_gate = torch.sigmoid((prior_penalty - prior_center) / prior_scale)
            std_gate = torch.sigmoid((embed_logit_std.detach() - base_logit_std - std_margin) / std_scale)
            entropy_gate = torch.sigmoid((base_entropy_mean - embed_entropy_mean - entropy_margin) / entropy_scale)
            raw_embed_gate = (flow_gate * prior_gate * std_gate * entropy_gate).clamp(0.0, 1.0)
            floor_t = embedding.new_tensor(float(np.clip(self.cfg.aptc_embedding_gate_floor, 0.0, 1.0)))
            embed_graph_gate = (floor_t + (1.0 - floor_t) * raw_embed_gate).detach()
            embed_graph_gate = embed_graph_gate * float(np.clip(getattr(self, "runtime_embedding_gate_multiplier", 1.0), 0.0, 1.0))
        else:
            flow_gate = embedding.new_tensor(1.0)
            prior_gate = embedding.new_tensor(1.0)
            std_gate = embedding.new_tensor(1.0)
            entropy_gate = embedding.new_tensor(1.0)
            raw_embed_gate = embedding.new_tensor(1.0)
            embed_graph_gate = embedding.new_tensor(1.0)
        if bool(self.cfg.aptc_embedding_view) and (
            bool(self.cfg.aptc_embedding_disagreement_node_gate) or bool(self.cfg.aptc_embedding_rank_node_gate)
        ):
            base_node_entropy = posterior_entropy(q_base_mix.detach()) / max(1e-8, math.log(float(max(2, self.n_clusters))))
            embed_node_entropy = posterior_entropy(q_embed.detach()) / max(1e-8, math.log(float(max(2, self.n_clusters))))
            node_kl = F.kl_div(q_embed.clamp_min(1e-8).log(), q_base_mix.detach(), reduction="none").sum(dim=1).detach()
            transport_margin = float(self.cfg.aptc_embedding_node_transport_margin)
            transport_scale = max(1e-4, float(self.cfg.aptc_embedding_node_transport_scale))
            refine_margin = float(self.cfg.aptc_embedding_node_refine_margin)
            refine_scale = max(1e-4, float(self.cfg.aptc_embedding_node_refine_scale))
            node_transport = F.kl_div(
                q_base_transport.clamp_min(1e-8).log(),
                q_base_mix.detach(),
                reduction="none",
            ).sum(dim=1).detach()
            node_refine = F.kl_div(
                q_base_refined.clamp_min(1e-8).log(),
                q_base_transport.detach(),
                reduction="none",
            ).sum(dim=1).detach()
            node_entropy_margin = float(self.cfg.aptc_embedding_node_entropy_margin)
            node_entropy_scale = max(1e-4, float(self.cfg.aptc_embedding_node_entropy_scale))
            node_kl_margin = float(self.cfg.aptc_embedding_node_kl_margin)
            node_kl_scale = max(1e-4, float(self.cfg.aptc_embedding_node_kl_scale))
            node_entropy_gate = torch.sigmoid((base_node_entropy - embed_node_entropy - node_entropy_margin) / node_entropy_scale)
            node_kl_gate = torch.sigmoid((node_kl - node_kl_margin) / node_kl_scale)
            node_transport_gate = torch.sigmoid((node_transport - transport_margin) / transport_scale)
            node_refine_gate = torch.sigmoid((node_refine - refine_margin) / refine_scale)
            heuristic_node_gate = (node_entropy_gate * node_kl_gate).clamp(0.0, 1.0)
            learned_node_gate = (node_transport_gate * node_refine_gate).clamp(0.0, 1.0)
            if bool(self.cfg.aptc_embedding_rank_node_gate):
                rank_quantile = float(np.clip(self.cfg.aptc_embedding_node_rank_quantile, 0.50, 0.98))
                rank_scale = max(1e-4, float(self.cfg.aptc_embedding_node_rank_scale))
                disagreement_score = (heuristic_node_gate * learned_node_gate).detach()
                if disagreement_score.numel() > 8:
                    cutoff = torch.quantile(disagreement_score, rank_quantile)
                else:
                    cutoff = disagreement_score.mean()
                rank_gate = torch.sigmoid((disagreement_score - cutoff) / rank_scale)
                embed_node_gate = (heuristic_node_gate * rank_gate).clamp(0.0, 1.0)
            else:
                rank_gate = embedding.new_ones(embedding.shape[0])
                embed_node_gate = (heuristic_node_gate * learned_node_gate).clamp(0.0, 1.0)
        elif bool(self.cfg.aptc_embedding_view) and (
            bool(self.cfg.aptc_embedding_node_gate)
            or bool(self.cfg.aptc_embedding_learned_node_gate)
            or bool(self.cfg.aptc_embedding_hybrid_node_gate)
        ):
            node_entropy_margin = float(self.cfg.aptc_embedding_node_entropy_margin)
            node_entropy_scale = max(1e-4, float(self.cfg.aptc_embedding_node_entropy_scale))
            node_kl_margin = float(self.cfg.aptc_embedding_node_kl_margin)
            node_kl_scale = max(1e-4, float(self.cfg.aptc_embedding_node_kl_scale))
            base_node_entropy = posterior_entropy(q_base_mix.detach()) / max(1e-8, math.log(float(max(2, self.n_clusters))))
            embed_node_entropy = posterior_entropy(q_embed.detach()) / max(1e-8, math.log(float(max(2, self.n_clusters))))
            node_kl = F.kl_div(q_embed.clamp_min(1e-8).log(), q_base_mix.detach(), reduction="none").sum(dim=1).detach()
            node_entropy_gate = torch.sigmoid((base_node_entropy - embed_node_entropy - node_entropy_margin) / node_entropy_scale)
            node_kl_gate = torch.sigmoid((node_kl - node_kl_margin) / node_kl_scale)
            node_transport_gate = embedding.new_ones(embedding.shape[0])
            node_refine_gate = embedding.new_ones(embedding.shape[0])
            rank_gate = embedding.new_ones(embedding.shape[0])
            heuristic_node_gate = (node_entropy_gate * node_kl_gate).clamp(0.0, 1.0)
            if bool(self.cfg.aptc_embedding_learned_node_gate) or bool(self.cfg.aptc_embedding_hybrid_node_gate):
                base_conf = q_base_mix.detach().max(dim=1).values
                embed_conf = q_embed.detach().max(dim=1).values
                gate_features = torch.stack(
                    [
                        base_node_entropy,
                        embed_node_entropy,
                        node_kl,
                        base_conf,
                        embed_conf,
                    ],
                    dim=1,
                )
                node_logits = self.embed_node_gate_mlp(gate_features).squeeze(1)
                learned_node_gate = torch.sigmoid(node_logits)
            else:
                learned_node_gate = embedding.new_ones(embedding.shape[0])
            if bool(self.cfg.aptc_embedding_hybrid_node_gate):
                embed_node_gate = (heuristic_node_gate * learned_node_gate).clamp(0.0, 1.0)
            elif bool(self.cfg.aptc_embedding_learned_node_gate):
                embed_node_gate = learned_node_gate
            else:
                embed_node_gate = heuristic_node_gate
        else:
            node_entropy_gate = embedding.new_ones(embedding.shape[0])
            node_kl_gate = embedding.new_ones(embedding.shape[0])
            node_transport_gate = embedding.new_ones(embedding.shape[0])
            node_refine_gate = embedding.new_ones(embedding.shape[0])
            rank_gate = embedding.new_ones(embedding.shape[0])
            heuristic_node_gate = embedding.new_ones(embedding.shape[0])
            learned_node_gate = embedding.new_ones(embedding.shape[0])
            embed_node_gate = embedding.new_ones(embedding.shape[0])
        student_posterior_gate = embedding.new_tensor(0.0)
        student_posterior_prior_gate = embedding.new_tensor(1.0)
        student_posterior_weight = 0.0
        max_student_posterior = float(np.clip(self.cfg.aptc_student_posterior_weight, 0.0, 1.0))
        if max_student_posterior > 0.0:
            entropy_center = float(self.cfg.aptc_student_posterior_entropy_center)
            entropy_scale = max(1e-4, float(self.cfg.aptc_student_posterior_entropy_scale))
            prior_center = float(self.cfg.aptc_student_posterior_prior_center)
            prior_scale = max(1e-4, float(self.cfg.aptc_student_posterior_prior_scale))
            student_posterior_gate = torch.sigmoid((base_entropy_mean - entropy_center) / entropy_scale)
            student_posterior_prior_gate = torch.sigmoid((prior_center - prior_penalty) / prior_scale)
            student_posterior_weight = max_student_posterior * float(
                (student_posterior_gate * student_posterior_prior_gate).detach().cpu()
            )
        q_flow_seed = self._assignment_flow(
            student_t_distribution(embedding, warm_prototypes),
            edge_index,
            edge_prior,
            degree,
            homo,
            hetero,
            hard,
        )
        flow_posterior_gate = embedding.new_tensor(0.0)
        flow_posterior_prior_gate = embedding.new_tensor(1.0)
        flow_posterior_weight = 0.0
        max_flow_posterior = float(np.clip(self.cfg.aptc_flow_posterior_weight, 0.0, 1.0))
        max_flow_posterior = max_flow_posterior * float(np.clip(getattr(self, "runtime_flow_posterior_multiplier", 1.0), 0.0, 1.0))
        if max_flow_posterior > 0.0:
            flow_center = float(self.cfg.aptc_flow_posterior_flow_center)
            flow_scale = max(1e-4, float(self.cfg.aptc_flow_posterior_flow_scale))
            prior_center = float(self.cfg.aptc_flow_posterior_prior_center)
            prior_scale = max(1e-4, float(self.cfg.aptc_flow_posterior_prior_scale))
            flow_posterior_gate = torch.sigmoid((torch.log1p(base_flow_kl) - flow_center) / flow_scale)
            flow_posterior_prior_gate = torch.sigmoid((prior_center - prior_penalty) / prior_scale)
            flow_posterior_weight = max_flow_posterior * float(
                (flow_posterior_gate * flow_posterior_prior_gate).detach().cpu()
            )
        entropies = torch.stack(entropy_views, dim=1)
        gate_logits = self.view_gate(torch.cat([embedding, entropies], dim=1))
        gate = F.softmax(gate_logits, dim=1)
        raw_gate = gate
        embed_amplitude_score = embedding.new_zeros(embedding.shape[0])
        embed_amplitude_gate = embedding.new_ones(embedding.shape[0])
        embed_amplitude_floor = embedding.new_zeros(embedding.shape[0])
        amplitude_blend_used = 0.0
        if bool(self.cfg.aptc_embedding_view):
            if bool(self.cfg.aptc_embedding_residual_fusion):
                gate = torch.cat([base_gate, raw_gate[:, 3:4]], dim=1)
                residual_scale = max(0.0, float(self.cfg.aptc_embedding_gate_residual_scale))
                embed_weight = (residual_scale * embed_graph_gate * embed_node_gate.unsqueeze(1)).clamp_min(0.0)
                if bool(self.cfg.aptc_embedding_residual_delta_gate):
                    delta_tau = max(1e-4, float(self.cfg.aptc_embedding_delta_temperature))
                    embed_delta = F.relu((q_embed - q_base_mix) / delta_tau)
                    if bool(self.cfg.aptc_embedding_amplitude_soft_gate):
                        amp_center = float(self.cfg.aptc_embedding_amplitude_center)
                        amp_scale = max(1e-4, float(self.cfg.aptc_embedding_amplitude_scale))
                        amp_floor = float(np.clip(self.cfg.aptc_embedding_amplitude_gate_floor, 0.0, 1.0))
                        amp_floor = amp_floor * float(
                            np.clip(getattr(self, "runtime_embedding_amplitude_floor_multiplier", 1.0), 0.0, 1.0)
                        )
                        embed_amplitude_score = embed_delta.sum(dim=1)
                        embed_amplitude_gate = torch.sigmoid((embed_amplitude_score - amp_center) / amp_scale)
                        if amp_floor > 0.0:
                            if bool(self.cfg.aptc_embedding_amplitude_graph_adaptive_floor):
                                embed_amplitude_floor = amp_floor * (1.0 - embed_graph_gate.detach()).expand_as(embed_amplitude_score)
                            elif bool(self.cfg.aptc_embedding_amplitude_inverse_node_floor):
                                node_floor_blend = float(np.clip(self.cfg.aptc_embedding_amplitude_node_floor_blend, 0.0, 1.0))
                                if bool(self.cfg.aptc_embedding_amplitude_graph_conditioned_blend):
                                    prior_center = float(self.cfg.aptc_embedding_amplitude_blend_prior_center)
                                    prior_scale = max(1e-4, float(self.cfg.aptc_embedding_amplitude_blend_prior_scale))
                                    prior_mix_gate = torch.sigmoid((prior_penalty - prior_center) / prior_scale)
                                    node_floor_blend = node_floor_blend * float(
                                        ((1.0 - embed_graph_gate.detach()) * prior_mix_gate).clamp(0.0, 1.0).item()
                                    )
                                amplitude_blend_used = node_floor_blend
                                node_floor = amp_floor * (1.0 - embed_node_gate.detach())
                                base_floor = embed_amplitude_score.new_full(embed_amplitude_score.shape, amp_floor)
                                embed_amplitude_floor = (1.0 - node_floor_blend) * base_floor + node_floor_blend * node_floor
                            elif bool(self.cfg.aptc_embedding_amplitude_node_adaptive_floor):
                                node_floor_blend = float(np.clip(self.cfg.aptc_embedding_amplitude_node_floor_blend, 0.0, 1.0))
                                if bool(self.cfg.aptc_embedding_amplitude_graph_conditioned_blend):
                                    prior_center = float(self.cfg.aptc_embedding_amplitude_blend_prior_center)
                                    prior_scale = max(1e-4, float(self.cfg.aptc_embedding_amplitude_blend_prior_scale))
                                    prior_mix_gate = torch.sigmoid((prior_penalty - prior_center) / prior_scale)
                                    node_floor_blend = node_floor_blend * float(
                                        ((1.0 - embed_graph_gate.detach()) * prior_mix_gate).clamp(0.0, 1.0).item()
                                    )
                                amplitude_blend_used = node_floor_blend
                                node_floor = amp_floor * embed_node_gate.detach()
                                base_floor = embed_amplitude_score.new_full(embed_amplitude_score.shape, amp_floor)
                                embed_amplitude_floor = (1.0 - node_floor_blend) * base_floor + node_floor_blend * node_floor
                            else:
                                embed_amplitude_floor = embed_amplitude_score.new_full(embed_amplitude_score.shape, amp_floor)
                            embed_amplitude_gate = embed_amplitude_floor + (1.0 - embed_amplitude_floor) * embed_amplitude_gate
                        embed_weight = embed_weight * embed_amplitude_gate.unsqueeze(1)
                    else:
                        embed_amplitude_score = embed_delta.sum(dim=1)
                    q_mix = q_base_mix + embed_weight * embed_delta
                else:
                    q_mix = q_base_mix + embed_weight * q_embed
                q_mix = q_mix / q_mix.sum(dim=1, keepdim=True).clamp_min(1e-8)
            else:
                gate = gate.clone()
                gate[:, 3:4] = gate[:, 3:4] * embed_graph_gate
                gate = gate / gate.sum(dim=1, keepdim=True).clamp_min(1e-8)
                q_mix = sum(gate[:, i : i + 1] * view for i, view in enumerate(q_views))
                q_mix = q_mix / q_mix.sum(dim=1, keepdim=True).clamp_min(1e-8)
        else:
            q_mix = sum(gate[:, i : i + 1] * view for i, view in enumerate(q_views))
            q_mix = q_mix / q_mix.sum(dim=1, keepdim=True).clamp_min(1e-8)
        if student_posterior_weight > 0.0:
            q_mix = (1.0 - student_posterior_weight) * q_mix + student_posterior_weight * q_student
            q_mix = q_mix / q_mix.sum(dim=1, keepdim=True).clamp_min(1e-8)
        if flow_posterior_weight > 0.0:
            q_mix = (1.0 - flow_posterior_weight) * q_mix + flow_posterior_weight * q_flow_seed
            q_mix = q_mix / q_mix.sum(dim=1, keepdim=True).clamp_min(1e-8)
        rayleigh = torch.stack(rayleigh_views)

        q_transport = sinkhorn_transport(
            q_mix,
            prior,
            epsilon=float(self.cfg.aptc_sinkhorn_epsilon),
            iters=int(self.cfg.aptc_sinkhorn_iters),
        )
        q_refined = self._refine(
            q_transport,
            prior,
            edge_index,
            edge_prior,
            degree,
            homo,
            hetero,
            hard,
        )
        dynamic_prototypes = adaptive_prototypes(embedding, q_refined)
        if self.training:
            with torch.no_grad():
                momentum = float(np.clip(self.cfg.aptc_proto_momentum, 0.0, 0.999))
                self.prototype_memory.mul_(momentum).add_((1.0 - momentum) * dynamic_prototypes.detach())
                self.prototype_memory.copy_(F.normalize(self.prototype_memory, p=2, dim=1))
        prototypes = F.normalize(
            (1.0 - float(self.cfg.aptc_dynamic_proto_weight)) * base_prototypes
            + float(self.cfg.aptc_dynamic_proto_weight) * dynamic_prototypes.detach(),
            p=2,
            dim=1,
        )
        compact = torch.sum(q_refined * torch.cdist(embedding, prototypes).pow(2), dim=1).mean()
        return {
            "q": q_mix,
            "q_transport": q_transport,
            "q_refined": q_refined,
            "q_attr": q_attr,
            "q_low": q_low,
            "q_high": q_high,
            "q_embed": q_embed,
            "view_gate": gate,
            "view_gate_raw": raw_gate,
            "base_view_gate": base_gate,
            "view_rayleigh": rayleigh,
            "view_logit_std": torch.stack([attr_logit_std, low_logit_std, high_logit_std, embed_logit_std]),
            "cluster_prior": prior,
            "cluster_centers": prototypes,
            "compact_loss": compact,
            "base_flow_kl": base_flow_kl,
            "student_posterior_gate": student_posterior_gate,
            "student_posterior_prior_gate": student_posterior_prior_gate,
            "student_posterior_weight": embedding.new_tensor(float(student_posterior_weight)),
            "flow_posterior_gate": flow_posterior_gate,
            "flow_posterior_prior_gate": flow_posterior_prior_gate,
            "flow_posterior_weight": embedding.new_tensor(float(flow_posterior_weight)),
            "prior_penalty": prior_penalty,
            "q_flow_seed": q_flow_seed,
            "embed_graph_gate": embed_graph_gate,
            "embed_graph_gate_raw": raw_embed_gate,
            "embed_flow_gate": flow_gate,
            "embed_prior_gate": prior_gate,
            "embed_std_gate": std_gate,
            "embed_entropy_gate": entropy_gate,
            "embed_node_gate": embed_node_gate,
            "embed_node_entropy_gate": node_entropy_gate,
            "embed_node_kl_gate": node_kl_gate,
            "embed_node_transport_gate": node_transport_gate,
            "embed_node_refine_gate": node_refine_gate,
            "embed_node_rank_gate": rank_gate,
            "embed_node_gate_heuristic": heuristic_node_gate,
            "embed_node_gate_learned": learned_node_gate,
            "embed_amplitude_score": embed_amplitude_score,
            "embed_amplitude_gate": embed_amplitude_gate,
            "embed_amplitude_floor": embed_amplitude_floor,
            "embed_amplitude_blend": embedding.new_tensor(float(amplitude_blend_used)),
            "base_entropy_mean": base_entropy_mean,
            "embed_entropy_mean": embed_entropy_mean,
            "base_logit_std": base_logit_std,
        }

    def _assignment_flow(
        self,
        q: torch.Tensor,
        edge_index: torch.Tensor,
        edge_prior: torch.Tensor,
        degree: torch.Tensor,
        homo: torch.Tensor,
        hetero: torch.Tensor,
        hard: torch.Tensor,
    ) -> torch.Tensor:
        power = max(1.0, float(self.cfg.assignment_sharpen_power))
        y0 = q.pow(power)
        y0 = y0 / y0.sum(dim=1, keepdim=True).clamp_min(1e-8)
        y = y0
        uniform = torch.full_like(q, 1.0 / self.n_clusters)
        attract_w = (homo + 0.25 * hard).clamp_min(1e-6)
        raw_floor = float(self.cfg.assignment_raw_repel_floor)
        raw_edge = (edge_prior >= 0.999).to(q.dtype)
        repel_w = (hetero + raw_floor * raw_edge).clamp_min(1e-6)
        for _ in range(max(0, int(self.cfg.assignment_flow_steps))):
            attract = normalized_spmm(edge_index, attract_w, y, degree.numel())
            repel = normalized_spmm(edge_index, repel_w, y, degree.numel())
            logits = (
                self.cfg.assignment_fidelity_weight * torch.log(y0.clamp_min(1e-8))
                + self.cfg.assignment_attract_weight * attract
                + self.cfg.assignment_repel_weight * (uniform - repel)
            )
            y = F.softmax(logits / max(1e-4, float(self.cfg.assignment_temperature)), dim=1)
        return y

    def _posterior_from_view(self, view: torch.Tensor, prototypes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mode = str(getattr(self.cfg, "aptc_posterior_mode", "cosine")).lower()
        if mode == "student_t":
            q = student_t_distribution(view, prototypes, alpha=float(getattr(self.cfg, "aptc_student_alpha", 1.0)))
            centered = q.clamp_min(1e-8).log()
            std = centered.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-6)
            return q, std.mean()
        logits = view @ prototypes.T
        logits = logits / max(1e-4, float(self.cfg.aptc_temperature))
        centered = logits - logits.mean(dim=1, keepdim=True)
        std = centered.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-6)
        floor = max(0.0, float(self.cfg.aptc_logit_std_floor))
        if floor > 0.0:
            strength = float(np.clip(self.cfg.aptc_logit_std_strength, 0.0, 1.0))
            lift = F.relu(centered.new_tensor(floor) - std) / std
            centered = centered * (1.0 + strength * lift)
        return F.softmax(centered, dim=1), std.mean()

    def _refine(
        self,
        q: torch.Tensor,
        prior: torch.Tensor,
        edge_index: torch.Tensor,
        edge_prior: torch.Tensor,
        degree: torch.Tensor,
        homo: torch.Tensor,
        hetero: torch.Tensor,
        hard: torch.Tensor,
    ) -> torch.Tensor:
        y = q
        uniform = prior.unsqueeze(0).expand_as(q)
        attract_w = (homo + float(self.cfg.aptc_hard_attract_weight) * hard).clamp_min(1e-6)
        raw_floor = float(self.cfg.assignment_raw_repel_floor)
        raw_edge = (edge_prior >= 0.999).to(q.dtype)
        repel_w = (hetero + raw_floor * raw_edge).clamp_min(1e-6)
        if bool(self.cfg.aptc_repel_mass_balance):
            repel_scale = (attract_w.mean().detach() / repel_w.mean().detach().clamp_min(1e-6)).clamp(
                float(self.cfg.aptc_min_repel_scale),
                float(self.cfg.aptc_max_repel_scale),
            )
        else:
            repel_scale = q.new_tensor(1.0)
        for _ in range(max(0, int(self.cfg.aptc_refine_steps))):
            attract = normalized_spmm(edge_index, attract_w, y, degree.numel())
            repel = normalized_spmm(edge_index, repel_w, y, degree.numel())
            logits = (
                float(self.cfg.assignment_fidelity_weight) * torch.log(q.clamp_min(1e-8))
                + float(self.cfg.assignment_attract_weight) * attract
                + float(self.cfg.assignment_repel_weight) * repel_scale * (uniform - repel)
            )
            y = sinkhorn_transport(
                F.softmax(logits / max(1e-4, float(self.cfg.assignment_temperature)), dim=1),
                prior,
                epsilon=float(self.cfg.aptc_sinkhorn_epsilon),
                iters=int(self.cfg.aptc_sinkhorn_iters),
            )
        return y


def student_t_distribution(z: torch.Tensor, centers: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    dist2 = torch.cdist(z, centers).pow(2)
    q = (1.0 + dist2 / float(alpha)).pow(-(float(alpha) + 1.0) / 2.0)
    return q / q.sum(dim=1, keepdim=True).clamp_min(1e-8)


def teacher_embedding_agreement(
    embedding: torch.Tensor,
    teacher: torch.Tensor,
    k: int,
) -> torch.Tensor:
    with torch.no_grad():
        z = F.normalize(embedding.detach(), p=2, dim=1)
        teacher_label = teacher.detach().argmax(dim=1)
        n = int(z.shape[0])
        if n <= 1:
            return z.new_ones(n)
        k_eff = min(max(1, int(k)), n - 1)
        if n <= 12_000:
            sim = z @ z.T
            sim.fill_diagonal_(-1.0)
            idx = sim.topk(k_eff, dim=1).indices
            neighbor_labels = teacher_label[idx]
        else:
            sample_size = min(4096, n)
            sample_idx = torch.linspace(0, n - 1, steps=sample_size, device=z.device).long()
            z_sample = z[sample_idx]
            label_sample = teacher_label[sample_idx]
            sim = z @ z_sample.T
            idx = sim.topk(min(k_eff, sample_size), dim=1).indices
            neighbor_labels = label_sample[idx]
        return (neighbor_labels == teacher_label[:, None]).to(z.dtype).mean(dim=1)


class EndToEndSECTCoCoModule(nn.Module):
    def __init__(
        self,
        in_dim: int,
        n_clusters: int,
        cfg: E2ESECTCoCoConfig,
        degree: torch.Tensor,
        edge_index: torch.Tensor,
        edge_prior: torch.Tensor,
    ):
        super().__init__()
        self.cfg = cfg
        self.n_clusters = int(n_clusters)
        self.register_buffer("degree", degree.float())
        self.register_buffer("edge_index", edge_index.long())
        self.register_buffer("edge_prior", edge_prior.float())
        self.register_buffer("init_teacher", torch.empty(0))
        self.register_buffer("init_prototypes", torch.empty(0))
        self.runtime_proto_readout_multiplier = 1.0
        self.runtime_embedding_gate_multiplier = 1.0
        self.runtime_embedding_amplitude_floor_multiplier = 1.0
        self.runtime_flow_posterior_multiplier = 1.0
        self.runtime_flow_mix_multiplier = 1.0
        self.encoder = SparseInputEncoder(in_dim, cfg.hidden_dim, cfg.embed_dim, cfg.dropout)
        self.raw_projector = nn.Linear(in_dim, cfg.embed_dim, bias=False)
        self.confidence = AdaptiveEdgeConfidence(
            edge_feature_dim=13,
            evidence_dim=4 if bool(cfg.edge_prior_evidence) else 3,
            recenter_strength=cfg.edge_logit_recenter_strength,
            recenter_scale=cfg.edge_logit_recenter_scale,
            min_std=cfg.edge_logit_recenter_min_std,
            alpha_smoothing=cfg.edge_alpha_smoothing,
        )
        self.contraction = DifferentiableTopologyContraction(
            cfg.init_low,
            cfg.init_high,
            cfg.min_threshold_gap,
            cfg.threshold_tau,
        )
        self.contraction.use_quantile_anchor = bool(cfg.quantile_threshold_anchor)
        self.contraction.quantile_anchor_weight = float(cfg.quantile_threshold_weight)
        self.low_filter_gate = nn.Parameter(torch.tensor(0.65, dtype=torch.float32))
        self.high_filter_gate = nn.Parameter(torch.tensor(0.25, dtype=torch.float32))
        self.highpass_scale_logit = nn.Parameter(torch.tensor(_logit((2.0 - 0.5) / (4.0 - 0.5)), dtype=torch.float32))
        self.raw_leakage_gate = nn.Parameter(torch.tensor(float(cfg.raw_leakage_init), dtype=torch.float32))
        self.projection = nn.Sequential(
            nn.Linear(cfg.embed_dim * 3, cfg.projection_dim),
            nn.LayerNorm(cfg.projection_dim),
            nn.GELU(),
            nn.Linear(cfg.projection_dim, cfg.projection_dim),
        )
        self.cluster_head = AdaptivePosteriorTransportHead(cfg.embed_dim, cfg.projection_dim, n_clusters, cfg)
        self.raw_skip = nn.Linear(in_dim, cfg.projection_dim, bias=False)
        self.raw_skip_gate = nn.Parameter(torch.tensor(_logit(cfg.raw_skip_weight), dtype=torch.float32))
        self.decoder = nn.Sequential(
            nn.Linear(cfg.projection_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, in_dim),
        )
        self._init_raw_skip()

    def _init_raw_skip(self) -> None:
        with torch.no_grad():
            self.raw_skip.weight.zero_()
            diag = min(self.raw_skip.weight.shape[0], self.raw_skip.weight.shape[1])
            self.raw_skip.weight[:diag, :diag] = torch.eye(diag, dtype=self.raw_skip.weight.dtype)

    def _frontend_pass(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z_attr = self.encoder(x)
        z_raw = F.normalize(self.raw_projector(x), p=2, dim=1)
        src, dst = self.edge_index
        edge_features, evidences = self._edge_features(z_attr, z_raw, src, dst)
        score, alpha, edge_logit = self.confidence(edge_features, evidences)
        homo, hetero, hard, low, high = self.contraction(score)
        support_weight = self._support_weights(score, homo, hetero, hard)
        raw_edge_weight = self.edge_prior.to(score.dtype).clamp(0.0, 1.0)
        raw_leak_beta = torch.sigmoid(self.raw_leakage_gate)
        low_support_weight = ((1.0 - raw_leak_beta) * support_weight + raw_leak_beta * raw_edge_weight).clamp_min(1e-6)
        low_view = self._diffuse(z_attr, low_support_weight, steps=self.cfg.lowpass_steps)
        hetero_view = self._signed_highpass(z_attr, hetero, steps=self.cfg.highpass_steps)
        z_cross_alignment = F.cosine_similarity(low_view, hetero_view, dim=1).mean()
        low_gate = torch.sigmoid(self.low_filter_gate)
        high_gate = torch.sigmoid(self.high_filter_gate)
        fused = torch.cat(
            [
                z_attr,
                low_gate * low_view + (1.0 - low_gate) * z_attr,
                high_gate * hetero_view,
            ],
            dim=1,
        )
        core_embedding = self.projection(fused)
        raw_skip = F.normalize(self.raw_skip(x), p=2, dim=1)
        skip_gate = torch.sigmoid(self.raw_skip_gate)
        embedding = F.normalize((1.0 - skip_gate) * core_embedding + skip_gate * raw_skip, p=2, dim=1)
        return {
            "z_attr": z_attr,
            "z_raw": z_raw,
            "low_view": low_view,
            "hetero_view": hetero_view,
            "z_cross_alignment": z_cross_alignment,
            "embedding": embedding,
            "score": score,
            "edge_logit": edge_logit,
            "alpha": alpha,
            "homo": homo,
            "hetero": hetero,
            "hard": hard,
            "low_threshold": low,
            "high_threshold": high,
            "support_weight": support_weight,
            "low_support_weight": low_support_weight,
            "raw_leak_beta": raw_leak_beta,
            "edge_features": edge_features,
            "evidences": evidences,
        }

    def _aptc_pass(self, frontend: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        embedding = frontend["embedding"]
        cluster_out = self.cluster_head(
            embedding=embedding,
            z_attr=frontend["z_attr"],
            low_view=frontend["low_view"],
            hetero_view=frontend["hetero_view"],
            edge_index=self.edge_index,
            edge_prior=self.edge_prior,
            degree=self.degree,
            homo=frontend["homo"],
            hetero=frontend["hetero"],
            hard=frontend["hard"],
        )
        q_flow_anchor = cluster_out["q_flow_seed"]
        flow_mix_gate = embedding.new_tensor(0.0)
        flow_mix_prior_gate = embedding.new_tensor(1.0)
        flow_mix_weight = 0.0
        q_main = cluster_out["q_refined"]
        main_flow_gate = embedding.new_tensor(0.0)
        main_prior_gate = embedding.new_tensor(1.0)
        main_entropy_gate = embedding.new_tensor(0.0)
        main_flow_weight = embedding.new_tensor(0.0)
        main_mode = str(getattr(self.cfg, "main_posterior_mode", "refined")).lower()
        if main_mode == "flow":
            q_main = q_flow_anchor
        elif main_mode == "blend":
            flow_main_weight = float(np.clip(getattr(self.cfg, "main_posterior_flow_weight", 0.5), 0.0, 1.0))
            main_flow_weight = embedding.new_tensor(flow_main_weight)
            q_main = (1.0 - flow_main_weight) * cluster_out["q_refined"] + flow_main_weight * q_flow_anchor
            q_main = q_main / q_main.sum(dim=1, keepdim=True).clamp_min(1e-8)
        elif main_mode == "adaptive_blend":
            flow_center = float(getattr(self.cfg, "main_posterior_flow_center", 1.0))
            flow_scale = max(1e-4, float(getattr(self.cfg, "main_posterior_flow_scale", 0.35)))
            prior_center = float(getattr(self.cfg, "main_posterior_prior_center", 0.12))
            prior_scale = max(1e-4, float(getattr(self.cfg, "main_posterior_prior_scale", 0.06)))
            entropy_center = float(getattr(self.cfg, "main_posterior_entropy_center", 0.82))
            entropy_scale = max(1e-4, float(getattr(self.cfg, "main_posterior_entropy_scale", 0.06)))
            entropy_rescue_weight = float(np.clip(getattr(self.cfg, "main_posterior_entropy_rescue_weight", 0.0), 0.0, 1.0))
            max_flow_main = float(np.clip(getattr(self.cfg, "main_posterior_flow_weight", 0.5), 0.0, 1.0))
            main_flow_gate = torch.sigmoid((torch.log1p(cluster_out["base_flow_kl"]) - flow_center) / flow_scale)
            main_prior_gate = torch.sigmoid((prior_center - cluster_out["prior_penalty"]) / prior_scale)
            main_entropy_gate = torch.sigmoid((cluster_out["base_entropy_mean"] - entropy_center) / entropy_scale)
            gated_prior = ((1.0 - entropy_rescue_weight) * main_prior_gate + entropy_rescue_weight * torch.maximum(main_prior_gate, main_entropy_gate)).clamp(0.0, 1.0)
            flow_main_weight = max_flow_main * float((main_flow_gate * gated_prior).detach().cpu())
            main_flow_weight = embedding.new_tensor(flow_main_weight)
            q_main = (1.0 - flow_main_weight) * cluster_out["q_refined"] + flow_main_weight * q_flow_anchor
            q_main = q_main / q_main.sum(dim=1, keepdim=True).clamp_min(1e-8)
        compact_loss = cluster_out["compact_loss"]
        if bool(getattr(self.cfg, "main_posterior_align_compact", False)):
            compact_loss = torch.sum(q_main * torch.cdist(embedding, cluster_out["cluster_centers"]).pow(2), dim=1).mean()
        q_final = q_main
        max_flow_mix = float(np.clip(self.cfg.aptc_flow_mix_weight, 0.0, 1.0))
        max_flow_mix = max_flow_mix * float(np.clip(getattr(self, "runtime_flow_mix_multiplier", 1.0), 0.0, 1.0))
        if max_flow_mix > 0.0:
            flow_center = float(self.cfg.aptc_flow_mix_flow_center)
            flow_scale = max(1e-4, float(self.cfg.aptc_flow_mix_flow_scale))
            prior_center = float(self.cfg.aptc_flow_mix_prior_center)
            prior_scale = max(1e-4, float(self.cfg.aptc_flow_mix_prior_scale))
            flow_mix_gate = torch.sigmoid((torch.log1p(cluster_out["base_flow_kl"]) - flow_center) / flow_scale)
            flow_mix_prior_gate = torch.sigmoid((prior_center - cluster_out["prior_penalty"]) / prior_scale)
            flow_mix_weight = max_flow_mix * float((flow_mix_gate * flow_mix_prior_gate).detach().cpu())
            q_final = (1.0 - flow_mix_weight) * cluster_out["q_refined"] + flow_mix_weight * q_flow_anchor
            q_final = q_final / q_final.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return {
            **frontend,
            "embedding": embedding,
            "q": cluster_out["q"],
            "q_transport": cluster_out["q_transport"],
            "q_refined": q_final,
            "q_flow": q_flow_anchor,
            "q_aptc_raw": cluster_out["q_refined"],
            "q_attr": cluster_out["q_attr"],
            "q_low": cluster_out["q_low"],
            "q_high": cluster_out["q_high"],
            "q_embed": cluster_out["q_embed"],
            "view_gate": cluster_out["view_gate"],
            "view_gate_raw": cluster_out["view_gate_raw"],
            "base_view_gate": cluster_out["base_view_gate"],
            "view_rayleigh": cluster_out["view_rayleigh"],
            "view_logit_std": cluster_out["view_logit_std"],
            "cluster_prior": cluster_out["cluster_prior"],
            "cluster_centers": cluster_out["cluster_centers"],
            "head_compact": compact_loss,
            "base_flow_kl": cluster_out["base_flow_kl"],
            "main_flow_gate": main_flow_gate,
            "main_prior_gate": main_prior_gate,
            "main_entropy_gate": main_entropy_gate,
            "main_flow_weight": main_flow_weight,
            "student_posterior_gate": cluster_out["student_posterior_gate"],
            "student_posterior_prior_gate": cluster_out["student_posterior_prior_gate"],
            "student_posterior_weight": cluster_out["student_posterior_weight"],
            "flow_posterior_gate": cluster_out["flow_posterior_gate"],
            "flow_posterior_prior_gate": cluster_out["flow_posterior_prior_gate"],
            "flow_posterior_weight": cluster_out["flow_posterior_weight"],
            "flow_mix_gate": flow_mix_gate,
            "flow_mix_prior_gate": flow_mix_prior_gate,
            "flow_mix_weight": embedding.new_tensor(float(flow_mix_weight)),
            "prior_penalty": cluster_out["prior_penalty"],
            "embed_graph_gate": cluster_out["embed_graph_gate"],
            "embed_graph_gate_raw": cluster_out["embed_graph_gate_raw"],
            "embed_flow_gate": cluster_out["embed_flow_gate"],
            "embed_prior_gate": cluster_out["embed_prior_gate"],
            "embed_std_gate": cluster_out["embed_std_gate"],
            "embed_entropy_gate": cluster_out["embed_entropy_gate"],
            "embed_node_gate": cluster_out["embed_node_gate"],
            "embed_node_entropy_gate": cluster_out["embed_node_entropy_gate"],
            "embed_node_kl_gate": cluster_out["embed_node_kl_gate"],
            "embed_node_transport_gate": cluster_out["embed_node_transport_gate"],
            "embed_node_refine_gate": cluster_out["embed_node_refine_gate"],
            "embed_node_rank_gate": cluster_out["embed_node_rank_gate"],
            "embed_node_gate_heuristic": cluster_out["embed_node_gate_heuristic"],
            "embed_node_gate_learned": cluster_out["embed_node_gate_learned"],
            "embed_amplitude_score": cluster_out["embed_amplitude_score"],
            "embed_amplitude_gate": cluster_out["embed_amplitude_gate"],
            "embed_amplitude_floor": cluster_out["embed_amplitude_floor"],
            "embed_amplitude_blend": cluster_out["embed_amplitude_blend"],
            "base_entropy_mean": cluster_out["base_entropy_mean"],
            "embed_entropy_mean": cluster_out["embed_entropy_mean"],
            "base_logit_std": cluster_out["base_logit_std"],
        }

    def encode_views(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        frontend = self._frontend_pass(x)
        return {**frontend, **self._aptc_pass(frontend)}

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.encode_views(x)

    def _select_loss_posterior(
        self,
        out: dict[str, torch.Tensor],
        source: str,
        *,
        flow_weight: float | None = None,
    ) -> tuple[torch.Tensor, float]:
        source = str(source).lower()
        blend_weight = float(np.clip(getattr(self.cfg, "loss_posterior_flow_weight", 0.0), 0.0, 1.0))
        if flow_weight is not None:
            blend_weight = float(np.clip(flow_weight, 0.0, 1.0))
        if source == "q_flow":
            return out["q_flow"], 0.0
        if source == "q_transport":
            return out["q_transport"], 0.0
        if source == "q_aptc_raw":
            return out["q_aptc_raw"], 0.0
        if source == "q_blend":
            q = (1.0 - blend_weight) * out["q_aptc_raw"] + blend_weight * out["q_flow"]
            q = q / q.sum(dim=1, keepdim=True).clamp_min(1e-8)
            return q, blend_weight
        return out["q_refined"], 0.0

    def _resolve_loss_sources(self) -> tuple[str, str]:
        cfg = self.cfg
        loss_source = str(getattr(cfg, "loss_posterior_source", "q_refined")).lower()
        reg_source = str(getattr(cfg, "loss_regularizer_source", "same")).lower()
        late_start = int(getattr(cfg, "loss_posterior_late_start_epoch", -1))
        current_epoch = int(getattr(self, "runtime_epoch", -1))
        if late_start >= 0 and current_epoch >= late_start:
            late_loss_source = str(getattr(cfg, "loss_posterior_late_source", "")).strip().lower()
            late_reg_source = str(getattr(cfg, "loss_regularizer_late_source", "")).strip().lower()
            if late_loss_source:
                loss_source = late_loss_source
            if late_reg_source:
                reg_source = late_reg_source
        return loss_source, reg_source

    def _resolve_target_bootstrap(self) -> tuple[str, float]:
        cfg = self.cfg
        target_source = str(getattr(cfg, "target_bootstrap_source", "q_refined")).lower()
        flow_weight = float(np.clip(getattr(cfg, "target_bootstrap_flow_weight", 0.0), 0.0, 1.0))
        late_start = int(getattr(cfg, "target_bootstrap_late_start_epoch", -1))
        current_epoch = int(getattr(self, "runtime_epoch", -1))
        if late_start >= 0 and current_epoch >= late_start:
            late_source = str(getattr(cfg, "target_bootstrap_late_source", "")).strip().lower()
            if late_source:
                target_source = late_source
                flow_weight = float(np.clip(getattr(cfg, "target_bootstrap_late_flow_weight", flow_weight), 0.0, 1.0))
        return target_source, flow_weight

    def _adaptive_target_flow_weight(self, out: dict[str, torch.Tensor], base_weight: float) -> float:
        if not bool(getattr(self.cfg, "target_bootstrap_adaptive_flow", False)):
            return float(np.clip(base_weight, 0.0, 1.0))
        min_w = float(np.clip(getattr(self.cfg, "target_bootstrap_flow_min", base_weight), 0.0, 1.0))
        max_w = float(np.clip(getattr(self.cfg, "target_bootstrap_flow_max", base_weight), min_w, 1.0))
        entropy_center = float(getattr(self.cfg, "target_bootstrap_entropy_center", 0.94))
        entropy_scale = max(1e-4, float(getattr(self.cfg, "target_bootstrap_entropy_scale", 0.04)))
        amp_center = float(getattr(self.cfg, "target_bootstrap_amplitude_center", 0.30))
        amp_scale = max(1e-4, float(getattr(self.cfg, "target_bootstrap_amplitude_scale", 0.08)))
        entropy = out["base_entropy_mean"].detach()
        amplitude = out["embed_amplitude_score"].mean().detach()
        entropy_gate = torch.sigmoid((entropy_center - entropy) / entropy_scale)
        amplitude_gate = torch.sigmoid((amplitude - amp_center) / amp_scale)
        gate = (entropy_gate * amplitude_gate).clamp(0.0, 1.0)
        weight = min_w + (max_w - min_w) * float(gate.cpu())
        return float(np.clip(weight, 0.0, 1.0))

    def loss(self, x: torch.Tensor, target: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, float]]:
        cfg = self.cfg
        out = self.forward(x)
        loss_source, reg_source = self._resolve_loss_sources()
        q_cluster, loss_flow_weight = self._select_loss_posterior(out, loss_source)
        if reg_source == "same":
            q_reg = q_cluster
        else:
            q_reg, _ = self._select_loss_posterior(out, reg_source, flow_weight=loss_flow_weight)
        if target is None:
            target = target_distribution(q_cluster).detach()
        cluster_loss = F.kl_div(q_cluster.clamp_min(1e-8).log(), target, reduction="batchmean")
        transport_loss = F.kl_div(q_reg.clamp_min(1e-8).log(), out["q_transport"].detach(), reduction="batchmean")
        teacher_conf = q_reg.new_zeros(q_reg.shape[0])
        teacher_margin = q_reg.new_zeros(q_reg.shape[0])
        teacher_agreement = q_reg.new_zeros(q_reg.shape[0])
        conf_weight = q_reg.new_zeros(q_reg.shape[0])
        conf_floor = float(cfg.aptc_teacher_conf_floor)
        teacher_conf_center = q_reg.new_tensor(float(cfg.aptc_teacher_conf_center))
        teacher_conf_scale = q_reg.new_tensor(max(1.0 - float(cfg.aptc_teacher_conf_center), 1e-8))
        teacher_reliability_mode_id = 0.0
        if self.init_teacher.numel() == q_reg.numel():
            teacher = self.init_teacher.detach().clamp_min(1e-8)
            q_log = q_reg.clamp_min(1e-8).log()
            per_node_kl = F.kl_div(q_log, teacher, reduction="none").sum(dim=1)
            topk = torch.topk(teacher, k=min(2, teacher.shape[1]), dim=1).values.detach()
            teacher_prob = topk[:, 0]
            if topk.shape[1] > 1:
                teacher_margin = (topk[:, 0] - topk[:, 1]).clamp_min(0.0)
            else:
                teacher_margin = teacher_prob
            reliability_mode = str(getattr(cfg, "aptc_teacher_reliability_mode", "prob")).lower()
            if reliability_mode == "margin":
                teacher_conf = teacher_margin.detach()
                teacher_reliability_mode_id = 1.0
            elif reliability_mode == "agreement":
                teacher_agreement = teacher_embedding_agreement(
                    out["embedding"],
                    teacher,
                    k=int(getattr(cfg, "aptc_teacher_agreement_k", 10)),
                ).detach()
                agreement_floor = float(getattr(cfg, "aptc_teacher_agreement_floor", 0.10))
                agreement_power = max(1e-4, float(getattr(cfg, "aptc_teacher_agreement_power", 1.0)))
                agreement_weight = agreement_floor + (1.0 - agreement_floor) * teacher_agreement.clamp(0.0, 1.0).pow(
                    agreement_power
                )
                teacher_conf = (teacher_margin * agreement_weight).detach()
                teacher_reliability_mode_id = 2.0
            else:
                teacher_conf = teacher_prob.detach()
                teacher_agreement = teacher_embedding_agreement(
                    out["embedding"],
                    teacher,
                    k=int(getattr(cfg, "aptc_teacher_agreement_k", 10)),
                ).detach()
            conf_center = float(cfg.aptc_teacher_conf_center)
            conf_power = max(1e-4, float(cfg.aptc_teacher_conf_power))
            conf_mode = str(getattr(cfg, "aptc_teacher_conf_center_mode", "fixed")).lower()
            if conf_mode in {"adaptive_quantile", "quantile"} and teacher_conf.numel() > 1:
                conf_q = float(np.clip(float(cfg.aptc_teacher_conf_quantile), 0.0, 0.99))
                teacher_conf_center = torch.quantile(teacher_conf, conf_q).detach()
                teacher_conf_scale = (teacher_conf.max().detach() - teacher_conf_center).clamp_min(
                    float(cfg.aptc_teacher_conf_min_scale)
                )
            elif conf_mode == "adaptive_mean" and teacher_conf.numel() > 0:
                teacher_conf_center = teacher_conf.mean().detach()
                teacher_conf_scale = (teacher_conf.max().detach() - teacher_conf_center).clamp_min(
                    float(cfg.aptc_teacher_conf_min_scale)
                )
            elif conf_mode == "adaptive_median" and teacher_conf.numel() > 0:
                teacher_conf_center = teacher_conf.median().detach()
                teacher_conf_scale = (teacher_conf.max().detach() - teacher_conf_center).clamp_min(
                    float(cfg.aptc_teacher_conf_min_scale)
                )
            else:
                teacher_conf_center = teacher_conf.new_tensor(conf_center)
                teacher_conf_scale = teacher_conf.new_tensor(max(1.0 - conf_center, 1e-8))
            conf_weight = ((teacher_conf - teacher_conf_center).clamp_min(0.0) / teacher_conf_scale).clamp(0.0, 1.0)
            conf_weight = conf_floor + (1.0 - conf_floor) * conf_weight.pow(conf_power)
            init_teacher_loss = (conf_weight * per_node_kl).sum() / conf_weight.sum().clamp_min(1e-8)
        else:
            init_teacher_loss = q_reg.new_tensor(0.0)
        proto_readout_loss, proto_readout_stats = prototype_readout_alignment_loss(
            q_reg,
            out["embedding"],
            out["cluster_centers"],
            cluster_prior=out["cluster_prior"],
            alpha=out["alpha"],
            q_base=out["q"],
            temperature=float(cfg.aptc_proto_readout_temperature),
            conf_power=float(cfg.aptc_proto_readout_conf_power),
            entropy_power=float(cfg.aptc_proto_readout_entropy_power),
            graph_gate=bool(cfg.aptc_proto_readout_graph_gate),
            prior_scale=float(cfg.aptc_proto_readout_prior_scale),
            alpha_floor=float(cfg.aptc_proto_readout_alpha_floor),
            alpha_span=float(cfg.aptc_proto_readout_alpha_span),
            gate_floor=float(cfg.aptc_proto_readout_gate_floor),
        )
        if self.init_prototypes.numel() == self.cluster_head.prototypes.numel():
            prototype_anchor_loss = F.mse_loss(
                F.normalize(self.cluster_head.prototypes, p=2, dim=1),
                F.normalize(self.init_prototypes.detach(), p=2, dim=1),
            )
        else:
            prototype_anchor_loss = q_reg.new_tensor(0.0)
        prototype_separation_loss = prototype_separation_regularizer(
            self.cluster_head.prototypes,
            margin=float(cfg.aptc_prototype_separation_margin),
        )
        compact_loss = out["head_compact"]
        if out["q_embed"].numel() > 0:
            if bool(cfg.aptc_embed_consistency_graph_gate):
                embed_consistency_weight = (
                    float(out["embed_graph_gate"].detach().cpu()) * float(out["embed_node_gate"].mean().detach().cpu())
                )
            else:
                embed_consistency_weight = 1.0
        else:
            embed_consistency_weight = 0.0
        view_weights = q_reg.new_tensor([1.0, 1.0, 1.0, embed_consistency_weight])
        view_consistency_loss = multi_view_consistency(
            q_reg,
            out["q_attr"],
            out["q_low"],
            out["q_high"],
            out["q_embed"],
            weights=view_weights,
        )
        edge_posterior_loss = edge_posterior_energy(q_reg, self.edge_index, out["homo"], out["hetero"], out["hard"])
        confidence_entropy_loss, confidence_entropy_stats = confidence_weighted_entropy_loss(
            q_reg,
            self.edge_index,
            out["score"],
            out["homo"],
            out["hetero"],
            out["hard"],
            power=float(cfg.confidence_entropy_power),
            hetero_weight=float(cfg.confidence_entropy_hetero_weight),
        )
        prior_entropy_loss = prior_entropy_regularizer(out["cluster_prior"])
        calib_alpha_loss, alpha_stats = evidence_attention_loss(
            out["alpha"],
            entropy_floor=float(cfg.calib_alpha_entropy_floor),
            usage_weight=float(cfg.calib_alpha_usage_weight),
            usage_floor=float(cfg.calib_alpha_usage_floor),
            usage_floor_weight=float(cfg.calib_alpha_usage_floor_weight),
        )
        calib_mask_loss, mask_stats = mask_diversity_loss(
            out["score"],
            out["homo"],
            out["hetero"],
            out["hard"],
            floor=float(cfg.calib_mask_floor),
            floor_weight=float(cfg.calib_mask_floor_weight),
        )
        calib_struct_attr_loss = structure_attribute_consistency_loss(
            out["score"],
            out["evidences"],
            self.edge_prior,
        )
        edge_rank_loss, rank_stats = order_preserving_edge_ranking_loss(
            out["score"],
            out["evidences"],
            self.edge_index,
            self.edge_prior,
            pos_quantile=float(cfg.edge_rank_pos_quantile),
            neg_quantile=float(cfg.edge_rank_neg_quantile),
            margin=float(cfg.edge_rank_margin),
            max_pairs=int(cfg.edge_rank_max_pairs),
            local_tau=float(cfg.edge_rank_local_tau),
            raw_teacher_weight=float(cfg.edge_rank_raw_teacher_weight),
            raw_gate_margin=float(cfg.edge_rank_raw_gate_margin),
            raw_gate_temperature=float(cfg.edge_rank_raw_gate_temperature),
        )
        edge_quantile_anchor_loss, qanchor_stats = quantile_threshold_coupling_loss(
            out["score"],
            out["low_threshold"],
            out["high_threshold"],
            rho=float(cfg.edge_quantile_anchor_rho),
        )
        partition_spread_loss, partition_spread_stats = partition_spread_pressure_loss(
            out["homo"],
            out["hetero"],
            out["hard"],
            out["low_threshold"],
            out["high_threshold"],
            min_spread=float(cfg.partition_min_spread),
            ambiguous_weight=float(cfg.partition_ambiguous_penalty_weight),
        )
        freq_separation_loss, freq_separation_stats = frequency_separation_pair_loss(
            out["low_view"],
            out["hetero_view"],
            self.edge_index,
            out["homo"],
            out["hetero"],
        )
        freq_ortho_loss = F.relu(out["z_cross_alignment"] - float(cfg.freq_ortho_target)).pow(2)
        homo_binary = out["score"] >= out["high_threshold"]
        hetero_binary = out["score"] <= out["low_threshold"]
        ambiguous_binary = (~homo_binary) & (~hetero_binary)
        homo_confidence_mean = (
            out["score"][homo_binary].mean()
            if bool(homo_binary.any())
            else out["score"].new_tensor(0.0)
        )
        hetero_confidence_mean = (
            out["score"][hetero_binary].mean()
            if bool(hetero_binary.any())
            else out["score"].new_tensor(0.0)
        )
        ambiguous_ratio = ambiguous_binary.to(out["score"].dtype).mean()
        subspace_loss, subspace_stats = self_expressive_subspace_loss(
            out["embedding"],
            temperature=float(cfg.subspace_temperature),
            l1_weight=float(cfg.subspace_l1_weight),
            max_nodes=int(cfg.subspace_max_nodes),
        )
        rayleigh_loss, rayleigh_stats = rayleigh_view_routing_loss(
            out["view_gate"],
            out["view_rayleigh"],
            temperature=float(cfg.rayleigh_temperature),
        )
        stitch_loss, stitch_stats = raw_posterior_stitching_loss(
            out["q_attr"],
            out["q_low"],
            self.edge_index,
            self.edge_prior,
            out["score"],
            out["homo"],
            out["hard"],
        )
        recon = self.decoder(out["embedding"])
        reconstruction_loss = F.mse_loss(recon, x)
        contrastive_loss = symmetric_info_nce(out["low_view"], out["hetero_view"])
        dirichlet_loss = edge_dirichlet(out["low_view"], self.edge_index, out["homo"] + 0.2 * out["hard"])
        emb_dirichlet_loss = edge_dirichlet(out["embedding"], self.edge_index, (out["homo"] + 0.2 * out["hard"]).detach().clamp_min(1e-6))
        zattr_dirichlet_loss = edge_dirichlet(out["z_attr"], self.edge_index, (out["homo"] + 0.2 * out["hard"]).detach().clamp_min(1e-6))
        highpass_loss = -edge_dirichlet(out["hetero_view"], self.edge_index, out["hetero"])
        if bool(cfg.balance_to_uniform):
            balance_target = torch.full_like(out["cluster_prior"], 1.0 / float(self.n_clusters))
        else:
            balance_target = out["cluster_prior"]
        balance_loss = ((q_reg.mean(dim=0) - balance_target) ** 2).sum()
        entropy_loss = -torch.sum(q_reg * q_reg.clamp_min(1e-8).log(), dim=1).mean()
        threshold_loss = threshold_regularizer(
            out["score"],
            out["homo"],
            out["hetero"],
            out["low_threshold"],
            out["high_threshold"],
            cfg.target_homo_ratio,
            cfg.target_hetero_ratio,
            adaptive=bool(cfg.adaptive_threshold_targets),
        )
        edge_prior_loss = F.binary_cross_entropy(
            out["score"].clamp(1e-5, 1.0 - 1e-5),
            self.edge_prior,
            reduction="mean",
        )
        soft_edge_label = (q_reg[self.edge_index[0]] * q_reg[self.edge_index[1]]).sum(dim=1).detach()
        edge_supervision_loss = F.binary_cross_entropy(out["score"].clamp(1e-5, 1.0 - 1e-5), soft_edge_label)
        total = (
            cfg.cluster_loss_weight * cluster_loss
            + cfg.aptc_transport_weight * transport_loss
            + cfg.aptc_flow_anchor_weight * F.kl_div(q_reg.clamp_min(1e-8).log(), out["q_flow"].detach(), reduction="batchmean")
            + cfg.aptc_init_teacher_weight * init_teacher_loss
            + cfg.aptc_proto_readout_weight * float(self.runtime_proto_readout_multiplier) * proto_readout_loss
            + cfg.aptc_prototype_anchor_weight * prototype_anchor_loss
            + cfg.aptc_prototype_separation_weight * prototype_separation_loss
            + cfg.compact_loss_weight * compact_loss
            + cfg.aptc_view_consistency_weight * view_consistency_loss
            + cfg.aptc_edge_posterior_weight * edge_posterior_loss
            + cfg.aptc_prior_entropy_weight * prior_entropy_loss
            + cfg.calib_alpha_weight * calib_alpha_loss
            + cfg.calib_mask_weight * calib_mask_loss
            + cfg.calib_struct_attr_weight * calib_struct_attr_loss
            + cfg.edge_rank_weight * edge_rank_loss
            + cfg.edge_quantile_anchor_weight * edge_quantile_anchor_loss
            + cfg.subspace_loss_weight * subspace_loss
            + cfg.rayleigh_routing_weight * rayleigh_loss
            + cfg.posterior_stitch_weight * stitch_loss
            + cfg.partition_spread_weight * partition_spread_loss
            + cfg.freq_separation_weight * freq_separation_loss
            + cfg.freq_ortho_weight * freq_ortho_loss
            + cfg.reconstruction_weight * reconstruction_loss
            + cfg.contrastive_weight * contrastive_loss
            + cfg.dirichlet_weight * dirichlet_loss
            + cfg.emb_dirichlet_weight * emb_dirichlet_loss
            + cfg.zattr_dirichlet_weight * zattr_dirichlet_loss
            + cfg.highpass_weight * highpass_loss
            + cfg.balance_weight * balance_loss
            + cfg.entropy_weight * entropy_loss
            + cfg.confidence_entropy_weight * confidence_entropy_loss
            + cfg.threshold_reg_weight * threshold_loss
            + cfg.edge_prior_weight * edge_prior_loss
            + cfg.edge_supervision_weight * edge_supervision_loss
            + cfg.assignment_loss_weight * F.kl_div(q_reg.clamp_min(1e-8).log(), out["q"].detach(), reduction="batchmean")
        )
        diagnostics = {
            "loss": float(total.detach().cpu()),
            "cluster": float(cluster_loss.detach().cpu()),
            "transport": float(transport_loss.detach().cpu()),
            "init_teacher": float(init_teacher_loss.detach().cpu()),
            "teacher_reliability_mode_id": teacher_reliability_mode_id,
            "teacher_margin_mean": float(teacher_margin.mean().detach().cpu()),
            "teacher_margin_std": float(teacher_margin.std(unbiased=False).detach().cpu()),
            "teacher_agreement_mean": float(teacher_agreement.mean().detach().cpu()),
            "teacher_agreement_std": float(teacher_agreement.std(unbiased=False).detach().cpu()),
            "teacher_agreement_active_ratio": float((teacher_agreement > 0.5).float().mean().detach().cpu()),
            "teacher_conf_mean": float(teacher_conf.mean().detach().cpu()),
            "teacher_conf_std": float(teacher_conf.std(unbiased=False).detach().cpu()),
            "teacher_conf_center": float(teacher_conf_center.detach().cpu()),
            "teacher_conf_scale": float(teacher_conf_scale.detach().cpu()),
            "teacher_weight_mean": float(conf_weight.mean().detach().cpu()),
            "teacher_weight_active_ratio": float((conf_weight > conf_floor + 1e-6).float().mean().detach().cpu()),
            "proto_readout": float(proto_readout_loss.detach().cpu()),
            "proto_readout_multiplier": float(self.runtime_proto_readout_multiplier),
            "prototype_anchor": float(prototype_anchor_loss.detach().cpu()),
            "prototype_separation": float(prototype_separation_loss.detach().cpu()),
            "compact": float(compact_loss.detach().cpu()),
            "view_consistency": float(view_consistency_loss.detach().cpu()),
            "edge_posterior": float(edge_posterior_loss.detach().cpu()),
            "prior_entropy": float(prior_entropy_loss.detach().cpu()),
            "calib_alpha": float(calib_alpha_loss.detach().cpu()),
            "calib_mask": float(calib_mask_loss.detach().cpu()),
            "calib_struct_attr": float(calib_struct_attr_loss.detach().cpu()),
            "edge_rank": float(edge_rank_loss.detach().cpu()),
            "edge_quantile_anchor": float(edge_quantile_anchor_loss.detach().cpu()),
            "subspace": float(subspace_loss.detach().cpu()),
            "rayleigh_route": float(rayleigh_loss.detach().cpu()),
            "posterior_stitch": float(stitch_loss.detach().cpu()),
            "partition_spread": float(partition_spread_loss.detach().cpu()),
            "frequency_separation": float(freq_separation_loss.detach().cpu()),
            "frequency_ortho": float(freq_ortho_loss.detach().cpu()),
            "highpass_scale": float((0.5 + 3.5 * torch.sigmoid(self.highpass_scale_logit)).detach().cpu()),
            "recon": float(reconstruction_loss.detach().cpu()),
            "contrast": float(contrastive_loss.detach().cpu()),
            "dirichlet": float(dirichlet_loss.detach().cpu()),
            "emb_dirichlet": float(emb_dirichlet_loss.detach().cpu()),
            "zattr_dirichlet": float(zattr_dirichlet_loss.detach().cpu()),
            "highpass": float(highpass_loss.detach().cpu()),
            "balance": float(balance_loss.detach().cpu()),
            "entropy": float(entropy_loss.detach().cpu()),
            "confidence_entropy": float(confidence_entropy_loss.detach().cpu()),
            "threshold_reg": float(threshold_loss.detach().cpu()),
            "edge_prior": float(edge_prior_loss.detach().cpu()),
            "edge_supervision": float(edge_supervision_loss.detach().cpu()),
            "flow_kl": float(F.kl_div(q_reg.clamp_min(1e-8).log(), out["q"].detach(), reduction="batchmean").detach().cpu()),
            "cluster_flow_kl": float(F.kl_div(q_cluster.clamp_min(1e-8).log(), out["q"].detach(), reduction="batchmean").detach().cpu()),
            "prior_min": float(out["cluster_prior"].min().detach().cpu()),
            "prior_max": float(out["cluster_prior"].max().detach().cpu()),
            "gate_attr": float(out["view_gate"][:, 0].mean().detach().cpu()),
            "gate_low": float(out["view_gate"][:, 1].mean().detach().cpu()),
            "gate_high": float(out["view_gate"][:, 2].mean().detach().cpu()),
            "gate_embed": float(out["view_gate"][:, 3].mean().detach().cpu()) if out["view_gate"].shape[1] > 3 else 0.0,
            "gate_embed_raw": float(out["view_gate_raw"][:, 3].mean().detach().cpu()) if out["view_gate_raw"].shape[1] > 3 else 0.0,
            "base_gate_attr": float(out["base_view_gate"][:, 0].mean().detach().cpu()),
            "base_gate_low": float(out["base_view_gate"][:, 1].mean().detach().cpu()),
            "base_gate_high": float(out["base_view_gate"][:, 2].mean().detach().cpu()),
            "embed_graph_gate": float(out["embed_graph_gate"].detach().cpu()),
            "embed_graph_gate_raw": float(out["embed_graph_gate_raw"].detach().cpu()),
            "embed_flow_gate": float(out["embed_flow_gate"].detach().cpu()),
            "embed_prior_gate": float(out["embed_prior_gate"].detach().cpu()),
            "embed_std_gate": float(out["embed_std_gate"].detach().cpu()),
            "embed_entropy_gate": float(out["embed_entropy_gate"].detach().cpu()),
            "embed_node_gate": float(out["embed_node_gate"].mean().detach().cpu()),
            "embed_node_entropy_gate": float(out["embed_node_entropy_gate"].mean().detach().cpu()),
            "embed_node_kl_gate": float(out["embed_node_kl_gate"].mean().detach().cpu()),
            "embed_node_transport_gate": float(out["embed_node_transport_gate"].mean().detach().cpu()),
            "embed_node_refine_gate": float(out["embed_node_refine_gate"].mean().detach().cpu()),
            "embed_node_rank_gate": float(out["embed_node_rank_gate"].mean().detach().cpu()),
            "embed_node_gate_heuristic": float(out["embed_node_gate_heuristic"].mean().detach().cpu()),
            "embed_node_gate_learned": float(out["embed_node_gate_learned"].mean().detach().cpu()),
            "embed_amplitude_score": float(out["embed_amplitude_score"].mean().detach().cpu()),
            "embed_amplitude_gate": float(out["embed_amplitude_gate"].mean().detach().cpu()),
            "embed_amplitude_floor": float(out["embed_amplitude_floor"].mean().detach().cpu()),
            "embed_amplitude_blend": float(out["embed_amplitude_blend"].detach().cpu()),
            "embed_amplitude_floor_multiplier": float(self.runtime_embedding_amplitude_floor_multiplier),
            "embed_gate_multiplier": float(self.runtime_embedding_gate_multiplier),
            "base_flow_kl": float(out["base_flow_kl"].detach().cpu()),
            "loss_flow_weight": float(loss_flow_weight),
            "target_flow_weight": float(getattr(self, "runtime_target_flow_weight", 0.0)),
            "loss_source_is_late": float(
                int(
                    int(getattr(cfg, "loss_posterior_late_start_epoch", -1)) >= 0
                    and int(getattr(self, "runtime_epoch", -1)) >= int(getattr(cfg, "loss_posterior_late_start_epoch", -1))
                )
            ),
            "main_flow_gate": float(out["main_flow_gate"].detach().cpu()),
            "main_prior_gate": float(out["main_prior_gate"].detach().cpu()),
            "main_entropy_gate": float(out["main_entropy_gate"].detach().cpu()),
            "main_flow_weight": float(out["main_flow_weight"].detach().cpu()),
            "student_posterior_gate": float(out["student_posterior_gate"].detach().cpu()),
            "student_posterior_prior_gate": float(out["student_posterior_prior_gate"].detach().cpu()),
            "student_posterior_weight": float(out["student_posterior_weight"].detach().cpu()),
            "flow_posterior_gate": float(out["flow_posterior_gate"].detach().cpu()),
            "flow_posterior_prior_gate": float(out["flow_posterior_prior_gate"].detach().cpu()),
            "flow_posterior_weight": float(out["flow_posterior_weight"].detach().cpu()),
            "flow_mix_gate": float(out["flow_mix_gate"].detach().cpu()),
            "flow_mix_prior_gate": float(out["flow_mix_prior_gate"].detach().cpu()),
            "flow_mix_weight": float(out["flow_mix_weight"].detach().cpu()),
            "base_entropy_mean": float(out["base_entropy_mean"].detach().cpu()),
            "embed_entropy_mean": float(out["embed_entropy_mean"].detach().cpu()),
            "base_logit_std": float(out["base_logit_std"].detach().cpu()),
            "low_threshold": float(out["low_threshold"].detach().cpu()),
            "high_threshold": float(out["high_threshold"].detach().cpu()),
            "homo_confidence_mean": float(homo_confidence_mean.detach().cpu()),
            "hetero_confidence_mean": float(hetero_confidence_mean.detach().cpu()),
            "z_cross_alignment": float(out["z_cross_alignment"].detach().cpu()),
            "z_low_norm": float(out["low_view"].norm(dim=1).mean().detach().cpu()),
            "z_high_norm": float(out["hetero_view"].norm(dim=1).mean().detach().cpu()),
            "ambiguous_ratio": float(ambiguous_ratio.detach().cpu()),
            "homo_ratio": float(out["homo"].mean().detach().cpu()),
            "hetero_ratio": float(out["hetero"].mean().detach().cpu()),
            "hard_ratio": float(out["hard"].mean().detach().cpu()),
            "edge_logit_mean": float(out["edge_logit"].mean().detach().cpu()),
            "edge_logit_std": float(out["edge_logit"].std(unbiased=False).detach().cpu()),
            "edge_score_mean": float(out["score"].mean().detach().cpu()),
            "edge_score_std": float(out["score"].std(unbiased=False).detach().cpu()),
            "alpha_attr": float(out["alpha"][:, 0].mean().detach().cpu()),
            "alpha_struct": float(out["alpha"][:, 1].mean().detach().cpu()),
            "alpha_gap": float(out["alpha"][:, 2].mean().detach().cpu()),
            "alpha_prior": float(out["alpha"][:, 3].mean().detach().cpu()) if out["alpha"].shape[1] > 3 else 0.0,
            "raw_leak_beta": float(out["raw_leak_beta"].detach().cpu()),
            "logit_std_attr": float(out["view_logit_std"][0].detach().cpu()),
            "logit_std_low": float(out["view_logit_std"][1].detach().cpu()),
            "logit_std_high": float(out["view_logit_std"][2].detach().cpu()),
            "logit_std_embed": float(out["view_logit_std"][3].detach().cpu()) if out["view_logit_std"].numel() > 3 else 0.0,
            **alpha_stats,
            **mask_stats,
            **subspace_stats,
            **rayleigh_stats,
            **stitch_stats,
            **confidence_entropy_stats,
            **rank_stats,
            **qanchor_stats,
            **partition_spread_stats,
            **freq_separation_stats,
            **proto_readout_stats,
        }
        return total, diagnostics

    def _edge_features(
        self,
        z_attr: torch.Tensor,
        z_raw: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attr_cos = (z_attr[src] * z_attr[dst]).sum(dim=1).clamp(-1.0, 1.0)
        raw_cos = (z_raw[src] * z_raw[dst]).sum(dim=1).clamp(-1.0, 1.0)
        attr01 = 0.5 * (attr_cos + 1.0)
        raw01 = 0.5 * (raw_cos + 1.0)
        log_deg = torch.log1p(self.degree)
        deg_i = log_deg[src]
        deg_j = log_deg[dst]
        deg_sim = 1.0 - (deg_i - deg_j).abs() / (deg_i + deg_j + 1.0)
        deg_sim = deg_sim.clamp(0.0, 1.0)
        prior = self.edge_prior
        gap = (attr01 - deg_sim).abs()
        product = attr01 * deg_sim
        edge_features = torch.stack(
            [
                attr01,
                raw01,
                deg_sim,
                prior,
                gap,
                product,
                torch.minimum(attr01, deg_sim),
                torch.maximum(attr01, deg_sim),
                1.0 / (deg_i + 1.0),
                1.0 / (deg_j + 1.0),
                (deg_i + deg_j) / (2.0 + log_deg.max().clamp_min(1.0)),
                attr01 - raw01,
                raw01 - deg_sim,
            ],
            dim=1,
        )
        evidence_items = [attr01, deg_sim, 1.0 - gap.clamp(0.0, 1.0)]
        if bool(self.cfg.edge_prior_evidence):
            evidence_items.append(prior.clamp(0.0, 1.0))
        evidences = torch.stack(evidence_items, dim=1)
        return edge_features, evidences

    def _support_weights(
        self,
        score: torch.Tensor,
        homo: torch.Tensor,
        hetero: torch.Tensor,
        hard: torch.Tensor,
    ) -> torch.Tensor:
        base = homo * score + hard * (0.35 + 0.30 * score) + hetero * 0.08 * (1.0 - score)
        return base.clamp_min(1e-6)

    def _diffuse(self, z: torch.Tensor, weight: torch.Tensor, steps: int) -> torch.Tensor:
        h = z
        out = z
        restart = float(self.cfg.diffusion_restart)
        for _ in range(max(1, int(steps))):
            h = normalized_spmm(self.edge_index, weight, h, self.degree.numel())
            h = (1.0 - restart) * h + restart * z
            out = out + h
        return F.normalize(out / float(max(1, int(steps)) + 1), p=2, dim=1)

    def _signed_highpass(self, z: torch.Tensor, hetero: torch.Tensor, steps: int) -> torch.Tensor:
        h = z
        out = z
        adaptive = bool(self.cfg.highpass_adaptive_scale)
        hetero_mass = hetero.mean().clamp(0.05, 1.0)
        hetero_scale = 0.5 + 3.5 * torch.sigmoid(self.highpass_scale_logit)
        for _ in range(max(1, int(steps))):
            smooth = normalized_spmm(self.edge_index, hetero.clamp_min(1e-6), h, self.degree.numel())
            if adaptive:
                h = z - hetero_mass * hetero_scale * smooth
            else:
                h = h - smooth
            out = out + h
        return F.normalize(out / float(max(1, int(steps)) + 1), p=2, dim=1)

    def _assignment_flow(
        self,
        q: torch.Tensor,
        homo: torch.Tensor,
        hetero: torch.Tensor,
        hard: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.cfg
        power = max(1.0, float(cfg.assignment_sharpen_power))
        y0 = q.pow(power)
        y0 = y0 / y0.sum(dim=1, keepdim=True).clamp_min(1e-8)
        y = y0
        uniform = torch.full_like(q, 1.0 / self.n_clusters)
        attract_w = (homo + 0.25 * hard).clamp_min(1e-6)
        raw_floor = float(cfg.assignment_raw_repel_floor)
        raw_edge = (self.edge_prior >= 0.999).to(q.dtype)
        repel_w = (hetero + raw_floor * raw_edge).clamp_min(1e-6)
        for _ in range(max(0, int(cfg.assignment_flow_steps))):
            attract = normalized_spmm(self.edge_index, attract_w, y, self.degree.numel())
            repel = normalized_spmm(self.edge_index, repel_w, y, self.degree.numel())
            logits = (
                cfg.assignment_fidelity_weight * torch.log(y0.clamp_min(1e-8))
                + cfg.assignment_attract_weight * attract
                + cfg.assignment_repel_weight * (uniform - repel)
            )
            y = F.softmax(logits / max(1e-4, float(cfg.assignment_temperature)), dim=1)
        return y


class E2ESECTCoCo:
    """Estimator wrapper that exposes the same fit_predict surface as the old pipeline."""

    def __init__(self, n_clusters: int, config: E2ESECTCoCoConfig):
        self.n_clusters = int(n_clusters)
        self.config = config
        self.model_: EndToEndSECTCoCoModule | None = None
        self.embedding_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self._true_labels_: np.ndarray | None = None
        self.diagnostics_: dict[str, Any] = {}

    def _log_frontend_diagnostics(
        self,
        frontend: dict[str, torch.Tensor],
        edge_index: torch.Tensor,
    ) -> dict[str, Any]:
        score = frontend["score"].detach()
        low_threshold = frontend["low_threshold"].detach()
        high_threshold = frontend["high_threshold"].detach()
        homo_mask = score >= high_threshold
        hetero_mask = score <= low_threshold
        ambiguous_mask = (~homo_mask) & (~hetero_mask)
        low_view = frontend["low_view"].detach()
        high_view = frontend["hetero_view"].detach()
        embedding = frontend["embedding"].detach()
        src, dst = edge_index

        if bool(homo_mask.any()):
            diff = low_view[src] - low_view[dst]
            dirichlet_energy_low = diff.pow(2).sum(dim=1)[homo_mask].mean().detach()
        else:
            dirichlet_energy_low = low_view.new_tensor(0.0)
        feature_var_high = high_view.var(dim=0, unbiased=False).mean().detach()

        intra_dist: float | None = None
        inter_dist: float | None = None
        cluster_separation: float | None = None
        labels = getattr(self, "_true_labels_", None)
        labels_np = None if labels is None else np.asarray(labels).reshape(-1)
        if labels_np is not None and labels_np.shape[0] == embedding.shape[0]:
            z = embedding.cpu()
            unique_labels = np.unique(labels_np)
            intra_sum = 0.0
            intra_count = 0
            for label in unique_labels:
                idx = np.flatnonzero(labels_np == label)
                if idx.size < 2:
                    continue
                block = z[torch.as_tensor(idx, dtype=torch.long)]
                pairwise = torch.pdist(block, p=2)
                if pairwise.numel() == 0:
                    continue
                intra_sum += float(pairwise.sum().item())
                intra_count += int(pairwise.numel())
            if intra_count > 0:
                intra_dist = intra_sum / float(intra_count)
                rng = np.random.default_rng(int(getattr(self.config, "seed", 0)))
                pairs: list[tuple[int, int]] = []
                attempts = 0
                n = int(labels_np.shape[0])
                while len(pairs) < 2000 and attempts < 50_000:
                    i = int(rng.integers(0, n))
                    j = int(rng.integers(0, n))
                    attempts += 1
                    if i == j or labels_np[i] == labels_np[j]:
                        continue
                    pairs.append((i, j))
                if pairs:
                    pair_idx = np.asarray(pairs, dtype=np.int64)
                    left = z[torch.as_tensor(pair_idx[:, 0], dtype=torch.long)]
                    right = z[torch.as_tensor(pair_idx[:, 1], dtype=torch.long)]
                    inter_dist = float(torch.norm(left - right, dim=1).mean().item())
                    cluster_separation = inter_dist / (intra_dist + 1e-8)

        return {
            "edge_homo_ratio": float(homo_mask.to(dtype=score.dtype).mean().detach().cpu()),
            "edge_hetero_ratio": float(hetero_mask.to(dtype=score.dtype).mean().detach().cpu()),
            "edge_ambiguous_ratio": float(ambiguous_mask.to(dtype=score.dtype).mean().detach().cpu()),
            "dirichlet_energy_low": float(dirichlet_energy_low.cpu()),
            "feature_var_high": float(feature_var_high.cpu()),
            "intra_dist": None if intra_dist is None else float(intra_dist),
            "inter_dist": None if inter_dist is None else float(inter_dist),
            "cluster_separation": None if cluster_separation is None else float(cluster_separation),
        }

    def fit_predict(
        self,
        adj: sp.spmatrix,
        features: sp.spmatrix | np.ndarray,
        graph_features_adj: sp.spmatrix | None = None,
        true_labels: np.ndarray | None = None,
    ) -> np.ndarray:
        set_seed(self.config.seed)
        start = time.perf_counter()
        cfg = self.config
        self._true_labels_ = None if true_labels is None else np.asarray(true_labels)
        device = resolve_device(cfg.device)
        adj = as_csr(adj)
        graph_source_adj = as_csr(graph_features_adj) if graph_features_adj is not None else adj
        graph_adj = graph_source_adj
        if bool(cfg.graph_input_transpose):
            graph_adj = graph_adj.T.tocsr()
        edge_adj = graph_source_adj if str(cfg.edge_graph_source).lower() == "graph" else adj
        x_np = prepare_dense_features(features, cfg, graph_adj)
        edge_index_np, edge_prior_np = build_candidate_edges(edge_adj, x_np, cfg)
        degree_np = np.asarray(edge_adj.sum(axis=1)).reshape(-1).astype(np.float32)

        x = torch.as_tensor(x_np, dtype=torch.float32, device=device)
        edge_index = torch.as_tensor(edge_index_np, dtype=torch.long, device=device)
        edge_prior = torch.as_tensor(edge_prior_np, dtype=torch.float32, device=device)
        degree = torch.as_tensor(degree_np, dtype=torch.float32, device=device)
        model = EndToEndSECTCoCoModule(x.shape[1], self.n_clusters, cfg, degree, edge_index, edge_prior).to(device)
        if bool(cfg.freeze_raw_skip):
            for param in model.raw_skip.parameters():
                param.requires_grad_(False)
            model.raw_skip_gate.requires_grad_(False)
        self.model_ = model

        model.eval()
        with torch.no_grad():
            init_out = model(x)
            init_emb = bootstrap_initial_embedding(init_out, cfg)
            init_labels, init_centers = cluster_numpy(init_emb, self.n_clusters, cfg)
            centers = torch.as_tensor(init_centers, dtype=torch.float32, device=device)
            model.cluster_head.prototypes.data.copy_(centers)
            model.cluster_head.prototype_memory.data.copy_(centers)
            model.init_prototypes = centers.detach().clone()
            init_counts = np.bincount(init_labels, minlength=self.n_clusters).astype(np.float32)
            init_prior = np.clip(init_counts / max(1.0, float(init_counts.sum())), 1e-4, 1.0)
            init_prior = init_prior / init_prior.sum()
            prior_blend = float(np.clip(cfg.init_prior_uniform_blend, 0.0, 1.0))
            if prior_blend > 0.0:
                if bool(cfg.init_prior_adaptive_blend):
                    entropy = -float(np.sum(init_prior * np.log(np.clip(init_prior, 1e-8, 1.0))))
                    entropy = entropy / max(1e-8, math.log(float(self.n_clusters)))
                    prior_blend *= float(np.clip(entropy, 0.0, 1.0))
                uniform_prior = np.full_like(init_prior, 1.0 / float(self.n_clusters))
                init_prior = (1.0 - prior_blend) * init_prior + prior_blend * uniform_prior
                init_prior = init_prior / init_prior.sum()
            model.cluster_head.cluster_prior_logits.data.copy_(torch.log(torch.as_tensor(init_prior, dtype=torch.float32, device=device)))
            teacher_out = model(x)
            model.init_teacher = teacher_out["q"].detach()

        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        target = None
        last_diag: dict[str, float] = {}
        frontend_snapshots: dict[str, float] = {}
        did_mid_init = False
        for epoch in range(int(cfg.epochs)):
            model.runtime_epoch = epoch
            model.train()
            refresh_interval = int(max(0, cfg.prototype_refresh_interval))
            if refresh_interval > 0 and epoch > 0 and epoch % refresh_interval == 0:
                model.eval()
                with torch.no_grad():
                    refresh_out = model(x)
                    refresh_mode = str(cfg.prototype_refresh_bootstrap_mode).strip()
                    old_mode = cfg.init_bootstrap_mode
                    if refresh_mode:
                        cfg.init_bootstrap_mode = refresh_mode
                    refresh_emb = bootstrap_initial_embedding(refresh_out, cfg)
                    cfg.init_bootstrap_mode = old_mode
                    refresh_labels, refresh_centers = cluster_numpy(refresh_emb, self.n_clusters, cfg)
                    refresh_centers_t = torch.as_tensor(refresh_centers, dtype=torch.float32, device=device)
                    proto_momentum = float(np.clip(cfg.prototype_refresh_momentum, 0.0, 0.999))
                    if proto_momentum > 0.0:
                        blended_proto = F.normalize(
                            proto_momentum * model.cluster_head.prototypes.data + (1.0 - proto_momentum) * refresh_centers_t,
                            p=2,
                            dim=1,
                        )
                        model.cluster_head.prototypes.data.copy_(blended_proto)
                        blended_mem = F.normalize(
                            proto_momentum * model.cluster_head.prototype_memory.data + (1.0 - proto_momentum) * refresh_centers_t,
                            p=2,
                            dim=1,
                        )
                        model.cluster_head.prototype_memory.data.copy_(blended_mem)
                    else:
                        model.cluster_head.prototypes.data.copy_(refresh_centers_t)
                        model.cluster_head.prototype_memory.data.copy_(refresh_centers_t)
                    counts = np.bincount(refresh_labels, minlength=self.n_clusters).astype(np.float32)
                    refresh_prior = np.clip(counts / max(1.0, float(counts.sum())), 1e-4, 1.0)
                    refresh_prior = refresh_prior / refresh_prior.sum()
                    refresh_prior_t = torch.log(torch.as_tensor(refresh_prior, dtype=torch.float32, device=device))
                    prior_momentum = float(np.clip(cfg.prior_refresh_momentum, 0.0, 0.999))
                    if prior_momentum > 0.0:
                        blended_prior = prior_momentum * model.cluster_head.cluster_prior_logits.data + (1.0 - prior_momentum) * refresh_prior_t
                        model.cluster_head.cluster_prior_logits.data.copy_(blended_prior)
                    else:
                        model.cluster_head.cluster_prior_logits.data.copy_(refresh_prior_t)
                    teacher_source = str(cfg.teacher_refresh_source).lower()
                    if teacher_source == "q_flow":
                        refresh_teacher = refresh_out["q_flow"]
                    elif teacher_source == "q_transport":
                        refresh_teacher = refresh_out["q_transport"]
                    else:
                        refresh_teacher = refresh_out["q_refined"]
                    teacher_momentum = float(np.clip(cfg.teacher_refresh_momentum, 0.0, 0.999))
                    if model.init_teacher.numel() == refresh_teacher.numel() and teacher_momentum > 0.0:
                        model.init_teacher = (
                            teacher_momentum * model.init_teacher.detach()
                            + (1.0 - teacher_momentum) * refresh_teacher.detach()
                        )
                    else:
                        model.init_teacher = refresh_teacher.detach()
                model.train()
            if (not did_mid_init) and int(cfg.mid_init_epoch) >= 0 and epoch >= int(cfg.mid_init_epoch):
                model.eval()
                with torch.no_grad():
                    mid_out = model(x)
                    old_mode = cfg.init_bootstrap_mode
                    cfg.init_bootstrap_mode = str(cfg.mid_init_bootstrap_mode)
                    mid_emb = bootstrap_initial_embedding(mid_out, cfg)
                    cfg.init_bootstrap_mode = old_mode
                    mid_labels, mid_centers = cluster_numpy(mid_emb, self.n_clusters, cfg)
                    centers = torch.as_tensor(mid_centers, dtype=torch.float32, device=device)
                    model.cluster_head.prototypes.data.copy_(centers)
                    model.cluster_head.prototype_memory.data.copy_(centers)
                    model.init_prototypes = centers.detach().clone()
                    init_counts = np.bincount(mid_labels, minlength=self.n_clusters).astype(np.float32)
                    init_prior = np.clip(init_counts / max(1.0, float(init_counts.sum())), 1e-4, 1.0)
                    init_prior = init_prior / init_prior.sum()
                    model.cluster_head.cluster_prior_logits.data.copy_(torch.log(torch.as_tensor(init_prior, dtype=torch.float32, device=device)))
                    teacher_out = model(x)
                    model.init_teacher = teacher_out["q"].detach()
                    target = None
                model.train()
                did_mid_init = True
            if epoch == 0 or epoch % int(max(1, cfg.cluster_update_interval)) == 0:
                with torch.no_grad():
                    out = model(x)
                    target_source, target_flow_weight = model._resolve_target_bootstrap()
                    target_flow_weight = model._adaptive_target_flow_weight(out, target_flow_weight)
                    model.runtime_target_flow_weight = target_flow_weight
                    if target_source == "q_flow":
                        target_input = out["q_flow"]
                    elif target_source == "q_transport":
                        target_input = out["q_transport"]
                    elif target_source == "q_base":
                        target_input = out["q"]
                    elif target_source == "q_blend":
                        target_input = (1.0 - target_flow_weight) * out["q_refined"] + target_flow_weight * out["q_flow"]
                        target_input = target_input / target_input.sum(dim=1, keepdim=True).clamp_min(1e-8)
                    else:
                        target_input = out["q_refined"]
                    target = target_distribution(target_input).detach()
                    if epoch < int(cfg.pretrain_epochs):
                        target = None
            warmup = int(max(0, cfg.aptc_proto_readout_warmup_epochs))
            ramp = int(max(1, cfg.aptc_proto_readout_ramp_epochs))
            if epoch < warmup:
                proto_readout_multiplier = 0.0
            else:
                proto_readout_multiplier = min(1.0, float(epoch - warmup + 1) / float(ramp))
            model.runtime_proto_readout_multiplier = proto_readout_multiplier
            embed_warmup = int(max(0, cfg.aptc_embedding_gate_warmup_epochs))
            embed_ramp = int(max(1, cfg.aptc_embedding_gate_ramp_epochs))
            if epoch < embed_warmup:
                embedding_gate_multiplier = 0.0
            else:
                embedding_gate_multiplier = min(1.0, float(epoch - embed_warmup + 1) / float(embed_ramp))
            model.runtime_embedding_gate_multiplier = embedding_gate_multiplier
            amp_floor_warmup = int(max(0, cfg.aptc_embedding_amplitude_floor_warmup_epochs))
            amp_floor_ramp = int(max(1, cfg.aptc_embedding_amplitude_floor_ramp_epochs))
            if epoch < amp_floor_warmup:
                amplitude_floor_multiplier = 0.0
            else:
                amplitude_floor_multiplier = min(1.0, float(epoch - amp_floor_warmup + 1) / float(amp_floor_ramp))
            model.runtime_embedding_amplitude_floor_multiplier = amplitude_floor_multiplier
            flow_posterior_warmup = int(max(0, cfg.aptc_flow_posterior_warmup_epochs))
            flow_posterior_ramp = int(max(1, cfg.aptc_flow_posterior_ramp_epochs))
            if epoch < flow_posterior_warmup:
                flow_posterior_multiplier = 0.0
            else:
                flow_posterior_multiplier = min(1.0, float(epoch - flow_posterior_warmup + 1) / float(flow_posterior_ramp))
            model.runtime_flow_posterior_multiplier = flow_posterior_multiplier
            flow_mix_warmup = int(max(0, cfg.aptc_flow_mix_warmup_epochs))
            flow_mix_ramp = int(max(1, cfg.aptc_flow_mix_ramp_epochs))
            if epoch < flow_mix_warmup:
                flow_mix_multiplier = 0.0
            else:
                flow_mix_multiplier = min(1.0, float(epoch - flow_mix_warmup + 1) / float(flow_mix_ramp))
            model.runtime_flow_mix_multiplier = flow_mix_multiplier
            optimizer.zero_grad(set_to_none=True)
            loss, diag = model.loss(x, target=target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            last_diag = diag
            if (epoch + 1) in {1, 40, 80}:
                suffix = f"epoch_{epoch + 1}"
                for key in (
                    "low_threshold",
                    "high_threshold",
                    "homo_ratio",
                    "hetero_ratio",
                    "hard_ratio",
                    "ambiguous_ratio",
                    "z_low_norm",
                    "z_high_norm",
                    "z_cross_alignment",
                    "edge_score_mean",
                    "edge_score_std",
                ):
                    if key in diag:
                        frontend_snapshots[f"{key}_{suffix}"] = float(diag[key])
            if bool(cfg.extras.get("verbose_training", False)) and (epoch % 50 == 0 or epoch + 1 == cfg.epochs):
                LOGGER.info("[%s] epoch=%d loss=%.4f H=%.3f L=%.3f", cfg.name, epoch, diag["loss"], diag["high_threshold"], diag["low_threshold"])

        model.eval()
        with torch.no_grad():
            frontend = model._frontend_pass(x)
            frontend_diag = self._log_frontend_diagnostics(frontend, model.edge_index)
            out = {**frontend, **model._aptc_pass(frontend)}
            emb = out["embedding"].detach().cpu().numpy()
            labels = out["q_refined"].argmax(dim=1).detach().cpu().numpy().astype(np.int64)
            _n_cl = int(out["q_refined"].shape[1])
            _sub_dim = min(2 * _n_cl, 20, emb.shape[0] - 1)
            _postproc_choice = "aptc_fallback"
            _sil_full = -2.0
            _sil_sub = -2.0
            _sil_best_sub = -2.0
            _adaptive_sub_dim = 0
            _postproc_error = ""
            try:
                from sklearn.preprocessing import normalize as _sk_norm
                from sklearn.metrics import silhouette_score as _sil
                _emb_np = out["embedding"].detach().cpu().float().numpy()
                _emb_np = _sk_norm(_emb_np, norm="l2", axis=1)
                _n_init = min(int(cfg.kmeans_n_init), 40)
                _km_full = KMeans(
                    n_clusters=_n_cl,
                    n_init=_n_init,
                    random_state=int(cfg.seed),
                    max_iter=300,
                )
                _lab_full = _km_full.fit_predict(_emb_np).astype(np.int64)
                _, _, _Vt = np.linalg.svd(_emb_np, full_matrices=False)
                _Z_sub = _sk_norm(_emb_np @ _Vt[:_sub_dim].T, norm="l2", axis=1)
                _km_sub = KMeans(
                    n_clusters=_n_cl,
                    n_init=_n_init,
                    random_state=int(cfg.seed),
                    max_iter=300,
                )
                _lab_sub = _km_sub.fit_predict(_Z_sub).astype(np.int64)
                _N = _emb_np.shape[0]
                _sil_n = min(3000, _N)
                _rng = np.random.default_rng(int(cfg.seed))
                _idx = _rng.choice(_N, _sil_n, replace=False) if _N > _sil_n else np.arange(_N)
                if len(np.unique(_lab_full[_idx])) >= 2:
                    _sil_full = float(_sil(_emb_np[_idx], _lab_full[_idx]))
                if len(np.unique(_lab_sub[_idx])) >= 2:
                    _sil_sub = float(_sil(_emb_np[_idx], _lab_sub[_idx]))
                if _sil_sub > _sil_full + float(cfg.postproc_subspace_margin):
                    labels = _lab_sub
                    _postproc_choice = "subspace"
                else:
                    labels = _lab_full
                    _postproc_choice = "full"
            except Exception as _e:
                _postproc_error = repr(_e)
                labels = out["q_refined"].argmax(dim=1).detach().cpu().numpy().astype(np.int64)
            _legacy_head_used = False
            _legacy_head_error = ""
            _label_mode = str(cfg.final_label_mode).lower()
            if _label_mode in {"legacy_subspace_refine", "subspace_refine_head", "adaptive_legacy_subspace"}:
                try:
                    _legacy_labels, _legacy_emb = run_legacy_subspace_refine_head(
                        cfg,
                        adj=adj,
                        features=features,
                        base_features=x_np,
                        embedding=emb,
                        edge_index=edge_index_np,
                        head_support={
                            "support": out["support_weight"].detach().cpu().numpy(),
                            "homo": out["homo"].detach().cpu().numpy(),
                            "hard": out["hard"].detach().cpu().numpy(),
                        },
                        n_clusters=self.n_clusters,
                        device=device,
                    )
                    if _label_mode == "adaptive_legacy_subspace":
                        labels, emb, _adaptive_stats = adaptive_subspace_kmeans_labels(
                            base_embedding=emb,
                            subspace_embedding=_legacy_emb,
                            n_clusters=self.n_clusters,
                            cfg=cfg,
                        )
                        _postproc_choice = str(_adaptive_stats["choice"])
                        _sil_full = float(_adaptive_stats["sil_full"])
                        _sil_best_sub = float(_adaptive_stats["sil_best_sub"])
                        _sil_sub = _sil_best_sub
                        _adaptive_sub_dim = int(_adaptive_stats["best_dim"])
                        _sub_dim = int(_adaptive_stats["best_dim"])
                    else:
                        labels, emb = _legacy_labels, _legacy_emb
                        _postproc_choice = "legacy_subspace_refine"
                    _legacy_head_used = True
                except Exception as _e:
                    _legacy_head_error = repr(_e)
        self.embedding_ = normalize(np.nan_to_num(emb), norm="l2", axis=1)
        self.labels_ = labels
        self.diagnostics_ = {
            "candidate": cfg.name,
            "nodes": int(adj.shape[0]),
            "edges": int(edge_adj.nnz if cfg.directed_candidate_edges else edge_adj.nnz // 2),
            "candidate_edges": int(edge_index_np.shape[1] if cfg.directed_candidate_edges else edge_index_np.shape[1] // 2),
            "input_dim": int(x_np.shape[1]),
            "embedding_dim": int(emb.shape[1]),
            "runtime_sec": round(time.perf_counter() - start, 3),
            **last_diag,
            **frontend_snapshots,
            **frontend_diag,
        }
        self.diagnostics_["selected_sub_dim"] = int(_sub_dim) if "_sub_dim" in locals() else 0
        self.diagnostics_["adaptive_sub_dim"] = int(_adaptive_sub_dim) if "_adaptive_sub_dim" in locals() else 0
        self.diagnostics_["postproc_choice"] = _postproc_choice if "_postproc_choice" in locals() else "unknown"
        self.diagnostics_["sil_full"] = float(_sil_full) if "_sil_full" in locals() else -2.0
        self.diagnostics_["sil_sub"] = float(_sil_sub) if "_sil_sub" in locals() else -2.0
        self.diagnostics_["sil_best_sub"] = float(_sil_best_sub) if "_sil_best_sub" in locals() else -2.0
        self.diagnostics_["postproc_error"] = _postproc_error if "_postproc_error" in locals() else ""
        self.diagnostics_["legacy_head_used"] = bool(_legacy_head_used) if "_legacy_head_used" in locals() else False
        self.diagnostics_["legacy_head_error"] = _legacy_head_error if "_legacy_head_error" in locals() else ""
        if true_labels is not None:
            true_labels_np = np.asarray(true_labels).reshape(-1)
            if true_labels_np.shape[0] == labels.shape[0]:
                final_metrics = evaluate_clustering(true_labels_np, labels)
                km = KMeans(
                    n_clusters=self.n_clusters,
                    random_state=cfg.seed,
                    n_init=10,
                    max_iter=300,
                )
                km_labels = km.fit_predict(np.asarray(emb))
                km_metrics = evaluate_clustering(true_labels_np, km_labels)
                self.diagnostics_.update(
                    {
                        "final_acc": float(final_metrics["acc"]),
                        "final_nmi": float(final_metrics["nmi"]),
                        "final_ari": float(final_metrics["ari"]),
                        "embedding_kmeans_acc": float(km_metrics["acc"]),
                        "embedding_kmeans_nmi": float(km_metrics["nmi"]),
                        "embedding_kmeans_ari": float(km_metrics["ari"]),
                        "embedding_posterior_gap": float(km_metrics["acc"] - final_metrics["acc"]),
                    }
                )
        return labels


def _logit(value: float) -> float:
    value = float(np.clip(value, 1e-6, 1.0 - 1e-6))
    return math.log(value / (1.0 - value))


def set_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def resolve_device(name: str) -> torch.device:
    if str(name).startswith("cuda") and torch.cuda.is_available():
        return torch.device(name)
    return torch.device("cpu")


def as_csr(x, dtype=np.float32) -> sp.csr_matrix:
    if sp.issparse(x):
        return x.astype(dtype).tocsr()
    return sp.csr_matrix(np.asarray(x, dtype=dtype))


def run_legacy_subspace_refine_head(
    cfg: E2ESECTCoCoConfig,
    *,
    adj: sp.csr_matrix,
    features: sp.spmatrix | np.ndarray,
    base_features: np.ndarray,
    embedding: np.ndarray,
    edge_index: np.ndarray,
    head_support: dict[str, np.ndarray],
    n_clusters: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    legacy_cfg = e2e_to_legacy_subspace_config(cfg)
    support = graph_from_directed_edges(adj.shape[0], edge_index, head_support["support"])
    homo = graph_from_directed_edges(adj.shape[0], edge_index, head_support["homo"])
    hard = graph_from_directed_edges(adj.shape[0], edge_index, head_support["hard"])
    denoised = (
        support
        + 0.25 * hard
        + float(cfg.extras.get("head_raw_graph_weight", 0.10)) * adj
        + sp.eye(adj.shape[0], dtype=np.float32, format="csr")
    ).tocsr()
    denoised.eliminate_zeros()
    adapter = LegacySubspaceHeadAdapter(
        n_clusters=n_clusters,
        input_features=features,
        base_features=base_features,
        embedding=embedding,
        raw_adj=adj,
        denoised_adj=denoised,
        homo_graph=homo,
    )
    labels = legacy_subspace_refine(adapter, legacy_cfg, device)
    return labels.astype(np.int64), np.asarray(adapter.embedding_, dtype=np.float32)


def adaptive_subspace_kmeans_labels(
    *,
    base_embedding: np.ndarray,
    subspace_embedding: np.ndarray,
    n_clusters: int,
    cfg: E2ESECTCoCoConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int | str]]:
    from sklearn.metrics import silhouette_score

    base = normalize(np.nan_to_num(np.asarray(base_embedding, dtype=np.float32)), norm="l2", axis=1)
    sub = normalize(np.nan_to_num(np.asarray(subspace_embedding, dtype=np.float32)), norm="l2", axis=1)
    n = int(base.shape[0])
    k = int(n_clusters)
    n_init = min(int(cfg.kmeans_n_init), 40)
    rng = np.random.default_rng(int(cfg.seed))
    sil_n = min(3000, n)
    idx = rng.choice(n, sil_n, replace=False) if n > sil_n else np.arange(n)

    km_full = KMeans(n_clusters=k, n_init=n_init, random_state=int(cfg.seed), max_iter=300)
    lab_full = km_full.fit_predict(base).astype(np.int64)
    sil_full = -1.0
    if len(np.unique(lab_full[idx])) >= 2:
        sil_full = float(silhouette_score(base[idx], lab_full[idx]))

    max_dim = max(1, min(sub.shape[0] - 1, sub.shape[1]))
    scan_dims = sorted({d for d in (k, 2 * k, 3 * k, 4 * k, 5 * k) if 1 <= d <= max_dim})
    if not scan_dims:
        return lab_full, base.astype(np.float32), {
            "choice": "full_kmeans",
            "sil_full": float(sil_full),
            "sil_best_sub": -2.0,
            "best_dim": 0,
        }

    _, _, vt = np.linalg.svd(sub, full_matrices=False)
    best_dim = 0
    best_sil = -2.0
    best_labels: np.ndarray | None = None
    best_z: np.ndarray | None = None
    for dim in scan_dims:
        z_dim = normalize(np.nan_to_num(sub @ vt[:dim].T), norm="l2", axis=1)
        km_dim = KMeans(n_clusters=k, n_init=n_init, random_state=int(cfg.seed), max_iter=200)
        labels_dim = km_dim.fit_predict(z_dim).astype(np.int64)
        if len(np.unique(labels_dim[idx])) < 2:
            continue
        sil = float(silhouette_score(base[idx], labels_dim[idx]))
        if sil > best_sil:
            best_sil = sil
            best_dim = int(dim)
            best_labels = labels_dim
            best_z = z_dim.astype(np.float32)

    threshold = float(cfg.subspace_sil_threshold)
    if best_labels is not None and best_z is not None and best_sil > threshold:
        return best_labels.astype(np.int64), best_z, {
            "choice": f"subspace_{best_dim}",
            "sil_full": float(sil_full),
            "sil_best_sub": float(best_sil),
            "best_dim": int(best_dim),
        }
    return lab_full, base.astype(np.float32), {
        "choice": "full_kmeans",
        "sil_full": float(sil_full),
        "sil_best_sub": float(best_sil),
        "best_dim": int(best_dim),
    }


def e2e_to_legacy_subspace_config(cfg: E2ESECTCoCoConfig) -> SECTCoCoConfig:
    extras = dict(cfg.extras)
    return SECTCoCoConfig(
        seed=cfg.seed,
        device=cfg.device,
        attr_dim=cfg.input_dim,
        cluster_dim=cfg.projection_dim,
        kmeans_n_init=cfg.kmeans_n_init,
        use_minibatch_kmeans=cfg.use_minibatch_kmeans,
        name=cfg.name,
        extras=extras,
    )


class LegacySubspaceHeadAdapter:
    def __init__(
        self,
        *,
        n_clusters: int,
        input_features: sp.spmatrix | np.ndarray,
        base_features: np.ndarray,
        embedding: np.ndarray,
        raw_adj: sp.csr_matrix,
        denoised_adj: sp.csr_matrix,
        homo_graph: sp.csr_matrix,
    ):
        self.n_clusters = int(n_clusters)
        self.input_features_ = as_csr(input_features)
        self.base_features_ = normalize(np.nan_to_num(base_features), norm="l2", axis=1).astype(np.float32)
        self.embedding_ = normalize(np.nan_to_num(embedding), norm="l2", axis=1).astype(np.float32)
        self.raw_adj_ = raw_adj
        self.denoised_adj_ = denoised_adj
        self.homo_graph_ = homo_graph


def graph_from_directed_edges(n: int, edge_index: np.ndarray, weight: np.ndarray) -> sp.csr_matrix:
    rows = edge_index[0].astype(np.int64)
    cols = edge_index[1].astype(np.int64)
    vals = np.asarray(weight, dtype=np.float32)
    graph = sp.coo_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float32).tocsr()
    graph = graph.maximum(graph.T).tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    return graph


def prepare_dense_features(
    features: sp.spmatrix | np.ndarray,
    cfg: E2ESECTCoCoConfig,
    adj: sp.csr_matrix | None = None,
) -> np.ndarray:
    if sp.issparse(features):
        feat = features.astype(np.float32).tocsr()
        max_dim = int(cfg.input_dim)
        if feat.shape[1] > max_dim:
            dim = min(max_dim, feat.shape[0] - 1, feat.shape[1] - 1)
            arr = TruncatedSVD(n_components=max(2, dim), random_state=cfg.seed).fit_transform(feat)
        else:
            arr = feat.toarray()
    else:
        arr = np.asarray(features, dtype=np.float32)
        if arr.shape[1] > int(cfg.input_dim):
            dim = min(int(cfg.input_dim), arr.shape[0] - 1, arr.shape[1] - 1)
            arr = TruncatedSVD(n_components=max(2, dim), random_state=cfg.seed).fit_transform(arr)
    arr = np.nan_to_num(arr).astype(np.float32)
    if bool(cfg.normalize_input_views):
        arr = normalize(arr, norm="l2", axis=1)
    graph_dim = int(cfg.graph_input_dim)
    if graph_dim > 0 and adj is not None and min(adj.shape) > 2:
        dim = min(graph_dim, adj.shape[0] - 1, adj.shape[1] - 1)
        graph_view = TruncatedSVD(n_components=max(2, dim), random_state=cfg.seed).fit_transform(adj.astype(np.float32))
        graph_view = np.nan_to_num(graph_view).astype(np.float32)
        if bool(cfg.normalize_input_views):
            graph_view = normalize(graph_view, norm="l2", axis=1)
        arr = np.hstack([arr, graph_view]).astype(np.float32)
    arr = normalize(np.nan_to_num(arr), norm="l2", axis=1)
    return arr.astype(np.float32)


def build_candidate_edges(adj: sp.csr_matrix, x: np.ndarray, cfg: E2ESECTCoCoConfig) -> tuple[np.ndarray, np.ndarray]:
    n = adj.shape[0]
    raw = adj.tocoo() if bool(cfg.directed_candidate_edges) else sp.triu(adj, k=1).tocoo()
    pairs: dict[tuple[int, int], float] = {}
    for i, j in zip(raw.row, raw.col):
        i = int(i)
        j = int(j)
        if i == j:
            continue
        if bool(cfg.directed_candidate_edges):
            pairs[(i, j)] = 1.0
        else:
            a, b = (i, j) if i < j else (j, i)
            pairs[(a, b)] = 1.0
    k = int(cfg.feature_knn)
    if k > 0 and n > 1:
        nn = NearestNeighbors(n_neighbors=min(k + 1, n), metric="cosine", algorithm="auto")
        nn.fit(x)
        distances, indices = nn.kneighbors(x, return_distance=True)
        count = 0
        max_feature = int(cfg.max_feature_edges)
        for i in range(n):
            for dist, j in zip(distances[i, 1:], indices[i, 1:]):
                if i == int(j):
                    continue
                if bool(cfg.directed_candidate_edges):
                    pairs.setdefault((i, int(j)), float(np.clip(1.0 - dist, 0.0, 1.0)))
                else:
                    a, b = (i, int(j)) if i < int(j) else (int(j), i)
                    pairs.setdefault((a, b), float(np.clip(1.0 - dist, 0.0, 1.0)))
                count += 1
                if count >= max_feature:
                    break
            if count >= max_feature:
                break
    max_edges = int(cfg.max_train_edges)
    if len(pairs) > max_edges:
        rng = np.random.default_rng(cfg.seed)
        keys = list(pairs.keys())
        raw_flags = np.asarray([pairs[key] >= 1.0 for key in keys], dtype=bool)
        raw_idx = np.flatnonzero(raw_flags)
        feat_idx = np.flatnonzero(~raw_flags)
        if raw_idx.size > max_edges:
            keep_raw = rng.choice(raw_idx, size=max_edges, replace=False)
        else:
            keep_raw = raw_idx
        remaining = max(0, max_edges - keep_raw.size)
        if feat_idx.size > remaining:
            feat_idx = rng.choice(feat_idx, size=remaining, replace=False)
        keep = np.concatenate([keep_raw, feat_idx])
        pairs = {keys[int(idx)]: pairs[keys[int(idx)]] for idx in keep}
    rows: list[int] = []
    cols: list[int] = []
    prior: list[float] = []
    for (i, j), value in pairs.items():
        if bool(cfg.directed_candidate_edges):
            rows.append(i)
            cols.append(j)
            prior.append(value)
        else:
            rows.extend([i, j])
            cols.extend([j, i])
            prior.extend([value, value])
    if not rows:
        rows = list(range(n))
        cols = list(range(n))
        prior = [1.0] * n
    edge_index = np.vstack([np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)])
    return edge_index, np.asarray(prior, dtype=np.float32)


def normalized_spmm(edge_index: torch.Tensor, weight: torch.Tensor, x: torch.Tensor, n: int) -> torch.Tensor:
    src, dst = edge_index
    deg = torch.zeros(n, device=x.device, dtype=x.dtype).scatter_add_(0, src, weight.to(x.dtype))
    norm = weight.to(x.dtype) / torch.sqrt((deg[src] + 1e-8) * (deg[dst] + 1e-8))
    out = torch.zeros_like(x)
    out.index_add_(0, dst, x[src] * norm.unsqueeze(1))
    return out


def edge_dirichlet(z: torch.Tensor, edge_index: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    src, dst = edge_index
    dist = (z[src] - z[dst]).pow(2).sum(dim=1)
    return (weight.to(z.dtype) * dist).mean()


def posterior_rayleigh(q: torch.Tensor, edge_index: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    src, dst = edge_index
    w = weight.to(q.dtype).clamp_min(0.0)
    dist = (q[src] - q[dst]).pow(2).sum(dim=1)
    return (w * dist).sum() / w.sum().clamp_min(1e-8)


def posterior_entropy(q: torch.Tensor) -> torch.Tensor:
    return -torch.sum(q * q.clamp_min(1e-8).log(), dim=1)


def sinkhorn_transport(q: torch.Tensor, prior: torch.Tensor, *, epsilon: float, iters: int) -> torch.Tensor:
    n = q.shape[0]
    eps = max(1e-3, float(epsilon))
    transport = q.clamp_min(1e-8).pow(1.0 / eps)
    col_target = prior.to(q.dtype).clamp_min(1e-8) * float(n)
    for _ in range(max(1, int(iters))):
        transport = transport / transport.sum(dim=1, keepdim=True).clamp_min(1e-8)
        transport = transport * (col_target / transport.sum(dim=0).clamp_min(1e-8)).unsqueeze(0)
    return transport / transport.sum(dim=1, keepdim=True).clamp_min(1e-8)


def multi_view_consistency(
    q_ref: torch.Tensor,
    *views: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    ref = q_ref.detach().clamp_min(1e-8)
    losses = [F.kl_div(view.clamp_min(1e-8).log(), ref, reduction="batchmean") for view in views]
    if weights is None:
        return sum(losses) / float(len(losses))
    if len(losses) == 0:
        return q_ref.sum() * 0.0
    weight_t = weights.to(q_ref.dtype)
    if weight_t.numel() != len(losses):
        raise ValueError("weights must match number of views")
    weight_t = weight_t / weight_t.sum().clamp_min(1e-8)
    total = q_ref.sum() * 0.0
    for w, loss in zip(weight_t, losses):
        total = total + w * loss
    return total


def rayleigh_view_routing_loss(
    gate: torch.Tensor,
    rayleigh: torch.Tensor,
    *,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    tau = max(1e-4, float(temperature))
    target = F.softmax((-rayleigh.detach() / tau).clamp(-30.0, 30.0), dim=0)
    usage = gate.mean(dim=0).clamp_min(1e-8)
    loss = F.kl_div(usage.log(), target.to(gate.dtype), reduction="sum")
    stats = {
        "rayleigh_attr": float(rayleigh[0].detach().cpu()),
        "rayleigh_low": float(rayleigh[1].detach().cpu()),
        "rayleigh_high": float(rayleigh[2].detach().cpu()),
        "rayleigh_embed": float(rayleigh[3].detach().cpu()) if rayleigh.numel() > 3 else 0.0,
        "rayleigh_target_attr": float(target[0].detach().cpu()),
        "rayleigh_target_low": float(target[1].detach().cpu()),
        "rayleigh_target_high": float(target[2].detach().cpu()),
        "rayleigh_target_embed": float(target[3].detach().cpu()) if target.numel() > 3 else 0.0,
    }
    return loss, stats


def raw_posterior_stitching_loss(
    q_attr: torch.Tensor,
    q_low: torch.Tensor,
    edge_index: torch.Tensor,
    edge_prior: torch.Tensor,
    score: torch.Tensor,
    homo: torch.Tensor,
    hard: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    src, dst = edge_index
    with torch.no_grad():
        raw = (edge_prior >= 0.999).to(q_low.dtype)
        weight = (raw * (homo + 0.5 * hard) * score).clamp_min(0.0)
        denom = weight.sum().clamp_min(1e-8)
        active = (weight > 1e-6).to(q_low.dtype).mean()
    low_dist = (q_low[src] - q_low[dst]).pow(2).sum(dim=1)
    attr_dist = (q_attr[src] - q_attr[dst]).pow(2).sum(dim=1)
    low_loss = (weight * low_dist).sum() / denom
    attr_loss = (weight * attr_dist).sum() / denom
    total = low_loss + 0.5 * attr_loss
    stats = {
        "stitch_low": float(low_loss.detach().cpu()),
        "stitch_attr": float(attr_loss.detach().cpu()),
        "stitch_weight_mean": float(weight.mean().detach().cpu()),
        "stitch_active_ratio": float(active.detach().cpu()),
    }
    return total, stats


def partition_spread_pressure_loss(
    homo: torch.Tensor,
    hetero: torch.Tensor,
    hard: torch.Tensor,
    low_threshold: torch.Tensor,
    high_threshold: torch.Tensor,
    *,
    min_spread: float,
    ambiguous_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    gap = high_threshold - low_threshold
    spread = F.relu(gap.new_tensor(float(min_spread)) - gap)
    ambiguous = hard.mean()
    decisive = (homo + hetero).mean()
    total = spread + float(ambiguous_weight) * ambiguous
    stats = {
        "partition_gap": float(gap.detach().cpu()),
        "partition_spread_gap_loss": float(spread.detach().cpu()),
        "partition_ambiguous_soft": float(ambiguous.detach().cpu()),
        "partition_decisive_soft": float(decisive.detach().cpu()),
    }
    return total, stats


def frequency_separation_pair_loss(
    z_low: torch.Tensor,
    z_high: torch.Tensor,
    edge_index: torch.Tensor,
    homo: torch.Tensor,
    hetero: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    src, dst = edge_index
    low = F.normalize(z_low, p=2, dim=1)
    high = F.normalize(z_high, p=2, dim=1)
    low_sim = (low[src] * low[dst]).sum(dim=1)
    high_sim = (high[src] * high[dst]).sum(dim=1)
    homo_w = homo.to(low.dtype).clamp_min(0.0)
    hetero_w = hetero.to(low.dtype).clamp_min(0.0)
    homo_norm = homo_w.sum().clamp_min(1e-6)
    hetero_norm = hetero_w.sum().clamp_min(1e-6)
    homo_loss = (homo_w * ((1.0 - low_sim) + (1.0 + high_sim))).sum() / homo_norm
    hetero_loss = (hetero_w * ((1.0 - high_sim) + (1.0 + low_sim))).sum() / hetero_norm
    total = 0.5 * (homo_loss + hetero_loss)
    stats = {
        "freq_low_edge_sim": float(low_sim.mean().detach().cpu()),
        "freq_high_edge_sim": float(high_sim.mean().detach().cpu()),
        "freq_homo_weight": float(homo_w.mean().detach().cpu()),
        "freq_hetero_weight": float(hetero_w.mean().detach().cpu()),
        "freq_homo_loss": float(homo_loss.detach().cpu()),
        "freq_hetero_loss": float(hetero_loss.detach().cpu()),
    }
    return total, stats


def self_expressive_subspace_loss(
    embedding: torch.Tensor,
    *,
    temperature: float,
    l1_weight: float,
    max_nodes: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    n = embedding.shape[0]
    if n <= 1:
        zero = embedding.sum() * 0.0
        return zero, {"subspace_recon": 0.0, "subspace_l1": 0.0, "subspace_nodes": float(n)}
    if n > int(max_nodes) > 0:
        idx = torch.randperm(n, device=embedding.device)[: int(max_nodes)]
        h = embedding[idx]
    else:
        h = embedding
    h = F.normalize(h, p=2, dim=1)
    logits = h @ h.T / max(1e-4, float(temperature))
    eye = torch.eye(h.shape[0], device=h.device, dtype=torch.bool)
    logits = logits.masked_fill(eye, -30.0)
    s = F.softmax(logits, dim=1)
    recon = s @ h
    recon_loss = F.mse_loss(recon, h)
    l1_loss = s.abs().mean()
    total = recon_loss + float(l1_weight) * l1_loss
    stats = {
        "subspace_recon": float(recon_loss.detach().cpu()),
        "subspace_l1": float(l1_loss.detach().cpu()),
        "subspace_nodes": float(h.shape[0]),
    }
    return total, stats


def edge_posterior_energy(
    q: torch.Tensor,
    edge_index: torch.Tensor,
    homo: torch.Tensor,
    hetero: torch.Tensor,
    hard: torch.Tensor,
) -> torch.Tensor:
    src, dst = edge_index
    agreement = torch.sum(q[src] * q[dst], dim=1)
    attract = (homo + 0.2 * hard).to(q.dtype) * (1.0 - agreement)
    repel = hetero.to(q.dtype) * agreement
    return attract.mean() + repel.mean()


def confidence_weighted_entropy_loss(
    q: torch.Tensor,
    edge_index: torch.Tensor,
    score: torch.Tensor,
    homo: torch.Tensor,
    hetero: torch.Tensor,
    hard: torch.Tensor,
    *,
    power: float,
    hetero_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    src, dst = edge_index
    dtype = q.dtype
    n = q.shape[0]
    hetero_w = float(np.clip(hetero_weight, 0.0, 1.0))
    decisive_mask = (homo + hetero_w * hetero).to(dtype)
    edge_certainty = (score.to(dtype) * decisive_mask * (1.0 - hard).to(dtype)).clamp_min(0.0)
    node_mass = torch.zeros(n, device=q.device, dtype=dtype)
    node_degree = torch.zeros(n, device=q.device, dtype=dtype)
    node_mass.index_add_(0, src, edge_certainty)
    node_mass.index_add_(0, dst, edge_certainty)
    ones = torch.ones_like(edge_certainty)
    node_degree.index_add_(0, src, ones)
    node_degree.index_add_(0, dst, ones)
    confidence = (node_mass / node_degree.clamp_min(1.0)).clamp(0.0, 1.0)
    if float(power) != 1.0:
        confidence = confidence.pow(max(1e-4, float(power)))
    entropy = -torch.sum(q * q.clamp_min(1e-8).log(), dim=1)
    weight = confidence.detach()
    loss = (weight * entropy).sum() / weight.sum().clamp_min(1e-8)
    stats = {
        "conf_entropy_node_weight": float(confidence.mean().detach().cpu()),
        "conf_entropy_node_max": float(confidence.max().detach().cpu()),
    }
    return loss, stats


def evidence_attention_loss(
    alpha: torch.Tensor,
    *,
    entropy_floor: float,
    usage_weight: float,
    usage_floor: float,
    usage_floor_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    eps = torch.finfo(alpha.dtype).eps
    k = alpha.shape[1]
    entropy = -torch.sum(alpha * alpha.clamp_min(eps).log(), dim=1) / math.log(float(k))
    entropy_floor_t = alpha.new_tensor(float(entropy_floor))
    entropy_loss = F.relu(entropy_floor_t - entropy).pow(2).mean()
    usage = alpha.mean(dim=0).clamp_min(eps)
    uniform = torch.full_like(usage, 1.0 / float(k))
    usage_loss = F.kl_div(usage.log(), uniform, reduction="sum")
    usage_floor_t = alpha.new_tensor(float(usage_floor))
    usage_floor_loss = F.relu(usage_floor_t - usage).pow(2).sum()
    total = entropy_loss + float(usage_weight) * usage_loss + float(usage_floor_weight) * usage_floor_loss
    stats = {
        "alpha_entropy": float(entropy.mean().detach().cpu()),
        "alpha_usage_kl": float(usage_loss.detach().cpu()),
        "alpha_usage_floor": float(usage_floor_loss.detach().cpu()),
    }
    return total, stats


def mask_diversity_loss(
    score: torch.Tensor,
    homo: torch.Tensor,
    hetero: torch.Tensor,
    hard: torch.Tensor,
    *,
    floor: float,
    floor_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    eps = torch.finfo(score.dtype).eps
    masses = torch.stack([homo.mean(), hetero.mean(), hard.mean()]).clamp_min(eps)
    with torch.no_grad():
        if score.numel() > 8:
            spread = (torch.quantile(score.detach(), 0.90) - torch.quantile(score.detach(), 0.10)).clamp(0.0, 1.0)
        else:
            spread = score.detach().new_tensor(0.5)
        tau_min = score.detach().new_tensor(0.15)
        tau_max = score.detach().new_tensor(0.45)
        tau_mid = score.detach().new_tensor(0.25)
        target = torch.stack(
            [
                tau_min + (tau_max - tau_min) * spread,
                tau_min + (tau_max - tau_min) * (1.0 - spread),
                tau_mid,
            ]
        )
        target = target / target.sum().clamp_min(eps)
    mass_loss = F.kl_div(masses.log(), target.to(score.dtype), reduction="sum")
    floor_t = score.new_tensor(float(floor))
    floor_loss = F.relu(floor_t - masses).pow(2).sum()
    total = mass_loss + float(floor_weight) * floor_loss
    stats = {
        "mask_target_homo": float(target[0].detach().cpu()),
        "mask_target_hetero": float(target[1].detach().cpu()),
        "mask_target_hard": float(target[2].detach().cpu()),
        "mask_floor_penalty": float(floor_loss.detach().cpu()),
    }
    return total, stats


def structure_attribute_consistency_loss(
    score: torch.Tensor,
    evidences: torch.Tensor,
    edge_prior: torch.Tensor,
) -> torch.Tensor:
    eps = torch.finfo(score.dtype).eps
    attr = evidences[:, 0].clamp(eps, 1.0 - eps)
    degree = evidences[:, 1].clamp(eps, 1.0 - eps)
    prior = edge_prior.to(score.dtype).clamp(eps, 1.0 - eps)
    with torch.no_grad():
        disagreement = (attr - degree).abs().clamp(0.0, 1.0)
        weights = torch.stack(
            [
                1.0 - disagreement,
                degree,
                prior,
            ],
            dim=1,
        ).clamp_min(eps)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(eps)
        teacher = (weights[:, 0] * attr + weights[:, 1] * degree + weights[:, 2] * prior).clamp(eps, 1.0 - eps)
        confidence = (teacher - 0.5).abs().mul(2.0).clamp(0.0, 1.0)
    bce = F.binary_cross_entropy(score.clamp(eps, 1.0 - eps), teacher, reduction="none")
    return (confidence * bce).mean()


def order_preserving_edge_ranking_loss(
    score: torch.Tensor,
    evidences: torch.Tensor,
    edge_index: torch.Tensor,
    edge_prior: torch.Tensor,
    *,
    pos_quantile: float,
    neg_quantile: float,
    margin: float,
    max_pairs: int,
    local_tau: float,
    raw_teacher_weight: float,
    raw_gate_margin: float,
    raw_gate_temperature: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    eps = torch.finfo(score.dtype).eps
    attr = evidences[:, 0].clamp(0.0, 1.0)
    degree = evidences[:, 1].clamp(0.0, 1.0)
    src = edge_index[0]
    with torch.no_grad():
        reliability = (1.0 - (attr - degree).abs()).clamp(0.0, 1.0)
        base_teacher = (attr * reliability).clamp(0.0, 1.0)
        prior = edge_prior.to(score.dtype).clamp(0.0, 1.0)
        raw_mask = prior >= 0.999
        non_raw_mask = ~raw_mask
        clean_evidence = (0.5 * (attr + degree) * reliability).clamp(0.0, 1.0)
        if bool(raw_mask.any()) and bool(non_raw_mask.any()):
            raw_clean = clean_evidence[raw_mask].mean()
            non_raw_clean = clean_evidence[non_raw_mask].mean()
            raw_advantage = raw_clean - non_raw_clean
            tau_gate = max(float(raw_gate_temperature), 1e-4)
            raw_gate = torch.sigmoid((raw_advantage - float(raw_gate_margin)) / tau_gate).clamp(0.0, 1.0)
        elif bool(raw_mask.any()):
            raw_clean_values = clean_evidence[raw_mask]
            raw_center = raw_clean_values.mean()
            raw_spread = raw_clean_values.std(unbiased=False)
            raw_advantage = raw_spread
            tau_gate = max(float(raw_gate_temperature), 1e-4)
            raw_gate = torch.sigmoid((raw_spread - float(raw_gate_margin)) / tau_gate).clamp(0.0, 1.0)
            centered = torch.sigmoid((clean_evidence - raw_center) / tau_gate)
            prior = prior * centered.clamp(0.0, 1.0)
        else:
            raw_advantage = clean_evidence.new_tensor(0.0)
            raw_gate = clean_evidence.new_tensor(0.0)
        raw_weight = (float(raw_teacher_weight) * raw_gate).clamp(0.0, 1.0)
        raw_teacher = (0.5 * prior + 0.5 * prior * reliability).clamp(0.0, 1.0)
        teacher_rank = ((1.0 - raw_weight) * base_teacher + raw_weight * raw_teacher).clamp(0.0, 1.0)
        if teacher_rank.numel() > int(max_pairs) > 0:
            keep = torch.randperm(teacher_rank.numel(), device=score.device)[: int(max_pairs)]
        else:
            keep = torch.arange(teacher_rank.numel(), device=score.device)
    if keep.numel() <= 1:
        zero = score.sum() * 0.0
        return zero, {
            "rank_pos_score": 0.0,
            "rank_neg_score": 0.0,
            "rank_gap": 0.0,
            "rank_pairs": 0.0,
            "rank_raw_gate": float(raw_gate.detach().cpu()),
            "rank_raw_advantage": float(raw_advantage.detach().cpu()),
            "rank_raw_weight": float(raw_weight.detach().cpu()),
        }
    src_k = src[keep]
    score_k = score[keep].clamp(eps, 1.0 - eps)
    teacher_k = teacher_rank[keep]
    tau = max(float(local_tau), 1e-4)
    n = int(src.max().detach().cpu()) + 1 if src.numel() else 0
    pos_logits = (teacher_k / tau).clamp(-30.0, 30.0)
    neg_logits = ((1.0 - teacher_k) / tau).clamp(-30.0, 30.0)
    pos_exp = torch.exp(pos_logits)
    neg_exp = torch.exp(neg_logits)
    pos_den = torch.zeros(n, device=score.device, dtype=score.dtype).scatter_add_(0, src_k, pos_exp).clamp_min(eps)
    neg_den = torch.zeros(n, device=score.device, dtype=score.dtype).scatter_add_(0, src_k, neg_exp).clamp_min(eps)
    pos_num = torch.zeros(n, device=score.device, dtype=score.dtype).scatter_add_(0, src_k, pos_exp * score_k)
    neg_num = torch.zeros(n, device=score.device, dtype=score.dtype).scatter_add_(0, src_k, neg_exp * score_k)
    counts = torch.zeros(n, device=score.device, dtype=score.dtype).scatter_add_(0, src_k, torch.ones_like(score_k))
    valid = counts > 1.0
    if not bool(valid.any()):
        zero = score.sum() * 0.0
        return zero, {
            "rank_pos_score": 0.0,
            "rank_neg_score": 0.0,
            "rank_gap": 0.0,
            "rank_pairs": 0.0,
            "rank_raw_gate": float(raw_gate.detach().cpu()),
            "rank_raw_advantage": float(raw_advantage.detach().cpu()),
            "rank_raw_weight": float(raw_weight.detach().cpu()),
        }
    pos_score = pos_num[valid] / pos_den[valid]
    neg_score = neg_num[valid] / neg_den[valid]
    loss = F.relu(neg_score - pos_score + float(margin)).mean()
    stats = {
        "rank_pos_score": float(pos_score.mean().detach().cpu()),
        "rank_neg_score": float(neg_score.mean().detach().cpu()),
        "rank_gap": float((pos_score.mean() - neg_score.mean()).detach().cpu()),
        "rank_pairs": float(valid.sum().detach().cpu()),
        "rank_raw_gate": float(raw_gate.detach().cpu()),
        "rank_raw_advantage": float(raw_advantage.detach().cpu()),
        "rank_raw_weight": float(raw_weight.detach().cpu()),
    }
    return loss, stats


def quantile_threshold_coupling_loss(
    score: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
    *,
    rho: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    rho = float(np.clip(rho, 1e-3, 0.49))
    if score.numel() > 8:
        with torch.no_grad():
            q_low = torch.quantile(score.detach(), rho)
            q_high = torch.quantile(score.detach(), 1.0 - rho)
    else:
        with torch.no_grad():
            q_low = score.detach().mean()
            q_high = score.detach().mean()
    loss = (high - q_high.to(high.dtype)).pow(2) + 0.5 * (low - q_low.to(low.dtype)).pow(2)
    stats = {
        "qanchor_low_target": float(q_low.detach().cpu()),
        "qanchor_high_target": float(q_high.detach().cpu()),
        "qanchor_high_gap": float((high.detach() - q_high.to(high.dtype)).cpu()),
        "qanchor_low_gap": float((low.detach() - q_low.to(low.dtype)).cpu()),
    }
    return loss, stats


def adaptive_prototypes(embedding: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    weights = q / q.sum(dim=0, keepdim=True).clamp_min(1e-8)
    prototypes = weights.T @ embedding
    return F.normalize(prototypes, p=2, dim=1)


def bootstrap_initial_embedding(init_out: dict[str, torch.Tensor], cfg: E2ESECTCoCoConfig) -> np.ndarray:
    mode = str(cfg.init_bootstrap_mode).lower()
    if mode not in {"multiview_svd", "embedding_low_svd"}:
        return init_out["embedding"].detach().cpu().numpy()
    if mode == "embedding_low_svd":
        views = [init_out["embedding"], init_out["low_view"]]
    else:
        views = [
            init_out["embedding"],
            init_out["z_attr"],
            init_out["low_view"],
            init_out["hetero_view"],
        ]
    arrays = [v.detach().cpu().numpy().astype(np.float32) for v in views]
    z = np.hstack(arrays).astype(np.float32)
    z = normalize(np.nan_to_num(z), norm="l2", axis=1)
    dim_cfg = int(cfg.init_bootstrap_dim) if int(cfg.init_bootstrap_dim) > 0 else int(cfg.projection_dim)
    dim = min(dim_cfg, z.shape[0] - 1, z.shape[1] - 1)
    if dim >= 2 and z.shape[1] > dim:
        z = TruncatedSVD(n_components=dim, random_state=cfg.seed).fit_transform(z)
    return normalize(np.nan_to_num(z).astype(np.float32), norm="l2", axis=1)


def prior_entropy_regularizer(prior: torch.Tensor) -> torch.Tensor:
    k = prior.numel()
    uniform = torch.full_like(prior, 1.0 / float(k))
    return F.kl_div(prior.clamp_min(1e-8).log(), uniform, reduction="sum")


def prototype_separation_regularizer(prototypes: torch.Tensor, *, margin: float) -> torch.Tensor:
    k = prototypes.shape[0]
    if k <= 1:
        return prototypes.sum() * 0.0
    p = F.normalize(prototypes, p=2, dim=1)
    sim = p @ p.T
    off_diag = ~torch.eye(k, device=prototypes.device, dtype=torch.bool)
    return F.relu(sim[off_diag] - float(margin)).pow(2).mean()


def prototype_readout_alignment_loss(
    q: torch.Tensor,
    embedding: torch.Tensor,
    prototypes: torch.Tensor,
    *,
    cluster_prior: torch.Tensor,
    alpha: torch.Tensor,
    q_base: torch.Tensor,
    temperature: float,
    conf_power: float,
    entropy_power: float,
    graph_gate: bool,
    prior_scale: float,
    alpha_floor: float,
    alpha_span: float,
    gate_floor: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    emb = F.normalize(embedding, p=2, dim=1)
    proto = F.normalize(prototypes, p=2, dim=1)
    logits = emb @ proto.T / max(1e-4, float(temperature))
    teacher = F.softmax(logits, dim=1).detach()
    per_node = F.kl_div(q.clamp_min(1e-8).log(), teacher, reduction="none").sum(dim=1)
    k = max(2, int(q.shape[1]))
    q_entropy = -torch.sum(q.detach() * q.detach().clamp_min(1e-8).log(), dim=1) / math.log(float(k))
    teacher_conf = (teacher.max(dim=1).values - (1.0 / float(k))) / max(1e-8, 1.0 - (1.0 / float(k)))
    weight = q_entropy.clamp(0.0, 1.0).pow(max(1e-4, float(entropy_power)))
    weight = weight * teacher_conf.clamp(0.0, 1.0).pow(max(1e-4, float(conf_power)))
    raw_loss = (weight * per_node).sum() / weight.sum().clamp_min(1e-8)
    if bool(graph_gate):
        prior_penalty = prior_entropy_regularizer(cluster_prior.detach()).clamp_min(0.0)
        prior_gate = torch.exp(-float(prior_scale) * prior_penalty).clamp(0.0, 1.0)
        alpha_attr = alpha.detach()[:, 0].mean().clamp(0.0, 1.0)
        alpha_gate = ((alpha_attr - float(alpha_floor)) / max(1e-4, float(alpha_span))).clamp(0.0, 1.0)
        flow_kl = F.kl_div(q.clamp_min(1e-8).log(), q_base.detach(), reduction="batchmean").detach().clamp_min(0.0)
        flow_gate = (1.0 / (1.0 + flow_kl)).clamp(0.0, 1.0)
        raw_gate = (prior_gate * alpha_gate * flow_gate).clamp(0.0, 1.0)
        floor_t = q.new_tensor(float(np.clip(gate_floor, 0.0, 1.0)))
        gate = (floor_t + (1.0 - floor_t) * raw_gate).detach()
    else:
        prior_gate = q.new_tensor(1.0)
        alpha_gate = q.new_tensor(1.0)
        flow_gate = q.new_tensor(1.0)
        raw_gate = q.new_tensor(1.0)
        gate = q.new_tensor(1.0)
    loss = gate * raw_loss
    entropy = -torch.sum(teacher * teacher.clamp_min(1e-8).log(), dim=1).mean()
    confidence = teacher.max(dim=1).values.mean()
    return loss, {
        "proto_readout_raw": float(raw_loss.detach().cpu()),
        "proto_readout_entropy": float(entropy.detach().cpu()),
        "proto_readout_conf": float(confidence.detach().cpu()),
        "proto_readout_weight": float(weight.mean().detach().cpu()),
        "proto_readout_graph_gate": float(gate.detach().cpu()),
        "proto_readout_raw_graph_gate": float(raw_gate.detach().cpu()),
        "proto_readout_prior_gate": float(prior_gate.detach().cpu()),
        "proto_readout_alpha_gate": float(alpha_gate.detach().cpu()),
        "proto_readout_flow_gate": float(flow_gate.detach().cpu()),
    }


def target_distribution(q: torch.Tensor) -> torch.Tensor:
    weight = q.pow(2) / q.sum(dim=0, keepdim=True).clamp_min(1e-8)
    return weight / weight.sum(dim=1, keepdim=True).clamp_min(1e-8)


def symmetric_info_nce(a: torch.Tensor, b: torch.Tensor, tau: float = 0.25) -> torch.Tensor:
    a = F.normalize(a, p=2, dim=1)
    b = F.normalize(b, p=2, dim=1)
    n = a.shape[0]
    if n > 4096:
        idx = torch.randperm(n, device=a.device)[:4096]
        a = a[idx]
        b = b[idx]
        n = a.shape[0]
    logits = a @ b.T / tau
    labels = torch.arange(n, device=a.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def threshold_regularizer(
    score: torch.Tensor,
    homo: torch.Tensor,
    hetero: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
    target_homo: float,
    target_hetero: float,
    adaptive: bool = False,
) -> torch.Tensor:
    margin_h = F.relu(high - score) * homo
    margin_l = F.relu(score - low) * hetero
    if adaptive and score.numel() > 8:
        target_homo_t, target_hetero_t = adaptive_threshold_occupancy(score)
        target_homo_value = target_homo_t.to(score.device, score.dtype)
        target_hetero_value = target_hetero_t.to(score.device, score.dtype)
    else:
        target_homo_value = torch.as_tensor(float(target_homo), device=score.device, dtype=score.dtype)
        target_hetero_value = torch.as_tensor(float(target_hetero), device=score.device, dtype=score.dtype)
    occupancy = (homo.mean() - target_homo_value).pow(2) + (hetero.mean() - target_hetero_value).pow(2)
    return margin_h.mean() + margin_l.mean() + occupancy


def adaptive_threshold_occupancy(score: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        spread = torch.quantile(score.detach(), 0.90) - torch.quantile(score.detach(), 0.10)
        hetero_target = (0.18 + 0.35 * (1.0 - spread).clamp(0.0, 1.0)).clamp(0.18, 0.50)
        homo_target = (0.18 + 0.35 * spread.clamp(0.0, 1.0)).clamp(0.18, 0.50)
        total = homo_target + hetero_target
        scale = torch.clamp(0.78 / total, max=1.0)
        return homo_target * scale, hetero_target * scale


def cluster_numpy(z: np.ndarray, n_clusters: int, cfg: E2ESECTCoCoConfig) -> tuple[np.ndarray, np.ndarray]:
    z = normalize(np.nan_to_num(z), norm="l2", axis=1)
    if cfg.use_minibatch_kmeans or z.shape[0] > int(cfg.extras.get("minibatch_threshold", 12000)):
        km = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=cfg.seed,
            n_init=max(3, min(10, int(cfg.kmeans_n_init))),
            batch_size=min(4096, max(512, z.shape[0] // 3)),
        )
    else:
        km = KMeans(n_clusters=n_clusters, random_state=cfg.seed, n_init=int(cfg.kmeans_n_init), max_iter=300)
    labels = km.fit_predict(z)
    centers = normalize(np.nan_to_num(km.cluster_centers_), norm="l2", axis=1).astype(np.float32)
    return labels.astype(np.int64), centers
