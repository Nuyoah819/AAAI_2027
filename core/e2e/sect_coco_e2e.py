from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
import time
from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from core.eval.metrics import evaluate_clustering
from core.legacy.sect_coco_legacy import (
    SECTCoCoConfig,
    _AnchorSubspaceHead,
    _elss_row_normalize,
    _elss_sym_normalize,
    _fast_elss_head as legacy_fast_elss_head,
    _scipy_to_torch64,
    _subspace_refine as legacy_subspace_refine,
)


LOGGER = logging.getLogger(__name__)


def _align_labels_to_reference(source_labels: np.ndarray, reference_labels: np.ndarray, n_clusters: int) -> np.ndarray:
    source = np.asarray(source_labels, dtype=np.int64).reshape(-1)
    reference = np.asarray(reference_labels, dtype=np.int64).reshape(-1)
    if source.shape[0] != reference.shape[0] or n_clusters <= 1:
        return source.copy()
    valid = (source >= 0) & (source < n_clusters) & (reference >= 0) & (reference < n_clusters)
    if not np.any(valid):
        return source.copy()
    overlap = np.zeros((n_clusters, n_clusters), dtype=np.int64)
    np.add.at(overlap, (source[valid], reference[valid]), 1)
    rows, cols = linear_sum_assignment(overlap.max() - overlap)
    mapping = np.arange(n_clusters, dtype=np.int64)
    mapping[rows] = cols
    aligned = source.copy()
    mapped = (aligned >= 0) & (aligned < n_clusters)
    aligned[mapped] = mapping[aligned[mapped]]
    return aligned


@dataclass
class E2ESECTCoCoConfig:
    seed: int = 42
    device: str = "cuda"
    small_graph_cpu_max_nodes: int = 0
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
    aptc_local_teacher: bool = False
    aptc_local_teacher_beta: float = 0.35
    aptc_local_teacher_pos_weight: float = 0.50
    aptc_local_teacher_hard_weight: float = 0.125
    aptc_local_teacher_neg_weight: float = 0.25
    aptc_local_teacher_temperature: float = 1.0
    aptc_local_teacher_use_prior_uniform: bool = False
    aptc_local_teacher_node_weight: str = "uniform"
    aptc_local_teacher_detach_masks: bool = True
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
    ideal_signed_embedding_weight: float = 0.0
    ideal_signed_homo_weight: float = 1.0
    ideal_signed_hetero_weight: float = 0.5
    ideal_signed_hard_weight: float = 0.10
    ideal_confidence_power: float = 1.0
    ideal_band_resolution_weight: float = 0.0
    ideal_band_center: float = 0.5
    ideal_band_width: float = 0.20
    ideal_highpass_energy_weight: float = 0.0
    ideal_highpass_conflict_power: float = 1.0
    v43b_conflict_margin_weight: float = 0.03
    v43b_conflict_margin: float = 0.25
    v43b_hard_conflict_weight: float = 0.5
    v43b_uncertainty_center: float = 0.40
    v43b_uncertainty_width: float = 0.40
    v43b_hard_clarity_floor: float = 0.25
    v43b_band_conflict_weight: float = 0.005
    v43b_highpass_energy_weight: float = 0.0
    v44_topology_band_resolution_weight: float = 0.0
    v44_conflict_highpass_corr_weight: float = 0.0
    v44_alpha_hard: float = 0.5
    v44_lambda_clear: float = 0.2
    v44_conflict_beta: float = 0.5
    v44_target_corr: float = 0.05
    v44_corr_eps: float = 1e-8
    v44b_pre_hp_corr_weight: float = 0.0
    v44b_conflict_beta: float = 0.5
    v44b_target_corr: float = 0.05
    v44b_corr_eps: float = 1e-8
    v45a_edge_freq_weight: float = 0.0
    v45a_band_guard_weight: float = 0.0
    v45a_warmup_epochs: int = 5
    v45a_band_gate_k: float = 20.0
    v45a_target_edge_gap: float = 0.0
    v45a_band_reference_delta: float = 0.0
    v45a_corr_eps: float = 1e-8
    v46a_band_cal_weight: float = 0.0
    v46a_balance_weight: float = 0.0
    v46a_spread_weight: float = 0.0
    v46a_entropy_floor: float = 0.60
    v46a_min_threshold_gap: float = 0.05
    v46a_corr_eps: float = 1e-8
    v47a_resolution_weight: float = 0.0
    v47a_usage_guard_weight: float = 0.0
    v47a_agree_high_quantile: float = 0.70
    v47a_agree_low_quantile: float = 0.30
    v47a_uncert_high_quantile: float = 0.70
    v47a_usage_entropy_floor: float = 0.60
    v47a_eps: float = 1e-8
    v48a_enabled: bool = False
    v48a_snapshot_sample_size: int = 20_000
    v48a_movement_eps: float = 1e-8
    v49a_enabled: bool = False
    v49a_tau_clear: float = 1.0
    v49a_tau_orient: float = 1.0
    v49a_snapshot_sample_size: int = 20_000
    v49a_movement_eps: float = 1e-8
    v50a_enabled: bool = False
    v50a_anchor_weight: float = 0.0
    v50a_anchor_source: str = "spectral"
    v50a_filter_steps: int = 2
    v50a_anchor_rank_multiplier: float = 1.0
    v50a_anchor_temperature: float = 0.35
    v50a_anchor_refresh: bool = False
    v51a_enabled: bool = False
    v51a_anchor_weight: float = 0.0
    v51a_reliability_floor: float = 0.10
    v51a_reliable_threshold: float = 0.20
    v51a_min_effective_mass: float = 0.10
    v52a_enabled: bool = False
    v52a_anchor_weight: float = 0.0
    v52a_reliability_floor: float = 0.10
    v52a_reliable_threshold: float = 0.20
    v52a_min_effective_mass: float = 0.10
    v52a_warmup_epochs: int = 20
    v52a_ramp_epochs: int = 40
    v53a_enabled: bool = False
    v53a_anchor_weight: float = 0.0
    v53a_reliability_floor: float = 0.10
    v53a_reliable_threshold: float = 0.20
    v53a_min_effective_mass: float = 0.10
    v53a_warmup_epochs: int = 20
    v53a_ramp_epochs: int = 40
    v53a_residual_beta: float = 0.50
    v54a_enabled: bool = False
    v54a_anchor_weight: float = 0.0
    v54a_reliability_floor: float = 0.10
    v54a_reliable_threshold: float = 0.20
    v54a_min_effective_mass: float = 0.10
    v54a_warmup_epochs: int = 20
    v54a_ramp_epochs: int = 40
    v54a_beta_min: float = 0.35
    v54a_beta_max: float = 0.70
    v55a_enabled: bool = False
    v55a_anchor_weight: float = 0.0
    v55a_reliability_floor: float = 0.10
    v55a_reliable_threshold: float = 0.20
    v55a_min_effective_mass: float = 0.10
    v55a_warmup_epochs: int = 20
    v55a_ramp_epochs: int = 40
    v55a_beta_min: float = 0.35
    v55a_beta_max: float = 0.70
    v55a_soft_power: float = 0.50
    v56a_enabled: bool = False
    v56a_anchor_weight: float = 0.0
    v56a_reliability_floor: float = 0.10
    v56a_reliable_threshold: float = 0.20
    v56a_min_effective_mass: float = 0.10
    v56a_warmup_epochs: int = 20
    v56a_ramp_epochs: int = 40
    v56a_beta_min: float = 0.35
    v56a_beta_max: float = 0.70
    v56a_soft_power: float = 0.50
    v56a_hybrid_compensation: float = 0.50
    v57a_enabled: bool = False
    v57a_anchor_weight: float = 0.0
    v57a_reliability_floor: float = 0.10
    v57a_reliable_threshold: float = 0.20
    v57a_min_effective_mass: float = 0.10
    v57a_warmup_epochs: int = 20
    v57a_ramp_epochs: int = 40
    v57a_beta_min: float = 0.35
    v57a_beta_max: float = 0.70
    v57a_soft_power: float = 0.50
    v57a_hybrid_compensation: float = 0.50
    v57a_target_mass: float = 0.08
    v57a_max_mass_scale: float = 1.50
    v57a_max_reliability_cap: float = 0.90
    v58a_enabled: bool = False
    v58a_anchor_weight: float = 0.0
    v58a_reliability_floor: float = 0.10
    v58a_reliable_threshold: float = 0.20
    v58a_min_effective_mass: float = 0.10
    v58a_warmup_epochs: int = 20
    v58a_ramp_epochs: int = 40
    v58a_beta_min: float = 0.35
    v58a_beta_max: float = 0.70
    v58a_soft_power: float = 0.50
    v58a_hybrid_compensation: float = 0.50
    v58a_target_mass: float = 0.08
    v58a_max_mass_scale: float = 1.50
    v58a_max_reliability_cap: float = 0.90
    v58a_release_warmup_epochs: int = 20
    v58a_release_ramp_epochs: int = 40
    v58a_release_hold_until_epoch: int = 80
    v58a_release_decay_epochs: int = 60
    v58a_release_floor: float = 0.25
    v59a_enabled: bool = False
    v59a_anchor_weight: float = 0.0
    v59a_reliability_floor: float = 0.10
    v59a_reliable_threshold: float = 0.20
    v59a_min_effective_mass: float = 0.10
    v59a_warmup_epochs: int = 20
    v59a_ramp_epochs: int = 40
    v59a_beta_min: float = 0.35
    v59a_beta_max: float = 0.70
    v59a_soft_power: float = 0.50
    v59a_hybrid_compensation: float = 0.50
    v59a_target_mass: float = 0.08
    v59a_max_mass_scale: float = 1.50
    v59a_max_reliability_cap: float = 0.90
    v59a_release_start_epoch: int = 80
    v59a_release_decay_epochs: int = 60
    v59a_release_floor: float = 0.25
    v60a_enabled: bool = False
    v60a_anchor_weight: float = 0.0
    v60a_guard_weight: float = 0.0
    v60a_confidence_threshold: float = 0.60
    v60a_start_epoch: int = 80
    v60a_guard_ramp_epochs: int = 20
    v60a_max_gamma: float = 1.0
    v60a_reliability_floor: float = 0.10
    v60a_reliable_threshold: float = 0.20
    v60a_min_effective_mass: float = 0.10
    v60a_warmup_epochs: int = 20
    v60a_ramp_epochs: int = 40
    v60a_beta_min: float = 0.35
    v60a_beta_max: float = 0.70
    v60a_soft_power: float = 0.50
    v60a_hybrid_compensation: float = 0.50
    v60a_target_mass: float = 0.08
    v60a_max_mass_scale: float = 1.50
    v60a_max_reliability_cap: float = 0.90
    v60a_release_start_epoch: int = 80
    v60a_release_decay_epochs: int = 60
    v60a_release_floor: float = 0.25
    v61a_enabled: bool = False
    v61a_anchor_weight: float = 0.0
    v61a_guard_weight: float = 0.0
    v61a_absolute_floor: float = 0.45
    v61a_min_teacher_coverage: float = 0.10
    v61a_start_epoch: int = 80
    v61a_guard_ramp_epochs: int = 20
    v61a_max_gamma: float = 1.0
    v61a_reliability_floor: float = 0.10
    v61a_reliable_threshold: float = 0.20
    v61a_min_effective_mass: float = 0.10
    v61a_warmup_epochs: int = 20
    v61a_ramp_epochs: int = 40
    v61a_beta_min: float = 0.35
    v61a_beta_max: float = 0.70
    v61a_soft_power: float = 0.50
    v61a_hybrid_compensation: float = 0.50
    v61a_target_mass: float = 0.08
    v61a_max_mass_scale: float = 1.50
    v61a_max_reliability_cap: float = 0.90
    v61a_release_start_epoch: int = 80
    v61a_release_decay_epochs: int = 60
    v61a_release_floor: float = 0.25
    v62a_enabled: bool = False
    v62a_anchor_weight: float = 0.0
    v62a_guard_weight: float = 0.0
    v62a_absolute_floor: float = 0.45
    v62a_min_teacher_coverage: float = 0.10
    v62a_start_epoch: int = 80
    v62a_guard_ramp_epochs: int = 20
    v62a_max_gamma: float = 1.0
    v62a_drift_start_epoch: int = 100
    v62a_drift_floor: float = 0.02
    v62a_drift_scale: float = 0.06
    v62a_drift_boost: float = 1.0
    v62a_max_effective_guard_multiplier: float = 2.0
    v62a_reliability_floor: float = 0.10
    v62a_reliable_threshold: float = 0.20
    v62a_min_effective_mass: float = 0.10
    v62a_warmup_epochs: int = 20
    v62a_ramp_epochs: int = 40
    v62a_beta_min: float = 0.35
    v62a_beta_max: float = 0.70
    v62a_soft_power: float = 0.50
    v62a_hybrid_compensation: float = 0.50
    v62a_target_mass: float = 0.08
    v62a_max_mass_scale: float = 1.50
    v62a_max_reliability_cap: float = 0.90
    v62a_release_start_epoch: int = 80
    v62a_release_decay_epochs: int = 60
    v62a_release_floor: float = 0.25
    v63b_enabled: bool = False
    v63b_edge_ood_weight: float = 0.0
    v63b_confusion_guard_weight: float = 0.0
    v63b_low_rescue_strength: float = 0.0
    v63b_high_suppress_strength: float = 0.0
    v63b_concordance_power: float = 1.0
    v63b_edge_margin: float = 0.12
    v63b_edge_pos_quantile: float = 0.80
    v63b_edge_neg_quantile: float = 0.20
    v63b_edge_max_pairs: int = 8192
    v63b_guard_floor: float = 0.25
    v63b_guard_power: float = 1.0
    v63b_guard_min_neighbor_count: float = 1.0
    v63b_graph_gate_center: float = 0.17
    v63b_graph_gate_scale: float = 0.025
    v63b_graph_gate_floor: float = 0.0
    v64a_enabled: bool = False
    v64a_subspace_source: str = "spectral"
    v64a_subspace_gram_weight: float = 0.0
    v64a_filter_steps: int = 2
    v64a_rank_multiplier: float = 4.0
    v64a_max_rank: int = 64
    v64a_gram_max_nodes: int = 1536
    v64a_start_epoch: int = 0
    v64a_ramp_epochs: int = 20
    v64a_release_start_epoch: int = 0
    v64a_release_decay_epochs: int = 0
    v64a_release_floor: float = 1.0
    v86a_v64_low_agreement_gate_enabled: bool = False
    v86a_v64_gate_center: float = 0.75
    v86a_v64_gate_scale: float = 0.05
    v86a_v64_gate_floor: float = 0.0
    v66a_elss_n_anchors: int = 300
    v66a_elss_power: int = 2
    v66a_elss_d: float = 0.875
    v66a_elss_alpha2: float = 0.00005
    v66a_elss_gamma: float = 0.003
    v66a_elss_filter_coef: float | None = None
    v66a_elss_return_k_rank: bool = False
    v66a_elss_k_rank: int = 0
    v66a_elss_q_norm: str = "l2"
    v67a_anchor_distrust_enabled: bool = False
    v67a_anchor_distrust_start_epoch: int = 100
    v67a_anchor_agreement_center: float = 0.75
    v67a_anchor_agreement_scale: float = 0.05
    v67a_anchor_distrust_floor: float = 0.10
    v90a_anchor_distrust_graph_gate_enabled: bool = False
    v90a_anchor_distrust_graph_noise_center: float = 0.20
    v90a_anchor_distrust_graph_noise_scale: float = 0.02
    v68a_low_agreement_teacher_boost_enabled: bool = False
    v68a_teacher_boost_start_epoch: int = 80
    v68a_teacher_boost_center: float = 0.75
    v68a_teacher_boost_scale: float = 0.05
    v68a_teacher_boost_max: float = 3.0
    v70a_low_agreement_entropy_guard_enabled: bool = False
    v70a_entropy_guard_weight: float = 0.0
    v70a_entropy_guard_start_epoch: int = 80
    v70a_entropy_guard_agreement_center: float = 0.75
    v70a_entropy_guard_agreement_scale: float = 0.05
    v70a_entropy_guard_floor: float = 0.35
    v71a_anchor_bypass_enabled: bool = False
    v71a_anchor_bypass_start_epoch: int = 80
    v71a_anchor_bypass_mix: float = 1.0
    v71a_anchor_bypass_soft_center: float = 0.55
    v71a_anchor_bypass_soft_scale: float = 0.08
    v71a_anchor_bypass_min_mass: float = 0.01
    v72a_stability_rollback_enabled: bool = False
    v72a_rollback_anchor_agreement_max: float = 0.55
    v72a_rollback_teacher_agreement_max: float = 0.92
    v74a_nc_weighted_readout_enabled: bool = False
    v74a_readout_weight_floor: float = 0.20
    v74a_readout_conf_power: float = 1.0
    v74a_readout_clarity_power: float = 1.5
    v75a_reliable_anchor_readout_enabled: bool = False
    v75a_anchor_readout_agreement_min: float = 0.95
    v75a_anchor_readout_cluster_separation_min: float = 0.0
    v78a_anchor_smoothing_enabled: bool = False
    v78a_anchor_smoothing_agreement_min: float = 0.0
    v78a_anchor_smoothing_conf_max: float = 0.95
    v78a_anchor_smoothing_majority_min: float = 0.70
    v78a_anchor_smoothing_max_change_ratio: float = 0.02
    v80a_anchor_smoothing_min_votes: int = 0
    v82a_anchor_diffusion_smoothing_enabled: bool = False
    v82a_anchor_diffusion_agreement_min: float = 0.97
    v82a_anchor_diffusion_steps: int = 2
    v82a_anchor_diffusion_restart: float = 0.65
    v82a_anchor_diffusion_conf_max: float = 0.98
    v82a_anchor_diffusion_margin_min: float = 0.20
    v82a_anchor_diffusion_max_change_ratio: float = 0.01
    v79a_consensus_smoothing_enabled: bool = False
    v79a_consensus_agreement_min: float = 0.97
    v79a_consensus_conf_max: float = 0.98
    v79a_consensus_majority_min: float = 0.62
    v79a_consensus_min_votes: int = 1
    v79a_consensus_max_change_ratio: float = 0.006
    v83a_final_neighbor_smoothing_enabled: bool = False
    v83a_final_neighbor_anchor_agreement_max: float = 0.90
    v83a_final_neighbor_majority_min: float = 0.70
    v83a_final_neighbor_max_change_ratio: float = 0.02
    v84a_raw_embedding_readout_enabled: bool = False
    v84a_raw_embedding_anchor_agreement_max: float = 0.90
    v84a_raw_embedding_kmeans_n_init: int = 10
    v84a_raw_embedding_kmeans_seed: int = -1
    v91a_spectral_anchor_readout_enabled: bool = False
    v91a_spectral_anchor_filter_steps: int = 5
    v91a_spectral_anchor_rank_multiplier: float = 4.0
    v91a_spectral_anchor_conf_min: float = 0.70
    v91a_spectral_anchor_entropy_max: float = 0.70
    v91a_spectral_anchor_balance_min: float = 0.40
    v93a_raw_feature_svd_readout_enabled: bool = False
    v93a_raw_feature_svd_anchor_agreement_max: float = 0.85
    v93a_raw_feature_svd_dim: int = 64
    v93a_raw_feature_svd_sil_min: float = 0.05
    v93a_raw_feature_svd_balance_min: float = 0.10
    v98a_gated_legacy_subspace_readout_enabled: bool = False
    v98a_legacy_anchor_agreement_min: float = 0.95
    v98a_legacy_hard_ratio_min: float = 0.75
    v98a_legacy_sil_min: float = 0.50
    v98a_legacy_balance_min: float = 0.20
    v99a_fast_elss_readout_enabled: bool = False
    v99a_fast_elss_require_spectral_anchor: bool = True
    v99a_fast_elss_sil_min: float = 0.45
    v99a_fast_elss_balance_min: float = 0.35
    v100a_embedding_svd_readout_enabled: bool = False
    v100a_embedding_svd_anchor_agreement_max: float = 0.58
    v100a_embedding_svd_dim: int = 20
    v100a_embedding_svd_kmeans_n_init: int = 10
    v100a_embedding_svd_kmeans_seed: int = 0
    v100a_embedding_svd_sil_min: float = 0.20
    v100a_embedding_svd_balance_min: float = 0.05
    v105a_size_pressure_enabled: bool = False
    v105a_size_pressure_ratio_min: float = 1.55
    v105a_size_pressure_target_ratio: float = 1.50
    v105a_size_pressure_max_change_ratio: float = 0.05
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
    subspace_sil_full_tolerance: float = 1.0
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


def _posterior_entropy(q: torch.Tensor) -> torch.Tensor:
    denom = math.log(float(max(2, q.shape[1])))
    return -(q * q.clamp_min(1e-8).log()).sum(dim=1) / max(denom, 1e-8)


def _edge_soft_agreement(q: torch.Tensor, edge_index: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if weight.numel() == 0:
        return q.new_tensor(0.0)
    src, dst = edge_index
    agree = (q[src] * q[dst]).sum(dim=1)
    return (weight.to(q.dtype) * agree).sum() / weight.to(q.dtype).sum().clamp_min(1e-8)


def _local_teacher_row_spmm(edge_index: torch.Tensor, weight: torch.Tensor, x: torch.Tensor, n: int) -> torch.Tensor:
    src, dst = edge_index
    w = weight.to(x.dtype).clamp_min(0.0)
    deg = torch.zeros(n, device=x.device, dtype=x.dtype).scatter_add_(0, dst, w).clamp_min(1e-8)
    out = torch.zeros_like(x)
    out.index_add_(0, dst, x[src] * (w / deg[dst]).unsqueeze(1))
    return out


def local_consensus_teacher_target(
    teacher: torch.Tensor,
    edge_index: torch.Tensor,
    degree: torch.Tensor,
    homo: torch.Tensor,
    hetero: torch.Tensor,
    hard: torch.Tensor,
    *,
    beta: float = 0.35,
    pos_weight: float = 0.50,
    hard_weight: float = 0.125,
    neg_weight: float = 0.25,
    temperature: float = 1.0,
    use_prior_uniform: bool = False,
    detach_masks: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    t0 = teacher.detach().clamp_min(1e-8)
    t0 = t0 / t0.sum(dim=1, keepdim=True).clamp_min(1e-8)
    mask_homo = homo.detach() if detach_masks else homo
    mask_hetero = hetero.detach() if detach_masks else hetero
    mask_hard = hard.detach() if detach_masks else hard
    pos_mask = (mask_homo + float(hard_weight) * mask_hard).clamp_min(1e-8)
    neg_mask = mask_hetero.clamp_min(1e-8)
    pos_msg = _local_teacher_row_spmm(edge_index, pos_mask, t0, degree.numel())
    neg_msg = _local_teacher_row_spmm(edge_index, neg_mask, t0, degree.numel())
    if bool(use_prior_uniform):
        prior_ref = torch.full_like(t0, 1.0 / float(t0.shape[1]))
        neg_term = prior_ref - neg_msg
    else:
        neg_term = neg_msg
    logits = (t0.clamp_min(1e-8).log() + float(pos_weight) * pos_msg - float(neg_weight) * neg_term) / max(
        1e-4,
        float(temperature),
    )
    t_local = F.softmax(logits, dim=1)
    beta_value = float(np.clip(beta, 0.0, 1.0))
    t_final = (1.0 - beta_value) * t0 + beta_value * t_local
    t_final = t_final / t_final.sum(dim=1, keepdim=True).clamp_min(1e-8)
    beta_tensor = t0.new_full((t0.shape[0],), beta_value)
    stats = {
        "local_teacher_beta": beta_tensor,
        "local_teacher_kl_to_t0": F.kl_div(t_final.clamp_min(1e-8).log(), t0, reduction="batchmean"),
        "local_teacher_entropy_t0": _posterior_entropy(t0).mean(),
        "local_teacher_entropy_local": _posterior_entropy(t_local).mean(),
        "local_teacher_entropy_final": _posterior_entropy(t_final).mean(),
        "local_teacher_pos_agree_t0": _edge_soft_agreement(t0, edge_index, pos_mask),
        "local_teacher_pos_agree_final": _edge_soft_agreement(t_final, edge_index, pos_mask),
        "local_teacher_hard_agree_final": _edge_soft_agreement(t_final, edge_index, mask_hard.clamp_min(1e-8)),
        "local_teacher_neg_overlap_t0": _edge_soft_agreement(t0, edge_index, neg_mask),
        "local_teacher_neg_overlap_final": _edge_soft_agreement(t_final, edge_index, neg_mask),
    }
    stats["local_teacher_pos_gain"] = stats["local_teacher_pos_agree_final"] - stats["local_teacher_pos_agree_t0"]
    stats["local_teacher_neg_reduction"] = stats["local_teacher_neg_overlap_t0"] - stats["local_teacher_neg_overlap_final"]
    return t_final, stats


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    weight = weight.to(value.dtype).clamp_min(0.0)
    return (value * weight).sum() / weight.sum().clamp_min(1e-8)


def ideal_signed_embedding_regularizer(
    z: torch.Tensor,
    edge_index: torch.Tensor,
    homo: torch.Tensor,
    hetero: torch.Tensor,
    hard: torch.Tensor,
    edge_confidence: torch.Tensor,
    *,
    homo_weight: float = 1.0,
    hetero_weight: float = 0.5,
    hard_weight: float = 0.10,
    confidence_power: float = 1.0,
    hard_margin: float = 0.25,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    src, dst = edge_index
    z_norm = F.normalize(z, p=2, dim=1)
    sim = (z_norm[src] * z_norm[dst]).sum(dim=1).clamp(-1.0, 1.0)
    conf = edge_confidence.detach().clamp(0.0, 1.0).pow(max(1e-4, float(confidence_power)))
    homo_w = homo.detach().clamp_min(0.0) * conf
    hetero_w = hetero.detach().clamp_min(0.0) * conf
    hard_w = hard.detach().clamp_min(0.0) * conf
    homo_loss = _weighted_mean(1.0 - sim, homo_w)
    hetero_loss = _weighted_mean(F.relu(sim), hetero_w)
    hard_loss = _weighted_mean(F.relu(sim - float(hard_margin)), hard_w)
    loss = float(homo_weight) * homo_loss + float(hetero_weight) * hetero_loss + float(hard_weight) * hard_loss
    stats = {
        "ideal_homo_sim_mean": _weighted_mean(sim, homo_w),
        "ideal_hetero_pos_overlap": _weighted_mean(F.relu(sim), hetero_w),
        "ideal_hard_pos_overlap": _weighted_mean(F.relu(sim - float(hard_margin)), hard_w),
    }
    return loss, stats


def ideal_band_resolution_regularizer(
    homo: torch.Tensor,
    hetero: torch.Tensor,
    hard: torch.Tensor,
    edge_confidence: torch.Tensor,
    *,
    confidence_power: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    conf = edge_confidence.detach().clamp(0.0, 1.0).pow(max(1e-4, float(confidence_power)))
    band_mass = _weighted_mean(hard, conf)
    clear_mass = _weighted_mean(torch.maximum(homo, hetero), conf)
    return band_mass, {"ideal_band_mass": band_mass, "ideal_clear_mass": clear_mass}


def ideal_highpass_energy_regularizer(
    low_view: torch.Tensor,
    high_view: torch.Tensor,
    edge_index: torch.Tensor,
    hetero: torch.Tensor,
    hard: torch.Tensor,
    *,
    conflict_power: float = 1.0,
    target_min_energy: float = 0.20,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    n = low_view.shape[0]
    high_energy = high_view.pow(2).sum(dim=1) / (low_view.pow(2).sum(dim=1) + high_view.pow(2).sum(dim=1) + 1e-8)
    src, dst = edge_index
    edge_conflict = (hetero.detach().clamp_min(0.0) + hard.detach().clamp_min(0.0)).clamp(0.0, 1.0)
    node_mass = high_view.new_zeros(n)
    node_degree = high_view.new_zeros(n)
    node_mass.index_add_(0, src, edge_conflict)
    node_mass.index_add_(0, dst, edge_conflict)
    node_degree.index_add_(0, src, torch.ones_like(edge_conflict))
    node_degree.index_add_(0, dst, torch.ones_like(edge_conflict))
    conflict = (node_mass / node_degree.clamp_min(1.0)).clamp(0.0, 1.0).pow(max(1e-4, float(conflict_power)))
    loss = F.relu(float(target_min_energy) * conflict - high_energy).mean()
    centered_conflict = conflict - conflict.mean()
    centered_energy = high_energy.detach() - high_energy.detach().mean()
    corr = (centered_conflict * centered_energy).mean() / (
        centered_conflict.pow(2).mean().sqrt() * centered_energy.pow(2).mean().sqrt() + 1e-8
    )
    stats = {
        "ideal_highpass_energy_mean": high_energy.mean(),
        "ideal_conflict_mean": conflict.mean(),
        "ideal_conflict_energy_corr": corr,
    }
    return loss, stats


def v64a_spectral_subspace_gram_alignment_loss(
    embedding: torch.Tensor,
    z_anchor: torch.Tensor,
    *,
    enabled: bool,
    current_epoch: int,
    start_epoch: int,
    ramp_epochs: int,
    release_start_epoch: int,
    release_decay_epochs: int,
    release_floor: float,
    max_nodes: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    zero = embedding.sum() * 0.0
    gamma_value = 0.0
    release_value = 1.0
    if bool(enabled):
        epoch_number = int(current_epoch) + 1
        if epoch_number >= int(start_epoch):
            ramp = max(1, int(ramp_epochs))
            gamma_value = min(1.0, max(0.0, (epoch_number - int(start_epoch) + 1) / float(ramp)))
        decay_epochs = max(0, int(release_decay_epochs))
        release_floor_value = float(np.clip(release_floor, 0.0, 1.0))
        if decay_epochs > 0 and epoch_number > int(release_start_epoch):
            progress = min(1.0, max(0.0, (epoch_number - int(release_start_epoch)) / float(decay_epochs)))
            release_value = max(release_floor_value, 1.0 - (1.0 - release_floor_value) * progress)
        else:
            release_value = 1.0
        gamma_value *= release_value
    gamma = embedding.new_tensor(float(gamma_value))
    release = embedding.new_tensor(float(release_value))
    stats_zero = {
        "v64a_subspace_gram_loss": zero.detach(),
        "v64a_subspace_gram_corr": zero.detach(),
        "v64a_subspace_sample_size": zero.detach(),
        "v64a_subspace_gamma": gamma.detach(),
        "v64a_subspace_release": release.detach(),
        "v64a_subspace_anchor_dim": zero.detach(),
    }
    if (
        (not bool(enabled))
        or gamma_value <= 0.0
        or z_anchor.numel() == 0
        or embedding.dim() != 2
        or z_anchor.dim() != 2
        or embedding.shape[0] != z_anchor.shape[0]
    ):
        return zero, stats_zero

    n = int(embedding.shape[0])
    sample_size = min(n, max(2, int(max_nodes)))
    if sample_size <= 1:
        return zero, stats_zero
    if n > sample_size:
        step = float(n) / float(sample_size)
        idx = (torch.arange(sample_size, device=embedding.device, dtype=torch.float32) * step).long()
        idx = idx.clamp_max(n - 1)
    else:
        idx = torch.arange(n, device=embedding.device)

    h = F.normalize(embedding[idx], p=2, dim=1, eps=1e-8)
    z = F.normalize(z_anchor.to(device=embedding.device, dtype=embedding.dtype)[idx].detach(), p=2, dim=1, eps=1e-8)
    gram_h = h @ h.T
    gram_z = z @ z.T
    gram_h = gram_h - gram_h.mean(dim=0, keepdim=True) - gram_h.mean(dim=1, keepdim=True) + gram_h.mean()
    gram_z = gram_z - gram_z.mean(dim=0, keepdim=True) - gram_z.mean(dim=1, keepdim=True) + gram_z.mean()
    raw_loss = F.mse_loss(gram_h, gram_z)
    denom = gram_h.pow(2).mean().sqrt() * gram_z.pow(2).mean().sqrt() + 1e-8
    corr = (gram_h * gram_z).mean() / denom
    loss = gamma * raw_loss
    stats = {
        "v64a_subspace_gram_loss": loss.detach(),
        "v64a_subspace_gram_corr": corr.detach(),
        "v64a_subspace_sample_size": embedding.new_tensor(float(idx.numel())).detach(),
        "v64a_subspace_gamma": gamma.detach(),
        "v64a_subspace_release": release.detach(),
        "v64a_subspace_anchor_dim": embedding.new_tensor(float(z_anchor.shape[1])).detach(),
    }
    return loss, stats


def v71a_hard_consensus_anchor_bypass_loss(
    q: torch.Tensor,
    q_anchor: torch.Tensor,
    q_embed: torch.Tensor | None,
    edge_index: torch.Tensor,
    *,
    enabled: bool,
    current_epoch: int,
    start_epoch: int,
    release_gamma: torch.Tensor,
    soft_center: float,
    soft_scale: float,
    min_mass: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    zero = q.new_tensor(0.0)
    stats_zero = {
        "v71a_anchor_bypass_loss": zero.detach(),
        "v71a_anchor_bypass_active": q.new_tensor(bool(enabled)).detach(),
        "v71a_anchor_bypass_gate_mean": zero.detach(),
        "v71a_anchor_bypass_hard_consensus": zero.detach(),
        "v71a_anchor_bypass_soft_consensus": zero.detach(),
        "v71a_anchor_bypass_reliability_mean": zero.detach(),
        "v71a_anchor_bypass_release_gamma": release_gamma.detach().to(q.device, dtype=q.dtype),
    }
    if (
        (not bool(enabled))
        or int(current_epoch) + 1 < int(start_epoch)
        or q_anchor.numel() != q.numel()
        or q.dim() != 2
    ):
        return zero, stats_zero

    anchor = q_anchor.to(device=q.device, dtype=q.dtype).detach().clamp_min(1e-8)
    anchor = anchor / anchor.sum(dim=1, keepdim=True).clamp_min(1e-8)
    student = q.clamp_min(1e-8)
    student = student / student.sum(dim=1, keepdim=True).clamp_min(1e-8)
    if q_embed is not None and q_embed.numel() == q.numel():
        embed = q_embed.to(device=q.device, dtype=q.dtype).detach().clamp_min(1e-8)
        embed = embed / embed.sum(dim=1, keepdim=True).clamp_min(1e-8)
    else:
        embed = student.detach()

    n = int(student.shape[0])
    k = max(2, int(student.shape[1]))
    inv_uniform_gap = float(k) / float(k - 1)
    anchor_label = anchor.argmax(dim=1)
    q_label = student.detach().argmax(dim=1)
    embed_label = embed.argmax(dim=1)
    hard_q = (q_label == anchor_label).to(q.dtype)
    hard_embed = (embed_label == anchor_label).to(q.dtype)
    hard_consensus = (0.5 * hard_q + 0.5 * hard_embed).detach()
    qa_norm = (((student.detach() * anchor).sum(dim=1) - (1.0 / float(k))) * inv_uniform_gap).clamp(0.0, 1.0)
    ea_norm = (((embed * anchor).sum(dim=1) - (1.0 / float(k))) * inv_uniform_gap).clamp(0.0, 1.0)
    soft_consensus = (0.5 * qa_norm + 0.5 * ea_norm).clamp(0.0, 1.0).detach()
    soft_gate = torch.sigmoid((soft_consensus - float(soft_center)) / max(1e-4, float(soft_scale))).detach()
    gate = (hard_consensus * (0.5 + 0.5 * soft_gate)).clamp(0.0, 1.0).detach()

    conf = ((anchor.max(dim=1).values - (1.0 / float(k))) * inv_uniform_gap).clamp(0.0, 1.0)
    if edge_index.numel() > 0:
        src, dst = edge_index[0].to(q.device), edge_index[1].to(q.device)
        sim = (anchor[src] * anchor[dst]).sum(dim=1)
        local_sum = q.new_zeros(n)
        local_cnt = q.new_zeros(n)
        one = torch.ones_like(sim)
        local_sum.index_add_(0, src, sim)
        local_sum.index_add_(0, dst, sim)
        local_cnt.index_add_(0, src, one)
        local_cnt.index_add_(0, dst, one)
        local = local_sum / local_cnt.clamp_min(1.0)
        local = torch.where(local_cnt > 0.0, local, conf)
    else:
        local = conf
    local_norm = ((local - (1.0 / float(k))) * inv_uniform_gap).clamp(0.0, 1.0)
    reliability = (gate * (0.5 * conf + 0.5 * local_norm)).clamp(0.0, 1.0).detach()
    per_node_kl = torch.sum(student * (student.clamp_min(1e-8).log() - anchor.clamp_min(1e-8).log()), dim=1)
    denom = reliability.sum().clamp_min(max(1e-8, float(min_mass)) * float(max(1, n)))
    release = release_gamma.to(device=q.device, dtype=q.dtype).detach()
    loss = release * torch.sum(reliability * per_node_kl) / denom
    stats = {
        "v71a_anchor_bypass_loss": loss.detach(),
        "v71a_anchor_bypass_active": q.new_tensor(True).detach(),
        "v71a_anchor_bypass_gate_mean": gate.mean().detach(),
        "v71a_anchor_bypass_hard_consensus": hard_consensus.mean().detach(),
        "v71a_anchor_bypass_soft_consensus": soft_consensus.mean().detach(),
        "v71a_anchor_bypass_reliability_mean": reliability.mean().detach(),
        "v71a_anchor_bypass_release_gamma": release.detach(),
    }
    return loss, stats


def v63b_edge_ood_ranking_loss(
    score: torch.Tensor,
    edge_logit: torch.Tensor,
    evidences: torch.Tensor,
    edge_prior: torch.Tensor,
    *,
    enabled: bool,
    margin: float,
    pos_quantile: float,
    neg_quantile: float,
    max_pairs: int,
    concordance_power: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    zero = score.sum() * 0.0
    stats_zero = {
        "v63b_edge_ood_loss": zero.detach(),
        "v63b_edge_rank_gap": zero.detach(),
        "v63b_edge_logit_gap": zero.detach(),
        "v63b_edge_pos_score": zero.detach(),
        "v63b_edge_neg_score": zero.detach(),
        "v63b_edge_clean_mass": zero.detach(),
        "v63b_edge_noise_mass": zero.detach(),
        "v63b_edge_pairs": zero.detach(),
    }
    if (not bool(enabled)) or score.numel() <= 1 or evidences.shape[0] != score.shape[0]:
        return zero, stats_zero

    with torch.no_grad():
        if score.numel() > int(max_pairs) > 0:
            step = float(score.numel()) / float(int(max_pairs))
            keep = (torch.arange(int(max_pairs), device=score.device, dtype=torch.float32) * step).long()
            keep = keep.clamp_max(score.numel() - 1)
        else:
            keep = torch.arange(score.numel(), device=score.device)
    if keep.numel() <= 1:
        return zero, stats_zero

    score_k = score[keep].clamp(1e-5, 1.0 - 1e-5)
    logit_k = edge_logit[keep]
    attr = evidences[keep, 0].detach().clamp(0.0, 1.0)
    degree = evidences[keep, 1].detach().clamp(0.0, 1.0) if evidences.shape[1] > 1 else attr
    concordance = evidences[keep, min(2, evidences.shape[1] - 1)].detach().clamp(0.0, 1.0)
    concordance = concordance.pow(max(1e-4, float(concordance_power)))
    prior = edge_prior[keep].to(score.dtype).detach().clamp(0.0, 1.0)
    feature_edge = (prior < 0.999).to(score.dtype)

    clean = (0.50 * attr + 0.25 * degree + 0.25 * prior) * concordance
    noise = (1.0 - attr) * (1.0 - concordance) * (0.5 + 0.5 * feature_edge)
    teacher = (clean - noise).clamp(-1.0, 1.0)
    if teacher.numel() > 1:
        pos_cut = torch.quantile(teacher, float(np.clip(pos_quantile, 0.0, 0.99))).detach()
        neg_cut = torch.quantile(teacher, float(np.clip(neg_quantile, 0.0, 0.99))).detach()
    else:
        pos_cut = teacher.mean().detach()
        neg_cut = teacher.mean().detach()
    pos_weight = (teacher >= pos_cut).to(score.dtype) * clean.clamp_min(0.0)
    neg_weight = (teacher <= neg_cut).to(score.dtype) * noise.clamp_min(0.0)
    if not bool((pos_weight.sum() > 1e-8 and neg_weight.sum() > 1e-8).detach().cpu()):
        return zero, stats_zero

    pos_score = _weighted_mean(score_k, pos_weight)
    neg_score = _weighted_mean(score_k, neg_weight)
    pos_logit = _weighted_mean(logit_k, pos_weight)
    neg_logit = _weighted_mean(logit_k, neg_weight)
    score_gap = pos_score - neg_score
    logit_gap = pos_logit - neg_logit
    margin_t = score.new_tensor(float(margin))
    rank_loss = F.relu(margin_t - score_gap)
    energy_loss = F.relu(2.0 * margin_t - logit_gap)
    loss = rank_loss + 0.25 * energy_loss
    stats = {
        "v63b_edge_ood_loss": loss.detach(),
        "v63b_edge_rank_gap": score_gap.detach(),
        "v63b_edge_logit_gap": logit_gap.detach(),
        "v63b_edge_pos_score": pos_score.detach(),
        "v63b_edge_neg_score": neg_score.detach(),
        "v63b_edge_clean_mass": clean.mean().detach(),
        "v63b_edge_noise_mass": noise.mean().detach(),
        "v63b_edge_pairs": score.new_tensor(float(keep.numel())),
    }
    return loss, stats


def v63b_confusion_aware_self_distillation_guard_loss(
    q: torch.Tensor,
    teacher_q: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    enabled: bool,
    teacher_ready: bool,
    teacher_epoch: int,
    current_epoch: int,
    guard_weight: float,
    absolute_floor: float,
    min_teacher_coverage: float,
    start_epoch: int,
    ramp_epochs: int,
    max_gamma: float,
    guard_floor: float,
    guard_power: float,
    min_neighbor_count: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    zero = q.new_tensor(0.0)
    one = q.new_tensor(1.0)
    gamma = q.new_tensor(
        _v60a_guard_gamma_value(
            current_epoch,
            start_epoch=start_epoch,
            ramp_epochs=ramp_epochs,
            max_gamma=max_gamma,
        )
        if bool(enabled) and bool(teacher_ready) and teacher_q.numel() == q.numel()
        else 0.0
    )
    guard_weight_t = q.new_tensor(float(guard_weight))
    teacher_epoch_t = q.new_tensor(float(teacher_epoch if bool(teacher_ready) else -1))
    floor_t = q.new_tensor(float(np.clip(absolute_floor, 0.0, 1.0)))
    coverage_t = q.new_tensor(float(np.clip(min_teacher_coverage, 0.0, 1.0)))
    guard_floor_t = q.new_tensor(float(np.clip(guard_floor, 0.0, 1.0)))
    if (not bool(enabled)) or (not bool(teacher_ready)) or teacher_q.numel() != q.numel():
        stats_zero = {
            "v63b_guard_enabled": q.new_tensor(bool(enabled)),
            "v63b_guard_teacher_ready": q.new_tensor(False),
            "v63b_guard_gamma": gamma.detach(),
            "v63b_guard_weight": guard_weight_t.detach(),
            "v63b_guard_loss": zero,
            "v63b_guard_kl": zero,
            "v63b_guard_node_gate_mean": zero,
            "v63b_guard_neighbor_agreement_mean": zero,
            "v63b_guard_neighbor_coverage": zero,
            "v63b_guard_active_ratio": zero,
            "v63b_guard_floor": guard_floor_t.detach(),
            "v63b_guard_teacher_epoch": teacher_epoch_t.detach(),
        }
        return zero, stats_zero

    teacher = teacher_q.to(device=q.device, dtype=q.dtype).detach().clamp_min(1e-8)
    teacher = teacher / teacher.sum(dim=1, keepdim=True).clamp_min(1e-8)
    student = q.clamp_min(1e-8)
    student = student / student.sum(dim=1, keepdim=True).clamp_min(1e-8)
    teacher_conf = teacher.max(dim=1).values.detach()
    floor_mask = (teacher_conf >= float(floor_t.detach().cpu())).detach()
    topk_mask = torch.zeros_like(floor_mask, dtype=torch.bool)
    n_nodes = int(teacher_conf.numel())
    coverage_value = float(coverage_t.detach().cpu())
    if n_nodes > 0 and coverage_value > 0.0:
        k = int(math.ceil(coverage_value * float(n_nodes)))
        k = max(1, min(n_nodes, k))
        top_idx = torch.topk(teacher_conf, k=k, largest=True, sorted=False).indices
        topk_mask[top_idx] = True
    active = (floor_mask | topk_mask).detach()

    src, dst = edge_index
    edge_agreement = (teacher[src] * teacher[dst]).sum(dim=1).detach().clamp(0.0, 1.0)
    node_sum = q.new_zeros(n_nodes)
    node_count = q.new_zeros(n_nodes)
    ones = torch.ones_like(edge_agreement)
    node_sum.index_add_(0, src, edge_agreement)
    node_sum.index_add_(0, dst, edge_agreement)
    node_count.index_add_(0, src, ones)
    node_count.index_add_(0, dst, ones)
    min_count = max(0.0, float(min_neighbor_count))
    has_neighbors = node_count >= min_count
    neighbor_agreement = node_sum / node_count.clamp_min(1.0)
    neighbor_agreement = torch.where(has_neighbors, neighbor_agreement, teacher_conf).clamp(0.0, 1.0).detach()
    k = max(2, int(teacher.shape[1]))
    conf_norm = ((teacher_conf - (1.0 / float(k))) * (float(k) / float(k - 1))).clamp(0.0, 1.0)
    reliability = (0.5 * conf_norm + 0.5 * neighbor_agreement).clamp(0.0, 1.0)
    node_gate = guard_floor_t + (one - guard_floor_t) * reliability.pow(max(1e-4, float(guard_power)))
    active_weight = active.to(q.dtype) * node_gate.detach()
    per_node_kl = torch.sum(teacher * (teacher.clamp_min(1e-8).log() - student.clamp_min(1e-8).log()), dim=1)
    guard_kl = _weighted_mean(per_node_kl, active_weight) if bool((active_weight.sum() > 1e-8).detach().cpu()) else zero
    guard_loss = gamma * guard_kl
    stats = {
        "v63b_guard_enabled": q.new_tensor(bool(enabled)),
        "v63b_guard_teacher_ready": q.new_tensor(True),
        "v63b_guard_gamma": gamma.detach(),
        "v63b_guard_weight": guard_weight_t.detach(),
        "v63b_guard_loss": guard_loss.detach(),
        "v63b_guard_kl": guard_kl.detach(),
        "v63b_guard_node_gate_mean": node_gate.mean().detach(),
        "v63b_guard_neighbor_agreement_mean": neighbor_agreement.mean().detach(),
        "v63b_guard_neighbor_coverage": has_neighbors.to(q.dtype).mean().detach(),
        "v63b_guard_active_ratio": (active_weight > 1e-8).to(q.dtype).mean().detach(),
        "v63b_guard_floor": guard_floor_t.detach(),
        "v63b_guard_teacher_epoch": teacher_epoch_t.detach(),
    }
    return guard_loss, stats


def v44_topology_band_resolution_regularizer(
    score: torch.Tensor,
    low_threshold: torch.Tensor,
    high_threshold: torch.Tensor,
    homo: torch.Tensor,
    hetero: torch.Tensor,
    hard: torch.Tensor,
    *,
    alpha_hard: float = 0.5,
    lambda_clear: float = 0.2,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    mid = 0.5 * (low_threshold + high_threshold)
    half_width = 0.5 * (high_threshold - low_threshold).abs().clamp_min(float(eps))
    score_uncertainty = (1.0 - (score - mid).abs() / half_width).clamp(0.0, 1.0)
    clear_mass = torch.maximum(homo, hetero).clamp(0.0, 1.0)
    band_mass = hard.clamp(0.0, 1.0)
    topology_conflict = (hetero + float(alpha_hard) * hard).clamp(0.0, 1.0)
    conflict_weight = (band_mass * score_uncertainty * topology_conflict).detach()
    loss = _weighted_mean(band_mass, conflict_weight) - float(lambda_clear) * _weighted_mean(clear_mass, conflict_weight)
    threshold_gap = (high_threshold - low_threshold).abs()
    decisive_mass = (homo + hetero).clamp(0.0, 1.0)
    score_uncertainty_detached = score_uncertainty.detach()
    stats = {
        "v44_band_loss": loss.detach(),
        "v44_band_mass": band_mass.detach().mean(),
        "v44_hard_ratio": hard.detach().mean(),
        "v44_ambiguous_ratio": ((score.detach() > low_threshold.detach()) & (score.detach() < high_threshold.detach())).to(score.dtype).mean(),
        "v44_clear_mass": clear_mass.detach().mean(),
        "v44_decisive_mass": decisive_mass.detach().mean(),
        "v44_score_uncertainty_mean": score_uncertainty_detached.mean(),
        "v44_score_uncertainty_p90": torch.quantile(score_uncertainty_detached, 0.90) if score_uncertainty_detached.numel() > 1 else score_uncertainty_detached.mean(),
        "v44_threshold_gap": threshold_gap.detach(),
        "v44_low_threshold": low_threshold.detach(),
        "v44_high_threshold": high_threshold.detach(),
        "v44_conflict_weight_mean": conflict_weight.mean(),
        "v44_topology_conflict_mean": topology_conflict.detach().mean(),
    }
    return loss, stats


def v44_conflict_coupled_highpass_regularizer(
    low_view: torch.Tensor,
    high_view: torch.Tensor,
    edge_index: torch.Tensor,
    hetero: torch.Tensor,
    hard: torch.Tensor,
    *,
    beta: float = 0.5,
    target_corr: float = 0.05,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    n = low_view.shape[0]
    src, dst = edge_index
    edge_conflict = (hetero + float(beta) * hard).detach().clamp(0.0, 1.0)
    node_mass = high_view.new_zeros(n)
    node_degree = high_view.new_zeros(n)
    node_mass.index_add_(0, src, edge_conflict)
    node_mass.index_add_(0, dst, edge_conflict)
    node_degree.index_add_(0, src, torch.ones_like(edge_conflict))
    node_degree.index_add_(0, dst, torch.ones_like(edge_conflict))
    valid = node_degree > 0
    node_conflict = node_mass / node_degree.clamp_min(1.0)
    high_sq = high_view.pow(2).sum(dim=1)
    low_sq = low_view.pow(2).sum(dim=1)
    high_energy = high_sq / (low_sq + high_sq + float(eps))
    if bool(valid.sum() > 1):
        conflict_valid = node_conflict[valid]
        energy_valid = high_energy[valid]
        centered_conflict = conflict_valid - conflict_valid.mean()
        centered_energy = energy_valid - energy_valid.mean()
        denom = centered_conflict.pow(2).mean().sqrt() * centered_energy.pow(2).mean().sqrt()
        corr = torch.where(
            denom > float(eps),
            (centered_conflict * centered_energy).mean() / (denom + float(eps)),
            high_view.new_tensor(0.0),
        )
    else:
        corr = high_view.new_tensor(0.0)
    corr = torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    loss = F.relu(high_view.new_tensor(float(target_corr)) - corr).pow(2)
    high_conflict = node_conflict.detach() >= node_conflict.detach().median()
    low_conflict = node_conflict.detach() < node_conflict.detach().median()
    high_conflict_energy = high_energy.detach()[high_conflict].mean() if bool(high_conflict.any()) else high_energy.detach().new_tensor(0.0)
    low_conflict_energy = high_energy.detach()[low_conflict].mean() if bool(low_conflict.any()) else high_energy.detach().new_tensor(0.0)
    stats = {
        "v44_highpass_loss": loss.detach(),
        "v44_conflict_energy_corr": corr.detach(),
        "v44_highpass_energy_mean": high_energy.detach().mean(),
        "v44_highpass_energy_std": high_energy.detach().std(unbiased=False),
        "v44_node_conflict_mean": node_conflict.detach().mean(),
        "v44_node_conflict_std": node_conflict.detach().std(unbiased=False),
        "v44_high_conflict_energy": high_conflict_energy,
        "v44_low_conflict_energy": low_conflict_energy,
        "v44_energy_gap": high_conflict_energy - low_conflict_energy,
    }
    return loss, stats


def v44b_pre_highpass_response_regularizer(
    pre_hp_response: torch.Tensor,
    edge_index: torch.Tensor,
    hetero: torch.Tensor,
    hard: torch.Tensor,
    low_view: torch.Tensor,
    high_view: torch.Tensor,
    *,
    beta: float = 0.5,
    target_corr: float = 0.05,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    n = pre_hp_response.shape[0]
    src, dst = edge_index
    edge_conflict = (hetero + float(beta) * hard).detach().clamp(0.0, 1.0)
    node_mass = pre_hp_response.new_zeros(n)
    node_degree = pre_hp_response.new_zeros(n)
    node_mass.index_add_(0, src, edge_conflict)
    node_mass.index_add_(0, dst, edge_conflict)
    node_degree.index_add_(0, src, torch.ones_like(edge_conflict))
    node_degree.index_add_(0, dst, torch.ones_like(edge_conflict))
    valid = node_degree > 0
    node_conflict = node_mass / node_degree.clamp_min(1.0)
    response = torch.nan_to_num(pre_hp_response, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    if bool(valid.sum() > 1):
        conflict_valid = node_conflict[valid]
        response_valid = response[valid]
        centered_conflict = conflict_valid - conflict_valid.mean()
        centered_response = response_valid - response_valid.mean()
        denom = centered_conflict.pow(2).mean().sqrt() * centered_response.pow(2).mean().sqrt()
        corr = torch.where(
            denom > float(eps),
            (centered_conflict * centered_response).mean() / (denom + float(eps)),
            response.new_tensor(0.0),
        )
    else:
        corr = response.new_tensor(0.0)
    corr = torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    loss = F.relu(response.new_tensor(float(target_corr)) - corr).pow(2)
    conflict_median = node_conflict.detach()[valid].median() if bool(valid.any()) else node_conflict.detach().median()
    high_conflict = node_conflict.detach() >= conflict_median
    low_conflict = node_conflict.detach() < conflict_median
    high_conflict_response = response.detach()[high_conflict].mean() if bool(high_conflict.any()) else response.detach().new_tensor(0.0)
    low_conflict_response = response.detach()[low_conflict].mean() if bool(low_conflict.any()) else response.detach().new_tensor(0.0)
    response_detached = response.detach()
    p10 = torch.quantile(response_detached, 0.10) if response_detached.numel() > 1 else response_detached.mean()
    p90 = torch.quantile(response_detached, 0.90) if response_detached.numel() > 1 else response_detached.mean()
    high_sq = high_view.detach().pow(2).sum(dim=1)
    low_sq = low_view.detach().pow(2).sum(dim=1)
    postnorm_energy = high_sq / (low_sq + high_sq + float(eps))
    post_high = postnorm_energy[high_conflict].mean() if bool(high_conflict.any()) else postnorm_energy.new_tensor(0.0)
    post_low = postnorm_energy[low_conflict].mean() if bool(low_conflict.any()) else postnorm_energy.new_tensor(0.0)
    stats = {
        "v44b_pre_hp_loss": loss.detach(),
        "v44b_pre_hp_response_mean": response_detached.mean(),
        "v44b_pre_hp_response_std": response_detached.std(unbiased=False),
        "v44b_pre_hp_response_p10": p10,
        "v44b_pre_hp_response_p90": p90,
        "v44b_pre_hp_response_ratio_p90_p10": p90 / p10.clamp_min(float(eps)),
        "v44b_conflict_response_corr": corr.detach(),
        "v44b_high_conflict_response": high_conflict_response,
        "v44b_low_conflict_response": low_conflict_response,
        "v44b_response_gap": high_conflict_response - low_conflict_response,
        "v44b_node_conflict_mean": node_conflict.detach().mean(),
        "v44b_node_conflict_std": node_conflict.detach().std(unbiased=False),
        "v44b_postnorm_hp_energy_mean": postnorm_energy.mean(),
        "v44b_postnorm_hp_energy_std": postnorm_energy.std(unbiased=False),
        "v44b_postnorm_energy_gap": post_high - post_low,
    }
    return loss, stats


def v45a_edge_local_band_guarded_frequency_regularizer(
    pre_hp_response: torch.Tensor,
    edge_index: torch.Tensor,
    homo: torch.Tensor,
    hetero: torch.Tensor,
    hard: torch.Tensor,
    band_reference: torch.Tensor,
    *,
    band_reference_ready: bool,
    active: bool,
    band_gate_k: float = 20.0,
    target_edge_gap: float = 0.0,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    src, dst = edge_index
    response = torch.nan_to_num(pre_hp_response, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    edge_response = 0.5 * (response[src] + response[dst])
    boundary_weight = (hetero.detach() + hard.detach()).clamp(0.0, 1.0)
    safe_homo_weight = homo.detach().clamp(0.0, 1.0)
    boundary_response_mean = _weighted_mean(edge_response, boundary_weight)
    safe_homo_response_mean = _weighted_mean(edge_response, safe_homo_weight)
    edge_response_gap = boundary_response_mean - safe_homo_response_mean
    edge_freq_loss_raw = F.relu(edge_response.new_tensor(float(target_edge_gap)) - edge_response_gap).pow(2)

    boundary_mass = boundary_weight.mean()
    safe_homo_mass = safe_homo_weight.mean()
    boundary_centered = boundary_weight - boundary_weight.mean()
    response_centered = edge_response.detach() - edge_response.detach().mean()
    denom = boundary_centered.pow(2).mean().sqrt() * response_centered.pow(2).mean().sqrt()
    edge_response_corr = torch.where(
        denom > float(eps),
        (boundary_centered * response_centered).mean() / (denom + float(eps)),
        edge_response.new_tensor(0.0),
    )
    edge_response_corr = torch.nan_to_num(edge_response_corr, nan=0.0, posinf=0.0, neginf=0.0)

    band_mass = hard.clamp(0.0, 1.0).mean()
    reference = band_reference.detach().to(device=band_mass.device, dtype=band_mass.dtype)
    safe_band_gate = torch.sigmoid(band_mass.new_tensor(float(band_gate_k)) * (reference - band_mass))
    band_guard_loss_raw = F.relu(band_mass - reference).pow(2)
    if bool(active) and bool(band_reference_ready):
        band_guard_loss = band_guard_loss_raw
        edge_freq_loss = safe_band_gate * edge_freq_loss_raw
    else:
        band_guard_loss = band_mass.new_tensor(0.0)
        edge_freq_loss = band_mass.new_tensor(0.0)
    stats = {
        "v45a_band_mass": band_mass.detach(),
        "v45a_band_reference": reference.detach(),
        "v45a_band_guard_loss": band_guard_loss.detach(),
        "v45a_safe_band_gate": safe_band_gate.detach(),
        "v45a_edge_freq_loss": edge_freq_loss.detach(),
        "v45a_edge_freq_loss_raw": edge_freq_loss_raw.detach(),
        "v45a_boundary_response_mean": boundary_response_mean.detach(),
        "v45a_safe_homo_response_mean": safe_homo_response_mean.detach(),
        "v45a_edge_response_gap": edge_response_gap.detach(),
        "v45a_edge_response_corr": edge_response_corr.detach(),
        "v45a_boundary_mass": boundary_mass.detach(),
        "v45a_safe_homo_mass": safe_homo_mass.detach(),
    }
    return edge_freq_loss, band_guard_loss, band_mass.detach(), stats


def v46a_topology_band_calibration_regularizer(
    homo: torch.Tensor,
    hetero: torch.Tensor,
    hard: torch.Tensor,
    low_threshold: torch.Tensor,
    high_threshold: torch.Tensor,
    *,
    entropy_floor: float = 0.60,
    min_threshold_gap: float = 0.05,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    band = hard.clamp(0.0, 1.0)
    band_cal_loss = band.pow(2).mean()
    raw_usage = torch.stack(
        [
            homo.clamp_min(0.0).mean(),
            hetero.clamp_min(0.0).mean(),
            band.mean(),
        ]
    ).clamp_min(float(eps))
    usage = raw_usage / raw_usage.sum().clamp_min(float(eps))
    usage_entropy = -torch.sum(usage * usage.clamp_min(float(eps)).log()) / math.log(3.0)
    balance_loss = F.relu(usage_entropy.new_tensor(float(entropy_floor)) - usage_entropy).pow(2)
    threshold_gap = (high_threshold - low_threshold).abs()
    spread_loss = F.relu(threshold_gap.new_tensor(float(min_threshold_gap)) - threshold_gap).pow(2)
    stats = {
        "v46a_band_cal_loss": band_cal_loss.detach(),
        "v46a_balance_loss": balance_loss.detach(),
        "v46a_spread_loss": spread_loss.detach(),
        "v46a_band_mass": band.detach().mean(),
        "v46a_homo_usage": usage[0].detach(),
        "v46a_hetero_usage": usage[1].detach(),
        "v46a_hard_usage": usage[2].detach(),
        "v46a_usage_entropy": usage_entropy.detach(),
        "v46a_threshold_gap": threshold_gap.detach(),
        "v46a_low_threshold": low_threshold.detach(),
        "v46a_high_threshold": high_threshold.detach(),
    }
    return band_cal_loss, balance_loss, spread_loss, stats


def v47a_posterior_guided_band_resolution_regularizer(
    q_posterior: torch.Tensor,
    edge_index: torch.Tensor,
    homo: torch.Tensor,
    hetero: torch.Tensor,
    hard: torch.Tensor,
    *,
    agree_high_quantile: float = 0.70,
    agree_low_quantile: float = 0.30,
    uncert_high_quantile: float = 0.70,
    usage_entropy_floor: float = 0.60,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    src, dst = edge_index
    q = q_posterior.detach().clamp_min(float(eps))
    q = q / q.sum(dim=1, keepdim=True).clamp_min(float(eps))
    qi = q[src]
    qj = q[dst]
    agreement = (qi * qj).sum(dim=1)
    denom = math.log(max(2, q.shape[1]))
    entropy_i = -torch.sum(qi * qi.clamp_min(float(eps)).log(), dim=1) / denom
    entropy_j = -torch.sum(qj * qj.clamp_min(float(eps)).log(), dim=1) / denom
    uncertainty = 0.5 * (entropy_i + entropy_j)
    if agreement.numel() > 1:
        agree_high = torch.quantile(agreement, float(agree_high_quantile))
        agree_low = torch.quantile(agreement, float(agree_low_quantile))
        uncert_high = torch.quantile(uncertainty, float(uncert_high_quantile))
    else:
        agree_high = agreement.mean()
        agree_low = agreement.mean()
        uncert_high = uncertainty.mean()

    homo_target = ((agreement >= agree_high) & (uncertainty < uncert_high)).to(homo.dtype)
    hetero_target = ((agreement <= agree_low) & (uncertainty < uncert_high)).to(homo.dtype)
    defer_target = (uncertainty >= uncert_high).to(homo.dtype)
    any_target = (homo_target + hetero_target + defer_target).clamp(0.0, 1.0)
    unassigned_target = (1.0 - any_target).clamp(0.0, 1.0)

    hard_weight = hard.clamp(0.0, 1.0)
    hard_mass = hard_weight.mean().clamp_min(float(eps))
    resolution_terms = (
        homo_target * (-homo.clamp_min(float(eps)).log())
        + hetero_target * (-hetero.clamp_min(float(eps)).log())
        + defer_target * (-hard.clamp_min(float(eps)).log())
    )
    resolution_loss = (hard_weight * resolution_terms).mean()

    raw_usage = torch.stack(
        [
            homo.clamp_min(0.0).mean(),
            hetero.clamp_min(0.0).mean(),
            hard_weight.mean(),
        ]
    ).clamp_min(float(eps))
    usage = raw_usage / raw_usage.sum().clamp_min(float(eps))
    usage_entropy = -torch.sum(usage * usage.clamp_min(float(eps)).log()) / math.log(3.0)
    usage_guard_loss = F.relu(usage_entropy.new_tensor(float(usage_entropy_floor)) - usage_entropy).pow(2)

    def _effective_mass(target: torch.Tensor) -> torch.Tensor:
        return (hard_weight.detach() * target.detach()).mean() / hard_mass.detach()

    stats = {
        "v47a_resolution_loss": resolution_loss.detach(),
        "v47a_usage_guard_loss": usage_guard_loss.detach(),
        "v47a_posterior_agreement_mean": agreement.mean().detach(),
        "v47a_posterior_agreement_std": agreement.std(unbiased=False).detach(),
        "v47a_posterior_uncertainty_mean": uncertainty.mean().detach(),
        "v47a_agree_high_threshold": agree_high.detach(),
        "v47a_agree_low_threshold": agree_low.detach(),
        "v47a_uncert_high_threshold": uncert_high.detach(),
        "v47a_homo_target_mass": _effective_mass(homo_target),
        "v47a_hetero_target_mass": _effective_mass(hetero_target),
        "v47a_defer_target_mass": _effective_mass(defer_target),
        "v47a_unassigned_target_mass": _effective_mass(unassigned_target),
        "v47a_raw_homo_target_mass": homo_target.detach().mean(),
        "v47a_raw_hetero_target_mass": hetero_target.detach().mean(),
        "v47a_raw_defer_target_mass": defer_target.detach().mean(),
        "v47a_raw_unassigned_target_mass": unassigned_target.detach().mean(),
        "v47a_effective_target_mass": _effective_mass(any_target),
        "v47a_band_mass": hard_weight.detach().mean(),
        "v47a_homo_usage": usage[0].detach(),
        "v47a_hetero_usage": usage[1].detach(),
        "v47a_hard_usage": usage[2].detach(),
        "v47a_usage_entropy": usage_entropy.detach(),
    }
    return resolution_loss, usage_guard_loss, stats


def conflict_margin_frontend_regularizer(
    z: torch.Tensor,
    edge_index: torch.Tensor,
    score: torch.Tensor,
    low_threshold: torch.Tensor,
    high_threshold: torch.Tensor,
    homo: torch.Tensor,
    hetero: torch.Tensor,
    hard: torch.Tensor,
    q_low: torch.Tensor,
    q_high: torch.Tensor,
    q_refined: torch.Tensor,
    low_view: torch.Tensor,
    high_view: torch.Tensor,
    *,
    margin: float = 0.25,
    hard_conflict_weight: float = 0.5,
    uncertainty_center: float = 0.40,
    uncertainty_width: float = 0.40,
    hard_clarity_floor: float = 0.25,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    src, dst = edge_index
    z_norm = F.normalize(z, p=2, dim=1)
    sim = (z_norm[src] * z_norm[dst]).sum(dim=1).clamp(-1.0, 1.0)
    structural_conflict = (hetero + float(hard_conflict_weight) * hard).clamp(0.0, 1.0)
    mid = 0.5 * (low_threshold + high_threshold)
    half_width = 0.5 * (high_threshold - low_threshold).abs().clamp_min(1e-6)
    clarity = ((score - mid).abs() / (half_width + 1e-8)).clamp(0.0, 1.0)
    clarity = torch.maximum(clarity, float(hard_clarity_floor) * hard.clamp(0.0, 1.0)).clamp(0.0, 1.0)
    low_agree = (q_low[src] * q_low[dst]).sum(dim=1)
    high_agree = (q_high[src] * q_high[dst]).sum(dim=1)
    view_disagreement = F.relu(low_agree - high_agree)
    q_entropy = _posterior_entropy(q_refined)
    edge_entropy = 0.5 * (q_entropy[src] + q_entropy[dst])
    uncertainty_gate = ((edge_entropy - float(uncertainty_center)) / max(1e-8, float(uncertainty_width))).clamp(0.0, 1.0)
    gate = (structural_conflict * clarity * view_disagreement * uncertainty_gate).detach()
    violation = F.relu(sim - float(margin))
    loss = _weighted_mean(violation.pow(2), gate)
    active = gate > 1e-6
    high_conflict = structural_conflict.detach() >= 0.5
    low_conflict = structural_conflict.detach() < 0.5
    high_energy = high_view.pow(2).sum(dim=1) / (low_view.pow(2).sum(dim=1) + high_view.pow(2).sum(dim=1) + 1e-8)
    edge_high_energy = 0.5 * (high_energy[src] + high_energy[dst])
    centered_conflict = structural_conflict.detach() - structural_conflict.detach().mean()
    centered_energy = edge_high_energy.detach() - edge_high_energy.detach().mean()
    conflict_energy_corr = (centered_conflict * centered_energy).mean() / (
        centered_conflict.pow(2).mean().sqrt() * centered_energy.pow(2).mean().sqrt() + 1e-8
    )
    band_mass = hard.detach().clamp_min(0.0).mean()
    conflict_band_mass = (hard.detach().clamp_min(0.0) * structural_conflict.detach()).mean()
    clear_mass = torch.maximum(homo.detach(), hetero.detach()).clamp_min(0.0).mean()
    band_conflict_loss = (hard.clamp_min(0.0) * structural_conflict).mean()
    stats = {
        "v43b_conflict_gate_mean": gate.mean(),
        "v43b_conflict_gate_std": gate.std(unbiased=False),
        "v43b_conflict_gate_active_ratio": active.to(gate.dtype).mean(),
        "v43b_conflict_gate_p90": torch.quantile(gate.detach(), 0.90) if gate.numel() > 1 else gate.detach().mean(),
        "v43b_structural_conflict_mean": structural_conflict.detach().mean(),
        "v43b_clarity_mean": clarity.detach().mean(),
        "v43b_view_disagreement_mean": view_disagreement.detach().mean(),
        "v43b_uncertainty_gate_mean": uncertainty_gate.detach().mean(),
        "v43b_conflict_margin_loss": loss.detach(),
        "v43b_conflict_margin_violation_mean": _weighted_mean(violation.detach(), gate),
        "v43b_conflict_margin_violation_ratio": ((violation.detach() > 0.0) & active).to(gate.dtype).sum() / active.to(gate.dtype).sum().clamp_min(1.0),
        "v43b_high_conflict_overlap": sim.detach()[high_conflict].mean() if bool(high_conflict.any()) else sim.new_tensor(0.0),
        "v43b_low_conflict_overlap": sim.detach()[low_conflict].mean() if bool(low_conflict.any()) else sim.new_tensor(0.0),
        "v43b_overlap_gap": (sim.detach()[high_conflict].mean() if bool(high_conflict.any()) else sim.new_tensor(0.0)) - (sim.detach()[low_conflict].mean() if bool(low_conflict.any()) else sim.new_tensor(0.0)),
        "v43b_band_conflict_loss": band_conflict_loss.detach(),
        "v43b_band_mass": band_mass,
        "v43b_conflict_band_mass": conflict_band_mass,
        "v43b_clear_mass": clear_mass,
        "v43b_low_high_disagreement_mean": (low_agree - high_agree).detach().mean(),
        "v43b_highpass_energy_mean": high_energy.detach().mean(),
        "v43b_conflict_energy_corr": conflict_energy_corr,
    }
    return loss, stats


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
        self.register_buffer("v60a_teacher_q", torch.empty(0))
        self.register_buffer("v60a_teacher_ready", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("v60a_teacher_epoch", torch.tensor(-1, dtype=torch.long))
        self.register_buffer("v61a_teacher_q", torch.empty(0))
        self.register_buffer("v61a_teacher_ready", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("v61a_teacher_epoch", torch.tensor(-1, dtype=torch.long))
        self.register_buffer("v62a_teacher_q", torch.empty(0))
        self.register_buffer("v62a_teacher_ready", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("v62a_teacher_epoch", torch.tensor(-1, dtype=torch.long))
        self.register_buffer("v45a_band_reference", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("v45a_warmup_band_sum", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("v45a_warmup_epoch_count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("v45a_last_warmup_epoch", torch.tensor(-1, dtype=torch.long))
        self.register_buffer("v45a_band_reference_ready", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("v48a_snapshot_ready", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("v48a_prev_homo", torch.empty(0))
        self.register_buffer("v48a_prev_hetero", torch.empty(0))
        self.register_buffer("v48a_prev_hard", torch.empty(0))
        self.register_buffer("v48a_prev_score", torch.empty(0))
        self.register_buffer("v48a_prev_low_threshold", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("v48a_prev_high_threshold", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("v49a_snapshot_ready", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("v49a_prev_homo", torch.empty(0))
        self.register_buffer("v49a_prev_hetero", torch.empty(0))
        self.register_buffer("v49a_prev_hard", torch.empty(0))
        self.register_buffer("v49a_prev_score", torch.empty(0))
        self.register_buffer("v50a_anchor_q", torch.empty(0))
        self.register_buffer("v64a_subspace_z", torch.empty(0))
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

    def set_v50a_anchor(self, q_anchor: torch.Tensor) -> None:
        self.v50a_anchor_q = q_anchor.detach().clone()

    def set_v64a_subspace_anchor(self, z_anchor: torch.Tensor) -> None:
        self.v64a_subspace_z = z_anchor.detach().clone()

    def _init_raw_skip(self) -> None:
        with torch.no_grad():
            self.raw_skip.weight.zero_()
            diag = min(self.raw_skip.weight.shape[0], self.raw_skip.weight.shape[1])
            self.raw_skip.weight[:diag, :diag] = torch.eye(diag, dtype=self.raw_skip.weight.dtype)

    def _v49a_reparameterized_topology(
        self,
        score: torch.Tensor,
        edge_logit: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        tau_clear = max(float(getattr(self.cfg, "v49a_tau_clear", 1.0)), 1e-6)
        tau_orient = max(float(getattr(self.cfg, "v49a_tau_orient", 1.0)), 1e-6)
        orient = torch.sigmoid(edge_logit / tau_orient)
        clear = torch.sigmoid(edge_logit.abs() / tau_clear)
        homo = clear * orient
        hetero = clear * (1.0 - orient)
        hard = (1.0 - clear).clamp(0.0, 1.0)
        return homo, hetero, hard, clear, orient

    def _v49a_topology_transition_diagnostics(self, out: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        enabled = bool(getattr(self.cfg, "v49a_enabled", False))
        score = out["score"]
        eps = float(getattr(self.cfg, "v49a_movement_eps", 1e-8))
        zero = score.new_tensor(0.0)
        sample_cap = int(max(1, getattr(self.cfg, "v49a_snapshot_sample_size", 20_000)))
        sample_size = int(min(score.numel(), sample_cap))
        clear = out.get("v49a_clear", score.new_zeros(score.shape))
        orient = out.get("v49a_orient", score.new_zeros(score.shape))

        if sample_size <= 0:
            return {
                "v49a_homo_usage": zero,
                "v49a_hetero_usage": zero,
                "v49a_hard_usage": zero,
                "v49a_band_mass": zero,
                "v49a_usage_entropy": zero,
                "v49a_clear_mean": zero,
                "v49a_clear_std": zero,
                "v49a_orient_mean": zero,
                "v49a_orient_std": zero,
                "v49a_has_prev_snapshot": torch.tensor(False, device=score.device),
                "v49a_sample_size": zero,
                "v49a_mean_abs_delta_homo": zero,
                "v49a_mean_abs_delta_hetero": zero,
                "v49a_mean_abs_delta_hard": zero,
                "v49a_mean_abs_delta_score": zero,
                "v49a_hard_mass_delta": zero,
                "v49a_hard_rank_corr_prev": zero,
                "v49a_homo_target_mass": zero,
                "v49a_hetero_target_mass": zero,
                "v49a_defer_target_mass": zero,
                "v49a_raw_homo_target_mass": zero,
                "v49a_raw_hetero_target_mass": zero,
                "v49a_raw_defer_target_mass": zero,
                "v49a_targeted_homo_delta": zero,
                "v49a_targeted_hetero_delta": zero,
                "v49a_targeted_hard_delta": zero,
            }

        sample = torch.arange(sample_size, device=score.device)
        homo = out["homo"][sample].detach()
        hetero = out["hetero"][sample].detach()
        hard = out["hard"][sample].detach()
        sampled_score = score[sample].detach()

        usage = torch.stack(
            [
                out["homo"].clamp_min(0.0).mean(),
                out["hetero"].clamp_min(0.0).mean(),
                out["hard"].clamp_min(0.0).mean(),
            ]
        )
        usage_norm = usage / usage.sum().clamp_min(eps)
        usage_entropy = -(usage_norm * usage_norm.clamp_min(eps).log()).sum() / math.log(3.0)

        q = out["q_refined"].detach().clamp_min(eps)
        q = q / q.sum(dim=1, keepdim=True).clamp_min(eps)
        src, dst = self.edge_index[:, sample]
        qi = q[src]
        qj = q[dst]
        agreement = (qi * qj).sum(dim=1)
        denom = math.log(max(2, q.shape[1]))
        entropy_i = -torch.sum(qi * qi.clamp_min(eps).log(), dim=1) / denom
        entropy_j = -torch.sum(qj * qj.clamp_min(eps).log(), dim=1) / denom
        uncertainty = 0.5 * (entropy_i + entropy_j)
        if agreement.numel() > 1:
            agree_high = torch.quantile(agreement, float(getattr(self.cfg, "v47a_agree_high_quantile", 0.70)))
            agree_low = torch.quantile(agreement, float(getattr(self.cfg, "v47a_agree_low_quantile", 0.30)))
            uncert_high = torch.quantile(uncertainty, float(getattr(self.cfg, "v47a_uncert_high_quantile", 0.70)))
        else:
            agree_high = agreement.mean()
            agree_low = agreement.mean()
            uncert_high = uncertainty.mean()
        homo_target = ((agreement >= agree_high) & (uncertainty < uncert_high)).to(score.dtype)
        hetero_target = ((agreement <= agree_low) & (uncertainty < uncert_high)).to(score.dtype)
        defer_target = (uncertainty >= uncert_high).to(score.dtype)
        hard_mass = hard.clamp(0.0, 1.0).mean().clamp_min(eps)

        def _effective_mass(target: torch.Tensor) -> torch.Tensor:
            return (hard.clamp(0.0, 1.0) * target).mean() / hard_mass

        has_prev = bool(self.v49a_snapshot_ready.detach().cpu()) and self.v49a_prev_homo.numel() == sample_size
        if has_prev:
            prev_homo = self.v49a_prev_homo.to(device=score.device, dtype=score.dtype)
            prev_hetero = self.v49a_prev_hetero.to(device=score.device, dtype=score.dtype)
            prev_hard = self.v49a_prev_hard.to(device=score.device, dtype=score.dtype)
            prev_score = self.v49a_prev_score.to(device=score.device, dtype=score.dtype)
            delta_homo = homo - prev_homo
            delta_hetero = hetero - prev_hetero
            delta_hard = hard - prev_hard
            delta_score = sampled_score - prev_score
            hard_center = hard - hard.mean()
            prev_hard_center = prev_hard - prev_hard.mean()
            corr_denom = hard_center.pow(2).mean().sqrt() * prev_hard_center.pow(2).mean().sqrt()
            hard_rank_corr = torch.where(
                corr_denom > eps,
                (hard_center * prev_hard_center).mean() / (corr_denom + eps),
                zero,
            )

            def _target_delta(delta: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
                target_sum = target.sum()
                if bool((target_sum > eps).detach().cpu()):
                    return (delta * target).sum() / target_sum.clamp_min(eps)
                return zero

            movement = {
                "v49a_has_prev_snapshot": torch.tensor(True, device=score.device),
                "v49a_sample_size": score.new_tensor(float(sample_size)),
                "v49a_mean_abs_delta_homo": delta_homo.abs().mean(),
                "v49a_mean_abs_delta_hetero": delta_hetero.abs().mean(),
                "v49a_mean_abs_delta_hard": delta_hard.abs().mean(),
                "v49a_mean_abs_delta_score": delta_score.abs().mean(),
                "v49a_hard_mass_delta": hard.mean() - prev_hard.mean(),
                "v49a_hard_rank_corr_prev": torch.nan_to_num(hard_rank_corr, nan=0.0, posinf=0.0, neginf=0.0),
                "v49a_targeted_homo_delta": _target_delta(delta_homo, homo_target),
                "v49a_targeted_hetero_delta": _target_delta(delta_hetero, hetero_target),
                "v49a_targeted_hard_delta": _target_delta(delta_hard, defer_target),
            }
        else:
            movement = {
                "v49a_has_prev_snapshot": torch.tensor(False, device=score.device),
                "v49a_sample_size": score.new_tensor(float(sample_size)),
                "v49a_mean_abs_delta_homo": zero,
                "v49a_mean_abs_delta_hetero": zero,
                "v49a_mean_abs_delta_hard": zero,
                "v49a_mean_abs_delta_score": zero,
                "v49a_hard_mass_delta": zero,
                "v49a_hard_rank_corr_prev": zero,
                "v49a_targeted_homo_delta": zero,
                "v49a_targeted_hetero_delta": zero,
                "v49a_targeted_hard_delta": zero,
            }

        stats = {
            "v49a_homo_usage": usage[0],
            "v49a_hetero_usage": usage[1],
            "v49a_hard_usage": usage[2],
            "v49a_band_mass": usage[2],
            "v49a_usage_entropy": usage_entropy,
            "v49a_clear_mean": clear.detach().mean(),
            "v49a_clear_std": clear.detach().std(unbiased=False),
            "v49a_orient_mean": orient.detach().mean(),
            "v49a_orient_std": orient.detach().std(unbiased=False),
            "v49a_homo_target_mass": _effective_mass(homo_target),
            "v49a_hetero_target_mass": _effective_mass(hetero_target),
            "v49a_defer_target_mass": _effective_mass(defer_target),
            "v49a_raw_homo_target_mass": homo_target.mean(),
            "v49a_raw_hetero_target_mass": hetero_target.mean(),
            "v49a_raw_defer_target_mass": defer_target.mean(),
        }
        stats.update(movement)

        if enabled and self.training:
            self.v49a_prev_homo = homo.detach().clone()
            self.v49a_prev_hetero = hetero.detach().clone()
            self.v49a_prev_hard = hard.detach().clone()
            self.v49a_prev_score = sampled_score.detach().clone()
            self.v49a_snapshot_ready.fill_(True)

        return {key: value.detach() for key, value in stats.items()}

    def _v48a_topology_dynamics_audit(self, out: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        enabled = bool(getattr(self.cfg, "v48a_enabled", False))
        score = out["score"]
        eps = float(getattr(self.cfg, "v48a_movement_eps", 1e-8))
        zero = score.new_tensor(0.0)
        sample_cap = int(max(1, getattr(self.cfg, "v48a_snapshot_sample_size", 20_000)))
        sample_size = int(min(score.numel(), sample_cap))
        if sample_size <= 0:
            return {
                "v48a_has_prev_snapshot": torch.tensor(False, device=score.device),
                "v48a_sample_size": zero,
                "v48a_mean_abs_delta_homo": zero,
                "v48a_mean_abs_delta_hetero": zero,
                "v48a_mean_abs_delta_hard": zero,
                "v48a_mean_abs_delta_score": zero,
                "v48a_hard_mass_delta": zero,
                "v48a_threshold_delta": zero,
                "v48a_hard_rank_corr_prev": zero,
                "v48a_homo_target_mass": zero,
                "v48a_hetero_target_mass": zero,
                "v48a_defer_target_mass": zero,
                "v48a_raw_homo_target_mass": zero,
                "v48a_raw_hetero_target_mass": zero,
                "v48a_raw_defer_target_mass": zero,
                "v48a_targeted_homo_delta": zero,
                "v48a_targeted_hetero_delta": zero,
                "v48a_targeted_hard_delta": zero,
            }

        sample = torch.arange(sample_size, device=score.device)
        homo = out["homo"][sample].detach()
        hetero = out["hetero"][sample].detach()
        hard = out["hard"][sample].detach()
        sampled_score = score[sample].detach()
        low_threshold = out["low_threshold"].detach()
        high_threshold = out["high_threshold"].detach()

        q = out["q_refined"].detach().clamp_min(eps)
        q = q / q.sum(dim=1, keepdim=True).clamp_min(eps)
        src, dst = self.edge_index[:, sample]
        qi = q[src]
        qj = q[dst]
        agreement = (qi * qj).sum(dim=1)
        denom = math.log(max(2, q.shape[1]))
        entropy_i = -torch.sum(qi * qi.clamp_min(eps).log(), dim=1) / denom
        entropy_j = -torch.sum(qj * qj.clamp_min(eps).log(), dim=1) / denom
        uncertainty = 0.5 * (entropy_i + entropy_j)
        if agreement.numel() > 1:
            agree_high = torch.quantile(agreement, float(getattr(self.cfg, "v47a_agree_high_quantile", 0.70)))
            agree_low = torch.quantile(agreement, float(getattr(self.cfg, "v47a_agree_low_quantile", 0.30)))
            uncert_high = torch.quantile(uncertainty, float(getattr(self.cfg, "v47a_uncert_high_quantile", 0.70)))
        else:
            agree_high = agreement.mean()
            agree_low = agreement.mean()
            uncert_high = uncertainty.mean()
        homo_target = ((agreement >= agree_high) & (uncertainty < uncert_high)).to(score.dtype)
        hetero_target = ((agreement <= agree_low) & (uncertainty < uncert_high)).to(score.dtype)
        defer_target = (uncertainty >= uncert_high).to(score.dtype)
        hard_mass = hard.clamp(0.0, 1.0).mean().clamp_min(eps)

        def _effective_mass(target: torch.Tensor) -> torch.Tensor:
            return (hard.clamp(0.0, 1.0) * target).mean() / hard_mass

        has_prev = bool(self.v48a_snapshot_ready.detach().cpu()) and self.v48a_prev_homo.numel() == sample_size
        if has_prev:
            prev_homo = self.v48a_prev_homo.to(device=score.device, dtype=score.dtype)
            prev_hetero = self.v48a_prev_hetero.to(device=score.device, dtype=score.dtype)
            prev_hard = self.v48a_prev_hard.to(device=score.device, dtype=score.dtype)
            prev_score = self.v48a_prev_score.to(device=score.device, dtype=score.dtype)
            prev_low = self.v48a_prev_low_threshold.to(device=score.device, dtype=score.dtype)
            prev_high = self.v48a_prev_high_threshold.to(device=score.device, dtype=score.dtype)
            delta_homo = homo - prev_homo
            delta_hetero = hetero - prev_hetero
            delta_hard = hard - prev_hard
            delta_score = sampled_score - prev_score
            hard_center = hard - hard.mean()
            prev_hard_center = prev_hard - prev_hard.mean()
            corr_denom = hard_center.pow(2).mean().sqrt() * prev_hard_center.pow(2).mean().sqrt()
            hard_rank_corr = torch.where(
                corr_denom > eps,
                (hard_center * prev_hard_center).mean() / (corr_denom + eps),
                zero,
            )

            def _target_delta(delta: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
                target_sum = target.sum()
                if bool((target_sum > eps).detach().cpu()):
                    return (delta * target).sum() / target_sum.clamp_min(eps)
                return zero

            stats = {
                "v48a_has_prev_snapshot": torch.tensor(True, device=score.device),
                "v48a_sample_size": score.new_tensor(float(sample_size)),
                "v48a_mean_abs_delta_homo": delta_homo.abs().mean(),
                "v48a_mean_abs_delta_hetero": delta_hetero.abs().mean(),
                "v48a_mean_abs_delta_hard": delta_hard.abs().mean(),
                "v48a_mean_abs_delta_score": delta_score.abs().mean(),
                "v48a_hard_mass_delta": hard.mean() - prev_hard.mean(),
                "v48a_threshold_delta": 0.5 * ((low_threshold - prev_low).abs() + (high_threshold - prev_high).abs()),
                "v48a_hard_rank_corr_prev": torch.nan_to_num(hard_rank_corr, nan=0.0, posinf=0.0, neginf=0.0),
                "v48a_targeted_homo_delta": _target_delta(delta_homo, homo_target),
                "v48a_targeted_hetero_delta": _target_delta(delta_hetero, hetero_target),
                "v48a_targeted_hard_delta": _target_delta(delta_hard, defer_target),
            }
        else:
            stats = {
                "v48a_has_prev_snapshot": torch.tensor(False, device=score.device),
                "v48a_sample_size": score.new_tensor(float(sample_size)),
                "v48a_mean_abs_delta_homo": zero,
                "v48a_mean_abs_delta_hetero": zero,
                "v48a_mean_abs_delta_hard": zero,
                "v48a_mean_abs_delta_score": zero,
                "v48a_hard_mass_delta": zero,
                "v48a_threshold_delta": zero,
                "v48a_hard_rank_corr_prev": zero,
                "v48a_targeted_homo_delta": zero,
                "v48a_targeted_hetero_delta": zero,
                "v48a_targeted_hard_delta": zero,
            }

        stats.update(
            {
                "v48a_homo_target_mass": _effective_mass(homo_target),
                "v48a_hetero_target_mass": _effective_mass(hetero_target),
                "v48a_defer_target_mass": _effective_mass(defer_target),
                "v48a_raw_homo_target_mass": homo_target.mean(),
                "v48a_raw_hetero_target_mass": hetero_target.mean(),
                "v48a_raw_defer_target_mass": defer_target.mean(),
            }
        )

        if enabled and self.training:
            self.v48a_prev_homo = homo.detach().clone()
            self.v48a_prev_hetero = hetero.detach().clone()
            self.v48a_prev_hard = hard.detach().clone()
            self.v48a_prev_score = sampled_score.detach().clone()
            self.v48a_prev_low_threshold.copy_(low_threshold.detach().to(self.v48a_prev_low_threshold.dtype))
            self.v48a_prev_high_threshold.copy_(high_threshold.detach().to(self.v48a_prev_high_threshold.dtype))
            self.v48a_snapshot_ready.fill_(True)

        return {key: value.detach() for key, value in stats.items()}

    def _frontend_pass(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z_attr = self.encoder(x)
        z_raw = F.normalize(self.raw_projector(x), p=2, dim=1)
        src, dst = self.edge_index
        edge_features, evidences = self._edge_features(z_attr, z_raw, src, dst)
        score, alpha, edge_logit = self.confidence(edge_features, evidences)
        homo, hetero, hard, low, high = self.contraction(score)
        v49a_clear = score.new_zeros(score.shape)
        v49a_orient = score.new_zeros(score.shape)
        if bool(getattr(self.cfg, "v49a_enabled", False)):
            homo, hetero, hard, v49a_clear, v49a_orient = self._v49a_reparameterized_topology(score, edge_logit)
        support_weight = self._support_weights(score, homo, hetero, hard)
        v63b_feature_rescue = score.new_zeros(score.shape)
        v63b_high_weight = hetero
        v63b_graph_noise = score.new_tensor(0.0)
        v63b_graph_gate = score.new_tensor(0.0)
        v63b_enabled = bool(getattr(self.cfg, "v63b_enabled", False))
        if v63b_enabled:
            attr_evidence = evidences[:, 0].detach().clamp(0.0, 1.0)
            concordance = evidences[:, min(2, evidences.shape[1] - 1)].detach().clamp(0.0, 1.0)
            concordance = concordance.pow(max(1e-4, float(getattr(self.cfg, "v63b_concordance_power", 1.0))))
            prior = self.edge_prior.to(score.dtype).detach().clamp(0.0, 1.0)
            feature_edge = (prior < 0.999).to(score.dtype)
            edge_noise = ((1.0 - attr_evidence) * (1.0 - concordance) * (0.5 + 0.5 * feature_edge)).clamp(0.0, 1.0)
            v63b_graph_noise = edge_noise.mean().detach()
            gate_center = float(getattr(self.cfg, "v63b_graph_gate_center", 0.17))
            gate_scale = max(1e-4, float(getattr(self.cfg, "v63b_graph_gate_scale", 0.025)))
            gate_floor = float(np.clip(getattr(self.cfg, "v63b_graph_gate_floor", 0.0), 0.0, 1.0))
            raw_graph_gate = torch.sigmoid((v63b_graph_noise - gate_center) / gate_scale)
            v63b_graph_gate = (gate_floor + (1.0 - gate_floor) * raw_graph_gate).clamp(0.0, 1.0).detach()
            v63b_feature_rescue = (v63b_graph_gate * feature_edge * prior * attr_evidence * concordance).clamp(0.0, 1.0)
            low_rescue_strength = max(0.0, float(getattr(self.cfg, "v63b_low_rescue_strength", 0.0)))
            high_suppress_strength = float(np.clip(getattr(self.cfg, "v63b_high_suppress_strength", 0.0), 0.0, 1.0))
            support_weight = (support_weight + low_rescue_strength * v63b_feature_rescue).clamp_min(1e-6)
            v63b_high_weight = (hetero * (1.0 - high_suppress_strength * v63b_feature_rescue)).clamp_min(1e-6)
        raw_edge_weight = self.edge_prior.to(score.dtype).clamp(0.0, 1.0)
        raw_leak_beta = torch.sigmoid(self.raw_leakage_gate)
        low_support_weight = ((1.0 - raw_leak_beta) * support_weight + raw_leak_beta * raw_edge_weight).clamp_min(1e-6)
        low_view = self._diffuse(z_attr, low_support_weight, steps=self.cfg.lowpass_steps)
        hetero_view = self._signed_highpass(z_attr, v63b_high_weight, steps=self.cfg.highpass_steps)
        low_smooth = normalized_spmm(self.edge_index, low_support_weight, z_attr, self.degree.numel())
        hetero_smooth = normalized_spmm(self.edge_index, v63b_high_weight.clamp_min(1e-6), z_attr, self.degree.numel())
        hetero_mass = v63b_high_weight.mean().clamp(0.05, 1.0)
        hetero_scale = 0.5 + 3.5 * torch.sigmoid(self.highpass_scale_logit)
        raw_high_response = (hetero_mass * hetero_scale * hetero_smooth).pow(2).sum(dim=1)
        raw_low_response = (low_smooth - z_attr).pow(2).sum(dim=1)
        pre_hp_response = torch.log1p(raw_high_response)
        pre_hp_energy = raw_high_response / (raw_high_response + raw_low_response + 1e-8)
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
            "v44b_pre_hp_response": pre_hp_response,
            "v44b_pre_hp_energy": pre_hp_energy,
            "v44b_raw_high_response": raw_high_response,
            "v44b_raw_low_response": raw_low_response,
            "z_cross_alignment": z_cross_alignment,
            "embedding": embedding,
            "score": score,
            "edge_logit": edge_logit,
            "alpha": alpha,
            "homo": homo,
            "hetero": hetero,
            "hard": hard,
            "v49a_clear": v49a_clear,
            "v49a_orient": v49a_orient,
            "v63b_feature_rescue": v63b_feature_rescue,
            "v63b_high_weight": v63b_high_weight,
            "v63b_graph_noise": v63b_graph_noise,
            "v63b_graph_gate": v63b_graph_gate,
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
        local_teacher_enabled = bool(getattr(cfg, "aptc_local_teacher", False))
        local_teacher_stats = {
            "local_teacher_beta": q_reg.new_zeros(q_reg.shape[0]),
            "local_teacher_kl_to_t0": q_reg.new_tensor(0.0),
            "local_teacher_entropy_t0": q_reg.new_tensor(0.0),
            "local_teacher_entropy_local": q_reg.new_tensor(0.0),
            "local_teacher_entropy_final": q_reg.new_tensor(0.0),
            "local_teacher_pos_agree_t0": q_reg.new_tensor(0.0),
            "local_teacher_pos_agree_final": q_reg.new_tensor(0.0),
            "local_teacher_hard_agree_final": q_reg.new_tensor(0.0),
            "local_teacher_neg_overlap_t0": q_reg.new_tensor(0.0),
            "local_teacher_neg_overlap_final": q_reg.new_tensor(0.0),
            "local_teacher_pos_gain": q_reg.new_tensor(0.0),
            "local_teacher_neg_reduction": q_reg.new_tensor(0.0),
        }
        local_teacher_qreg_kl = q_reg.new_tensor(0.0)
        local_teacher_qrefined_kl = q_reg.new_tensor(0.0)
        local_teacher_qmix_kl = q_reg.new_tensor(0.0)
        if self.init_teacher.numel() == q_reg.numel():
            teacher_t0 = self.init_teacher.detach().clamp_min(1e-8)
            teacher_t0 = teacher_t0 / teacher_t0.sum(dim=1, keepdim=True).clamp_min(1e-8)
            teacher = teacher_t0
            if local_teacher_enabled:
                teacher, local_teacher_stats = local_consensus_teacher_target(
                    teacher_t0,
                    self.edge_index,
                    self.degree,
                    out["homo"],
                    out["hetero"],
                    out["hard"],
                    beta=float(getattr(cfg, "aptc_local_teacher_beta", 0.35)),
                    pos_weight=float(getattr(cfg, "aptc_local_teacher_pos_weight", 0.50)),
                    hard_weight=float(getattr(cfg, "aptc_local_teacher_hard_weight", 0.125)),
                    neg_weight=float(getattr(cfg, "aptc_local_teacher_neg_weight", 0.25)),
                    temperature=float(getattr(cfg, "aptc_local_teacher_temperature", 1.0)),
                    use_prior_uniform=bool(getattr(cfg, "aptc_local_teacher_use_prior_uniform", False)),
                    detach_masks=bool(getattr(cfg, "aptc_local_teacher_detach_masks", True)),
                )
                local_teacher_qreg_kl = F.kl_div(q_reg.clamp_min(1e-8).log(), teacher, reduction="batchmean")
                local_teacher_qrefined_kl = F.kl_div(out["q_refined"].clamp_min(1e-8).log(), teacher, reduction="batchmean")
                local_teacher_qmix_kl = F.kl_div(out["q"].clamp_min(1e-8).log(), teacher, reduction="batchmean")
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
            node_weight_mode = str(getattr(cfg, "aptc_local_teacher_node_weight", "uniform")).lower()
            if local_teacher_enabled and node_weight_mode == "uniform":
                conf_weight = torch.ones_like(teacher_conf)
                teacher_conf_center = teacher_conf.new_tensor(float(cfg.aptc_teacher_conf_center))
                teacher_conf_scale = teacher_conf.new_tensor(max(1.0 - float(cfg.aptc_teacher_conf_center), 1e-8))
            elif conf_mode in {"adaptive_quantile", "quantile"} and teacher_conf.numel() > 1:
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
            if not (local_teacher_enabled and node_weight_mode == "uniform"):
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
        ideal_enabled = (
            float(getattr(cfg, "ideal_signed_embedding_weight", 0.0)) > 0.0
            or float(getattr(cfg, "ideal_band_resolution_weight", 0.0)) > 0.0
            or float(getattr(cfg, "ideal_highpass_energy_weight", 0.0)) > 0.0
        )
        ideal_signed_embedding_loss, ideal_signed_stats = ideal_signed_embedding_regularizer(
            out["embedding"],
            self.edge_index,
            out["homo"],
            out["hetero"],
            out["hard"],
            out["score"],
            homo_weight=float(getattr(cfg, "ideal_signed_homo_weight", 1.0)),
            hetero_weight=float(getattr(cfg, "ideal_signed_hetero_weight", 0.5)),
            hard_weight=float(getattr(cfg, "ideal_signed_hard_weight", 0.10)),
            confidence_power=float(getattr(cfg, "ideal_confidence_power", 1.0)),
        )
        ideal_band_resolution_loss, ideal_band_stats = ideal_band_resolution_regularizer(
            out["homo"],
            out["hetero"],
            out["hard"],
            out["score"],
            confidence_power=float(getattr(cfg, "ideal_confidence_power", 1.0)),
        )
        ideal_highpass_energy_loss, ideal_highpass_stats = ideal_highpass_energy_regularizer(
            out["low_view"],
            out["hetero_view"],
            self.edge_index,
            out["hetero"],
            out["hard"],
            conflict_power=float(getattr(cfg, "ideal_highpass_conflict_power", 1.0)),
        )
        v43b_enabled = float(getattr(cfg, "v43b_conflict_margin_weight", 0.0)) > 0.0
        v43b_conflict_margin_loss, v43b_stats = conflict_margin_frontend_regularizer(
            out["embedding"],
            self.edge_index,
            out["score"],
            out["low_threshold"],
            out["high_threshold"],
            out["homo"],
            out["hetero"],
            out["hard"],
            out["q_low"],
            out["q_high"],
            out["q_refined"],
            out["low_view"],
            out["hetero_view"],
            margin=float(getattr(cfg, "v43b_conflict_margin", 0.25)),
            hard_conflict_weight=float(getattr(cfg, "v43b_hard_conflict_weight", 0.5)),
            uncertainty_center=float(getattr(cfg, "v43b_uncertainty_center", 0.40)),
            uncertainty_width=float(getattr(cfg, "v43b_uncertainty_width", 0.40)),
            hard_clarity_floor=float(getattr(cfg, "v43b_hard_clarity_floor", 0.25)),
        )
        v44_enabled = (
            float(getattr(cfg, "v44_topology_band_resolution_weight", 0.0)) > 0.0
            or float(getattr(cfg, "v44_conflict_highpass_corr_weight", 0.0)) > 0.0
        )
        v44b_enabled = float(getattr(cfg, "v44b_pre_hp_corr_weight", 0.0)) > 0.0
        v44_band_loss, v44_band_stats = v44_topology_band_resolution_regularizer(
            out["score"],
            out["low_threshold"],
            out["high_threshold"],
            out["homo"],
            out["hetero"],
            out["hard"],
            alpha_hard=float(getattr(cfg, "v44_alpha_hard", 0.5)),
            lambda_clear=float(getattr(cfg, "v44_lambda_clear", 0.2)),
            eps=float(getattr(cfg, "v44_corr_eps", 1e-8)),
        )
        v44_highpass_loss, v44_highpass_stats = v44_conflict_coupled_highpass_regularizer(
            out["low_view"],
            out["hetero_view"],
            self.edge_index,
            out["hetero"],
            out["hard"],
            beta=float(getattr(cfg, "v44_conflict_beta", 0.5)),
            target_corr=float(getattr(cfg, "v44_target_corr", 0.05)),
            eps=float(getattr(cfg, "v44_corr_eps", 1e-8)),
        )
        v44b_pre_hp_loss, v44b_stats = v44b_pre_highpass_response_regularizer(
            out["v44b_pre_hp_response"],
            self.edge_index,
            out["hetero"],
            out["hard"],
            out["low_view"],
            out["hetero_view"],
            beta=float(getattr(cfg, "v44b_conflict_beta", 0.5)),
            target_corr=float(getattr(cfg, "v44b_target_corr", 0.05)),
            eps=float(getattr(cfg, "v44b_corr_eps", 1e-8)),
        )
        v45a_enabled = (
            float(getattr(cfg, "v45a_edge_freq_weight", 0.0)) > 0.0
            or float(getattr(cfg, "v45a_band_guard_weight", 0.0)) > 0.0
        )
        v45a_warmup_epochs = int(max(0, getattr(cfg, "v45a_warmup_epochs", 5)))
        current_epoch = int(getattr(self, "runtime_epoch", -1))
        v45a_observed_band_mass = out["hard"].detach().clamp(0.0, 1.0).mean()
        if (
            bool(self.training)
            and bool(v45a_enabled)
            and current_epoch >= 0
            and current_epoch < v45a_warmup_epochs
            and int(self.v45a_last_warmup_epoch.detach().cpu()) != current_epoch
        ):
            self.v45a_warmup_band_sum.add_(v45a_observed_band_mass.to(self.v45a_warmup_band_sum.dtype))
            self.v45a_warmup_epoch_count.add_(1)
            self.v45a_last_warmup_epoch.fill_(current_epoch)
            if int(self.v45a_warmup_epoch_count.detach().cpu()) >= v45a_warmup_epochs:
                reference = self.v45a_warmup_band_sum / self.v45a_warmup_epoch_count.clamp_min(1).to(self.v45a_warmup_band_sum.dtype)
                reference = reference - float(getattr(cfg, "v45a_band_reference_delta", 0.0))
                self.v45a_band_reference.copy_(reference.detach())
                self.v45a_band_reference_ready.fill_(True)
        elif bool(v45a_enabled) and v45a_warmup_epochs <= 0 and not bool(self.v45a_band_reference_ready.detach().cpu()):
            reference = v45a_observed_band_mass.to(self.v45a_band_reference.dtype) - float(getattr(cfg, "v45a_band_reference_delta", 0.0))
            self.v45a_band_reference.copy_(reference.detach())
            self.v45a_band_reference_ready.fill_(True)
        v45a_losses_active = bool(v45a_enabled) and bool(self.v45a_band_reference_ready.detach().cpu()) and current_epoch >= v45a_warmup_epochs
        v45a_edge_freq_loss, v45a_band_guard_loss, _, v45a_stats = v45a_edge_local_band_guarded_frequency_regularizer(
            out["v44b_pre_hp_response"],
            self.edge_index,
            out["homo"],
            out["hetero"],
            out["hard"],
            self.v45a_band_reference,
            band_reference_ready=bool(self.v45a_band_reference_ready.detach().cpu()),
            active=v45a_losses_active,
            band_gate_k=float(getattr(cfg, "v45a_band_gate_k", 20.0)),
            target_edge_gap=float(getattr(cfg, "v45a_target_edge_gap", 0.0)),
            eps=float(getattr(cfg, "v45a_corr_eps", 1e-8)),
        )
        v46a_enabled = (
            float(getattr(cfg, "v46a_band_cal_weight", 0.0)) > 0.0
            or float(getattr(cfg, "v46a_balance_weight", 0.0)) > 0.0
            or float(getattr(cfg, "v46a_spread_weight", 0.0)) > 0.0
        )
        v46a_band_cal_loss, v46a_balance_loss, v46a_spread_loss, v46a_stats = v46a_topology_band_calibration_regularizer(
            out["homo"],
            out["hetero"],
            out["hard"],
            out["low_threshold"],
            out["high_threshold"],
            entropy_floor=float(getattr(cfg, "v46a_entropy_floor", 0.60)),
            min_threshold_gap=float(getattr(cfg, "v46a_min_threshold_gap", 0.05)),
            eps=float(getattr(cfg, "v46a_corr_eps", 1e-8)),
        )
        v47a_enabled = (
            float(getattr(cfg, "v47a_resolution_weight", 0.0)) > 0.0
            or float(getattr(cfg, "v47a_usage_guard_weight", 0.0)) > 0.0
        )
        v47a_resolution_loss, v47a_usage_guard_loss, v47a_stats = v47a_posterior_guided_band_resolution_regularizer(
            out["q_refined"],
            self.edge_index,
            out["homo"],
            out["hetero"],
            out["hard"],
            agree_high_quantile=float(getattr(cfg, "v47a_agree_high_quantile", 0.70)),
            agree_low_quantile=float(getattr(cfg, "v47a_agree_low_quantile", 0.30)),
            uncert_high_quantile=float(getattr(cfg, "v47a_uncert_high_quantile", 0.70)),
            usage_entropy_floor=float(getattr(cfg, "v47a_usage_entropy_floor", 0.60)),
            eps=float(getattr(cfg, "v47a_eps", 1e-8)),
        )
        v48a_enabled = bool(getattr(cfg, "v48a_enabled", False))
        v48a_stats = self._v48a_topology_dynamics_audit(out)
        v49a_enabled = bool(getattr(cfg, "v49a_enabled", False))
        v49a_stats = self._v49a_topology_transition_diagnostics(out)
        v50a_enabled = bool(getattr(cfg, "v50a_enabled", False))
        v50a_anchor_loss, v50a_stats = spectral_anchor_alignment_loss(
            out["q_refined"],
            self.v50a_anchor_q,
            q_embed=out["q_embed"],
            enabled=v50a_enabled,
            effective_weight=float(getattr(cfg, "v50a_anchor_weight", 0.0)),
        )
        v51a_enabled = bool(getattr(cfg, "v51a_enabled", False))
        v51a_anchor_loss, v51a_stats = reliability_gated_spectral_anchor_loss(
            out["q_refined"],
            self.v50a_anchor_q,
            q_embed=out["q_embed"],
            edge_index=self.edge_index,
            enabled=v51a_enabled,
            effective_weight=float(getattr(cfg, "v51a_anchor_weight", 0.0)),
            reliability_floor=float(getattr(cfg, "v51a_reliability_floor", 0.10)),
            reliable_threshold=float(getattr(cfg, "v51a_reliable_threshold", 0.20)),
            min_effective_mass=float(getattr(cfg, "v51a_min_effective_mass", 0.10)),
        )
        v52a_enabled = bool(getattr(cfg, "v52a_enabled", False))
        v52a_anchor_loss, v52a_stats = curriculum_reliability_spectral_anchor_loss(
            out["q_refined"],
            self.v50a_anchor_q,
            q_embed=out["q_embed"],
            edge_index=self.edge_index,
            enabled=v52a_enabled,
            current_epoch=current_epoch,
            effective_weight=float(getattr(cfg, "v52a_anchor_weight", 0.0)),
            reliability_floor=float(getattr(cfg, "v52a_reliability_floor", 0.10)),
            reliable_threshold=float(getattr(cfg, "v52a_reliable_threshold", 0.20)),
            min_effective_mass=float(getattr(cfg, "v52a_min_effective_mass", 0.10)),
            warmup_epochs=int(getattr(cfg, "v52a_warmup_epochs", 20)),
            ramp_epochs=int(getattr(cfg, "v52a_ramp_epochs", 40)),
        )
        v53a_enabled = bool(getattr(cfg, "v53a_enabled", False))
        v53a_anchor_loss, v53a_stats = residual_curriculum_spectral_anchor_loss(
            out["q_refined"],
            self.v50a_anchor_q,
            q_embed=out["q_embed"],
            edge_index=self.edge_index,
            enabled=v53a_enabled,
            current_epoch=current_epoch,
            effective_weight=float(getattr(cfg, "v53a_anchor_weight", 0.0)),
            reliability_floor=float(getattr(cfg, "v53a_reliability_floor", 0.10)),
            reliable_threshold=float(getattr(cfg, "v53a_reliable_threshold", 0.20)),
            min_effective_mass=float(getattr(cfg, "v53a_min_effective_mass", 0.10)),
            warmup_epochs=int(getattr(cfg, "v53a_warmup_epochs", 20)),
            ramp_epochs=int(getattr(cfg, "v53a_ramp_epochs", 40)),
            residual_beta=float(getattr(cfg, "v53a_residual_beta", 0.50)),
        )
        v54a_enabled = bool(getattr(cfg, "v54a_enabled", False))
        v54a_anchor_loss, v54a_stats = consensus_bounded_residual_spectral_anchor_loss(
            out["q_refined"],
            self.v50a_anchor_q,
            q_embed=out["q_embed"],
            edge_index=self.edge_index,
            enabled=v54a_enabled,
            current_epoch=current_epoch,
            effective_weight=float(getattr(cfg, "v54a_anchor_weight", 0.0)),
            reliability_floor=float(getattr(cfg, "v54a_reliability_floor", 0.10)),
            reliable_threshold=float(getattr(cfg, "v54a_reliable_threshold", 0.20)),
            min_effective_mass=float(getattr(cfg, "v54a_min_effective_mass", 0.10)),
            warmup_epochs=int(getattr(cfg, "v54a_warmup_epochs", 20)),
            ramp_epochs=int(getattr(cfg, "v54a_ramp_epochs", 40)),
            beta_min=float(getattr(cfg, "v54a_beta_min", 0.35)),
            beta_max=float(getattr(cfg, "v54a_beta_max", 0.70)),
        )
        v55a_enabled = bool(getattr(cfg, "v55a_enabled", False))
        v55a_anchor_loss, v55a_stats = soft_consensus_bounded_residual_spectral_anchor_loss(
            out["q_refined"],
            self.v50a_anchor_q,
            q_embed=out["q_embed"],
            edge_index=self.edge_index,
            enabled=v55a_enabled,
            current_epoch=current_epoch,
            effective_weight=float(getattr(cfg, "v55a_anchor_weight", 0.0)),
            reliability_floor=float(getattr(cfg, "v55a_reliability_floor", 0.10)),
            reliable_threshold=float(getattr(cfg, "v55a_reliable_threshold", 0.20)),
            min_effective_mass=float(getattr(cfg, "v55a_min_effective_mass", 0.10)),
            warmup_epochs=int(getattr(cfg, "v55a_warmup_epochs", 20)),
            ramp_epochs=int(getattr(cfg, "v55a_ramp_epochs", 40)),
            beta_min=float(getattr(cfg, "v55a_beta_min", 0.35)),
            beta_max=float(getattr(cfg, "v55a_beta_max", 0.70)),
            soft_power=float(getattr(cfg, "v55a_soft_power", 0.50)),
        )
        v56a_enabled = bool(getattr(cfg, "v56a_enabled", False))
        v56a_anchor_loss, v56a_stats = hybrid_consensus_floor_residual_spectral_anchor_loss(
            out["q_refined"],
            self.v50a_anchor_q,
            q_embed=out["q_embed"],
            edge_index=self.edge_index,
            enabled=v56a_enabled,
            current_epoch=current_epoch,
            effective_weight=float(getattr(cfg, "v56a_anchor_weight", 0.0)),
            reliability_floor=float(getattr(cfg, "v56a_reliability_floor", 0.10)),
            reliable_threshold=float(getattr(cfg, "v56a_reliable_threshold", 0.20)),
            min_effective_mass=float(getattr(cfg, "v56a_min_effective_mass", 0.10)),
            warmup_epochs=int(getattr(cfg, "v56a_warmup_epochs", 20)),
            ramp_epochs=int(getattr(cfg, "v56a_ramp_epochs", 40)),
            beta_min=float(getattr(cfg, "v56a_beta_min", 0.35)),
            beta_max=float(getattr(cfg, "v56a_beta_max", 0.70)),
            soft_power=float(getattr(cfg, "v56a_soft_power", 0.50)),
            hybrid_compensation=float(getattr(cfg, "v56a_hybrid_compensation", 0.50)),
        )
        v57a_enabled = bool(getattr(cfg, "v57a_enabled", False))
        v57a_anchor_loss, v57a_stats = mass_floor_normalized_residual_spectral_anchor_loss(
            out["q_refined"],
            self.v50a_anchor_q,
            q_embed=out["q_embed"],
            edge_index=self.edge_index,
            enabled=v57a_enabled,
            current_epoch=current_epoch,
            effective_weight=float(getattr(cfg, "v57a_anchor_weight", 0.0)),
            reliability_floor=float(getattr(cfg, "v57a_reliability_floor", 0.10)),
            reliable_threshold=float(getattr(cfg, "v57a_reliable_threshold", 0.20)),
            min_effective_mass=float(getattr(cfg, "v57a_min_effective_mass", 0.10)),
            warmup_epochs=int(getattr(cfg, "v57a_warmup_epochs", 20)),
            ramp_epochs=int(getattr(cfg, "v57a_ramp_epochs", 40)),
            beta_min=float(getattr(cfg, "v57a_beta_min", 0.35)),
            beta_max=float(getattr(cfg, "v57a_beta_max", 0.70)),
            soft_power=float(getattr(cfg, "v57a_soft_power", 0.50)),
            hybrid_compensation=float(getattr(cfg, "v57a_hybrid_compensation", 0.50)),
            target_mass=float(getattr(cfg, "v57a_target_mass", 0.08)),
            max_mass_scale=float(getattr(cfg, "v57a_max_mass_scale", 1.50)),
            max_reliability_cap=float(getattr(cfg, "v57a_max_reliability_cap", 0.90)),
        )
        v58a_enabled = bool(getattr(cfg, "v58a_enabled", False))
        v58a_anchor_loss, v58a_stats = anchor_release_residual_spectral_anchor_loss(
            out["q_refined"],
            self.v50a_anchor_q,
            q_embed=out["q_embed"],
            edge_index=self.edge_index,
            enabled=v58a_enabled,
            current_epoch=current_epoch,
            effective_weight=float(getattr(cfg, "v58a_anchor_weight", 0.0)),
            reliability_floor=float(getattr(cfg, "v58a_reliability_floor", 0.10)),
            reliable_threshold=float(getattr(cfg, "v58a_reliable_threshold", 0.20)),
            min_effective_mass=float(getattr(cfg, "v58a_min_effective_mass", 0.10)),
            warmup_epochs=int(getattr(cfg, "v58a_warmup_epochs", 20)),
            ramp_epochs=int(getattr(cfg, "v58a_ramp_epochs", 40)),
            beta_min=float(getattr(cfg, "v58a_beta_min", 0.35)),
            beta_max=float(getattr(cfg, "v58a_beta_max", 0.70)),
            soft_power=float(getattr(cfg, "v58a_soft_power", 0.50)),
            hybrid_compensation=float(getattr(cfg, "v58a_hybrid_compensation", 0.50)),
            target_mass=float(getattr(cfg, "v58a_target_mass", 0.08)),
            max_mass_scale=float(getattr(cfg, "v58a_max_mass_scale", 1.50)),
            max_reliability_cap=float(getattr(cfg, "v58a_max_reliability_cap", 0.90)),
            release_warmup_epochs=int(getattr(cfg, "v58a_release_warmup_epochs", 20)),
            release_ramp_epochs=int(getattr(cfg, "v58a_release_ramp_epochs", 40)),
            release_hold_until_epoch=int(getattr(cfg, "v58a_release_hold_until_epoch", 80)),
            release_decay_epochs=int(getattr(cfg, "v58a_release_decay_epochs", 60)),
            release_floor=float(getattr(cfg, "v58a_release_floor", 0.25)),
        )
        v59a_enabled = bool(getattr(cfg, "v59a_enabled", False))
        v59a_anchor_loss, v59a_stats = post80_anchor_release_residual_spectral_anchor_loss(
            out["q_refined"],
            self.v50a_anchor_q,
            q_embed=out["q_embed"],
            edge_index=self.edge_index,
            enabled=v59a_enabled,
            current_epoch=current_epoch,
            effective_weight=float(getattr(cfg, "v59a_anchor_weight", 0.0)),
            reliability_floor=float(getattr(cfg, "v59a_reliability_floor", 0.10)),
            reliable_threshold=float(getattr(cfg, "v59a_reliable_threshold", 0.20)),
            min_effective_mass=float(getattr(cfg, "v59a_min_effective_mass", 0.10)),
            warmup_epochs=int(getattr(cfg, "v59a_warmup_epochs", 20)),
            ramp_epochs=int(getattr(cfg, "v59a_ramp_epochs", 40)),
            beta_min=float(getattr(cfg, "v59a_beta_min", 0.35)),
            beta_max=float(getattr(cfg, "v59a_beta_max", 0.70)),
            soft_power=float(getattr(cfg, "v59a_soft_power", 0.50)),
            hybrid_compensation=float(getattr(cfg, "v59a_hybrid_compensation", 0.50)),
            target_mass=float(getattr(cfg, "v59a_target_mass", 0.08)),
            max_mass_scale=float(getattr(cfg, "v59a_max_mass_scale", 1.50)),
            max_reliability_cap=float(getattr(cfg, "v59a_max_reliability_cap", 0.90)),
            release_start_epoch=int(getattr(cfg, "v59a_release_start_epoch", 80)),
            release_decay_epochs=int(getattr(cfg, "v59a_release_decay_epochs", 60)),
            release_floor=float(getattr(cfg, "v59a_release_floor", 0.25)),
        )
        v60a_enabled = bool(getattr(cfg, "v60a_enabled", False))
        v60a_anchor_loss, v60a_anchor_stats = v60a_post80_anchor_release_residual_spectral_anchor_loss(
            out["q_refined"],
            self.v50a_anchor_q,
            q_embed=out["q_embed"],
            edge_index=self.edge_index,
            enabled=v60a_enabled,
            current_epoch=current_epoch,
            effective_weight=float(getattr(cfg, "v60a_anchor_weight", 0.0)),
            reliability_floor=float(getattr(cfg, "v60a_reliability_floor", 0.10)),
            reliable_threshold=float(getattr(cfg, "v60a_reliable_threshold", 0.20)),
            min_effective_mass=float(getattr(cfg, "v60a_min_effective_mass", 0.10)),
            warmup_epochs=int(getattr(cfg, "v60a_warmup_epochs", 20)),
            ramp_epochs=int(getattr(cfg, "v60a_ramp_epochs", 40)),
            beta_min=float(getattr(cfg, "v60a_beta_min", 0.35)),
            beta_max=float(getattr(cfg, "v60a_beta_max", 0.70)),
            soft_power=float(getattr(cfg, "v60a_soft_power", 0.50)),
            hybrid_compensation=float(getattr(cfg, "v60a_hybrid_compensation", 0.50)),
            target_mass=float(getattr(cfg, "v60a_target_mass", 0.08)),
            max_mass_scale=float(getattr(cfg, "v60a_max_mass_scale", 1.50)),
            max_reliability_cap=float(getattr(cfg, "v60a_max_reliability_cap", 0.90)),
            release_start_epoch=int(getattr(cfg, "v60a_release_start_epoch", 80)),
            release_decay_epochs=int(getattr(cfg, "v60a_release_decay_epochs", 60)),
            release_floor=float(getattr(cfg, "v60a_release_floor", 0.25)),
        )
        v60a_guard_loss, v60a_guard_stats = v60a_self_distillation_guard_loss(
            out["q_refined"],
            self.v60a_teacher_q,
            enabled=v60a_enabled,
            teacher_ready=bool(self.v60a_teacher_ready.detach().cpu()),
            teacher_epoch=int(self.v60a_teacher_epoch.detach().cpu()),
            current_epoch=current_epoch,
            guard_weight=float(getattr(cfg, "v60a_guard_weight", 0.0)),
            confidence_threshold=float(getattr(cfg, "v60a_confidence_threshold", 0.60)),
            start_epoch=int(getattr(cfg, "v60a_start_epoch", 80)),
            ramp_epochs=int(getattr(cfg, "v60a_guard_ramp_epochs", 20)),
            max_gamma=float(getattr(cfg, "v60a_max_gamma", 1.0)),
        )
        v61a_enabled = bool(getattr(cfg, "v61a_enabled", False))
        v61a_anchor_loss, v61a_anchor_stats = v61a_post80_anchor_release_residual_spectral_anchor_loss(
            out["q_refined"],
            self.v50a_anchor_q,
            q_embed=out["q_embed"],
            edge_index=self.edge_index,
            enabled=v61a_enabled,
            current_epoch=current_epoch,
            effective_weight=float(getattr(cfg, "v61a_anchor_weight", 0.0)),
            reliability_floor=float(getattr(cfg, "v61a_reliability_floor", 0.10)),
            reliable_threshold=float(getattr(cfg, "v61a_reliable_threshold", 0.20)),
            min_effective_mass=float(getattr(cfg, "v61a_min_effective_mass", 0.10)),
            warmup_epochs=int(getattr(cfg, "v61a_warmup_epochs", 20)),
            ramp_epochs=int(getattr(cfg, "v61a_ramp_epochs", 40)),
            beta_min=float(getattr(cfg, "v61a_beta_min", 0.35)),
            beta_max=float(getattr(cfg, "v61a_beta_max", 0.70)),
            soft_power=float(getattr(cfg, "v61a_soft_power", 0.50)),
            hybrid_compensation=float(getattr(cfg, "v61a_hybrid_compensation", 0.50)),
            target_mass=float(getattr(cfg, "v61a_target_mass", 0.08)),
            max_mass_scale=float(getattr(cfg, "v61a_max_mass_scale", 1.50)),
            max_reliability_cap=float(getattr(cfg, "v61a_max_reliability_cap", 0.90)),
            release_start_epoch=int(getattr(cfg, "v61a_release_start_epoch", 80)),
            release_decay_epochs=int(getattr(cfg, "v61a_release_decay_epochs", 60)),
            release_floor=float(getattr(cfg, "v61a_release_floor", 0.25)),
        )
        v61a_guard_loss, v61a_guard_stats = v61a_quantile_coverage_self_distillation_guard_loss(
            out["q_refined"],
            self.v61a_teacher_q,
            enabled=v61a_enabled,
            teacher_ready=bool(self.v61a_teacher_ready.detach().cpu()),
            teacher_epoch=int(self.v61a_teacher_epoch.detach().cpu()),
            current_epoch=current_epoch,
            guard_weight=float(getattr(cfg, "v61a_guard_weight", 0.0)),
            absolute_floor=float(getattr(cfg, "v61a_absolute_floor", 0.45)),
            min_teacher_coverage=float(getattr(cfg, "v61a_min_teacher_coverage", 0.10)),
            start_epoch=int(getattr(cfg, "v61a_start_epoch", 80)),
            ramp_epochs=int(getattr(cfg, "v61a_guard_ramp_epochs", 20)),
            max_gamma=float(getattr(cfg, "v61a_max_gamma", 1.0)),
        )
        v62a_enabled = bool(getattr(cfg, "v62a_enabled", False))
        v62a_anchor_loss, v62a_anchor_stats = v62a_post80_anchor_release_residual_spectral_anchor_loss(
            out["q_refined"],
            self.v50a_anchor_q,
            q_embed=out["q_embed"],
            edge_index=self.edge_index,
            enabled=v62a_enabled,
            current_epoch=current_epoch,
            effective_weight=float(getattr(cfg, "v62a_anchor_weight", 0.0)),
            reliability_floor=float(getattr(cfg, "v62a_reliability_floor", 0.10)),
            reliable_threshold=float(getattr(cfg, "v62a_reliable_threshold", 0.20)),
            min_effective_mass=float(getattr(cfg, "v62a_min_effective_mass", 0.10)),
            warmup_epochs=int(getattr(cfg, "v62a_warmup_epochs", 20)),
            ramp_epochs=int(getattr(cfg, "v62a_ramp_epochs", 40)),
            beta_min=float(getattr(cfg, "v62a_beta_min", 0.35)),
            beta_max=float(getattr(cfg, "v62a_beta_max", 0.70)),
            soft_power=float(getattr(cfg, "v62a_soft_power", 0.50)),
            hybrid_compensation=float(getattr(cfg, "v62a_hybrid_compensation", 0.50)),
            target_mass=float(getattr(cfg, "v62a_target_mass", 0.08)),
            max_mass_scale=float(getattr(cfg, "v62a_max_mass_scale", 1.50)),
            max_reliability_cap=float(getattr(cfg, "v62a_max_reliability_cap", 0.90)),
            release_start_epoch=int(getattr(cfg, "v62a_release_start_epoch", 80)),
            release_decay_epochs=int(getattr(cfg, "v62a_release_decay_epochs", 60)),
            release_floor=float(getattr(cfg, "v62a_release_floor", 0.25)),
        )
        v62a_guard_loss, v62a_guard_stats = v62a_drift_responsive_self_distillation_guard_loss(
            out["q_refined"],
            self.v62a_teacher_q,
            enabled=v62a_enabled,
            teacher_ready=bool(self.v62a_teacher_ready.detach().cpu()),
            teacher_epoch=int(self.v62a_teacher_epoch.detach().cpu()),
            current_epoch=current_epoch,
            guard_weight=float(getattr(cfg, "v62a_guard_weight", 0.0)),
            absolute_floor=float(getattr(cfg, "v62a_absolute_floor", 0.45)),
            min_teacher_coverage=float(getattr(cfg, "v62a_min_teacher_coverage", 0.10)),
            start_epoch=int(getattr(cfg, "v62a_start_epoch", 80)),
            ramp_epochs=int(getattr(cfg, "v62a_guard_ramp_epochs", 20)),
            max_gamma=float(getattr(cfg, "v62a_max_gamma", 1.0)),
            drift_start_epoch=int(getattr(cfg, "v62a_drift_start_epoch", 100)),
            drift_floor=float(getattr(cfg, "v62a_drift_floor", 0.02)),
            drift_scale=float(getattr(cfg, "v62a_drift_scale", 0.06)),
            drift_boost=float(getattr(cfg, "v62a_drift_boost", 1.0)),
            max_effective_guard_multiplier=float(getattr(cfg, "v62a_max_effective_guard_multiplier", 2.0)),
        )
        v71a_enabled = bool(getattr(cfg, "v71a_anchor_bypass_enabled", False))
        v71a_anchor_bypass_loss, v71a_stats = v71a_hard_consensus_anchor_bypass_loss(
            out["q_refined"],
            self.v50a_anchor_q,
            out["q_embed"],
            self.edge_index,
            enabled=v71a_enabled,
            current_epoch=current_epoch,
            start_epoch=int(getattr(cfg, "v71a_anchor_bypass_start_epoch", 80)),
            release_gamma=v62a_anchor_stats["v62a_release_gamma"],
            soft_center=float(getattr(cfg, "v71a_anchor_bypass_soft_center", 0.55)),
            soft_scale=float(getattr(cfg, "v71a_anchor_bypass_soft_scale", 0.08)),
            min_mass=float(getattr(cfg, "v71a_anchor_bypass_min_mass", 0.01)),
        )
        if bool(v71a_enabled) and int(current_epoch) + 1 >= int(getattr(cfg, "v71a_anchor_bypass_start_epoch", 80)):
            mix = float(np.clip(getattr(cfg, "v71a_anchor_bypass_mix", 1.0), 0.0, 1.0))
            v62a_anchor_loss = (1.0 - mix) * v62a_anchor_loss + mix * v71a_anchor_bypass_loss
        v67a_enabled = bool(getattr(cfg, "v67a_anchor_distrust_enabled", False))
        v67a_anchor_gate = out["q_refined"].new_tensor(1.0)
        anchor_agreement_for_gate = v62a_anchor_stats["v62a_weighted_q_anchor_agreement"].detach()
        epoch_number = int(current_epoch) + 1
        if v67a_enabled and epoch_number >= int(getattr(cfg, "v67a_anchor_distrust_start_epoch", 100)):
            center = out["q_refined"].new_tensor(float(getattr(cfg, "v67a_anchor_agreement_center", 0.75)))
            scale = out["q_refined"].new_tensor(max(1e-4, float(getattr(cfg, "v67a_anchor_agreement_scale", 0.05))))
            floor = out["q_refined"].new_tensor(float(np.clip(getattr(cfg, "v67a_anchor_distrust_floor", 0.10), 0.0, 1.0)))
            v67a_anchor_gate = floor + (1.0 - floor) * torch.sigmoid((anchor_agreement_for_gate - center) / scale)
            if bool(getattr(cfg, "v90a_anchor_distrust_graph_gate_enabled", False)):
                graph_center = out["q_refined"].new_tensor(
                    float(getattr(cfg, "v90a_anchor_distrust_graph_noise_center", 0.20))
                )
                graph_scale = out["q_refined"].new_tensor(
                    max(1e-4, float(getattr(cfg, "v90a_anchor_distrust_graph_noise_scale", 0.02)))
                )
                graph_distrust = torch.sigmoid((out["v63b_graph_noise"].detach() - graph_center) / graph_scale)
                v67a_anchor_gate = (1.0 - graph_distrust) + graph_distrust * v67a_anchor_gate
            v62a_anchor_loss = v62a_anchor_loss * v67a_anchor_gate
        v68a_enabled = bool(getattr(cfg, "v68a_low_agreement_teacher_boost_enabled", False))
        v68a_teacher_boost = out["q_refined"].new_tensor(1.0)
        if v68a_enabled and epoch_number >= int(getattr(cfg, "v68a_teacher_boost_start_epoch", 80)):
            center = out["q_refined"].new_tensor(float(getattr(cfg, "v68a_teacher_boost_center", 0.75)))
            scale = out["q_refined"].new_tensor(max(1e-4, float(getattr(cfg, "v68a_teacher_boost_scale", 0.05))))
            max_boost = out["q_refined"].new_tensor(max(1.0, float(getattr(cfg, "v68a_teacher_boost_max", 3.0))))
            distrust = 1.0 - torch.sigmoid((anchor_agreement_for_gate - center) / scale)
            v68a_teacher_boost = 1.0 + (max_boost - 1.0) * distrust
            v62a_guard_loss = v62a_guard_loss * v68a_teacher_boost
        v70a_enabled = bool(getattr(cfg, "v70a_low_agreement_entropy_guard_enabled", False))
        v70a_entropy_guard_loss = out["q_refined"].new_tensor(0.0)
        v70a_entropy_guard_gate = out["q_refined"].new_tensor(0.0)
        v70a_normalized_entropy = out["q_refined"].new_tensor(0.0)
        if v70a_enabled and epoch_number >= int(getattr(cfg, "v70a_entropy_guard_start_epoch", 80)):
            q_guard = out["q_refined"].clamp_min(1e-8)
            q_guard = q_guard / q_guard.sum(dim=1, keepdim=True).clamp_min(1e-8)
            k_guard = max(2, int(q_guard.shape[1]))
            entropy_raw = -torch.sum(q_guard * q_guard.clamp_min(1e-8).log(), dim=1).mean()
            v70a_normalized_entropy = entropy_raw / math.log(float(k_guard))
            center = out["q_refined"].new_tensor(float(getattr(cfg, "v70a_entropy_guard_agreement_center", 0.75)))
            scale = out["q_refined"].new_tensor(max(1e-4, float(getattr(cfg, "v70a_entropy_guard_agreement_scale", 0.05))))
            floor = out["q_refined"].new_tensor(float(np.clip(getattr(cfg, "v70a_entropy_guard_floor", 0.35), 0.0, 1.0)))
            v70a_entropy_guard_gate = 1.0 - torch.sigmoid((anchor_agreement_for_gate - center) / scale)
            v70a_entropy_guard_loss = v70a_entropy_guard_gate * F.relu(floor - v70a_normalized_entropy).pow(2)
        v63b_enabled = bool(getattr(cfg, "v63b_enabled", False))
        v63b_confusion_guard_loss, v63b_guard_stats = v63b_confusion_aware_self_distillation_guard_loss(
            out["q_refined"],
            self.v62a_teacher_q,
            self.edge_index,
            enabled=v63b_enabled,
            teacher_ready=bool(self.v62a_teacher_ready.detach().cpu()),
            teacher_epoch=int(self.v62a_teacher_epoch.detach().cpu()),
            current_epoch=current_epoch,
            guard_weight=float(getattr(cfg, "v63b_confusion_guard_weight", 0.0)),
            absolute_floor=float(getattr(cfg, "v62a_absolute_floor", 0.45)),
            min_teacher_coverage=float(getattr(cfg, "v62a_min_teacher_coverage", 0.10)),
            start_epoch=int(getattr(cfg, "v62a_start_epoch", 80)),
            ramp_epochs=int(getattr(cfg, "v62a_guard_ramp_epochs", 20)),
            max_gamma=float(getattr(cfg, "v62a_max_gamma", 1.0)),
            guard_floor=float(getattr(cfg, "v63b_guard_floor", 0.25)),
            guard_power=float(getattr(cfg, "v63b_guard_power", 1.0)),
            min_neighbor_count=float(getattr(cfg, "v63b_guard_min_neighbor_count", 1.0)),
        )
        v64a_enabled = bool(getattr(cfg, "v64a_enabled", False))
        v64a_subspace_gram_loss, v64a_stats = v64a_spectral_subspace_gram_alignment_loss(
            out["embedding"],
            self.v64a_subspace_z,
            enabled=v64a_enabled,
            current_epoch=current_epoch,
            start_epoch=int(getattr(cfg, "v64a_start_epoch", 0)),
            ramp_epochs=int(getattr(cfg, "v64a_ramp_epochs", 20)),
            release_start_epoch=int(getattr(cfg, "v64a_release_start_epoch", 0)),
            release_decay_epochs=int(getattr(cfg, "v64a_release_decay_epochs", 0)),
            release_floor=float(getattr(cfg, "v64a_release_floor", 1.0)),
            max_nodes=int(getattr(cfg, "v64a_gram_max_nodes", 1536)),
        )
        v86a_v64_gate = out["q_refined"].new_tensor(1.0)
        if bool(getattr(cfg, "v86a_v64_low_agreement_gate_enabled", False)):
            center = out["q_refined"].new_tensor(float(getattr(cfg, "v86a_v64_gate_center", 0.75)))
            scale = out["q_refined"].new_tensor(max(1e-4, float(getattr(cfg, "v86a_v64_gate_scale", 0.05))))
            floor = out["q_refined"].new_tensor(float(np.clip(getattr(cfg, "v86a_v64_gate_floor", 0.0), 0.0, 1.0)))
            low_agreement = 1.0 - torch.sigmoid((anchor_agreement_for_gate - center) / scale)
            v86a_v64_gate = (floor + (1.0 - floor) * low_agreement).clamp(0.0, 1.0)
            v64a_subspace_gram_loss = v64a_subspace_gram_loss * v86a_v64_gate
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
        v63b_edge_ood_loss, v63b_edge_stats = v63b_edge_ood_ranking_loss(
            out["score"],
            out["edge_logit"],
            out["evidences"],
            self.edge_prior,
            enabled=v63b_enabled,
            margin=float(getattr(cfg, "v63b_edge_margin", 0.12)),
            pos_quantile=float(getattr(cfg, "v63b_edge_pos_quantile", 0.80)),
            neg_quantile=float(getattr(cfg, "v63b_edge_neg_quantile", 0.20)),
            max_pairs=int(getattr(cfg, "v63b_edge_max_pairs", 8192)),
            concordance_power=float(getattr(cfg, "v63b_concordance_power", 1.0)),
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
            + cfg.ideal_signed_embedding_weight * ideal_signed_embedding_loss
            + cfg.ideal_band_resolution_weight * ideal_band_resolution_loss
            + cfg.ideal_highpass_energy_weight * ideal_highpass_energy_loss
            + cfg.v43b_conflict_margin_weight * v43b_conflict_margin_loss
            + cfg.v43b_band_conflict_weight * v43b_stats["v43b_band_conflict_loss"]
            + cfg.v43b_highpass_energy_weight * ideal_highpass_energy_loss
            + cfg.v44_topology_band_resolution_weight * v44_band_loss
            + cfg.v44_conflict_highpass_corr_weight * v44_highpass_loss
            + cfg.v44b_pre_hp_corr_weight * v44b_pre_hp_loss
            + cfg.v45a_edge_freq_weight * v45a_edge_freq_loss
            + cfg.v45a_band_guard_weight * v45a_band_guard_loss
            + cfg.v46a_band_cal_weight * v46a_band_cal_loss
            + cfg.v46a_balance_weight * v46a_balance_loss
            + cfg.v46a_spread_weight * v46a_spread_loss
            + cfg.v47a_resolution_weight * v47a_resolution_loss
            + cfg.v47a_usage_guard_weight * v47a_usage_guard_loss
            + cfg.v50a_anchor_weight * v50a_anchor_loss
            + cfg.v51a_anchor_weight * v51a_anchor_loss
            + cfg.v52a_anchor_weight * v52a_anchor_loss
            + cfg.v53a_anchor_weight * v53a_anchor_loss
            + cfg.v54a_anchor_weight * v54a_anchor_loss
            + cfg.v55a_anchor_weight * v55a_anchor_loss
            + cfg.v56a_anchor_weight * v56a_anchor_loss
            + cfg.v57a_anchor_weight * v57a_anchor_loss
            + cfg.v58a_anchor_weight * v58a_anchor_loss
            + cfg.v59a_anchor_weight * v59a_anchor_loss
            + cfg.v60a_anchor_weight * v60a_anchor_loss
            + cfg.v60a_guard_weight * v60a_guard_loss
            + cfg.v61a_anchor_weight * v61a_anchor_loss
            + cfg.v61a_guard_weight * v61a_guard_loss
            + cfg.v62a_anchor_weight * v62a_anchor_loss
            + cfg.v62a_guard_weight * v62a_guard_loss
            + cfg.v63b_confusion_guard_weight * v63b_confusion_guard_loss
            + cfg.v64a_subspace_gram_weight * v64a_subspace_gram_loss
            + cfg.v70a_entropy_guard_weight * v70a_entropy_guard_loss
            + cfg.aptc_prior_entropy_weight * prior_entropy_loss
            + cfg.calib_alpha_weight * calib_alpha_loss
            + cfg.calib_mask_weight * calib_mask_loss
            + cfg.calib_struct_attr_weight * calib_struct_attr_loss
            + cfg.edge_rank_weight * edge_rank_loss
            + cfg.v63b_edge_ood_weight * v63b_edge_ood_loss
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
            "local_teacher_enabled": bool(local_teacher_enabled),
            "local_teacher_beta_mean": float(local_teacher_stats["local_teacher_beta"].mean().detach().cpu()),
            "local_teacher_beta_std": float(local_teacher_stats["local_teacher_beta"].std(unbiased=False).detach().cpu()),
            "local_teacher_kl_to_t0": float(local_teacher_stats["local_teacher_kl_to_t0"].detach().cpu()),
            "local_teacher_entropy_t0": float(local_teacher_stats["local_teacher_entropy_t0"].detach().cpu()),
            "local_teacher_entropy_local": float(local_teacher_stats["local_teacher_entropy_local"].detach().cpu()),
            "local_teacher_entropy_final": float(local_teacher_stats["local_teacher_entropy_final"].detach().cpu()),
            "local_teacher_pos_agree_t0": float(local_teacher_stats["local_teacher_pos_agree_t0"].detach().cpu()),
            "local_teacher_pos_agree_final": float(local_teacher_stats["local_teacher_pos_agree_final"].detach().cpu()),
            "local_teacher_hard_agree_final": float(local_teacher_stats["local_teacher_hard_agree_final"].detach().cpu()),
            "local_teacher_neg_overlap_t0": float(local_teacher_stats["local_teacher_neg_overlap_t0"].detach().cpu()),
            "local_teacher_neg_overlap_final": float(local_teacher_stats["local_teacher_neg_overlap_final"].detach().cpu()),
            "local_teacher_pos_gain": float(local_teacher_stats["local_teacher_pos_gain"].detach().cpu()),
            "local_teacher_neg_reduction": float(local_teacher_stats["local_teacher_neg_reduction"].detach().cpu()),
            "local_teacher_qreg_kl": float(local_teacher_qreg_kl.detach().cpu()),
            "local_teacher_qrefined_kl": float(local_teacher_qrefined_kl.detach().cpu()),
            "local_teacher_qmix_kl": float(local_teacher_qmix_kl.detach().cpu()),
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
            "ideal_enabled": bool(ideal_enabled),
            "ideal_signed_embedding_loss": float(ideal_signed_embedding_loss.detach().cpu()),
            "ideal_band_resolution_loss": float(ideal_band_resolution_loss.detach().cpu()),
            "ideal_highpass_energy_loss": float(ideal_highpass_energy_loss.detach().cpu()),
            "ideal_homo_sim_mean": float(ideal_signed_stats["ideal_homo_sim_mean"].detach().cpu()),
            "ideal_hetero_pos_overlap": float(ideal_signed_stats["ideal_hetero_pos_overlap"].detach().cpu()),
            "ideal_hard_pos_overlap": float(ideal_signed_stats["ideal_hard_pos_overlap"].detach().cpu()),
            "ideal_band_mass": float(ideal_band_stats["ideal_band_mass"].detach().cpu()),
            "ideal_clear_mass": float(ideal_band_stats["ideal_clear_mass"].detach().cpu()),
            "ideal_highpass_energy_mean": float(ideal_highpass_stats["ideal_highpass_energy_mean"].detach().cpu()),
            "ideal_conflict_mean": float(ideal_highpass_stats["ideal_conflict_mean"].detach().cpu()),
            "ideal_conflict_energy_corr": float(ideal_highpass_stats["ideal_conflict_energy_corr"].detach().cpu()),
            "v43b_enabled": bool(v43b_enabled),
            "v43b_conflict_gate_mean": float(v43b_stats["v43b_conflict_gate_mean"].detach().cpu()),
            "v43b_conflict_gate_std": float(v43b_stats["v43b_conflict_gate_std"].detach().cpu()),
            "v43b_conflict_gate_active_ratio": float(v43b_stats["v43b_conflict_gate_active_ratio"].detach().cpu()),
            "v43b_conflict_gate_p90": float(v43b_stats["v43b_conflict_gate_p90"].detach().cpu()),
            "v43b_structural_conflict_mean": float(v43b_stats["v43b_structural_conflict_mean"].detach().cpu()),
            "v43b_clarity_mean": float(v43b_stats["v43b_clarity_mean"].detach().cpu()),
            "v43b_view_disagreement_mean": float(v43b_stats["v43b_view_disagreement_mean"].detach().cpu()),
            "v43b_uncertainty_gate_mean": float(v43b_stats["v43b_uncertainty_gate_mean"].detach().cpu()),
            "v43b_conflict_margin_loss": float(v43b_stats["v43b_conflict_margin_loss"].detach().cpu()),
            "v43b_conflict_margin_violation_mean": float(v43b_stats["v43b_conflict_margin_violation_mean"].detach().cpu()),
            "v43b_conflict_margin_violation_ratio": float(v43b_stats["v43b_conflict_margin_violation_ratio"].detach().cpu()),
            "v43b_high_conflict_overlap": float(v43b_stats["v43b_high_conflict_overlap"].detach().cpu()),
            "v43b_low_conflict_overlap": float(v43b_stats["v43b_low_conflict_overlap"].detach().cpu()),
            "v43b_overlap_gap": float(v43b_stats["v43b_overlap_gap"].detach().cpu()),
            "v43b_band_conflict_loss": float(v43b_stats["v43b_band_conflict_loss"].detach().cpu()),
            "v43b_band_mass": float(v43b_stats["v43b_band_mass"].detach().cpu()),
            "v43b_conflict_band_mass": float(v43b_stats["v43b_conflict_band_mass"].detach().cpu()),
            "v43b_clear_mass": float(v43b_stats["v43b_clear_mass"].detach().cpu()),
            "v43b_low_high_disagreement_mean": float(v43b_stats["v43b_low_high_disagreement_mean"].detach().cpu()),
            "v43b_highpass_energy_mean": float(v43b_stats["v43b_highpass_energy_mean"].detach().cpu()),
            "v43b_conflict_energy_corr": float(v43b_stats["v43b_conflict_energy_corr"].detach().cpu()),
            "v44_enabled": bool(v44_enabled),
            "v44_band_loss": float(v44_band_stats["v44_band_loss"].detach().cpu()),
            "v44_band_mass": float(v44_band_stats["v44_band_mass"].detach().cpu()),
            "v44_hard_ratio": float(v44_band_stats["v44_hard_ratio"].detach().cpu()),
            "v44_ambiguous_ratio": float(v44_band_stats["v44_ambiguous_ratio"].detach().cpu()),
            "v44_clear_mass": float(v44_band_stats["v44_clear_mass"].detach().cpu()),
            "v44_decisive_mass": float(v44_band_stats["v44_decisive_mass"].detach().cpu()),
            "v44_score_uncertainty_mean": float(v44_band_stats["v44_score_uncertainty_mean"].detach().cpu()),
            "v44_score_uncertainty_p90": float(v44_band_stats["v44_score_uncertainty_p90"].detach().cpu()),
            "v44_threshold_gap": float(v44_band_stats["v44_threshold_gap"].detach().cpu()),
            "v44_low_threshold": float(v44_band_stats["v44_low_threshold"].detach().cpu()),
            "v44_high_threshold": float(v44_band_stats["v44_high_threshold"].detach().cpu()),
            "v44_conflict_weight_mean": float(v44_band_stats["v44_conflict_weight_mean"].detach().cpu()),
            "v44_topology_conflict_mean": float(v44_band_stats["v44_topology_conflict_mean"].detach().cpu()),
            "v44_highpass_loss": float(v44_highpass_stats["v44_highpass_loss"].detach().cpu()),
            "v44_conflict_energy_corr": float(v44_highpass_stats["v44_conflict_energy_corr"].detach().cpu()),
            "v44_highpass_energy_mean": float(v44_highpass_stats["v44_highpass_energy_mean"].detach().cpu()),
            "v44_highpass_energy_std": float(v44_highpass_stats["v44_highpass_energy_std"].detach().cpu()),
            "v44_node_conflict_mean": float(v44_highpass_stats["v44_node_conflict_mean"].detach().cpu()),
            "v44_node_conflict_std": float(v44_highpass_stats["v44_node_conflict_std"].detach().cpu()),
            "v44_high_conflict_energy": float(v44_highpass_stats["v44_high_conflict_energy"].detach().cpu()),
            "v44_low_conflict_energy": float(v44_highpass_stats["v44_low_conflict_energy"].detach().cpu()),
            "v44_energy_gap": float(v44_highpass_stats["v44_energy_gap"].detach().cpu()),
            "v44b_enabled": bool(v44b_enabled),
            "v44b_pre_hp_loss": float(v44b_stats["v44b_pre_hp_loss"].detach().cpu()),
            "v44b_pre_hp_response_mean": float(v44b_stats["v44b_pre_hp_response_mean"].detach().cpu()),
            "v44b_pre_hp_response_std": float(v44b_stats["v44b_pre_hp_response_std"].detach().cpu()),
            "v44b_pre_hp_response_p10": float(v44b_stats["v44b_pre_hp_response_p10"].detach().cpu()),
            "v44b_pre_hp_response_p90": float(v44b_stats["v44b_pre_hp_response_p90"].detach().cpu()),
            "v44b_pre_hp_response_ratio_p90_p10": float(v44b_stats["v44b_pre_hp_response_ratio_p90_p10"].detach().cpu()),
            "v44b_conflict_response_corr": float(v44b_stats["v44b_conflict_response_corr"].detach().cpu()),
            "v44b_high_conflict_response": float(v44b_stats["v44b_high_conflict_response"].detach().cpu()),
            "v44b_low_conflict_response": float(v44b_stats["v44b_low_conflict_response"].detach().cpu()),
            "v44b_response_gap": float(v44b_stats["v44b_response_gap"].detach().cpu()),
            "v44b_node_conflict_mean": float(v44b_stats["v44b_node_conflict_mean"].detach().cpu()),
            "v44b_node_conflict_std": float(v44b_stats["v44b_node_conflict_std"].detach().cpu()),
            "v44b_postnorm_hp_energy_mean": float(v44b_stats["v44b_postnorm_hp_energy_mean"].detach().cpu()),
            "v44b_postnorm_hp_energy_std": float(v44b_stats["v44b_postnorm_hp_energy_std"].detach().cpu()),
            "v44b_postnorm_energy_gap": float(v44b_stats["v44b_postnorm_energy_gap"].detach().cpu()),
            "v45a_enabled": bool(v45a_enabled),
            "v45a_losses_active": bool(v45a_losses_active),
            "v45a_band_mass": float(v45a_stats["v45a_band_mass"].detach().cpu()),
            "v45a_band_reference": float(v45a_stats["v45a_band_reference"].detach().cpu()),
            "v45a_band_reference_ready": bool(self.v45a_band_reference_ready.detach().cpu()),
            "v45a_warmup_epoch_count": int(self.v45a_warmup_epoch_count.detach().cpu()),
            "v45a_band_guard_loss": float(v45a_stats["v45a_band_guard_loss"].detach().cpu()),
            "v45a_safe_band_gate": float(v45a_stats["v45a_safe_band_gate"].detach().cpu()),
            "v45a_edge_freq_loss": float(v45a_stats["v45a_edge_freq_loss"].detach().cpu()),
            "v45a_edge_freq_loss_raw": float(v45a_stats["v45a_edge_freq_loss_raw"].detach().cpu()),
            "v45a_boundary_response_mean": float(v45a_stats["v45a_boundary_response_mean"].detach().cpu()),
            "v45a_safe_homo_response_mean": float(v45a_stats["v45a_safe_homo_response_mean"].detach().cpu()),
            "v45a_edge_response_gap": float(v45a_stats["v45a_edge_response_gap"].detach().cpu()),
            "v45a_edge_response_corr": float(v45a_stats["v45a_edge_response_corr"].detach().cpu()),
            "v45a_boundary_mass": float(v45a_stats["v45a_boundary_mass"].detach().cpu()),
            "v45a_safe_homo_mass": float(v45a_stats["v45a_safe_homo_mass"].detach().cpu()),
            "v46a_enabled": bool(v46a_enabled),
            "v46a_band_cal_loss": float(v46a_stats["v46a_band_cal_loss"].detach().cpu()),
            "v46a_balance_loss": float(v46a_stats["v46a_balance_loss"].detach().cpu()),
            "v46a_spread_loss": float(v46a_stats["v46a_spread_loss"].detach().cpu()),
            "v46a_band_mass": float(v46a_stats["v46a_band_mass"].detach().cpu()),
            "v46a_homo_usage": float(v46a_stats["v46a_homo_usage"].detach().cpu()),
            "v46a_hetero_usage": float(v46a_stats["v46a_hetero_usage"].detach().cpu()),
            "v46a_hard_usage": float(v46a_stats["v46a_hard_usage"].detach().cpu()),
            "v46a_usage_entropy": float(v46a_stats["v46a_usage_entropy"].detach().cpu()),
            "v46a_threshold_gap": float(v46a_stats["v46a_threshold_gap"].detach().cpu()),
            "v46a_low_threshold": float(v46a_stats["v46a_low_threshold"].detach().cpu()),
            "v46a_high_threshold": float(v46a_stats["v46a_high_threshold"].detach().cpu()),
            "v47a_enabled": bool(v47a_enabled),
            "v47a_resolution_loss": float(v47a_stats["v47a_resolution_loss"].detach().cpu()),
            "v47a_usage_guard_loss": float(v47a_stats["v47a_usage_guard_loss"].detach().cpu()),
            "v47a_posterior_agreement_mean": float(v47a_stats["v47a_posterior_agreement_mean"].detach().cpu()),
            "v47a_posterior_agreement_std": float(v47a_stats["v47a_posterior_agreement_std"].detach().cpu()),
            "v47a_posterior_uncertainty_mean": float(v47a_stats["v47a_posterior_uncertainty_mean"].detach().cpu()),
            "v47a_agree_high_threshold": float(v47a_stats["v47a_agree_high_threshold"].detach().cpu()),
            "v47a_agree_low_threshold": float(v47a_stats["v47a_agree_low_threshold"].detach().cpu()),
            "v47a_uncert_high_threshold": float(v47a_stats["v47a_uncert_high_threshold"].detach().cpu()),
            "v47a_homo_target_mass": float(v47a_stats["v47a_homo_target_mass"].detach().cpu()),
            "v47a_hetero_target_mass": float(v47a_stats["v47a_hetero_target_mass"].detach().cpu()),
            "v47a_defer_target_mass": float(v47a_stats["v47a_defer_target_mass"].detach().cpu()),
            "v47a_unassigned_target_mass": float(v47a_stats["v47a_unassigned_target_mass"].detach().cpu()),
            "v47a_raw_homo_target_mass": float(v47a_stats["v47a_raw_homo_target_mass"].detach().cpu()),
            "v47a_raw_hetero_target_mass": float(v47a_stats["v47a_raw_hetero_target_mass"].detach().cpu()),
            "v47a_raw_defer_target_mass": float(v47a_stats["v47a_raw_defer_target_mass"].detach().cpu()),
            "v47a_raw_unassigned_target_mass": float(v47a_stats["v47a_raw_unassigned_target_mass"].detach().cpu()),
            "v47a_effective_target_mass": float(v47a_stats["v47a_effective_target_mass"].detach().cpu()),
            "v47a_band_mass": float(v47a_stats["v47a_band_mass"].detach().cpu()),
            "v47a_homo_usage": float(v47a_stats["v47a_homo_usage"].detach().cpu()),
            "v47a_hetero_usage": float(v47a_stats["v47a_hetero_usage"].detach().cpu()),
            "v47a_hard_usage": float(v47a_stats["v47a_hard_usage"].detach().cpu()),
            "v47a_usage_entropy": float(v47a_stats["v47a_usage_entropy"].detach().cpu()),
            "v48a_enabled": bool(v48a_enabled),
            "v48a_has_prev_snapshot": bool(v48a_stats["v48a_has_prev_snapshot"].detach().cpu()),
            "v48a_sample_size": int(v48a_stats["v48a_sample_size"].detach().cpu()),
            "v48a_mean_abs_delta_homo": float(v48a_stats["v48a_mean_abs_delta_homo"].detach().cpu()),
            "v48a_mean_abs_delta_hetero": float(v48a_stats["v48a_mean_abs_delta_hetero"].detach().cpu()),
            "v48a_mean_abs_delta_hard": float(v48a_stats["v48a_mean_abs_delta_hard"].detach().cpu()),
            "v48a_mean_abs_delta_score": float(v48a_stats["v48a_mean_abs_delta_score"].detach().cpu()),
            "v48a_hard_mass_delta": float(v48a_stats["v48a_hard_mass_delta"].detach().cpu()),
            "v48a_threshold_delta": float(v48a_stats["v48a_threshold_delta"].detach().cpu()),
            "v48a_hard_rank_corr_prev": float(v48a_stats["v48a_hard_rank_corr_prev"].detach().cpu()),
            "v48a_homo_target_mass": float(v48a_stats["v48a_homo_target_mass"].detach().cpu()),
            "v48a_hetero_target_mass": float(v48a_stats["v48a_hetero_target_mass"].detach().cpu()),
            "v48a_defer_target_mass": float(v48a_stats["v48a_defer_target_mass"].detach().cpu()),
            "v48a_raw_homo_target_mass": float(v48a_stats["v48a_raw_homo_target_mass"].detach().cpu()),
            "v48a_raw_hetero_target_mass": float(v48a_stats["v48a_raw_hetero_target_mass"].detach().cpu()),
            "v48a_raw_defer_target_mass": float(v48a_stats["v48a_raw_defer_target_mass"].detach().cpu()),
            "v48a_targeted_homo_delta": float(v48a_stats["v48a_targeted_homo_delta"].detach().cpu()),
            "v48a_targeted_hetero_delta": float(v48a_stats["v48a_targeted_hetero_delta"].detach().cpu()),
            "v48a_targeted_hard_delta": float(v48a_stats["v48a_targeted_hard_delta"].detach().cpu()),
            "v49a_enabled": bool(v49a_enabled),
            "v49a_homo_usage": float(v49a_stats["v49a_homo_usage"].detach().cpu()),
            "v49a_hetero_usage": float(v49a_stats["v49a_hetero_usage"].detach().cpu()),
            "v49a_hard_usage": float(v49a_stats["v49a_hard_usage"].detach().cpu()),
            "v49a_band_mass": float(v49a_stats["v49a_band_mass"].detach().cpu()),
            "v49a_usage_entropy": float(v49a_stats["v49a_usage_entropy"].detach().cpu()),
            "v49a_clear_mean": float(v49a_stats["v49a_clear_mean"].detach().cpu()),
            "v49a_clear_std": float(v49a_stats["v49a_clear_std"].detach().cpu()),
            "v49a_orient_mean": float(v49a_stats["v49a_orient_mean"].detach().cpu()),
            "v49a_orient_std": float(v49a_stats["v49a_orient_std"].detach().cpu()),
            "v49a_has_prev_snapshot": bool(v49a_stats["v49a_has_prev_snapshot"].detach().cpu()),
            "v49a_sample_size": int(v49a_stats["v49a_sample_size"].detach().cpu()),
            "v49a_mean_abs_delta_homo": float(v49a_stats["v49a_mean_abs_delta_homo"].detach().cpu()),
            "v49a_mean_abs_delta_hetero": float(v49a_stats["v49a_mean_abs_delta_hetero"].detach().cpu()),
            "v49a_mean_abs_delta_hard": float(v49a_stats["v49a_mean_abs_delta_hard"].detach().cpu()),
            "v49a_mean_abs_delta_score": float(v49a_stats["v49a_mean_abs_delta_score"].detach().cpu()),
            "v49a_hard_mass_delta": float(v49a_stats["v49a_hard_mass_delta"].detach().cpu()),
            "v49a_hard_rank_corr_prev": float(v49a_stats["v49a_hard_rank_corr_prev"].detach().cpu()),
            "v49a_homo_target_mass": float(v49a_stats["v49a_homo_target_mass"].detach().cpu()),
            "v49a_hetero_target_mass": float(v49a_stats["v49a_hetero_target_mass"].detach().cpu()),
            "v49a_defer_target_mass": float(v49a_stats["v49a_defer_target_mass"].detach().cpu()),
            "v49a_raw_homo_target_mass": float(v49a_stats["v49a_raw_homo_target_mass"].detach().cpu()),
            "v49a_raw_hetero_target_mass": float(v49a_stats["v49a_raw_hetero_target_mass"].detach().cpu()),
            "v49a_raw_defer_target_mass": float(v49a_stats["v49a_raw_defer_target_mass"].detach().cpu()),
            "v49a_targeted_homo_delta": float(v49a_stats["v49a_targeted_homo_delta"].detach().cpu()),
            "v49a_targeted_hetero_delta": float(v49a_stats["v49a_targeted_hetero_delta"].detach().cpu()),
            "v49a_targeted_hard_delta": float(v49a_stats["v49a_targeted_hard_delta"].detach().cpu()),
            "v50a_enabled": bool(v50a_enabled),
            "v50a_anchor_loss": float(v50a_stats["v50a_anchor_loss"].detach().cpu()),
            "v50a_q_anchor_kl": float(v50a_stats["v50a_q_anchor_kl"].detach().cpu()),
            "v50a_q_anchor_agreement": float(v50a_stats["v50a_q_anchor_agreement"].detach().cpu()),
            "v50a_embedding_anchor_agreement": float(v50a_stats["v50a_embedding_anchor_agreement"].detach().cpu()),
            "v50a_anchor_entropy": float(v50a_stats["v50a_anchor_entropy"].detach().cpu()),
            "v50a_anchor_confidence": float(v50a_stats["v50a_anchor_confidence"].detach().cpu()),
            "v50a_anchor_cluster_usage_entropy": float(v50a_stats["v50a_anchor_cluster_usage_entropy"].detach().cpu()),
            "v50a_anchor_effective_weight": float(v50a_stats["v50a_anchor_effective_weight"].detach().cpu()),
            "v51a_enabled": bool(v51a_enabled),
            "v51a_anchor_loss": float(v51a_stats["v51a_anchor_loss"].detach().cpu()),
            "v51a_weighted_q_anchor_kl": float(v51a_stats["v51a_weighted_q_anchor_kl"].detach().cpu()),
            "v51a_weighted_q_anchor_agreement": float(v51a_stats["v51a_weighted_q_anchor_agreement"].detach().cpu()),
            "v51a_unweighted_q_anchor_agreement": float(v51a_stats["v51a_unweighted_q_anchor_agreement"].detach().cpu()),
            "v51a_embedding_anchor_agreement": float(v51a_stats["v51a_embedding_anchor_agreement"].detach().cpu()),
            "v51a_anchor_entropy": float(v51a_stats["v51a_anchor_entropy"].detach().cpu()),
            "v51a_anchor_confidence": float(v51a_stats["v51a_anchor_confidence"].detach().cpu()),
            "v51a_anchor_cluster_usage_entropy": float(v51a_stats["v51a_anchor_cluster_usage_entropy"].detach().cpu()),
            "v51a_anchor_effective_weight": float(v51a_stats["v51a_anchor_effective_weight"].detach().cpu()),
            "v51a_reliability_mean": float(v51a_stats["v51a_reliability_mean"].detach().cpu()),
            "v51a_reliability_std": float(v51a_stats["v51a_reliability_std"].detach().cpu()),
            "v51a_reliability_p10": float(v51a_stats["v51a_reliability_p10"].detach().cpu()),
            "v51a_reliability_p50": float(v51a_stats["v51a_reliability_p50"].detach().cpu()),
            "v51a_reliability_p90": float(v51a_stats["v51a_reliability_p90"].detach().cpu()),
            "v51a_reliable_node_ratio": float(v51a_stats["v51a_reliable_node_ratio"].detach().cpu()),
            "v51a_effective_anchor_mass": float(v51a_stats["v51a_effective_anchor_mass"].detach().cpu()),
            "v51a_confidence_component_mean": float(v51a_stats["v51a_confidence_component_mean"].detach().cpu()),
            "v51a_q_anchor_component_mean": float(v51a_stats["v51a_q_anchor_component_mean"].detach().cpu()),
            "v51a_embed_anchor_component_mean": float(v51a_stats["v51a_embed_anchor_component_mean"].detach().cpu()),
            "v51a_local_component_mean": float(v51a_stats["v51a_local_component_mean"].detach().cpu()),
            "v52a_enabled": bool(v52a_enabled),
            "v52a_gamma": float(v52a_stats["v52a_gamma"].detach().cpu()),
            "v52a_anchor_loss": float(v52a_stats["v52a_anchor_loss"].detach().cpu()),
            "v52a_weighted_q_anchor_kl": float(v52a_stats["v52a_weighted_q_anchor_kl"].detach().cpu()),
            "v52a_weighted_q_anchor_agreement": float(v52a_stats["v52a_weighted_q_anchor_agreement"].detach().cpu()),
            "v52a_unweighted_q_anchor_agreement": float(v52a_stats["v52a_unweighted_q_anchor_agreement"].detach().cpu()),
            "v52a_embedding_anchor_agreement": float(v52a_stats["v52a_embedding_anchor_agreement"].detach().cpu()),
            "v52a_anchor_entropy": float(v52a_stats["v52a_anchor_entropy"].detach().cpu()),
            "v52a_anchor_confidence": float(v52a_stats["v52a_anchor_confidence"].detach().cpu()),
            "v52a_anchor_cluster_usage_entropy": float(v52a_stats["v52a_anchor_cluster_usage_entropy"].detach().cpu()),
            "v52a_anchor_effective_weight": float(v52a_stats["v52a_anchor_effective_weight"].detach().cpu()),
            "v52a_reliability_mean": float(v52a_stats["v52a_reliability_mean"].detach().cpu()),
            "v52a_reliability_std": float(v52a_stats["v52a_reliability_std"].detach().cpu()),
            "v52a_reliability_p10": float(v52a_stats["v52a_reliability_p10"].detach().cpu()),
            "v52a_reliability_p50": float(v52a_stats["v52a_reliability_p50"].detach().cpu()),
            "v52a_reliability_p90": float(v52a_stats["v52a_reliability_p90"].detach().cpu()),
            "v52a_reliable_node_ratio": float(v52a_stats["v52a_reliable_node_ratio"].detach().cpu()),
            "v52a_effective_anchor_mass": float(v52a_stats["v52a_effective_anchor_mass"].detach().cpu()),
            "v52a_base_reliability_mean": float(v52a_stats["v52a_base_reliability_mean"].detach().cpu()),
            "v52a_agreement_reliability_mean": float(v52a_stats["v52a_agreement_reliability_mean"].detach().cpu()),
            "v52a_confidence_component_mean": float(v52a_stats["v52a_confidence_component_mean"].detach().cpu()),
            "v52a_q_anchor_component_mean": float(v52a_stats["v52a_q_anchor_component_mean"].detach().cpu()),
            "v52a_embed_anchor_component_mean": float(v52a_stats["v52a_embed_anchor_component_mean"].detach().cpu()),
            "v52a_local_component_mean": float(v52a_stats["v52a_local_component_mean"].detach().cpu()),
            "v53a_enabled": bool(v53a_enabled),
            "v53a_gamma": float(v53a_stats["v53a_gamma"].detach().cpu()),
            "v53a_residual_beta": float(v53a_stats["v53a_residual_beta"].detach().cpu()),
            "v53a_residual_multiplier_mean": float(v53a_stats["v53a_residual_multiplier_mean"].detach().cpu()),
            "v53a_anchor_loss": float(v53a_stats["v53a_anchor_loss"].detach().cpu()),
            "v53a_weighted_q_anchor_kl": float(v53a_stats["v53a_weighted_q_anchor_kl"].detach().cpu()),
            "v53a_weighted_q_anchor_agreement": float(v53a_stats["v53a_weighted_q_anchor_agreement"].detach().cpu()),
            "v53a_unweighted_q_anchor_agreement": float(v53a_stats["v53a_unweighted_q_anchor_agreement"].detach().cpu()),
            "v53a_embedding_anchor_agreement": float(v53a_stats["v53a_embedding_anchor_agreement"].detach().cpu()),
            "v53a_anchor_entropy": float(v53a_stats["v53a_anchor_entropy"].detach().cpu()),
            "v53a_anchor_confidence": float(v53a_stats["v53a_anchor_confidence"].detach().cpu()),
            "v53a_anchor_cluster_usage_entropy": float(v53a_stats["v53a_anchor_cluster_usage_entropy"].detach().cpu()),
            "v53a_anchor_effective_weight": float(v53a_stats["v53a_anchor_effective_weight"].detach().cpu()),
            "v53a_reliability_mean": float(v53a_stats["v53a_reliability_mean"].detach().cpu()),
            "v53a_reliability_std": float(v53a_stats["v53a_reliability_std"].detach().cpu()),
            "v53a_reliability_p10": float(v53a_stats["v53a_reliability_p10"].detach().cpu()),
            "v53a_reliability_p50": float(v53a_stats["v53a_reliability_p50"].detach().cpu()),
            "v53a_reliability_p90": float(v53a_stats["v53a_reliability_p90"].detach().cpu()),
            "v53a_reliable_node_ratio": float(v53a_stats["v53a_reliable_node_ratio"].detach().cpu()),
            "v53a_effective_anchor_mass": float(v53a_stats["v53a_effective_anchor_mass"].detach().cpu()),
            "v53a_base_reliability_mean": float(v53a_stats["v53a_base_reliability_mean"].detach().cpu()),
            "v53a_agreement_reliability_mean": float(v53a_stats["v53a_agreement_reliability_mean"].detach().cpu()),
            "v53a_confidence_component_mean": float(v53a_stats["v53a_confidence_component_mean"].detach().cpu()),
            "v53a_q_anchor_component_mean": float(v53a_stats["v53a_q_anchor_component_mean"].detach().cpu()),
            "v53a_embed_anchor_component_mean": float(v53a_stats["v53a_embed_anchor_component_mean"].detach().cpu()),
            "v53a_local_component_mean": float(v53a_stats["v53a_local_component_mean"].detach().cpu()),
            "v54a_enabled": bool(v54a_enabled),
            "v54a_gamma": float(v54a_stats["v54a_gamma"].detach().cpu()),
            "v54a_beta_min": float(v54a_stats["v54a_beta_min"].detach().cpu()),
            "v54a_beta_max": float(v54a_stats["v54a_beta_max"].detach().cpu()),
            "v54a_beta_mean": float(v54a_stats["v54a_beta_mean"].detach().cpu()),
            "v54a_beta_p10": float(v54a_stats["v54a_beta_p10"].detach().cpu()),
            "v54a_beta_p50": float(v54a_stats["v54a_beta_p50"].detach().cpu()),
            "v54a_beta_p90": float(v54a_stats["v54a_beta_p90"].detach().cpu()),
            "v54a_hard_q_anchor_match_ratio": float(v54a_stats["v54a_hard_q_anchor_match_ratio"].detach().cpu()),
            "v54a_hard_embed_anchor_match_ratio": float(v54a_stats["v54a_hard_embed_anchor_match_ratio"].detach().cpu()),
            "v54a_hard_both_anchor_match_ratio": float(v54a_stats["v54a_hard_both_anchor_match_ratio"].detach().cpu()),
            "v54a_residual_multiplier_mean": float(v54a_stats["v54a_residual_multiplier_mean"].detach().cpu()),
            "v54a_anchor_loss": float(v54a_stats["v54a_anchor_loss"].detach().cpu()),
            "v54a_weighted_q_anchor_kl": float(v54a_stats["v54a_weighted_q_anchor_kl"].detach().cpu()),
            "v54a_weighted_q_anchor_agreement": float(v54a_stats["v54a_weighted_q_anchor_agreement"].detach().cpu()),
            "v54a_unweighted_q_anchor_agreement": float(v54a_stats["v54a_unweighted_q_anchor_agreement"].detach().cpu()),
            "v54a_embedding_anchor_agreement": float(v54a_stats["v54a_embedding_anchor_agreement"].detach().cpu()),
            "v54a_anchor_entropy": float(v54a_stats["v54a_anchor_entropy"].detach().cpu()),
            "v54a_anchor_confidence": float(v54a_stats["v54a_anchor_confidence"].detach().cpu()),
            "v54a_anchor_cluster_usage_entropy": float(v54a_stats["v54a_anchor_cluster_usage_entropy"].detach().cpu()),
            "v54a_anchor_effective_weight": float(v54a_stats["v54a_anchor_effective_weight"].detach().cpu()),
            "v54a_reliability_mean": float(v54a_stats["v54a_reliability_mean"].detach().cpu()),
            "v54a_reliability_std": float(v54a_stats["v54a_reliability_std"].detach().cpu()),
            "v54a_reliability_p10": float(v54a_stats["v54a_reliability_p10"].detach().cpu()),
            "v54a_reliability_p50": float(v54a_stats["v54a_reliability_p50"].detach().cpu()),
            "v54a_reliability_p90": float(v54a_stats["v54a_reliability_p90"].detach().cpu()),
            "v54a_reliable_node_ratio": float(v54a_stats["v54a_reliable_node_ratio"].detach().cpu()),
            "v54a_effective_anchor_mass": float(v54a_stats["v54a_effective_anchor_mass"].detach().cpu()),
            "v54a_base_reliability_mean": float(v54a_stats["v54a_base_reliability_mean"].detach().cpu()),
            "v54a_agreement_reliability_mean": float(v54a_stats["v54a_agreement_reliability_mean"].detach().cpu()),
            "v54a_confidence_component_mean": float(v54a_stats["v54a_confidence_component_mean"].detach().cpu()),
            "v54a_q_anchor_component_mean": float(v54a_stats["v54a_q_anchor_component_mean"].detach().cpu()),
            "v54a_embed_anchor_component_mean": float(v54a_stats["v54a_embed_anchor_component_mean"].detach().cpu()),
            "v54a_local_component_mean": float(v54a_stats["v54a_local_component_mean"].detach().cpu()),
            "v55a_enabled": bool(v55a_enabled),
            "v55a_gamma": float(v55a_stats["v55a_gamma"].detach().cpu()),
            "v55a_beta_min": float(v55a_stats["v55a_beta_min"].detach().cpu()),
            "v55a_beta_max": float(v55a_stats["v55a_beta_max"].detach().cpu()),
            "v55a_soft_power": float(v55a_stats["v55a_soft_power"].detach().cpu()),
            "v55a_soft_consensus_mean": float(v55a_stats["v55a_soft_consensus_mean"].detach().cpu()),
            "v55a_soft_consensus_p10": float(v55a_stats["v55a_soft_consensus_p10"].detach().cpu()),
            "v55a_soft_consensus_p50": float(v55a_stats["v55a_soft_consensus_p50"].detach().cpu()),
            "v55a_soft_consensus_p90": float(v55a_stats["v55a_soft_consensus_p90"].detach().cpu()),
            "v55a_beta_mean": float(v55a_stats["v55a_beta_mean"].detach().cpu()),
            "v55a_beta_p10": float(v55a_stats["v55a_beta_p10"].detach().cpu()),
            "v55a_beta_p50": float(v55a_stats["v55a_beta_p50"].detach().cpu()),
            "v55a_beta_p90": float(v55a_stats["v55a_beta_p90"].detach().cpu()),
            "v55a_residual_multiplier_mean": float(v55a_stats["v55a_residual_multiplier_mean"].detach().cpu()),
            "v55a_anchor_loss": float(v55a_stats["v55a_anchor_loss"].detach().cpu()),
            "v55a_weighted_q_anchor_kl": float(v55a_stats["v55a_weighted_q_anchor_kl"].detach().cpu()),
            "v55a_weighted_q_anchor_agreement": float(v55a_stats["v55a_weighted_q_anchor_agreement"].detach().cpu()),
            "v55a_unweighted_q_anchor_agreement": float(v55a_stats["v55a_unweighted_q_anchor_agreement"].detach().cpu()),
            "v55a_embedding_anchor_agreement": float(v55a_stats["v55a_embedding_anchor_agreement"].detach().cpu()),
            "v55a_anchor_entropy": float(v55a_stats["v55a_anchor_entropy"].detach().cpu()),
            "v55a_anchor_confidence": float(v55a_stats["v55a_anchor_confidence"].detach().cpu()),
            "v55a_anchor_cluster_usage_entropy": float(v55a_stats["v55a_anchor_cluster_usage_entropy"].detach().cpu()),
            "v55a_anchor_effective_weight": float(v55a_stats["v55a_anchor_effective_weight"].detach().cpu()),
            "v55a_reliability_mean": float(v55a_stats["v55a_reliability_mean"].detach().cpu()),
            "v55a_reliability_std": float(v55a_stats["v55a_reliability_std"].detach().cpu()),
            "v55a_reliability_p10": float(v55a_stats["v55a_reliability_p10"].detach().cpu()),
            "v55a_reliability_p50": float(v55a_stats["v55a_reliability_p50"].detach().cpu()),
            "v55a_reliability_p90": float(v55a_stats["v55a_reliability_p90"].detach().cpu()),
            "v55a_reliable_node_ratio": float(v55a_stats["v55a_reliable_node_ratio"].detach().cpu()),
            "v55a_effective_anchor_mass": float(v55a_stats["v55a_effective_anchor_mass"].detach().cpu()),
            "v55a_base_reliability_mean": float(v55a_stats["v55a_base_reliability_mean"].detach().cpu()),
            "v55a_agreement_reliability_mean": float(v55a_stats["v55a_agreement_reliability_mean"].detach().cpu()),
            "v55a_confidence_component_mean": float(v55a_stats["v55a_confidence_component_mean"].detach().cpu()),
            "v55a_q_anchor_component_mean": float(v55a_stats["v55a_q_anchor_component_mean"].detach().cpu()),
            "v55a_embed_anchor_component_mean": float(v55a_stats["v55a_embed_anchor_component_mean"].detach().cpu()),
            "v55a_local_component_mean": float(v55a_stats["v55a_local_component_mean"].detach().cpu()),
            "v56a_enabled": bool(v56a_enabled),
            "v56a_gamma": float(v56a_stats["v56a_gamma"].detach().cpu()),
            "v56a_beta_min": float(v56a_stats["v56a_beta_min"].detach().cpu()),
            "v56a_beta_max": float(v56a_stats["v56a_beta_max"].detach().cpu()),
            "v56a_soft_power": float(v56a_stats["v56a_soft_power"].detach().cpu()),
            "v56a_hybrid_compensation": float(v56a_stats["v56a_hybrid_compensation"].detach().cpu()),
            "v56a_hard_consensus_mean": float(v56a_stats["v56a_hard_consensus_mean"].detach().cpu()),
            "v56a_soft_consensus_mean": float(v56a_stats["v56a_soft_consensus_mean"].detach().cpu()),
            "v56a_lifted_soft_consensus_mean": float(v56a_stats["v56a_lifted_soft_consensus_mean"].detach().cpu()),
            "v56a_compensation_mean": float(v56a_stats["v56a_compensation_mean"].detach().cpu()),
            "v56a_compensation_active_ratio": float(v56a_stats["v56a_compensation_active_ratio"].detach().cpu()),
            "v56a_hybrid_consensus_mean": float(v56a_stats["v56a_hybrid_consensus_mean"].detach().cpu()),
            "v56a_beta_mean": float(v56a_stats["v56a_beta_mean"].detach().cpu()),
            "v56a_beta_p10": float(v56a_stats["v56a_beta_p10"].detach().cpu()),
            "v56a_beta_p50": float(v56a_stats["v56a_beta_p50"].detach().cpu()),
            "v56a_beta_p90": float(v56a_stats["v56a_beta_p90"].detach().cpu()),
            "v56a_residual_multiplier_mean": float(v56a_stats["v56a_residual_multiplier_mean"].detach().cpu()),
            "v56a_anchor_loss": float(v56a_stats["v56a_anchor_loss"].detach().cpu()),
            "v56a_weighted_q_anchor_kl": float(v56a_stats["v56a_weighted_q_anchor_kl"].detach().cpu()),
            "v56a_weighted_q_anchor_agreement": float(v56a_stats["v56a_weighted_q_anchor_agreement"].detach().cpu()),
            "v56a_unweighted_q_anchor_agreement": float(v56a_stats["v56a_unweighted_q_anchor_agreement"].detach().cpu()),
            "v56a_embedding_anchor_agreement": float(v56a_stats["v56a_embedding_anchor_agreement"].detach().cpu()),
            "v56a_anchor_entropy": float(v56a_stats["v56a_anchor_entropy"].detach().cpu()),
            "v56a_anchor_confidence": float(v56a_stats["v56a_anchor_confidence"].detach().cpu()),
            "v56a_anchor_cluster_usage_entropy": float(v56a_stats["v56a_anchor_cluster_usage_entropy"].detach().cpu()),
            "v56a_anchor_effective_weight": float(v56a_stats["v56a_anchor_effective_weight"].detach().cpu()),
            "v56a_reliability_mean": float(v56a_stats["v56a_reliability_mean"].detach().cpu()),
            "v56a_reliability_std": float(v56a_stats["v56a_reliability_std"].detach().cpu()),
            "v56a_reliability_p10": float(v56a_stats["v56a_reliability_p10"].detach().cpu()),
            "v56a_reliability_p50": float(v56a_stats["v56a_reliability_p50"].detach().cpu()),
            "v56a_reliability_p90": float(v56a_stats["v56a_reliability_p90"].detach().cpu()),
            "v56a_reliable_node_ratio": float(v56a_stats["v56a_reliable_node_ratio"].detach().cpu()),
            "v56a_effective_anchor_mass": float(v56a_stats["v56a_effective_anchor_mass"].detach().cpu()),
            "v56a_base_reliability_mean": float(v56a_stats["v56a_base_reliability_mean"].detach().cpu()),
            "v56a_agreement_reliability_mean": float(v56a_stats["v56a_agreement_reliability_mean"].detach().cpu()),
            "v56a_confidence_component_mean": float(v56a_stats["v56a_confidence_component_mean"].detach().cpu()),
            "v56a_q_anchor_component_mean": float(v56a_stats["v56a_q_anchor_component_mean"].detach().cpu()),
            "v56a_embed_anchor_component_mean": float(v56a_stats["v56a_embed_anchor_component_mean"].detach().cpu()),
            "v56a_local_component_mean": float(v56a_stats["v56a_local_component_mean"].detach().cpu()),
            "v57a_enabled": bool(v57a_enabled),
            "v57a_gamma": float(v57a_stats["v57a_gamma"].detach().cpu()),
            "v57a_beta_min": float(v57a_stats["v57a_beta_min"].detach().cpu()),
            "v57a_beta_max": float(v57a_stats["v57a_beta_max"].detach().cpu()),
            "v57a_soft_power": float(v57a_stats["v57a_soft_power"].detach().cpu()),
            "v57a_hybrid_compensation": float(v57a_stats["v57a_hybrid_compensation"].detach().cpu()),
            "v57a_target_mass": float(v57a_stats["v57a_target_mass"].detach().cpu()),
            "v57a_max_mass_scale": float(v57a_stats["v57a_max_mass_scale"].detach().cpu()),
            "v57a_max_reliability_cap": float(v57a_stats["v57a_max_reliability_cap"].detach().cpu()),
            "v57a_hard_consensus_mean": float(v57a_stats["v57a_hard_consensus_mean"].detach().cpu()),
            "v57a_soft_consensus_mean": float(v57a_stats["v57a_soft_consensus_mean"].detach().cpu()),
            "v57a_lifted_soft_consensus_mean": float(v57a_stats["v57a_lifted_soft_consensus_mean"].detach().cpu()),
            "v57a_compensation_mean": float(v57a_stats["v57a_compensation_mean"].detach().cpu()),
            "v57a_compensation_active_ratio": float(v57a_stats["v57a_compensation_active_ratio"].detach().cpu()),
            "v57a_hybrid_consensus_mean": float(v57a_stats["v57a_hybrid_consensus_mean"].detach().cpu()),
            "v57a_beta_mean": float(v57a_stats["v57a_beta_mean"].detach().cpu()),
            "v57a_raw_reliability_mean": float(v57a_stats["v57a_raw_reliability_mean"].detach().cpu()),
            "v57a_mass_scale": float(v57a_stats["v57a_mass_scale"].detach().cpu()),
            "v57a_scaled_reliability_mean": float(v57a_stats["v57a_scaled_reliability_mean"].detach().cpu()),
            "v57a_residual_multiplier_mean": float(v57a_stats["v57a_residual_multiplier_mean"].detach().cpu()),
            "v57a_anchor_loss": float(v57a_stats["v57a_anchor_loss"].detach().cpu()),
            "v57a_weighted_q_anchor_kl": float(v57a_stats["v57a_weighted_q_anchor_kl"].detach().cpu()),
            "v57a_weighted_q_anchor_agreement": float(v57a_stats["v57a_weighted_q_anchor_agreement"].detach().cpu()),
            "v57a_unweighted_q_anchor_agreement": float(v57a_stats["v57a_unweighted_q_anchor_agreement"].detach().cpu()),
            "v57a_embedding_anchor_agreement": float(v57a_stats["v57a_embedding_anchor_agreement"].detach().cpu()),
            "v57a_anchor_entropy": float(v57a_stats["v57a_anchor_entropy"].detach().cpu()),
            "v57a_anchor_confidence": float(v57a_stats["v57a_anchor_confidence"].detach().cpu()),
            "v57a_anchor_cluster_usage_entropy": float(v57a_stats["v57a_anchor_cluster_usage_entropy"].detach().cpu()),
            "v57a_anchor_effective_weight": float(v57a_stats["v57a_anchor_effective_weight"].detach().cpu()),
            "v57a_reliability_mean": float(v57a_stats["v57a_reliability_mean"].detach().cpu()),
            "v57a_reliability_std": float(v57a_stats["v57a_reliability_std"].detach().cpu()),
            "v57a_reliability_p10": float(v57a_stats["v57a_reliability_p10"].detach().cpu()),
            "v57a_reliability_p50": float(v57a_stats["v57a_reliability_p50"].detach().cpu()),
            "v57a_reliability_p90": float(v57a_stats["v57a_reliability_p90"].detach().cpu()),
            "v57a_reliable_node_ratio": float(v57a_stats["v57a_reliable_node_ratio"].detach().cpu()),
            "v57a_effective_anchor_mass": float(v57a_stats["v57a_effective_anchor_mass"].detach().cpu()),
            "v57a_base_reliability_mean": float(v57a_stats["v57a_base_reliability_mean"].detach().cpu()),
            "v57a_agreement_reliability_mean": float(v57a_stats["v57a_agreement_reliability_mean"].detach().cpu()),
            "v57a_confidence_component_mean": float(v57a_stats["v57a_confidence_component_mean"].detach().cpu()),
            "v57a_q_anchor_component_mean": float(v57a_stats["v57a_q_anchor_component_mean"].detach().cpu()),
            "v57a_embed_anchor_component_mean": float(v57a_stats["v57a_embed_anchor_component_mean"].detach().cpu()),
            "v57a_local_component_mean": float(v57a_stats["v57a_local_component_mean"].detach().cpu()),
            "v58a_enabled": bool(v58a_enabled),
            "v58a_release_gamma": float(v58a_stats["v58a_release_gamma"].detach().cpu()),
            "v58a_release_warmup_epochs": float(v58a_stats["v58a_release_warmup_epochs"].detach().cpu()),
            "v58a_release_ramp_epochs": float(v58a_stats["v58a_release_ramp_epochs"].detach().cpu()),
            "v58a_release_hold_until_epoch": float(v58a_stats["v58a_release_hold_until_epoch"].detach().cpu()),
            "v58a_release_decay_epochs": float(v58a_stats["v58a_release_decay_epochs"].detach().cpu()),
            "v58a_release_floor": float(v58a_stats["v58a_release_floor"].detach().cpu()),
            "v58a_gamma": float(v58a_stats["v58a_gamma"].detach().cpu()),
            "v58a_beta_min": float(v58a_stats["v58a_beta_min"].detach().cpu()),
            "v58a_beta_max": float(v58a_stats["v58a_beta_max"].detach().cpu()),
            "v58a_soft_power": float(v58a_stats["v58a_soft_power"].detach().cpu()),
            "v58a_hybrid_compensation": float(v58a_stats["v58a_hybrid_compensation"].detach().cpu()),
            "v58a_target_mass": float(v58a_stats["v58a_target_mass"].detach().cpu()),
            "v58a_max_mass_scale": float(v58a_stats["v58a_max_mass_scale"].detach().cpu()),
            "v58a_max_reliability_cap": float(v58a_stats["v58a_max_reliability_cap"].detach().cpu()),
            "v58a_hard_consensus_mean": float(v58a_stats["v58a_hard_consensus_mean"].detach().cpu()),
            "v58a_soft_consensus_mean": float(v58a_stats["v58a_soft_consensus_mean"].detach().cpu()),
            "v58a_lifted_soft_consensus_mean": float(v58a_stats["v58a_lifted_soft_consensus_mean"].detach().cpu()),
            "v58a_compensation_mean": float(v58a_stats["v58a_compensation_mean"].detach().cpu()),
            "v58a_compensation_active_ratio": float(v58a_stats["v58a_compensation_active_ratio"].detach().cpu()),
            "v58a_hybrid_consensus_mean": float(v58a_stats["v58a_hybrid_consensus_mean"].detach().cpu()),
            "v58a_beta_mean": float(v58a_stats["v58a_beta_mean"].detach().cpu()),
            "v58a_raw_reliability_mean": float(v58a_stats["v58a_raw_reliability_mean"].detach().cpu()),
            "v58a_mass_scale": float(v58a_stats["v58a_mass_scale"].detach().cpu()),
            "v58a_scaled_reliability_mean": float(v58a_stats["v58a_scaled_reliability_mean"].detach().cpu()),
            "v58a_residual_multiplier_mean": float(v58a_stats["v58a_residual_multiplier_mean"].detach().cpu()),
            "v58a_anchor_loss": float(v58a_stats["v58a_anchor_loss"].detach().cpu()),
            "v58a_pre_release_anchor_loss": float(v58a_stats["v58a_pre_release_anchor_loss"].detach().cpu()),
            "v58a_weighted_q_anchor_kl": float(v58a_stats["v58a_weighted_q_anchor_kl"].detach().cpu()),
            "v58a_pre_release_weighted_q_anchor_kl": float(v58a_stats["v58a_pre_release_weighted_q_anchor_kl"].detach().cpu()),
            "v58a_weighted_q_anchor_agreement": float(v58a_stats["v58a_weighted_q_anchor_agreement"].detach().cpu()),
            "v58a_unweighted_q_anchor_agreement": float(v58a_stats["v58a_unweighted_q_anchor_agreement"].detach().cpu()),
            "v58a_embedding_anchor_agreement": float(v58a_stats["v58a_embedding_anchor_agreement"].detach().cpu()),
            "v58a_anchor_entropy": float(v58a_stats["v58a_anchor_entropy"].detach().cpu()),
            "v58a_anchor_confidence": float(v58a_stats["v58a_anchor_confidence"].detach().cpu()),
            "v58a_anchor_cluster_usage_entropy": float(v58a_stats["v58a_anchor_cluster_usage_entropy"].detach().cpu()),
            "v58a_anchor_effective_weight": float(v58a_stats["v58a_anchor_effective_weight"].detach().cpu()),
            "v58a_reliability_mean": float(v58a_stats["v58a_reliability_mean"].detach().cpu()),
            "v58a_reliability_std": float(v58a_stats["v58a_reliability_std"].detach().cpu()),
            "v58a_reliability_p10": float(v58a_stats["v58a_reliability_p10"].detach().cpu()),
            "v58a_reliability_p50": float(v58a_stats["v58a_reliability_p50"].detach().cpu()),
            "v58a_reliability_p90": float(v58a_stats["v58a_reliability_p90"].detach().cpu()),
            "v58a_reliable_node_ratio": float(v58a_stats["v58a_reliable_node_ratio"].detach().cpu()),
            "v58a_effective_anchor_mass": float(v58a_stats["v58a_effective_anchor_mass"].detach().cpu()),
            "v58a_base_reliability_mean": float(v58a_stats["v58a_base_reliability_mean"].detach().cpu()),
            "v58a_agreement_reliability_mean": float(v58a_stats["v58a_agreement_reliability_mean"].detach().cpu()),
            "v58a_confidence_component_mean": float(v58a_stats["v58a_confidence_component_mean"].detach().cpu()),
            "v58a_q_anchor_component_mean": float(v58a_stats["v58a_q_anchor_component_mean"].detach().cpu()),
            "v58a_embed_anchor_component_mean": float(v58a_stats["v58a_embed_anchor_component_mean"].detach().cpu()),
            "v58a_local_component_mean": float(v58a_stats["v58a_local_component_mean"].detach().cpu()),
            "v59a_enabled": bool(v59a_enabled),
            "v59a_release_gamma": float(v59a_stats["v59a_release_gamma"].detach().cpu()),
            "v59a_release_start_epoch": float(v59a_stats["v59a_release_start_epoch"].detach().cpu()),
            "v59a_release_decay_epochs": float(v59a_stats["v59a_release_decay_epochs"].detach().cpu()),
            "v59a_release_floor": float(v59a_stats["v59a_release_floor"].detach().cpu()),
            "v59a_gamma": float(v59a_stats["v59a_gamma"].detach().cpu()),
            "v59a_beta_min": float(v59a_stats["v59a_beta_min"].detach().cpu()),
            "v59a_beta_max": float(v59a_stats["v59a_beta_max"].detach().cpu()),
            "v59a_soft_power": float(v59a_stats["v59a_soft_power"].detach().cpu()),
            "v59a_hybrid_compensation": float(v59a_stats["v59a_hybrid_compensation"].detach().cpu()),
            "v59a_target_mass": float(v59a_stats["v59a_target_mass"].detach().cpu()),
            "v59a_max_mass_scale": float(v59a_stats["v59a_max_mass_scale"].detach().cpu()),
            "v59a_max_reliability_cap": float(v59a_stats["v59a_max_reliability_cap"].detach().cpu()),
            "v59a_hard_consensus_mean": float(v59a_stats["v59a_hard_consensus_mean"].detach().cpu()),
            "v59a_soft_consensus_mean": float(v59a_stats["v59a_soft_consensus_mean"].detach().cpu()),
            "v59a_lifted_soft_consensus_mean": float(v59a_stats["v59a_lifted_soft_consensus_mean"].detach().cpu()),
            "v59a_compensation_mean": float(v59a_stats["v59a_compensation_mean"].detach().cpu()),
            "v59a_compensation_active_ratio": float(v59a_stats["v59a_compensation_active_ratio"].detach().cpu()),
            "v59a_hybrid_consensus_mean": float(v59a_stats["v59a_hybrid_consensus_mean"].detach().cpu()),
            "v59a_beta_mean": float(v59a_stats["v59a_beta_mean"].detach().cpu()),
            "v59a_raw_reliability_mean": float(v59a_stats["v59a_raw_reliability_mean"].detach().cpu()),
            "v59a_mass_scale": float(v59a_stats["v59a_mass_scale"].detach().cpu()),
            "v59a_scaled_reliability_mean": float(v59a_stats["v59a_scaled_reliability_mean"].detach().cpu()),
            "v59a_residual_multiplier_mean": float(v59a_stats["v59a_residual_multiplier_mean"].detach().cpu()),
            "v59a_anchor_loss": float(v59a_stats["v59a_anchor_loss"].detach().cpu()),
            "v59a_pre_release_anchor_loss": float(v59a_stats["v59a_pre_release_anchor_loss"].detach().cpu()),
            "v59a_weighted_q_anchor_kl": float(v59a_stats["v59a_weighted_q_anchor_kl"].detach().cpu()),
            "v59a_pre_release_weighted_q_anchor_kl": float(v59a_stats["v59a_pre_release_weighted_q_anchor_kl"].detach().cpu()),
            "v59a_weighted_q_anchor_agreement": float(v59a_stats["v59a_weighted_q_anchor_agreement"].detach().cpu()),
            "v59a_unweighted_q_anchor_agreement": float(v59a_stats["v59a_unweighted_q_anchor_agreement"].detach().cpu()),
            "v59a_embedding_anchor_agreement": float(v59a_stats["v59a_embedding_anchor_agreement"].detach().cpu()),
            "v59a_anchor_entropy": float(v59a_stats["v59a_anchor_entropy"].detach().cpu()),
            "v59a_anchor_confidence": float(v59a_stats["v59a_anchor_confidence"].detach().cpu()),
            "v59a_anchor_cluster_usage_entropy": float(v59a_stats["v59a_anchor_cluster_usage_entropy"].detach().cpu()),
            "v59a_anchor_effective_weight": float(v59a_stats["v59a_anchor_effective_weight"].detach().cpu()),
            "v59a_reliability_mean": float(v59a_stats["v59a_reliability_mean"].detach().cpu()),
            "v59a_reliability_std": float(v59a_stats["v59a_reliability_std"].detach().cpu()),
            "v59a_reliability_p10": float(v59a_stats["v59a_reliability_p10"].detach().cpu()),
            "v59a_reliability_p50": float(v59a_stats["v59a_reliability_p50"].detach().cpu()),
            "v59a_reliability_p90": float(v59a_stats["v59a_reliability_p90"].detach().cpu()),
            "v59a_reliable_node_ratio": float(v59a_stats["v59a_reliable_node_ratio"].detach().cpu()),
            "v59a_effective_anchor_mass": float(v59a_stats["v59a_effective_anchor_mass"].detach().cpu()),
            "v59a_base_reliability_mean": float(v59a_stats["v59a_base_reliability_mean"].detach().cpu()),
            "v59a_agreement_reliability_mean": float(v59a_stats["v59a_agreement_reliability_mean"].detach().cpu()),
            "v59a_confidence_component_mean": float(v59a_stats["v59a_confidence_component_mean"].detach().cpu()),
            "v59a_q_anchor_component_mean": float(v59a_stats["v59a_q_anchor_component_mean"].detach().cpu()),
            "v59a_embed_anchor_component_mean": float(v59a_stats["v59a_embed_anchor_component_mean"].detach().cpu()),
            "v59a_local_component_mean": float(v59a_stats["v59a_local_component_mean"].detach().cpu()),
            "v60a_enabled": bool(v60a_enabled),
            "v60a_release_gamma": float(v60a_anchor_stats["v60a_release_gamma"].detach().cpu()),
            "v60a_release_start_epoch": float(v60a_anchor_stats["v60a_release_start_epoch"].detach().cpu()),
            "v60a_release_decay_epochs": float(v60a_anchor_stats["v60a_release_decay_epochs"].detach().cpu()),
            "v60a_release_floor": float(v60a_anchor_stats["v60a_release_floor"].detach().cpu()),
            "v60a_gamma": float(v60a_anchor_stats["v60a_gamma"].detach().cpu()),
            "v60a_beta_min": float(v60a_anchor_stats["v60a_beta_min"].detach().cpu()),
            "v60a_beta_max": float(v60a_anchor_stats["v60a_beta_max"].detach().cpu()),
            "v60a_soft_power": float(v60a_anchor_stats["v60a_soft_power"].detach().cpu()),
            "v60a_hybrid_compensation": float(v60a_anchor_stats["v60a_hybrid_compensation"].detach().cpu()),
            "v60a_target_mass": float(v60a_anchor_stats["v60a_target_mass"].detach().cpu()),
            "v60a_max_mass_scale": float(v60a_anchor_stats["v60a_max_mass_scale"].detach().cpu()),
            "v60a_max_reliability_cap": float(v60a_anchor_stats["v60a_max_reliability_cap"].detach().cpu()),
            "v60a_hard_consensus_mean": float(v60a_anchor_stats["v60a_hard_consensus_mean"].detach().cpu()),
            "v60a_soft_consensus_mean": float(v60a_anchor_stats["v60a_soft_consensus_mean"].detach().cpu()),
            "v60a_lifted_soft_consensus_mean": float(v60a_anchor_stats["v60a_lifted_soft_consensus_mean"].detach().cpu()),
            "v60a_compensation_mean": float(v60a_anchor_stats["v60a_compensation_mean"].detach().cpu()),
            "v60a_compensation_active_ratio": float(v60a_anchor_stats["v60a_compensation_active_ratio"].detach().cpu()),
            "v60a_hybrid_consensus_mean": float(v60a_anchor_stats["v60a_hybrid_consensus_mean"].detach().cpu()),
            "v60a_beta_mean": float(v60a_anchor_stats["v60a_beta_mean"].detach().cpu()),
            "v60a_raw_reliability_mean": float(v60a_anchor_stats["v60a_raw_reliability_mean"].detach().cpu()),
            "v60a_mass_scale": float(v60a_anchor_stats["v60a_mass_scale"].detach().cpu()),
            "v60a_scaled_reliability_mean": float(v60a_anchor_stats["v60a_scaled_reliability_mean"].detach().cpu()),
            "v60a_residual_multiplier_mean": float(v60a_anchor_stats["v60a_residual_multiplier_mean"].detach().cpu()),
            "v60a_anchor_loss": float(v60a_anchor_stats["v60a_anchor_loss"].detach().cpu()),
            "v60a_pre_release_anchor_loss": float(v60a_anchor_stats["v60a_pre_release_anchor_loss"].detach().cpu()),
            "v60a_weighted_q_anchor_kl": float(v60a_anchor_stats["v60a_weighted_q_anchor_kl"].detach().cpu()),
            "v60a_pre_release_weighted_q_anchor_kl": float(v60a_anchor_stats["v60a_pre_release_weighted_q_anchor_kl"].detach().cpu()),
            "v60a_weighted_q_anchor_agreement": float(v60a_anchor_stats["v60a_weighted_q_anchor_agreement"].detach().cpu()),
            "v60a_unweighted_q_anchor_agreement": float(v60a_anchor_stats["v60a_unweighted_q_anchor_agreement"].detach().cpu()),
            "v60a_embedding_anchor_agreement": float(v60a_anchor_stats["v60a_embedding_anchor_agreement"].detach().cpu()),
            "v60a_anchor_entropy": float(v60a_anchor_stats["v60a_anchor_entropy"].detach().cpu()),
            "v60a_anchor_confidence": float(v60a_anchor_stats["v60a_anchor_confidence"].detach().cpu()),
            "v60a_anchor_cluster_usage_entropy": float(v60a_anchor_stats["v60a_anchor_cluster_usage_entropy"].detach().cpu()),
            "v60a_anchor_effective_weight": float(v60a_anchor_stats["v60a_anchor_effective_weight"].detach().cpu()),
            "v60a_reliability_mean": float(v60a_anchor_stats["v60a_reliability_mean"].detach().cpu()),
            "v60a_reliability_std": float(v60a_anchor_stats["v60a_reliability_std"].detach().cpu()),
            "v60a_reliability_p10": float(v60a_anchor_stats["v60a_reliability_p10"].detach().cpu()),
            "v60a_reliability_p50": float(v60a_anchor_stats["v60a_reliability_p50"].detach().cpu()),
            "v60a_reliability_p90": float(v60a_anchor_stats["v60a_reliability_p90"].detach().cpu()),
            "v60a_reliable_node_ratio": float(v60a_anchor_stats["v60a_reliable_node_ratio"].detach().cpu()),
            "v60a_effective_anchor_mass": float(v60a_anchor_stats["v60a_effective_anchor_mass"].detach().cpu()),
            "v60a_base_reliability_mean": float(v60a_anchor_stats["v60a_base_reliability_mean"].detach().cpu()),
            "v60a_agreement_reliability_mean": float(v60a_anchor_stats["v60a_agreement_reliability_mean"].detach().cpu()),
            "v60a_confidence_component_mean": float(v60a_anchor_stats["v60a_confidence_component_mean"].detach().cpu()),
            "v60a_q_anchor_component_mean": float(v60a_anchor_stats["v60a_q_anchor_component_mean"].detach().cpu()),
            "v60a_embed_anchor_component_mean": float(v60a_anchor_stats["v60a_embed_anchor_component_mean"].detach().cpu()),
            "v60a_local_component_mean": float(v60a_anchor_stats["v60a_local_component_mean"].detach().cpu()),
            "v60a_guard_enabled": bool(v60a_guard_stats["v60a_guard_enabled"].detach().cpu()),
            "v60a_teacher_ready": bool(v60a_guard_stats["v60a_teacher_ready"].detach().cpu()),
            "v60a_teacher_epoch": float(v60a_guard_stats["v60a_teacher_epoch"].detach().cpu()),
            "v60a_guard_gamma": float(v60a_guard_stats["v60a_guard_gamma"].detach().cpu()),
            "v60a_guard_weight": float(v60a_guard_stats["v60a_guard_weight"].detach().cpu()),
            "v60a_confidence_threshold": float(v60a_guard_stats["v60a_confidence_threshold"].detach().cpu()),
            "v60a_teacher_confidence_mean": float(v60a_guard_stats["v60a_teacher_confidence_mean"].detach().cpu()),
            "v60a_teacher_active_ratio": float(v60a_guard_stats["v60a_teacher_active_ratio"].detach().cpu()),
            "v60a_guard_kl": float(v60a_guard_stats["v60a_guard_kl"].detach().cpu()),
            "v60a_guard_loss": float(v60a_guard_stats["v60a_guard_loss"].detach().cpu()),
            "v60a_q_teacher_agreement": float(v60a_guard_stats["v60a_q_teacher_agreement"].detach().cpu()),
            "v60a_q_teacher_kl": float(v60a_guard_stats["v60a_q_teacher_kl"].detach().cpu()),
            "v61a_enabled": bool(v61a_enabled),
            "v61a_release_gamma": float(v61a_anchor_stats["v61a_release_gamma"].detach().cpu()),
            "v61a_release_start_epoch": float(v61a_anchor_stats["v61a_release_start_epoch"].detach().cpu()),
            "v61a_release_decay_epochs": float(v61a_anchor_stats["v61a_release_decay_epochs"].detach().cpu()),
            "v61a_release_floor": float(v61a_anchor_stats["v61a_release_floor"].detach().cpu()),
            "v61a_gamma": float(v61a_anchor_stats["v61a_gamma"].detach().cpu()),
            "v61a_raw_reliability_mean": float(v61a_anchor_stats["v61a_raw_reliability_mean"].detach().cpu()),
            "v61a_mass_scale": float(v61a_anchor_stats["v61a_mass_scale"].detach().cpu()),
            "v61a_scaled_reliability_mean": float(v61a_anchor_stats["v61a_scaled_reliability_mean"].detach().cpu()),
            "v61a_anchor_loss": float(v61a_anchor_stats["v61a_anchor_loss"].detach().cpu()),
            "v61a_pre_release_anchor_loss": float(v61a_anchor_stats["v61a_pre_release_anchor_loss"].detach().cpu()),
            "v61a_weighted_q_anchor_kl": float(v61a_anchor_stats["v61a_weighted_q_anchor_kl"].detach().cpu()),
            "v61a_pre_release_weighted_q_anchor_kl": float(v61a_anchor_stats["v61a_pre_release_weighted_q_anchor_kl"].detach().cpu()),
            "v61a_weighted_q_anchor_agreement": float(v61a_anchor_stats["v61a_weighted_q_anchor_agreement"].detach().cpu()),
            "v61a_reliability_mean": float(v61a_anchor_stats["v61a_reliability_mean"].detach().cpu()),
            "v61a_effective_anchor_mass": float(v61a_anchor_stats["v61a_effective_anchor_mass"].detach().cpu()),
            "v61a_base_reliability_mean": float(v61a_anchor_stats["v61a_base_reliability_mean"].detach().cpu()),
            "v61a_agreement_reliability_mean": float(v61a_anchor_stats["v61a_agreement_reliability_mean"].detach().cpu()),
            "v61a_guard_enabled": bool(v61a_guard_stats["v61a_guard_enabled"].detach().cpu()),
            "v61a_teacher_ready": bool(v61a_guard_stats["v61a_teacher_ready"].detach().cpu()),
            "v61a_teacher_epoch": float(v61a_guard_stats["v61a_teacher_epoch"].detach().cpu()),
            "v61a_guard_gamma": float(v61a_guard_stats["v61a_guard_gamma"].detach().cpu()),
            "v61a_guard_weight": float(v61a_guard_stats["v61a_guard_weight"].detach().cpu()),
            "v61a_absolute_floor": float(v61a_guard_stats["v61a_absolute_floor"].detach().cpu()),
            "v61a_min_teacher_coverage": float(v61a_guard_stats["v61a_min_teacher_coverage"].detach().cpu()),
            "v61a_teacher_confidence_mean": float(v61a_guard_stats["v61a_teacher_confidence_mean"].detach().cpu()),
            "v61a_teacher_active_ratio": float(v61a_guard_stats["v61a_teacher_active_ratio"].detach().cpu()),
            "v61a_teacher_floor_active_ratio": float(v61a_guard_stats["v61a_teacher_floor_active_ratio"].detach().cpu()),
            "v61a_teacher_topk_active_ratio": float(v61a_guard_stats["v61a_teacher_topk_active_ratio"].detach().cpu()),
            "v61a_guard_kl": float(v61a_guard_stats["v61a_guard_kl"].detach().cpu()),
            "v61a_guard_loss": float(v61a_guard_stats["v61a_guard_loss"].detach().cpu()),
            "v61a_q_teacher_agreement": float(v61a_guard_stats["v61a_q_teacher_agreement"].detach().cpu()),
            "v61a_q_teacher_kl": float(v61a_guard_stats["v61a_q_teacher_kl"].detach().cpu()),
            "v62a_enabled": bool(v62a_enabled),
            "v62a_release_gamma": float(v62a_anchor_stats["v62a_release_gamma"].detach().cpu()),
            "v62a_release_start_epoch": float(v62a_anchor_stats["v62a_release_start_epoch"].detach().cpu()),
            "v62a_release_decay_epochs": float(v62a_anchor_stats["v62a_release_decay_epochs"].detach().cpu()),
            "v62a_release_floor": float(v62a_anchor_stats["v62a_release_floor"].detach().cpu()),
            "v62a_gamma": float(v62a_anchor_stats["v62a_gamma"].detach().cpu()),
            "v62a_raw_reliability_mean": float(v62a_anchor_stats["v62a_raw_reliability_mean"].detach().cpu()),
            "v62a_mass_scale": float(v62a_anchor_stats["v62a_mass_scale"].detach().cpu()),
            "v62a_scaled_reliability_mean": float(v62a_anchor_stats["v62a_scaled_reliability_mean"].detach().cpu()),
            "v62a_anchor_loss": float(v62a_anchor_stats["v62a_anchor_loss"].detach().cpu()),
            "v62a_pre_release_anchor_loss": float(v62a_anchor_stats["v62a_pre_release_anchor_loss"].detach().cpu()),
            "v62a_weighted_q_anchor_kl": float(v62a_anchor_stats["v62a_weighted_q_anchor_kl"].detach().cpu()),
            "v62a_pre_release_weighted_q_anchor_kl": float(v62a_anchor_stats["v62a_pre_release_weighted_q_anchor_kl"].detach().cpu()),
            "v62a_weighted_q_anchor_agreement": float(v62a_anchor_stats["v62a_weighted_q_anchor_agreement"].detach().cpu()),
            "v62a_reliability_mean": float(v62a_anchor_stats["v62a_reliability_mean"].detach().cpu()),
            "v62a_effective_anchor_mass": float(v62a_anchor_stats["v62a_effective_anchor_mass"].detach().cpu()),
            "v62a_base_reliability_mean": float(v62a_anchor_stats["v62a_base_reliability_mean"].detach().cpu()),
            "v62a_agreement_reliability_mean": float(v62a_anchor_stats["v62a_agreement_reliability_mean"].detach().cpu()),
            "v62a_guard_enabled": bool(v62a_guard_stats["v62a_guard_enabled"].detach().cpu()),
            "v62a_teacher_ready": bool(v62a_guard_stats["v62a_teacher_ready"].detach().cpu()),
            "v62a_teacher_epoch": float(v62a_guard_stats["v62a_teacher_epoch"].detach().cpu()),
            "v62a_guard_gamma": float(v62a_guard_stats["v62a_guard_gamma"].detach().cpu()),
            "v62a_guard_weight": float(v62a_guard_stats["v62a_guard_weight"].detach().cpu()),
            "v62a_absolute_floor": float(v62a_guard_stats["v62a_absolute_floor"].detach().cpu()),
            "v62a_min_teacher_coverage": float(v62a_guard_stats["v62a_min_teacher_coverage"].detach().cpu()),
            "v62a_teacher_confidence_mean": float(v62a_guard_stats["v62a_teacher_confidence_mean"].detach().cpu()),
            "v62a_teacher_active_ratio": float(v62a_guard_stats["v62a_teacher_active_ratio"].detach().cpu()),
            "v62a_teacher_floor_active_ratio": float(v62a_guard_stats["v62a_teacher_floor_active_ratio"].detach().cpu()),
            "v62a_teacher_topk_active_ratio": float(v62a_guard_stats["v62a_teacher_topk_active_ratio"].detach().cpu()),
            "v62a_guard_kl": float(v62a_guard_stats["v62a_guard_kl"].detach().cpu()),
            "v62a_guard_loss": float(v62a_guard_stats["v62a_guard_loss"].detach().cpu()),
            "v62a_q_teacher_agreement": float(v62a_guard_stats["v62a_q_teacher_agreement"].detach().cpu()),
            "v62a_q_teacher_kl": float(v62a_guard_stats["v62a_q_teacher_kl"].detach().cpu()),
            "v62a_drift_score": float(v62a_guard_stats["v62a_drift_score"].detach().cpu()),
            "v62a_drift_gamma": float(v62a_guard_stats["v62a_drift_gamma"].detach().cpu()),
            "v62a_drift_floor": float(v62a_guard_stats["v62a_drift_floor"].detach().cpu()),
            "v62a_drift_scale": float(v62a_guard_stats["v62a_drift_scale"].detach().cpu()),
            "v62a_drift_boost": float(v62a_guard_stats["v62a_drift_boost"].detach().cpu()),
            "v62a_effective_guard_multiplier": float(v62a_guard_stats["v62a_effective_guard_multiplier"].detach().cpu()),
            "v62a_max_effective_guard_multiplier": float(v62a_guard_stats["v62a_max_effective_guard_multiplier"].detach().cpu()),
            "v71a_anchor_bypass_enabled": bool(v71a_enabled),
            "v71a_anchor_bypass_loss": float(v71a_anchor_bypass_loss.detach().cpu()),
            "v71a_anchor_bypass_gate_mean": float(v71a_stats["v71a_anchor_bypass_gate_mean"].detach().cpu()),
            "v71a_anchor_bypass_hard_consensus": float(v71a_stats["v71a_anchor_bypass_hard_consensus"].detach().cpu()),
            "v71a_anchor_bypass_soft_consensus": float(v71a_stats["v71a_anchor_bypass_soft_consensus"].detach().cpu()),
            "v71a_anchor_bypass_reliability_mean": float(v71a_stats["v71a_anchor_bypass_reliability_mean"].detach().cpu()),
            "v71a_anchor_bypass_release_gamma": float(v71a_stats["v71a_anchor_bypass_release_gamma"].detach().cpu()),
            "v67a_anchor_distrust_enabled": bool(v67a_enabled),
            "v67a_anchor_gate": float(v67a_anchor_gate.detach().cpu()),
            "v68a_teacher_boost_enabled": bool(v68a_enabled),
            "v68a_teacher_boost": float(v68a_teacher_boost.detach().cpu()),
            "v70a_entropy_guard_enabled": bool(v70a_enabled),
            "v70a_entropy_guard": float(v70a_entropy_guard_loss.detach().cpu()),
            "v70a_entropy_guard_gate": float(v70a_entropy_guard_gate.detach().cpu()),
            "v70a_normalized_entropy": float(v70a_normalized_entropy.detach().cpu()),
            "v63b_enabled": bool(v63b_enabled),
            "v63b_graph_noise": float(out["v63b_graph_noise"].detach().cpu()),
            "v63b_graph_gate": float(out["v63b_graph_gate"].detach().cpu()),
            "v63b_feature_rescue_mean": float(out["v63b_feature_rescue"].detach().mean().cpu()),
            "v63b_feature_rescue_active_ratio": float((out["v63b_feature_rescue"].detach() > 1e-6).to(out["score"].dtype).mean().cpu()),
            "v63b_high_weight_mean": float(out["v63b_high_weight"].detach().mean().cpu()),
            "v63b_edge_ood": float(v63b_edge_ood_loss.detach().cpu()),
            "v63b_edge_rank_gap": float(v63b_edge_stats["v63b_edge_rank_gap"].detach().cpu()),
            "v63b_edge_logit_gap": float(v63b_edge_stats["v63b_edge_logit_gap"].detach().cpu()),
            "v63b_edge_pos_score": float(v63b_edge_stats["v63b_edge_pos_score"].detach().cpu()),
            "v63b_edge_neg_score": float(v63b_edge_stats["v63b_edge_neg_score"].detach().cpu()),
            "v63b_edge_clean_mass": float(v63b_edge_stats["v63b_edge_clean_mass"].detach().cpu()),
            "v63b_edge_noise_mass": float(v63b_edge_stats["v63b_edge_noise_mass"].detach().cpu()),
            "v63b_edge_pairs": float(v63b_edge_stats["v63b_edge_pairs"].detach().cpu()),
            "v63b_confusion_guard": float(v63b_confusion_guard_loss.detach().cpu()),
            "v63b_guard_kl": float(v63b_guard_stats["v63b_guard_kl"].detach().cpu()),
            "v63b_guard_node_gate_mean": float(v63b_guard_stats["v63b_guard_node_gate_mean"].detach().cpu()),
            "v63b_guard_neighbor_agreement_mean": float(v63b_guard_stats["v63b_guard_neighbor_agreement_mean"].detach().cpu()),
            "v63b_guard_neighbor_coverage": float(v63b_guard_stats["v63b_guard_neighbor_coverage"].detach().cpu()),
            "v63b_guard_active_ratio": float(v63b_guard_stats["v63b_guard_active_ratio"].detach().cpu()),
            "v64a_enabled": bool(v64a_enabled),
            "v64a_subspace_gram": float(v64a_subspace_gram_loss.detach().cpu()),
            "v64a_subspace_gram_corr": float(v64a_stats["v64a_subspace_gram_corr"].detach().cpu()),
            "v64a_subspace_sample_size": float(v64a_stats["v64a_subspace_sample_size"].detach().cpu()),
            "v64a_subspace_gamma": float(v64a_stats["v64a_subspace_gamma"].detach().cpu()),
            "v64a_subspace_release": float(v64a_stats["v64a_subspace_release"].detach().cpu()),
            "v64a_subspace_anchor_dim": float(v64a_stats["v64a_subspace_anchor_dim"].detach().cpu()),
            "v86a_v64_gate": float(v86a_v64_gate.detach().cpu()),
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
        adj = as_csr(adj)
        if (
            int(getattr(cfg, "small_graph_cpu_max_nodes", 0)) > 0
            and adj.shape[0] <= int(getattr(cfg, "small_graph_cpu_max_nodes", 0))
        ):
            cfg.device = "cpu"
        device = resolve_device(cfg.device)
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
        if (
            bool(getattr(cfg, "v50a_enabled", False))
            or bool(getattr(cfg, "v51a_enabled", False))
            or bool(getattr(cfg, "v52a_enabled", False))
            or bool(getattr(cfg, "v53a_enabled", False))
            or bool(getattr(cfg, "v54a_enabled", False))
            or bool(getattr(cfg, "v55a_enabled", False))
            or bool(getattr(cfg, "v56a_enabled", False))
            or bool(getattr(cfg, "v57a_enabled", False))
            or bool(getattr(cfg, "v58a_enabled", False))
            or bool(getattr(cfg, "v59a_enabled", False))
            or bool(getattr(cfg, "v60a_enabled", False))
            or bool(getattr(cfg, "v61a_enabled", False))
            or bool(getattr(cfg, "v62a_enabled", False))
        ):
            v50a_source = str(getattr(cfg, "v50a_anchor_source", "spectral")).lower()
            if v50a_source in {"elss", "elss_anchor", "anchor_subspace"}:
                z_for_q_anchor = build_elss_anchor_subspace_embedding(x_np, graph_adj, self.n_clusters, cfg, device)
                q_anchor_np = soft_cluster_distribution_from_embedding(
                    z_for_q_anchor,
                    self.n_clusters,
                    cfg,
                    temperature=float(getattr(cfg, "v50a_anchor_temperature", 0.35)),
                )
            else:
                q_anchor_np = build_spectral_compactness_anchor(x_np, graph_adj, self.n_clusters, cfg)
            if q_anchor_np.shape == (x_np.shape[0], self.n_clusters):
                model.set_v50a_anchor(torch.as_tensor(q_anchor_np, dtype=torch.float32, device=device))
        if bool(getattr(cfg, "v64a_enabled", False)):
            v64a_source = str(getattr(cfg, "v64a_subspace_source", "spectral")).lower()
            if v64a_source in {"elss", "elss_anchor", "anchor_subspace"}:
                z_anchor_np = build_elss_anchor_subspace_embedding(x_np, graph_adj, self.n_clusters, cfg, device)
            else:
                z_anchor_np = build_spectral_subspace_embedding(x_np, graph_adj, self.n_clusters, cfg)
            if z_anchor_np.shape[0] == x_np.shape[0] and z_anchor_np.ndim == 2:
                model.set_v64a_subspace_anchor(torch.as_tensor(z_anchor_np, dtype=torch.float32, device=device))
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
        v72a_snapshot_state: dict[str, torch.Tensor] | None = None
        v72a_snapshot_epoch = -1
        v72a_rollback_used = False
        v72a_rollback_reason = ""
        v72a_rollback_anchor_agreement = 1.0
        v72a_rollback_teacher_agreement = 1.0
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
            if bool(getattr(cfg, "v60a_enabled", False)) and not bool(model.v60a_teacher_ready.detach().cpu()):
                v60a_start = int(getattr(cfg, "v60a_start_epoch", 80))
                if (epoch + 1) >= v60a_start:
                    model.eval()
                    with torch.no_grad():
                        snapshot_out = model(x)
                    model.train()
                    model.v60a_teacher_q = snapshot_out["q_refined"].detach().clone()
                    model.v60a_teacher_ready = torch.tensor(True, dtype=torch.bool, device=device)
                    model.v60a_teacher_epoch = torch.tensor(epoch + 1, dtype=torch.long, device=device)
                    teacher_conf = model.v60a_teacher_q.max(dim=1).values
                    threshold = float(getattr(cfg, "v60a_confidence_threshold", 0.60))
                    teacher_active = (teacher_conf >= threshold).to(teacher_conf.dtype)
                    diag["v60a_teacher_ready"] = True
                    diag["v60a_teacher_epoch"] = float(epoch + 1)
                    diag["v60a_teacher_confidence_mean"] = float(teacher_conf.mean().detach().cpu())
                    diag["v60a_teacher_active_ratio"] = float(teacher_active.mean().detach().cpu())
                    diag["v60a_q_teacher_agreement"] = 1.0
                    diag["v60a_q_teacher_kl"] = 0.0
            if bool(getattr(cfg, "v61a_enabled", False)) and not bool(model.v61a_teacher_ready.detach().cpu()):
                v61a_start = int(getattr(cfg, "v61a_start_epoch", 80))
                if (epoch + 1) >= v61a_start:
                    model.eval()
                    with torch.no_grad():
                        snapshot_out = model(x)
                    model.train()
                    model.v61a_teacher_q = snapshot_out["q_refined"].detach().clone()
                    model.v61a_teacher_ready = torch.tensor(True, dtype=torch.bool, device=device)
                    model.v61a_teacher_epoch = torch.tensor(epoch + 1, dtype=torch.long, device=device)
                    teacher_conf = model.v61a_teacher_q.max(dim=1).values
                    floor_value = float(getattr(cfg, "v61a_absolute_floor", 0.45))
                    coverage_value = float(np.clip(float(getattr(cfg, "v61a_min_teacher_coverage", 0.10)), 0.0, 1.0))
                    floor_active = teacher_conf >= floor_value
                    topk_active = torch.zeros_like(floor_active, dtype=torch.bool)
                    if teacher_conf.numel() > 0 and coverage_value > 0.0:
                        k = int(math.ceil(coverage_value * float(teacher_conf.numel())))
                        k = max(1, min(int(teacher_conf.numel()), k))
                        top_idx = torch.topk(teacher_conf, k=k, largest=True, sorted=False).indices
                        topk_active[top_idx] = True
                    teacher_active = (floor_active | topk_active).to(teacher_conf.dtype)
                    diag["v61a_teacher_ready"] = True
                    diag["v61a_teacher_epoch"] = float(epoch + 1)
                    diag["v61a_teacher_confidence_mean"] = float(teacher_conf.mean().detach().cpu())
                    diag["v61a_teacher_active_ratio"] = float(teacher_active.mean().detach().cpu())
                    diag["v61a_teacher_floor_active_ratio"] = float(floor_active.to(teacher_conf.dtype).mean().detach().cpu())
                    diag["v61a_teacher_topk_active_ratio"] = float(topk_active.to(teacher_conf.dtype).mean().detach().cpu())
                    diag["v61a_q_teacher_agreement"] = 1.0
                    diag["v61a_q_teacher_kl"] = 0.0
            if bool(getattr(cfg, "v62a_enabled", False)) and not bool(model.v62a_teacher_ready.detach().cpu()):
                v62a_start = int(getattr(cfg, "v62a_start_epoch", 80))
                if (epoch + 1) >= v62a_start:
                    model.eval()
                    with torch.no_grad():
                        snapshot_out = model(x)
                    model.train()
                    model.v62a_teacher_q = snapshot_out["q_refined"].detach().clone()
                    model.v62a_teacher_ready = torch.tensor(True, dtype=torch.bool, device=device)
                    model.v62a_teacher_epoch = torch.tensor(epoch + 1, dtype=torch.long, device=device)
                    teacher_conf = model.v62a_teacher_q.max(dim=1).values
                    floor_value = float(getattr(cfg, "v62a_absolute_floor", 0.45))
                    coverage_value = float(np.clip(float(getattr(cfg, "v62a_min_teacher_coverage", 0.10)), 0.0, 1.0))
                    floor_active = teacher_conf >= floor_value
                    topk_active = torch.zeros_like(floor_active, dtype=torch.bool)
                    if teacher_conf.numel() > 0 and coverage_value > 0.0:
                        k = int(math.ceil(coverage_value * float(teacher_conf.numel())))
                        k = max(1, min(int(teacher_conf.numel()), k))
                        top_idx = torch.topk(teacher_conf, k=k, largest=True, sorted=False).indices
                        topk_active[top_idx] = True
                    teacher_active = (floor_active | topk_active).to(teacher_conf.dtype)
                    diag["v62a_teacher_ready"] = True
                    diag["v62a_teacher_epoch"] = float(epoch + 1)
                    diag["v62a_teacher_confidence_mean"] = float(teacher_conf.mean().detach().cpu())
                    diag["v62a_teacher_active_ratio"] = float(teacher_active.mean().detach().cpu())
                    diag["v62a_teacher_floor_active_ratio"] = float(floor_active.to(teacher_conf.dtype).mean().detach().cpu())
                    diag["v62a_teacher_topk_active_ratio"] = float(topk_active.to(teacher_conf.dtype).mean().detach().cpu())
                    diag["v62a_q_teacher_agreement"] = 1.0
                    diag["v62a_q_teacher_kl"] = 0.0
                    if bool(getattr(cfg, "v72a_stability_rollback_enabled", False)):
                        v72a_snapshot_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                        v72a_snapshot_epoch = int(epoch + 1)
            if (epoch + 1) in {1, 20, 40, 60, 80, 100, 260}:
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
                    "v50a_q_anchor_agreement",
                    "v50a_q_anchor_kl",
                    "v51a_weighted_q_anchor_agreement",
                    "v51a_weighted_q_anchor_kl",
                    "v51a_reliability_mean",
                    "v52a_gamma",
                    "v52a_weighted_q_anchor_agreement",
                    "v52a_weighted_q_anchor_kl",
                    "v52a_reliability_mean",
                    "v52a_base_reliability_mean",
                    "v52a_agreement_reliability_mean",
                    "v53a_gamma",
                    "v53a_residual_multiplier_mean",
                    "v53a_weighted_q_anchor_agreement",
                    "v53a_weighted_q_anchor_kl",
                    "v53a_reliability_mean",
                    "v53a_base_reliability_mean",
                    "v53a_agreement_reliability_mean",
                    "v54a_gamma",
                    "v54a_beta_mean",
                    "v54a_residual_multiplier_mean",
                    "v54a_weighted_q_anchor_agreement",
                    "v54a_weighted_q_anchor_kl",
                    "v54a_reliability_mean",
                    "v54a_base_reliability_mean",
                    "v54a_agreement_reliability_mean",
                    "v55a_gamma",
                    "v55a_soft_consensus_mean",
                    "v55a_beta_mean",
                    "v55a_residual_multiplier_mean",
                    "v55a_weighted_q_anchor_agreement",
                    "v55a_weighted_q_anchor_kl",
                    "v55a_reliability_mean",
                    "v55a_base_reliability_mean",
                    "v55a_agreement_reliability_mean",
                    "v56a_gamma",
                    "v56a_hard_consensus_mean",
                    "v56a_soft_consensus_mean",
                    "v56a_lifted_soft_consensus_mean",
                    "v56a_compensation_mean",
                    "v56a_hybrid_consensus_mean",
                    "v56a_beta_mean",
                    "v56a_residual_multiplier_mean",
                    "v56a_weighted_q_anchor_agreement",
                    "v56a_weighted_q_anchor_kl",
                    "v56a_reliability_mean",
                    "v56a_base_reliability_mean",
                    "v56a_agreement_reliability_mean",
                    "v57a_gamma",
                    "v57a_hard_consensus_mean",
                    "v57a_soft_consensus_mean",
                    "v57a_lifted_soft_consensus_mean",
                    "v57a_compensation_mean",
                    "v57a_hybrid_consensus_mean",
                    "v57a_beta_mean",
                    "v57a_raw_reliability_mean",
                    "v57a_mass_scale",
                    "v57a_scaled_reliability_mean",
                    "v57a_residual_multiplier_mean",
                    "v57a_weighted_q_anchor_agreement",
                    "v57a_weighted_q_anchor_kl",
                    "v57a_reliability_mean",
                    "v57a_base_reliability_mean",
                    "v57a_agreement_reliability_mean",
                    "v58a_release_gamma",
                    "v58a_gamma",
                    "v58a_hard_consensus_mean",
                    "v58a_soft_consensus_mean",
                    "v58a_lifted_soft_consensus_mean",
                    "v58a_compensation_mean",
                    "v58a_hybrid_consensus_mean",
                    "v58a_beta_mean",
                    "v58a_raw_reliability_mean",
                    "v58a_mass_scale",
                    "v58a_scaled_reliability_mean",
                    "v58a_residual_multiplier_mean",
                    "v58a_weighted_q_anchor_agreement",
                    "v58a_weighted_q_anchor_kl",
                    "v58a_pre_release_weighted_q_anchor_kl",
                    "v58a_reliability_mean",
                    "v58a_base_reliability_mean",
                    "v58a_agreement_reliability_mean",
                    "v59a_release_gamma",
                    "v59a_gamma",
                    "v59a_hard_consensus_mean",
                    "v59a_soft_consensus_mean",
                    "v59a_lifted_soft_consensus_mean",
                    "v59a_compensation_mean",
                    "v59a_hybrid_consensus_mean",
                    "v59a_beta_mean",
                    "v59a_raw_reliability_mean",
                    "v59a_mass_scale",
                    "v59a_scaled_reliability_mean",
                    "v59a_residual_multiplier_mean",
                    "v59a_weighted_q_anchor_agreement",
                    "v59a_weighted_q_anchor_kl",
                    "v59a_pre_release_weighted_q_anchor_kl",
                    "v59a_reliability_mean",
                    "v59a_base_reliability_mean",
                    "v59a_agreement_reliability_mean",
                    "v60a_release_gamma",
                    "v60a_gamma",
                    "v60a_raw_reliability_mean",
                    "v60a_mass_scale",
                    "v60a_scaled_reliability_mean",
                    "v60a_weighted_q_anchor_agreement",
                    "v60a_weighted_q_anchor_kl",
                    "v60a_pre_release_weighted_q_anchor_kl",
                    "v60a_reliability_mean",
                    "v60a_base_reliability_mean",
                    "v60a_agreement_reliability_mean",
                    "v60a_guard_gamma",
                    "v60a_teacher_ready",
                    "v60a_teacher_active_ratio",
                    "v60a_q_teacher_agreement",
                    "v60a_guard_loss",
                    "v60a_guard_kl",
                    "v61a_release_gamma",
                    "v61a_gamma",
                    "v61a_raw_reliability_mean",
                    "v61a_mass_scale",
                    "v61a_scaled_reliability_mean",
                    "v61a_weighted_q_anchor_agreement",
                    "v61a_weighted_q_anchor_kl",
                    "v61a_pre_release_weighted_q_anchor_kl",
                    "v61a_reliability_mean",
                    "v61a_base_reliability_mean",
                    "v61a_agreement_reliability_mean",
                    "v61a_guard_gamma",
                    "v61a_teacher_ready",
                    "v61a_teacher_active_ratio",
                    "v61a_teacher_floor_active_ratio",
                    "v61a_teacher_topk_active_ratio",
                    "v61a_q_teacher_agreement",
                    "v61a_guard_loss",
                    "v61a_guard_kl",
                    "v62a_release_gamma",
                    "v62a_gamma",
                    "v62a_raw_reliability_mean",
                    "v62a_mass_scale",
                    "v62a_scaled_reliability_mean",
                    "v62a_weighted_q_anchor_agreement",
                    "v62a_weighted_q_anchor_kl",
                    "v62a_pre_release_weighted_q_anchor_kl",
                    "v62a_reliability_mean",
                    "v62a_base_reliability_mean",
                    "v62a_agreement_reliability_mean",
                    "v62a_guard_gamma",
                    "v62a_teacher_ready",
                    "v62a_teacher_active_ratio",
                    "v62a_teacher_floor_active_ratio",
                    "v62a_teacher_topk_active_ratio",
                    "v62a_q_teacher_agreement",
                    "v62a_guard_loss",
                    "v62a_guard_kl",
                    "v62a_drift_score",
                    "v62a_drift_gamma",
                    "v62a_effective_guard_multiplier",
                ):
                    if key in diag:
                        frontend_snapshots[f"{key}_{suffix}"] = float(diag[key])
            if bool(cfg.extras.get("verbose_training", False)) and (epoch % 50 == 0 or epoch + 1 == cfg.epochs):
                LOGGER.info("[%s] epoch=%d loss=%.4f H=%.3f L=%.3f", cfg.name, epoch, diag["loss"], diag["high_threshold"], diag["low_threshold"])

        if bool(getattr(cfg, "v72a_stability_rollback_enabled", False)) and v72a_snapshot_state is not None:
            v72a_rollback_anchor_agreement = float(last_diag.get("v62a_weighted_q_anchor_agreement", 1.0))
            v72a_rollback_teacher_agreement = float(last_diag.get("v62a_q_teacher_agreement", 1.0))
            anchor_max = float(getattr(cfg, "v72a_rollback_anchor_agreement_max", 0.55))
            teacher_max = float(getattr(cfg, "v72a_rollback_teacher_agreement_max", 0.92))
            if v72a_rollback_anchor_agreement <= anchor_max and v72a_rollback_teacher_agreement <= teacher_max:
                model.load_state_dict(v72a_snapshot_state)
                v72a_rollback_used = True
                v72a_rollback_reason = (
                    f"anchor_agreement={v72a_rollback_anchor_agreement:.4f},"
                    f"teacher_agreement={v72a_rollback_teacher_agreement:.4f}"
                )

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
            _v74a_readout_active = False
            _v74a_readout_weight_mean = 0.0
            _v74a_readout_weight_min = 0.0
            _v74a_readout_weight_max = 0.0
            _v74a_readout_nc_mean = 0.0
            _v74a_readout_conf_mean = 0.0
            _v75a_anchor_readout_used = False
            _v75a_anchor_readout_agreement = float(last_diag.get("v62a_weighted_q_anchor_agreement", 0.0))
            _v75a_anchor_readout_cluster_separation = float(last_diag.get("cluster_separation", 0.0))
            _v78a_anchor_smoothing_changed_ratio = 0.0
            _v78a_anchor_smoothing_candidate_ratio = 0.0
            _v78a_anchor_smoothing_majority_mean = 0.0
            _v80a_anchor_smoothing_vote_mean = 0.0
            _v82a_anchor_diffusion_changed_ratio = 0.0
            _v82a_anchor_diffusion_candidate_ratio = 0.0
            _v82a_anchor_diffusion_margin_mean = 0.0
            _v79a_consensus_smoothing_changed_ratio = 0.0
            _v79a_consensus_smoothing_candidate_ratio = 0.0
            _v79a_consensus_smoothing_vote_mean = 0.0
            _v79a_consensus_smoothing_majority_mean = 0.0
            _v83a_final_neighbor_smoothing_changed_ratio = 0.0
            _v83a_final_neighbor_smoothing_candidate_ratio = 0.0
            _v83a_final_neighbor_smoothing_majority_mean = 0.0
            _v84a_raw_embedding_readout_used = False
            _v91a_spectral_anchor_readout_used = False
            _v91a_spectral_anchor_conf_mean = 0.0
            _v91a_spectral_anchor_entropy = 1.0
            _v91a_spectral_anchor_balance = 0.0
            _v93a_raw_feature_svd_readout_used = False
            _v93a_raw_feature_svd_sil = -2.0
            _v93a_raw_feature_svd_balance = 0.0
            _v98a_legacy_subspace_readout_used = False
            _v98a_legacy_subspace_pre_gate = False
            _v98a_legacy_subspace_sil = -2.0
            _v98a_legacy_subspace_balance = 0.0
            _v98a_legacy_subspace_hard_ratio = float(last_diag.get("hard_ratio", 0.0))
            _v98a_legacy_subspace_error = ""
            _v99a_fast_elss_readout_used = False
            _v99a_fast_elss_pre_gate = False
            _v99a_fast_elss_sil = -2.0
            _v99a_fast_elss_balance = 0.0
            _v99a_fast_elss_error = ""
            _v100a_embedding_svd_readout_used = False
            _v100a_embedding_svd_sil = -2.0
            _v100a_embedding_svd_balance = 0.0
            _v105a_size_pressure_used = False
            _v105a_size_pressure_source_ratio = 0.0
            _v105a_size_pressure_changed_ratio = 0.0
            _v105a_size_pressure_nmove = 0
            _readout_kmeans_labels = None
            try:
                from sklearn.preprocessing import normalize as _sk_norm
                from sklearn.metrics import silhouette_score as _sil
                _emb_np = out["embedding"].detach().cpu().float().numpy()
                _emb_np = _sk_norm(_emb_np, norm="l2", axis=1)
                _N = _emb_np.shape[0]
                _v74a_weight = None
                if bool(getattr(cfg, "v74a_nc_weighted_readout_enabled", False)) and _N > 0 and _n_cl > 1:
                    _q_np = out["q_refined"].detach().cpu().float().numpy()
                    _pseudo = _q_np.argmax(axis=1).astype(np.int64)
                    _conf = _q_np.max(axis=1).astype(np.float32)
                    _hist = np.zeros((_N, _n_cl), dtype=np.float32)
                    np.add.at(_hist, (np.arange(_N), _pseudo), 1.0)
                    _ei = np.asarray(edge_index_np, dtype=np.int64)
                    if _ei.ndim == 2 and _ei.shape[0] == 2 and _ei.shape[1] > 0:
                        _src = np.clip(_ei[0], 0, _N - 1)
                        _dst = np.clip(_ei[1], 0, _N - 1)
                        np.add.at(_hist, (_src, _pseudo[_dst]), 1.0)
                        np.add.at(_hist, (_dst, _pseudo[_src]), 1.0)
                    _den = _hist.sum(axis=1, keepdims=True).clip(min=1e-6)
                    _pmax = (_hist / _den).max(axis=1).clip(min=1e-6, max=1.0)
                    _nc = (-np.log(_pmax) / max(1e-6, math.log(float(_n_cl)))).clip(0.0, 1.0)
                    _clarity = (1.0 - _nc).clip(0.0, 1.0)
                    _floor = float(np.clip(getattr(cfg, "v74a_readout_weight_floor", 0.20), 0.0, 1.0))
                    _conf_power = float(max(0.0, getattr(cfg, "v74a_readout_conf_power", 1.0)))
                    _clarity_power = float(max(0.0, getattr(cfg, "v74a_readout_clarity_power", 1.5)))
                    _v74a_weight = _floor + (1.0 - _floor) * np.power(_conf, _conf_power) * np.power(_clarity, _clarity_power)
                    _v74a_weight = np.nan_to_num(_v74a_weight, nan=_floor, posinf=1.0, neginf=_floor).astype(np.float64)
                    _v74a_readout_active = True
                    _v74a_readout_weight_mean = float(_v74a_weight.mean())
                    _v74a_readout_weight_min = float(_v74a_weight.min())
                    _v74a_readout_weight_max = float(_v74a_weight.max())
                    _v74a_readout_nc_mean = float(_nc.mean())
                    _v74a_readout_conf_mean = float(_conf.mean())
                _n_init = min(int(cfg.kmeans_n_init), 40)
                _km_full = KMeans(
                    n_clusters=_n_cl,
                    n_init=_n_init,
                    random_state=int(cfg.seed),
                    max_iter=300,
                )
                if _v74a_weight is None:
                    _lab_full = _km_full.fit_predict(_emb_np).astype(np.int64)
                else:
                    _km_full.fit(_emb_np, sample_weight=_v74a_weight)
                    _lab_full = _km_full.predict(_emb_np).astype(np.int64)
                _readout_kmeans_labels = _lab_full.copy()
                _, _, _Vt = np.linalg.svd(_emb_np, full_matrices=False)
                _Z_sub = _sk_norm(_emb_np @ _Vt[:_sub_dim].T, norm="l2", axis=1)
                _km_sub = KMeans(
                    n_clusters=_n_cl,
                    n_init=_n_init,
                    random_state=int(cfg.seed),
                    max_iter=300,
                )
                if _v74a_weight is None:
                    _lab_sub = _km_sub.fit_predict(_Z_sub).astype(np.int64)
                else:
                    _km_sub.fit(_Z_sub, sample_weight=_v74a_weight)
                    _lab_sub = _km_sub.predict(_Z_sub).astype(np.int64)
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
            if (
                bool(getattr(cfg, "v75a_reliable_anchor_readout_enabled", False))
                and model.v50a_anchor_q.numel() == out["q_refined"].numel()
                and _v75a_anchor_readout_agreement >= float(getattr(cfg, "v75a_anchor_readout_agreement_min", 0.95))
                and _v75a_anchor_readout_cluster_separation
                >= float(getattr(cfg, "v75a_anchor_readout_cluster_separation_min", 0.0))
            ):
                _anchor_labels = model.v50a_anchor_q.argmax(dim=1).detach().cpu().numpy().astype(np.int64)
                if np.unique(_anchor_labels).shape[0] >= 2:
                    labels = _anchor_labels
                    _postproc_choice = "elss_anchor_readout"
                    _v75a_anchor_readout_used = True
                    if (
                        bool(getattr(cfg, "v78a_anchor_smoothing_enabled", False))
                        and _v75a_anchor_readout_agreement
                        >= float(getattr(cfg, "v78a_anchor_smoothing_agreement_min", 0.0))
                    ):
                        _q_anchor_np = model.v50a_anchor_q.detach().cpu().float().numpy()
                        _anchor_conf = _q_anchor_np.max(axis=1)
                        _n_nodes = int(_anchor_labels.shape[0])
                        _n_classes = int(out["q_refined"].shape[1])
                        _hist = np.zeros((_n_nodes, _n_classes), dtype=np.float32)
                        _ei = np.asarray(edge_index_np, dtype=np.int64)
                        if _ei.ndim == 2 and _ei.shape[0] == 2 and _ei.shape[1] > 0:
                            _src = np.clip(_ei[0], 0, _n_nodes - 1)
                            _dst = np.clip(_ei[1], 0, _n_nodes - 1)
                            np.add.at(_hist, (_src, _anchor_labels[_dst]), 1.0)
                            np.add.at(_hist, (_dst, _anchor_labels[_src]), 1.0)
                        _deg = _hist.sum(axis=1)
                        _majority = _hist.argmax(axis=1).astype(np.int64)
                        _majority_ratio = np.divide(
                            _hist.max(axis=1),
                            np.maximum(_deg, 1e-6),
                            out=np.zeros(_n_nodes, dtype=np.float32),
                            where=_deg > 0,
                        )
                        _conf_max = float(np.clip(getattr(cfg, "v78a_anchor_smoothing_conf_max", 0.95), 0.0, 1.0))
                        _maj_min = float(np.clip(getattr(cfg, "v78a_anchor_smoothing_majority_min", 0.70), 0.0, 1.0))
                        _candidate = (
                            (_deg > 0)
                            & (_majority != _anchor_labels)
                            & (_anchor_conf <= _conf_max)
                            & (_majority_ratio >= _maj_min)
                        )
                        _min_votes = int(np.clip(getattr(cfg, "v80a_anchor_smoothing_min_votes", 0), 0, 2))
                        if _min_votes > 0:
                            _q_labels = out["q_refined"].argmax(dim=1).detach().cpu().numpy().astype(np.int64)
                            _q_aligned = _align_labels_to_reference(_q_labels, _anchor_labels, _n_classes)
                            if _readout_kmeans_labels is None:
                                _km_aligned = _q_aligned
                            else:
                                _km_aligned = _align_labels_to_reference(
                                    _readout_kmeans_labels, _anchor_labels, _n_classes
                                )
                            _votes = (_q_aligned == _majority).astype(np.int32) + (
                                _km_aligned == _majority
                            ).astype(np.int32)
                            _candidate = _candidate & (_votes >= _min_votes)
                            _v80a_anchor_smoothing_vote_mean = (
                                float(_votes[_candidate].mean()) if np.any(_candidate) else 0.0
                            )
                        _v78a_anchor_smoothing_candidate_ratio = float(_candidate.mean()) if _candidate.size else 0.0
                        _v78a_anchor_smoothing_majority_mean = (
                            float(_majority_ratio[_candidate].mean()) if np.any(_candidate) else 0.0
                        )
                        _max_ratio = float(np.clip(getattr(cfg, "v78a_anchor_smoothing_max_change_ratio", 0.02), 0.0, 1.0))
                        _max_changes = int(math.floor(_max_ratio * float(_n_nodes)))
                        if _max_changes > 0 and np.any(_candidate):
                            _cand_idx = np.flatnonzero(_candidate)
                            _order = np.argsort(_anchor_conf[_cand_idx], kind="stable")
                            _chosen = _cand_idx[_order[: min(_max_changes, _cand_idx.shape[0])]]
                            _smoothed = labels.copy()
                            _smoothed[_chosen] = _majority[_chosen]
                            if np.unique(_smoothed).shape[0] >= 2:
                                labels = _smoothed
                                _postproc_choice = "elss_anchor_smoothed_readout"
                                _v78a_anchor_smoothing_changed_ratio = float(_chosen.shape[0]) / float(max(1, _n_nodes))
                    if (
                        bool(getattr(cfg, "v82a_anchor_diffusion_smoothing_enabled", False))
                        and _v75a_anchor_readout_agreement
                        >= float(getattr(cfg, "v82a_anchor_diffusion_agreement_min", 0.97))
                    ):
                        _q_anchor_np = model.v50a_anchor_q.detach().cpu().float().numpy()
                        _anchor_conf = _q_anchor_np.max(axis=1)
                        _n_nodes = int(labels.shape[0])
                        _n_classes = int(out["q_refined"].shape[1])
                        _base_y = np.zeros((_n_nodes, _n_classes), dtype=np.float32)
                        _base_y[np.arange(_n_nodes), np.clip(labels, 0, _n_classes - 1)] = 1.0
                        _y = _base_y.copy()
                        _ei = np.asarray(edge_index_np, dtype=np.int64)
                        _src = np.empty(0, dtype=np.int64)
                        _dst = np.empty(0, dtype=np.int64)
                        if _ei.ndim == 2 and _ei.shape[0] == 2 and _ei.shape[1] > 0:
                            _src = np.clip(_ei[0], 0, _n_nodes - 1)
                            _dst = np.clip(_ei[1], 0, _n_nodes - 1)
                        _restart = float(np.clip(getattr(cfg, "v82a_anchor_diffusion_restart", 0.65), 0.0, 1.0))
                        _steps = int(max(1, getattr(cfg, "v82a_anchor_diffusion_steps", 2)))
                        _deg = np.ones(_n_nodes, dtype=np.float32)
                        if _src.size:
                            np.add.at(_deg, _src, 1.0)
                            np.add.at(_deg, _dst, 1.0)
                        for _ in range(_steps):
                            _msg = _y.copy()
                            if _src.size:
                                np.add.at(_msg, _src, _y[_dst])
                                np.add.at(_msg, _dst, _y[_src])
                            _neigh = _msg / np.maximum(_deg[:, None], 1e-6)
                            _y = _restart * _base_y + (1.0 - _restart) * _neigh
                        _prop_label = _y.argmax(axis=1).astype(np.int64)
                        if _n_classes > 1:
                            _part = np.partition(_y, -2, axis=1)
                            _margin = (_part[:, -1] - _part[:, -2]).astype(np.float32)
                        else:
                            _margin = np.ones(_n_nodes, dtype=np.float32)
                        _conf_max = float(np.clip(getattr(cfg, "v82a_anchor_diffusion_conf_max", 0.98), 0.0, 1.0))
                        _margin_min = float(max(0.0, getattr(cfg, "v82a_anchor_diffusion_margin_min", 0.20)))
                        _candidate = (
                            (_prop_label != labels)
                            & (_anchor_conf <= _conf_max)
                            & (_margin >= _margin_min)
                        )
                        _v82a_anchor_diffusion_candidate_ratio = float(_candidate.mean()) if _candidate.size else 0.0
                        _v82a_anchor_diffusion_margin_mean = (
                            float(_margin[_candidate].mean()) if np.any(_candidate) else 0.0
                        )
                        _max_ratio = float(
                            np.clip(getattr(cfg, "v82a_anchor_diffusion_max_change_ratio", 0.01), 0.0, 1.0)
                        )
                        _max_changes = int(math.floor(_max_ratio * float(_n_nodes)))
                        if _max_changes > 0 and np.any(_candidate):
                            _cand_idx = np.flatnonzero(_candidate)
                            _score = _margin[_cand_idx] + 0.02 * (1.0 - _anchor_conf[_cand_idx])
                            _order = np.argsort(-_score, kind="stable")
                            _chosen = _cand_idx[_order[: min(_max_changes, _cand_idx.shape[0])]]
                            _smoothed = labels.copy()
                            _smoothed[_chosen] = _prop_label[_chosen]
                            if np.unique(_smoothed).shape[0] >= 2:
                                labels = _smoothed
                                _postproc_choice = "elss_anchor_diffusion_smoothed_readout"
                                _v82a_anchor_diffusion_changed_ratio = (
                                    float(_chosen.shape[0]) / float(max(1, _n_nodes))
                                )
                    if (
                        bool(getattr(cfg, "v79a_consensus_smoothing_enabled", False))
                        and _v75a_anchor_readout_agreement
                        >= float(getattr(cfg, "v79a_consensus_agreement_min", 0.97))
                    ):
                        _q_anchor_np = model.v50a_anchor_q.detach().cpu().float().numpy()
                        _anchor_conf = _q_anchor_np.max(axis=1)
                        _n_nodes = int(labels.shape[0])
                        _n_classes = int(out["q_refined"].shape[1])
                        _hist = np.zeros((_n_nodes, _n_classes), dtype=np.float32)
                        _ei = np.asarray(edge_index_np, dtype=np.int64)
                        if _ei.ndim == 2 and _ei.shape[0] == 2 and _ei.shape[1] > 0:
                            _src = np.clip(_ei[0], 0, _n_nodes - 1)
                            _dst = np.clip(_ei[1], 0, _n_nodes - 1)
                            np.add.at(_hist, (_src, labels[_dst]), 1.0)
                            np.add.at(_hist, (_dst, labels[_src]), 1.0)
                        _deg = _hist.sum(axis=1)
                        _majority = _hist.argmax(axis=1).astype(np.int64)
                        _majority_ratio = np.divide(
                            _hist.max(axis=1),
                            np.maximum(_deg, 1e-6),
                            out=np.zeros(_n_nodes, dtype=np.float32),
                            where=_deg > 0,
                        )
                        _q_labels = out["q_refined"].argmax(dim=1).detach().cpu().numpy().astype(np.int64)
                        _q_aligned = _align_labels_to_reference(_q_labels, labels, _n_classes)
                        if _readout_kmeans_labels is None:
                            _km_aligned = _q_aligned
                        else:
                            _km_aligned = _align_labels_to_reference(_readout_kmeans_labels, labels, _n_classes)
                        _votes = (_q_aligned == _majority).astype(np.int32) + (
                            _km_aligned == _majority
                        ).astype(np.int32)
                        _conf_max = float(np.clip(getattr(cfg, "v79a_consensus_conf_max", 0.98), 0.0, 1.0))
                        _maj_min = float(np.clip(getattr(cfg, "v79a_consensus_majority_min", 0.62), 0.0, 1.0))
                        _min_votes = int(np.clip(getattr(cfg, "v79a_consensus_min_votes", 1), 1, 2))
                        _candidate = (
                            (_deg > 0)
                            & (_majority != labels)
                            & (_anchor_conf <= _conf_max)
                            & (_majority_ratio >= _maj_min)
                            & (_votes >= _min_votes)
                        )
                        _v79a_consensus_smoothing_candidate_ratio = (
                            float(_candidate.mean()) if _candidate.size else 0.0
                        )
                        _v79a_consensus_smoothing_vote_mean = (
                            float(_votes[_candidate].mean()) if np.any(_candidate) else 0.0
                        )
                        _v79a_consensus_smoothing_majority_mean = (
                            float(_majority_ratio[_candidate].mean()) if np.any(_candidate) else 0.0
                        )
                        _max_ratio = float(
                            np.clip(getattr(cfg, "v79a_consensus_max_change_ratio", 0.006), 0.0, 1.0)
                        )
                        _max_changes = int(math.floor(_max_ratio * float(_n_nodes)))
                        if _max_changes > 0 and np.any(_candidate):
                            _cand_idx = np.flatnonzero(_candidate)
                            _score = (
                                _majority_ratio[_cand_idx]
                                + 0.04 * _votes[_cand_idx].astype(np.float32)
                                + 0.02 * (1.0 - _anchor_conf[_cand_idx])
                            )
                            _order = np.argsort(-_score, kind="stable")
                            _chosen = _cand_idx[_order[: min(_max_changes, _cand_idx.shape[0])]]
                            _smoothed = labels.copy()
                            _smoothed[_chosen] = _majority[_chosen]
                            if np.unique(_smoothed).shape[0] >= 2:
                                labels = _smoothed
                                _postproc_choice = "elss_anchor_consensus_smoothed_readout"
                                _v79a_consensus_smoothing_changed_ratio = (
                                    float(_chosen.shape[0]) / float(max(1, _n_nodes))
                                )
            if bool(getattr(cfg, "v91a_spectral_anchor_readout_enabled", False)):
                _old_filter_steps = int(getattr(cfg, "v50a_filter_steps", 2))
                _old_rank_multiplier = float(getattr(cfg, "v50a_anchor_rank_multiplier", 1.0))
                try:
                    cfg.v50a_filter_steps = int(getattr(cfg, "v91a_spectral_anchor_filter_steps", 5))
                    cfg.v50a_anchor_rank_multiplier = float(
                        getattr(cfg, "v91a_spectral_anchor_rank_multiplier", 4.0)
                    )
                    _spectral_q = build_spectral_compactness_anchor(x_np, graph_adj, self.n_clusters, cfg)
                finally:
                    cfg.v50a_filter_steps = _old_filter_steps
                    cfg.v50a_anchor_rank_multiplier = _old_rank_multiplier
                if _spectral_q.shape == (x_np.shape[0], self.n_clusters):
                    _spectral_labels = _spectral_q.argmax(axis=1).astype(np.int64)
                    _spectral_conf = _spectral_q.max(axis=1)
                    _v91a_spectral_anchor_conf_mean = float(_spectral_conf.mean()) if _spectral_conf.size else 0.0
                    if self.n_clusters > 1 and _spectral_q.size:
                        _spectral_entropy = -np.sum(
                            _spectral_q * np.log(np.clip(_spectral_q, 1e-8, None)), axis=1
                        )
                        _v91a_spectral_anchor_entropy = float(
                            _spectral_entropy.mean() / max(1e-8, math.log(float(self.n_clusters)))
                        )
                    _counts = np.bincount(_spectral_labels, minlength=self.n_clusters).astype(np.float64)
                    _counts = _counts / max(1.0, float(_spectral_labels.shape[0]))
                    _positive = _counts[_counts > 0.0]
                    _v91a_spectral_anchor_balance = (
                        float(_positive.min() / max(_positive.max(), 1e-8)) if _positive.size else 0.0
                    )
                    if (
                        np.unique(_spectral_labels).shape[0] >= 2
                        and _v91a_spectral_anchor_conf_mean
                        >= float(getattr(cfg, "v91a_spectral_anchor_conf_min", 0.70))
                        and _v91a_spectral_anchor_entropy
                        <= float(getattr(cfg, "v91a_spectral_anchor_entropy_max", 0.70))
                        and _v91a_spectral_anchor_balance
                        >= float(getattr(cfg, "v91a_spectral_anchor_balance_min", 0.40))
                    ):
                        labels = _spectral_labels
                        _postproc_choice = "spectral_anchor_readout"
                        _v91a_spectral_anchor_readout_used = True
            if (
                bool(getattr(cfg, "v93a_raw_feature_svd_readout_enabled", False))
                and not bool(_v75a_anchor_readout_used)
                and not bool(_v91a_spectral_anchor_readout_used)
                and _v75a_anchor_readout_agreement
                <= float(getattr(cfg, "v93a_raw_feature_svd_anchor_agreement_max", 0.85))
            ):
                _raw_x = normalize(np.nan_to_num(np.asarray(x_np, dtype=np.float32)), norm="l2", axis=1)
                _svd_dim = int(max(1, getattr(cfg, "v93a_raw_feature_svd_dim", 64)))
                _svd_dim = min(_svd_dim, max(1, min(_raw_x.shape) - 1))
                if _svd_dim > 0 and _svd_dim < min(_raw_x.shape):
                    _raw_z = TruncatedSVD(n_components=_svd_dim, random_state=int(cfg.seed)).fit_transform(_raw_x)
                else:
                    _raw_z = _raw_x
                _raw_z = normalize(np.nan_to_num(np.asarray(_raw_z, dtype=np.float32)), norm="l2", axis=1)
                _raw_km = KMeans(
                    n_clusters=_n_cl,
                    n_init=min(max(10, int(getattr(cfg, "kmeans_n_init", 10))), 40),
                    random_state=int(cfg.seed),
                    max_iter=300,
                )
                _raw_labels = _raw_km.fit_predict(_raw_z).astype(np.int64)
                _counts = np.bincount(_raw_labels, minlength=_n_cl).astype(np.float64)
                _counts = _counts / max(1.0, float(_raw_labels.shape[0]))
                _positive = _counts[_counts > 0.0]
                _v93a_raw_feature_svd_balance = (
                    float(_positive.min() / max(_positive.max(), 1e-8)) if _positive.size else 0.0
                )
                _sil_n = min(3000, int(_raw_z.shape[0]))
                _rng = np.random.default_rng(int(cfg.seed))
                _idx = _rng.choice(_raw_z.shape[0], _sil_n, replace=False) if _raw_z.shape[0] > _sil_n else np.arange(_raw_z.shape[0])
                if len(np.unique(_raw_labels[_idx])) >= 2:
                    _v93a_raw_feature_svd_sil = float(silhouette_score(_raw_z[_idx], _raw_labels[_idx]))
                if (
                    np.unique(_raw_labels).shape[0] >= 2
                    and _v93a_raw_feature_svd_sil >= float(getattr(cfg, "v93a_raw_feature_svd_sil_min", 0.05))
                    and _v93a_raw_feature_svd_balance
                    >= float(getattr(cfg, "v93a_raw_feature_svd_balance_min", 0.10))
                ):
                    labels = _raw_labels
                    _postproc_choice = "raw_feature_svd_readout"
                    _v93a_raw_feature_svd_readout_used = True
            _v99a_fast_elss_pre_gate = (
                bool(getattr(cfg, "v99a_fast_elss_readout_enabled", False))
                and not bool(_v93a_raw_feature_svd_readout_used)
                and (
                    not bool(getattr(cfg, "v99a_fast_elss_require_spectral_anchor", True))
                    or bool(_v91a_spectral_anchor_readout_used)
                )
            )
            if _v99a_fast_elss_pre_gate:
                try:
                    _fast_labels, _fast_emb = run_legacy_fast_elss_head(
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
                    _fast_labels = np.asarray(_fast_labels, dtype=np.int64)
                    _fast_emb = normalize(
                        np.nan_to_num(np.asarray(_fast_emb, dtype=np.float32)),
                        norm="l2",
                        axis=1,
                    )
                    _counts = np.bincount(_fast_labels, minlength=_n_cl).astype(np.float64)
                    _counts = _counts / max(1.0, float(_fast_labels.shape[0]))
                    _positive = _counts[_counts > 0.0]
                    _v99a_fast_elss_balance = (
                        float(_positive.min() / max(_positive.max(), 1e-8)) if _positive.size else 0.0
                    )
                    _sil_n = min(3000, int(_fast_emb.shape[0]))
                    _rng = np.random.default_rng(int(cfg.seed))
                    _idx = (
                        _rng.choice(_fast_emb.shape[0], _sil_n, replace=False)
                        if _fast_emb.shape[0] > _sil_n
                        else np.arange(_fast_emb.shape[0])
                    )
                    if len(np.unique(_fast_labels[_idx])) >= 2:
                        _v99a_fast_elss_sil = float(silhouette_score(_fast_emb[_idx], _fast_labels[_idx]))
                    if (
                        np.unique(_fast_labels).shape[0] >= 2
                        and _v99a_fast_elss_sil >= float(getattr(cfg, "v99a_fast_elss_sil_min", 0.45))
                        and _v99a_fast_elss_balance
                        >= float(getattr(cfg, "v99a_fast_elss_balance_min", 0.35))
                    ):
                        labels = _fast_labels
                        emb = _fast_emb.astype(np.float32)
                        _postproc_choice = "fast_elss_readout"
                        _v99a_fast_elss_readout_used = True
                except Exception as _e:
                    _v99a_fast_elss_error = repr(_e)
            _v98a_legacy_subspace_pre_gate = (
                bool(getattr(cfg, "v98a_gated_legacy_subspace_readout_enabled", False))
                and bool(_v75a_anchor_readout_used)
                and not bool(_v91a_spectral_anchor_readout_used)
                and not bool(_v93a_raw_feature_svd_readout_used)
                and not bool(_v99a_fast_elss_readout_used)
                and _v75a_anchor_readout_agreement
                >= float(getattr(cfg, "v98a_legacy_anchor_agreement_min", 0.95))
                and _v98a_legacy_subspace_hard_ratio
                >= float(getattr(cfg, "v98a_legacy_hard_ratio_min", 0.75))
            )
            if _v98a_legacy_subspace_pre_gate:
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
                    _legacy_labels = np.asarray(_legacy_labels, dtype=np.int64)
                    _legacy_emb = normalize(
                        np.nan_to_num(np.asarray(_legacy_emb, dtype=np.float32)),
                        norm="l2",
                        axis=1,
                    )
                    _counts = np.bincount(_legacy_labels, minlength=_n_cl).astype(np.float64)
                    _counts = _counts / max(1.0, float(_legacy_labels.shape[0]))
                    _positive = _counts[_counts > 0.0]
                    _v98a_legacy_subspace_balance = (
                        float(_positive.min() / max(_positive.max(), 1e-8)) if _positive.size else 0.0
                    )
                    _sil_n = min(3000, int(_legacy_emb.shape[0]))
                    _rng = np.random.default_rng(int(cfg.seed))
                    _idx = (
                        _rng.choice(_legacy_emb.shape[0], _sil_n, replace=False)
                        if _legacy_emb.shape[0] > _sil_n
                        else np.arange(_legacy_emb.shape[0])
                    )
                    if len(np.unique(_legacy_labels[_idx])) >= 2:
                        _v98a_legacy_subspace_sil = float(
                            silhouette_score(_legacy_emb[_idx], _legacy_labels[_idx])
                        )
                    if (
                        np.unique(_legacy_labels).shape[0] >= 2
                        and _v98a_legacy_subspace_sil
                        >= float(getattr(cfg, "v98a_legacy_sil_min", 0.50))
                        and _v98a_legacy_subspace_balance
                        >= float(getattr(cfg, "v98a_legacy_balance_min", 0.20))
                    ):
                        labels = _legacy_labels
                        emb = _legacy_emb.astype(np.float32)
                        _postproc_choice = "gated_legacy_subspace_readout"
                        _v98a_legacy_subspace_readout_used = True
                except Exception as _e:
                    _v98a_legacy_subspace_error = repr(_e)
            if (
                bool(getattr(cfg, "v84a_raw_embedding_readout_enabled", False))
                and not bool(_v75a_anchor_readout_used)
                and not bool(_v91a_spectral_anchor_readout_used)
                and not bool(_v93a_raw_feature_svd_readout_used)
                and not bool(_v98a_legacy_subspace_readout_used)
                and not bool(_v99a_fast_elss_readout_used)
                and _v75a_anchor_readout_agreement
                <= float(getattr(cfg, "v84a_raw_embedding_anchor_agreement_max", 0.90))
            ):
                _raw_emb = np.nan_to_num(np.asarray(emb, dtype=np.float32))
                if _raw_emb.shape[0] >= _n_cl and _n_cl > 1:
                    _raw_n_init = min(int(getattr(cfg, "v84a_raw_embedding_kmeans_n_init", 10)), 40)
                    _raw_seed = int(getattr(cfg, "v84a_raw_embedding_kmeans_seed", -1))
                    if _raw_seed < 0:
                        _raw_seed = int(cfg.seed)
                    _raw_km = KMeans(
                        n_clusters=_n_cl,
                        n_init=_raw_n_init,
                        random_state=_raw_seed,
                        max_iter=300,
                    )
                    _raw_labels = _raw_km.fit_predict(_raw_emb).astype(np.int64)
                    if np.unique(_raw_labels).shape[0] >= 2:
                        labels = _raw_labels
                        _postproc_choice = "raw_embedding_full"
                        _v84a_raw_embedding_readout_used = True
            if (
                bool(getattr(cfg, "v100a_embedding_svd_readout_enabled", False))
                and not bool(_v75a_anchor_readout_used)
                and not bool(_v91a_spectral_anchor_readout_used)
                and not bool(_v93a_raw_feature_svd_readout_used)
                and not bool(_v98a_legacy_subspace_readout_used)
                and not bool(_v99a_fast_elss_readout_used)
                and not bool(_v84a_raw_embedding_readout_used)
                and _v75a_anchor_readout_agreement
                <= float(getattr(cfg, "v100a_embedding_svd_anchor_agreement_max", 0.58))
            ):
                _svd_emb = np.nan_to_num(np.asarray(emb, dtype=np.float32))
                _svd_dim = int(max(1, getattr(cfg, "v100a_embedding_svd_dim", 20)))
                _svd_dim = min(_svd_dim, max(1, min(_svd_emb.shape) - 1))
                if _svd_dim > 0 and _svd_dim < min(_svd_emb.shape):
                    _svd_z = TruncatedSVD(n_components=_svd_dim, random_state=int(cfg.seed)).fit_transform(_svd_emb)
                else:
                    _svd_z = _svd_emb
                _svd_z = normalize(np.nan_to_num(np.asarray(_svd_z, dtype=np.float32)), norm="l2", axis=1)
                _svd_km = KMeans(
                    n_clusters=_n_cl,
                    n_init=max(1, int(getattr(cfg, "v100a_embedding_svd_kmeans_n_init", 10))),
                    random_state=int(getattr(cfg, "v100a_embedding_svd_kmeans_seed", 0)),
                    max_iter=500,
                )
                _svd_labels = _svd_km.fit_predict(_svd_z).astype(np.int64)
                _counts = np.bincount(_svd_labels, minlength=_n_cl).astype(np.float64)
                _counts = _counts / max(1.0, float(_svd_labels.shape[0]))
                _positive = _counts[_counts > 0.0]
                _v100a_embedding_svd_balance = (
                    float(_positive.min() / max(_positive.max(), 1e-8)) if _positive.size else 0.0
                )
                _sil_n = min(3000, int(_svd_z.shape[0]))
                _rng = np.random.default_rng(int(cfg.seed))
                _idx = (
                    _rng.choice(_svd_z.shape[0], _sil_n, replace=False)
                    if _svd_z.shape[0] > _sil_n
                    else np.arange(_svd_z.shape[0])
                )
                if len(np.unique(_svd_labels[_idx])) >= 2:
                    _v100a_embedding_svd_sil = float(silhouette_score(_svd_z[_idx], _svd_labels[_idx]))
                if (
                    np.unique(_svd_labels).shape[0] >= 2
                    and _v100a_embedding_svd_sil
                    >= float(getattr(cfg, "v100a_embedding_svd_sil_min", 0.20))
                    and _v100a_embedding_svd_balance
                    >= float(getattr(cfg, "v100a_embedding_svd_balance_min", 0.05))
                ):
                    labels = _svd_labels
                    emb = _svd_z.astype(np.float32)
                    _postproc_choice = "embedding_svd_readout"
                    _v100a_embedding_svd_readout_used = True
            if bool(getattr(cfg, "v105a_size_pressure_enabled", False)) and bool(_v99a_fast_elss_readout_used):
                _pressure_labels = np.asarray(labels, dtype=np.int64).reshape(-1)
                _pressure_emb = np.nan_to_num(np.asarray(emb, dtype=np.float32))
                if (
                    _pressure_labels.shape[0] == _pressure_emb.shape[0]
                    and _pressure_emb.ndim == 2
                    and _pressure_emb.shape[0] > 0
                ):
                    _unique_labels, _inverse_labels, _cluster_counts = np.unique(
                        _pressure_labels,
                        return_inverse=True,
                        return_counts=True,
                    )
                    if _unique_labels.shape[0] >= 3:
                        _count_order = np.argsort(-_cluster_counts, kind="stable")
                        _largest_pos = int(_count_order[0])
                        _second_pos = int(_count_order[1])
                        _largest_count = int(_cluster_counts[_largest_pos])
                        _second_count = int(_cluster_counts[_second_pos])
                        _v105a_size_pressure_source_ratio = float(_largest_count) / float(max(1, _second_count))
                        _ratio_min = float(max(1.0, getattr(cfg, "v105a_size_pressure_ratio_min", 1.55)))
                        _target_ratio = float(max(1.0, getattr(cfg, "v105a_size_pressure_target_ratio", 1.50)))
                        _max_change_ratio = float(
                            np.clip(getattr(cfg, "v105a_size_pressure_max_change_ratio", 0.05), 0.0, 1.0)
                        )
                        _target_largest = int(math.floor(_target_ratio * float(_second_count)))
                        _max_changes = int(math.floor(_max_change_ratio * float(_pressure_labels.shape[0])))
                        _requested_moves = int(max(0, _largest_count - max(_second_count, _target_largest)))
                        _requested_moves = int(min(_requested_moves, _max_changes))
                        if (
                            _requested_moves > 0
                            and _v105a_size_pressure_source_ratio >= _ratio_min
                            and _largest_count - _requested_moves > 0
                        ):
                            _z_pressure = normalize(_pressure_emb, norm="l2", axis=1)
                            _centroids = np.zeros((_unique_labels.shape[0], _z_pressure.shape[1]), dtype=np.float32)
                            for _cluster_pos in range(_unique_labels.shape[0]):
                                _cluster_mask = _inverse_labels == _cluster_pos
                                if np.any(_cluster_mask):
                                    _centroids[_cluster_pos] = _z_pressure[_cluster_mask].mean(axis=0)
                            _centroids = normalize(np.nan_to_num(_centroids), norm="l2", axis=1)
                            _similarity = np.asarray(_z_pressure @ _centroids.T, dtype=np.float32)
                            _own_similarity = _similarity[np.arange(_similarity.shape[0]), _inverse_labels]
                            _non_own_similarity = _similarity.copy()
                            _non_own_similarity[np.arange(_non_own_similarity.shape[0]), _inverse_labels] = -np.inf
                            _nearest_other_pos = _non_own_similarity.argmax(axis=1).astype(np.int64)
                            _margin = _own_similarity - _similarity[
                                np.arange(_similarity.shape[0]),
                                _nearest_other_pos,
                            ]
                            _candidate_idx = np.flatnonzero(
                                (_inverse_labels == _largest_pos) & (_nearest_other_pos == _second_pos)
                            )
                            if _candidate_idx.size > 0:
                                _order = np.argsort(_margin[_candidate_idx], kind="stable")
                                _chosen = _candidate_idx[_order[: min(_requested_moves, _candidate_idx.size)]]
                                _pressure_new = _pressure_labels.copy()
                                _pressure_new[_chosen] = int(_unique_labels[_second_pos])
                                _new_unique, _new_counts = np.unique(_pressure_new, return_counts=True)
                                if _new_unique.shape[0] == _unique_labels.shape[0] and np.all(_new_counts > 0):
                                    labels = _pressure_new
                                    _postproc_choice = f"{_postproc_choice}_size_pressure"
                                    _v105a_size_pressure_used = True
                                    _v105a_size_pressure_nmove = int(_chosen.shape[0])
                                    _v105a_size_pressure_changed_ratio = float(_chosen.shape[0]) / float(
                                        max(1, _pressure_labels.shape[0])
                                    )
            if (
                bool(getattr(cfg, "v83a_final_neighbor_smoothing_enabled", False))
                and not bool(_v75a_anchor_readout_used)
                and _v75a_anchor_readout_agreement
                <= float(getattr(cfg, "v83a_final_neighbor_anchor_agreement_max", 0.90))
            ):
                _n_nodes = int(labels.shape[0])
                _n_classes = int(out["q_refined"].shape[1])
                _hist = np.zeros((_n_nodes, _n_classes), dtype=np.float32)
                _ei = np.asarray(edge_index_np, dtype=np.int64)
                if _ei.ndim == 2 and _ei.shape[0] == 2 and _ei.shape[1] > 0:
                    _src = np.clip(_ei[0], 0, _n_nodes - 1)
                    _dst = np.clip(_ei[1], 0, _n_nodes - 1)
                    np.add.at(_hist, (_src, labels[_dst]), 1.0)
                    np.add.at(_hist, (_dst, labels[_src]), 1.0)
                _deg = _hist.sum(axis=1)
                _majority = _hist.argmax(axis=1).astype(np.int64)
                _majority_ratio = np.divide(
                    _hist.max(axis=1),
                    np.maximum(_deg, 1e-6),
                    out=np.zeros(_n_nodes, dtype=np.float32),
                    where=_deg > 0,
                )
                _maj_min = float(np.clip(getattr(cfg, "v83a_final_neighbor_majority_min", 0.70), 0.0, 1.0))
                _candidate = (_deg > 0) & (_majority != labels) & (_majority_ratio >= _maj_min)
                _v83a_final_neighbor_smoothing_candidate_ratio = float(_candidate.mean()) if _candidate.size else 0.0
                _v83a_final_neighbor_smoothing_majority_mean = (
                    float(_majority_ratio[_candidate].mean()) if np.any(_candidate) else 0.0
                )
                _max_ratio = float(
                    np.clip(getattr(cfg, "v83a_final_neighbor_max_change_ratio", 0.02), 0.0, 1.0)
                )
                _max_changes = int(math.floor(_max_ratio * float(_n_nodes)))
                if _max_changes > 0 and np.any(_candidate):
                    _cand_idx = np.flatnonzero(_candidate)
                    _order = np.argsort(-_majority_ratio[_cand_idx], kind="stable")
                    _chosen = _cand_idx[_order[: min(_max_changes, _cand_idx.shape[0])]]
                    _smoothed = labels.copy()
                    _smoothed[_chosen] = _majority[_chosen]
                    if np.unique(_smoothed).shape[0] >= 2:
                        labels = _smoothed
                        _postproc_choice = f"{_postproc_choice}_final_neighbor_smoothed"
                        _v83a_final_neighbor_smoothing_changed_ratio = (
                            float(_chosen.shape[0]) / float(max(1, _n_nodes))
                        )
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
                        _adaptive_sil_full_tolerance = float(_adaptive_stats.get("sil_full_tolerance", 1.0))
                    else:
                        labels, emb = _legacy_labels, _legacy_emb
                        _postproc_choice = "legacy_subspace_refine"
                    _legacy_head_used = True
                except Exception as _e:
                    _legacy_head_error = repr(_e)
        self.embedding_raw_ = np.nan_to_num(np.asarray(emb, dtype=np.float32))
        self.embedding_ = normalize(np.nan_to_num(emb), norm="l2", axis=1)
        self.labels_ = labels
        self.diagnostics_ = {
            "candidate": cfg.name,
            "nodes": int(adj.shape[0]),
            "edges": int(edge_adj.nnz if cfg.directed_candidate_edges else edge_adj.nnz // 2),
            "candidate_edges": int(edge_index_np.shape[1] if cfg.directed_candidate_edges else edge_index_np.shape[1] // 2),
            "input_dim": int(x_np.shape[1]),
            "embedding_dim": int(emb.shape[1]),
            "effective_device": str(device),
            "small_graph_cpu_max_nodes": int(getattr(cfg, "small_graph_cpu_max_nodes", 0)),
            "runtime_sec": round(time.perf_counter() - start, 3),
            **last_diag,
            **frontend_snapshots,
            **frontend_diag,
        }
        self.diagnostics_["v72a_rollback_enabled"] = bool(getattr(cfg, "v72a_stability_rollback_enabled", False))
        self.diagnostics_["v72a_rollback_used"] = bool(v72a_rollback_used)
        self.diagnostics_["v72a_snapshot_epoch"] = int(v72a_snapshot_epoch)
        self.diagnostics_["v72a_rollback_reason"] = str(v72a_rollback_reason)
        self.diagnostics_["v72a_rollback_anchor_agreement"] = float(v72a_rollback_anchor_agreement)
        self.diagnostics_["v72a_rollback_teacher_agreement"] = float(v72a_rollback_teacher_agreement)
        self.diagnostics_["v74a_nc_weighted_readout_enabled"] = bool(
            getattr(cfg, "v74a_nc_weighted_readout_enabled", False)
        )
        self.diagnostics_["v74a_readout_active"] = bool(_v74a_readout_active) if "_v74a_readout_active" in locals() else False
        self.diagnostics_["v74a_readout_weight_mean"] = (
            float(_v74a_readout_weight_mean) if "_v74a_readout_weight_mean" in locals() else 0.0
        )
        self.diagnostics_["v74a_readout_weight_min"] = (
            float(_v74a_readout_weight_min) if "_v74a_readout_weight_min" in locals() else 0.0
        )
        self.diagnostics_["v74a_readout_weight_max"] = (
            float(_v74a_readout_weight_max) if "_v74a_readout_weight_max" in locals() else 0.0
        )
        self.diagnostics_["v74a_readout_nc_mean"] = (
            float(_v74a_readout_nc_mean) if "_v74a_readout_nc_mean" in locals() else 0.0
        )
        self.diagnostics_["v74a_readout_conf_mean"] = (
            float(_v74a_readout_conf_mean) if "_v74a_readout_conf_mean" in locals() else 0.0
        )
        self.diagnostics_["v75a_reliable_anchor_readout_enabled"] = bool(
            getattr(cfg, "v75a_reliable_anchor_readout_enabled", False)
        )
        self.diagnostics_["v75a_anchor_readout_used"] = (
            bool(_v75a_anchor_readout_used) if "_v75a_anchor_readout_used" in locals() else False
        )
        self.diagnostics_["v75a_anchor_readout_agreement"] = (
            float(_v75a_anchor_readout_agreement) if "_v75a_anchor_readout_agreement" in locals() else 0.0
        )
        self.diagnostics_["v75a_anchor_readout_cluster_separation"] = (
            float(_v75a_anchor_readout_cluster_separation)
            if "_v75a_anchor_readout_cluster_separation" in locals()
            else 0.0
        )
        self.diagnostics_["v78a_anchor_smoothing_enabled"] = bool(getattr(cfg, "v78a_anchor_smoothing_enabled", False))
        self.diagnostics_["v78a_anchor_smoothing_changed_ratio"] = (
            float(_v78a_anchor_smoothing_changed_ratio)
            if "_v78a_anchor_smoothing_changed_ratio" in locals()
            else 0.0
        )
        self.diagnostics_["v78a_anchor_smoothing_candidate_ratio"] = (
            float(_v78a_anchor_smoothing_candidate_ratio)
            if "_v78a_anchor_smoothing_candidate_ratio" in locals()
            else 0.0
        )
        self.diagnostics_["v78a_anchor_smoothing_majority_mean"] = (
            float(_v78a_anchor_smoothing_majority_mean)
            if "_v78a_anchor_smoothing_majority_mean" in locals()
            else 0.0
        )
        self.diagnostics_["v80a_anchor_smoothing_min_votes"] = int(
            getattr(cfg, "v80a_anchor_smoothing_min_votes", 0)
        )
        self.diagnostics_["v80a_anchor_smoothing_vote_mean"] = (
            float(_v80a_anchor_smoothing_vote_mean) if "_v80a_anchor_smoothing_vote_mean" in locals() else 0.0
        )
        self.diagnostics_["v82a_anchor_diffusion_smoothing_enabled"] = bool(
            getattr(cfg, "v82a_anchor_diffusion_smoothing_enabled", False)
        )
        self.diagnostics_["v82a_anchor_diffusion_changed_ratio"] = (
            float(_v82a_anchor_diffusion_changed_ratio)
            if "_v82a_anchor_diffusion_changed_ratio" in locals()
            else 0.0
        )
        self.diagnostics_["v82a_anchor_diffusion_candidate_ratio"] = (
            float(_v82a_anchor_diffusion_candidate_ratio)
            if "_v82a_anchor_diffusion_candidate_ratio" in locals()
            else 0.0
        )
        self.diagnostics_["v82a_anchor_diffusion_margin_mean"] = (
            float(_v82a_anchor_diffusion_margin_mean) if "_v82a_anchor_diffusion_margin_mean" in locals() else 0.0
        )
        self.diagnostics_["v79a_consensus_smoothing_enabled"] = bool(
            getattr(cfg, "v79a_consensus_smoothing_enabled", False)
        )
        self.diagnostics_["v79a_consensus_smoothing_changed_ratio"] = (
            float(_v79a_consensus_smoothing_changed_ratio)
            if "_v79a_consensus_smoothing_changed_ratio" in locals()
            else 0.0
        )
        self.diagnostics_["v79a_consensus_smoothing_candidate_ratio"] = (
            float(_v79a_consensus_smoothing_candidate_ratio)
            if "_v79a_consensus_smoothing_candidate_ratio" in locals()
            else 0.0
        )
        self.diagnostics_["v79a_consensus_smoothing_vote_mean"] = (
            float(_v79a_consensus_smoothing_vote_mean)
            if "_v79a_consensus_smoothing_vote_mean" in locals()
            else 0.0
        )
        self.diagnostics_["v79a_consensus_smoothing_majority_mean"] = (
            float(_v79a_consensus_smoothing_majority_mean)
            if "_v79a_consensus_smoothing_majority_mean" in locals()
            else 0.0
        )
        self.diagnostics_["v83a_final_neighbor_smoothing_enabled"] = bool(
            getattr(cfg, "v83a_final_neighbor_smoothing_enabled", False)
        )
        self.diagnostics_["v83a_final_neighbor_smoothing_changed_ratio"] = (
            float(_v83a_final_neighbor_smoothing_changed_ratio)
            if "_v83a_final_neighbor_smoothing_changed_ratio" in locals()
            else 0.0
        )
        self.diagnostics_["v83a_final_neighbor_smoothing_candidate_ratio"] = (
            float(_v83a_final_neighbor_smoothing_candidate_ratio)
            if "_v83a_final_neighbor_smoothing_candidate_ratio" in locals()
            else 0.0
        )
        self.diagnostics_["v83a_final_neighbor_smoothing_majority_mean"] = (
            float(_v83a_final_neighbor_smoothing_majority_mean)
            if "_v83a_final_neighbor_smoothing_majority_mean" in locals()
            else 0.0
        )
        self.diagnostics_["v84a_raw_embedding_readout_enabled"] = bool(
            getattr(cfg, "v84a_raw_embedding_readout_enabled", False)
        )
        self.diagnostics_["v84a_raw_embedding_readout_used"] = (
            bool(_v84a_raw_embedding_readout_used) if "_v84a_raw_embedding_readout_used" in locals() else False
        )
        self.diagnostics_["v91a_spectral_anchor_readout_enabled"] = bool(
            getattr(cfg, "v91a_spectral_anchor_readout_enabled", False)
        )
        self.diagnostics_["v91a_spectral_anchor_readout_used"] = (
            bool(_v91a_spectral_anchor_readout_used) if "_v91a_spectral_anchor_readout_used" in locals() else False
        )
        self.diagnostics_["v91a_spectral_anchor_conf_mean"] = (
            float(_v91a_spectral_anchor_conf_mean) if "_v91a_spectral_anchor_conf_mean" in locals() else 0.0
        )
        self.diagnostics_["v91a_spectral_anchor_entropy"] = (
            float(_v91a_spectral_anchor_entropy) if "_v91a_spectral_anchor_entropy" in locals() else 1.0
        )
        self.diagnostics_["v91a_spectral_anchor_balance"] = (
            float(_v91a_spectral_anchor_balance) if "_v91a_spectral_anchor_balance" in locals() else 0.0
        )
        self.diagnostics_["v93a_raw_feature_svd_readout_enabled"] = bool(
            getattr(cfg, "v93a_raw_feature_svd_readout_enabled", False)
        )
        self.diagnostics_["v93a_raw_feature_svd_readout_used"] = (
            bool(_v93a_raw_feature_svd_readout_used)
            if "_v93a_raw_feature_svd_readout_used" in locals()
            else False
        )
        self.diagnostics_["v93a_raw_feature_svd_sil"] = (
            float(_v93a_raw_feature_svd_sil) if "_v93a_raw_feature_svd_sil" in locals() else -2.0
        )
        self.diagnostics_["v93a_raw_feature_svd_balance"] = (
            float(_v93a_raw_feature_svd_balance) if "_v93a_raw_feature_svd_balance" in locals() else 0.0
        )
        self.diagnostics_["v98a_gated_legacy_subspace_readout_enabled"] = bool(
            getattr(cfg, "v98a_gated_legacy_subspace_readout_enabled", False)
        )
        self.diagnostics_["v98a_legacy_subspace_pre_gate"] = (
            bool(_v98a_legacy_subspace_pre_gate) if "_v98a_legacy_subspace_pre_gate" in locals() else False
        )
        self.diagnostics_["v98a_legacy_subspace_readout_used"] = (
            bool(_v98a_legacy_subspace_readout_used)
            if "_v98a_legacy_subspace_readout_used" in locals()
            else False
        )
        self.diagnostics_["v98a_legacy_subspace_sil"] = (
            float(_v98a_legacy_subspace_sil) if "_v98a_legacy_subspace_sil" in locals() else -2.0
        )
        self.diagnostics_["v98a_legacy_subspace_balance"] = (
            float(_v98a_legacy_subspace_balance) if "_v98a_legacy_subspace_balance" in locals() else 0.0
        )
        self.diagnostics_["v98a_legacy_subspace_hard_ratio"] = (
            float(_v98a_legacy_subspace_hard_ratio) if "_v98a_legacy_subspace_hard_ratio" in locals() else 0.0
        )
        self.diagnostics_["v98a_legacy_subspace_error"] = (
            _v98a_legacy_subspace_error if "_v98a_legacy_subspace_error" in locals() else ""
        )
        self.diagnostics_["v99a_fast_elss_readout_enabled"] = bool(
            getattr(cfg, "v99a_fast_elss_readout_enabled", False)
        )
        self.diagnostics_["v99a_fast_elss_pre_gate"] = (
            bool(_v99a_fast_elss_pre_gate) if "_v99a_fast_elss_pre_gate" in locals() else False
        )
        self.diagnostics_["v99a_fast_elss_readout_used"] = (
            bool(_v99a_fast_elss_readout_used) if "_v99a_fast_elss_readout_used" in locals() else False
        )
        self.diagnostics_["v99a_fast_elss_sil"] = (
            float(_v99a_fast_elss_sil) if "_v99a_fast_elss_sil" in locals() else -2.0
        )
        self.diagnostics_["v99a_fast_elss_balance"] = (
            float(_v99a_fast_elss_balance) if "_v99a_fast_elss_balance" in locals() else 0.0
        )
        self.diagnostics_["v99a_fast_elss_error"] = (
            _v99a_fast_elss_error if "_v99a_fast_elss_error" in locals() else ""
        )
        self.diagnostics_["v100a_embedding_svd_readout_enabled"] = bool(
            getattr(cfg, "v100a_embedding_svd_readout_enabled", False)
        )
        self.diagnostics_["v100a_embedding_svd_readout_used"] = (
            bool(_v100a_embedding_svd_readout_used)
            if "_v100a_embedding_svd_readout_used" in locals()
            else False
        )
        self.diagnostics_["v100a_embedding_svd_sil"] = (
            float(_v100a_embedding_svd_sil) if "_v100a_embedding_svd_sil" in locals() else -2.0
        )
        self.diagnostics_["v100a_embedding_svd_balance"] = (
            float(_v100a_embedding_svd_balance) if "_v100a_embedding_svd_balance" in locals() else 0.0
        )
        self.diagnostics_["v105a_size_pressure_enabled"] = bool(
            getattr(cfg, "v105a_size_pressure_enabled", False)
        )
        self.diagnostics_["v105a_size_pressure_used"] = (
            bool(_v105a_size_pressure_used) if "_v105a_size_pressure_used" in locals() else False
        )
        self.diagnostics_["v105a_size_pressure_source_ratio"] = (
            float(_v105a_size_pressure_source_ratio) if "_v105a_size_pressure_source_ratio" in locals() else 0.0
        )
        self.diagnostics_["v105a_size_pressure_changed_ratio"] = (
            float(_v105a_size_pressure_changed_ratio)
            if "_v105a_size_pressure_changed_ratio" in locals()
            else 0.0
        )
        self.diagnostics_["v105a_size_pressure_nmove"] = (
            int(_v105a_size_pressure_nmove) if "_v105a_size_pressure_nmove" in locals() else 0
        )
        self.diagnostics_["v105a_size_pressure_target_ratio"] = float(
            getattr(cfg, "v105a_size_pressure_target_ratio", 1.50)
        )
        self.diagnostics_["selected_sub_dim"] = int(_sub_dim) if "_sub_dim" in locals() else 0
        self.diagnostics_["adaptive_sub_dim"] = int(_adaptive_sub_dim) if "_adaptive_sub_dim" in locals() else 0
        self.diagnostics_["postproc_choice"] = _postproc_choice if "_postproc_choice" in locals() else "unknown"
        self.diagnostics_["sil_full"] = float(_sil_full) if "_sil_full" in locals() else -2.0
        self.diagnostics_["sil_sub"] = float(_sil_sub) if "_sil_sub" in locals() else -2.0
        self.diagnostics_["sil_best_sub"] = float(_sil_best_sub) if "_sil_best_sub" in locals() else -2.0
        self.diagnostics_["adaptive_sil_full_tolerance"] = (
            float(_adaptive_sil_full_tolerance) if "_adaptive_sil_full_tolerance" in locals() else 1.0
        )
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
                q_refined_labels = out["q_refined"].argmax(dim=1).detach().cpu().numpy().astype(np.int64)
                q_refined_metrics = evaluate_clustering(true_labels_np, q_refined_labels)
                anchor_metrics = None
                if (
                    bool(getattr(cfg, "v50a_enabled", False))
                    or bool(getattr(cfg, "v51a_enabled", False))
                    or bool(getattr(cfg, "v52a_enabled", False))
                    or bool(getattr(cfg, "v53a_enabled", False))
                    or bool(getattr(cfg, "v54a_enabled", False))
                    or bool(getattr(cfg, "v55a_enabled", False))
                    or bool(getattr(cfg, "v56a_enabled", False))
                    or bool(getattr(cfg, "v57a_enabled", False))
                    or bool(getattr(cfg, "v58a_enabled", False))
                    or bool(getattr(cfg, "v59a_enabled", False))
                    or bool(getattr(cfg, "v60a_enabled", False))
                    or bool(getattr(cfg, "v61a_enabled", False))
                    or bool(getattr(cfg, "v62a_enabled", False))
                ) and model.v50a_anchor_q.numel() == out["q_refined"].numel():
                    anchor_labels = model.v50a_anchor_q.argmax(dim=1).detach().cpu().numpy().astype(np.int64)
                    anchor_metrics = evaluate_clustering(true_labels_np, anchor_labels)
                self.diagnostics_.update(
                    {
                        "final_acc": float(final_metrics["acc"]),
                        "final_nmi": float(final_metrics["nmi"]),
                        "final_ari": float(final_metrics["ari"]),
                        "embedding_kmeans_acc": float(km_metrics["acc"]),
                        "embedding_kmeans_nmi": float(km_metrics["nmi"]),
                        "embedding_kmeans_ari": float(km_metrics["ari"]),
                        "q_refined_acc_diagnostic": float(q_refined_metrics["acc"]),
                        "q_refined_nmi_diagnostic": float(q_refined_metrics["nmi"]),
                        "q_refined_ari_diagnostic": float(q_refined_metrics["ari"]),
                        "embedding_posterior_gap": float(km_metrics["acc"] - final_metrics["acc"]),
                        "v50a_anchor_acc_diagnostic": 0.0 if anchor_metrics is None else float(anchor_metrics["acc"]),
                        "v50a_anchor_nmi_diagnostic": 0.0 if anchor_metrics is None else float(anchor_metrics["nmi"]),
                        "v50a_anchor_ari_diagnostic": 0.0 if anchor_metrics is None else float(anchor_metrics["ari"]),
                        "v51a_anchor_acc_diagnostic": 0.0 if anchor_metrics is None else float(anchor_metrics["acc"]),
                        "v51a_anchor_nmi_diagnostic": 0.0 if anchor_metrics is None else float(anchor_metrics["nmi"]),
                        "v51a_anchor_ari_diagnostic": 0.0 if anchor_metrics is None else float(anchor_metrics["ari"]),
                        "v52a_anchor_acc_diagnostic": 0.0 if anchor_metrics is None else float(anchor_metrics["acc"]),
                        "v52a_anchor_nmi_diagnostic": 0.0 if anchor_metrics is None else float(anchor_metrics["nmi"]),
                        "v52a_anchor_ari_diagnostic": 0.0 if anchor_metrics is None else float(anchor_metrics["ari"]),
                        "v53a_anchor_acc_diagnostic": 0.0 if anchor_metrics is None else float(anchor_metrics["acc"]),
                        "v53a_anchor_nmi_diagnostic": 0.0 if anchor_metrics is None else float(anchor_metrics["nmi"]),
                        "v53a_anchor_ari_diagnostic": 0.0 if anchor_metrics is None else float(anchor_metrics["ari"]),
                        "v54a_anchor_acc_diagnostic": 0.0 if anchor_metrics is None else float(anchor_metrics["acc"]),
                        "v54a_anchor_nmi_diagnostic": 0.0 if anchor_metrics is None else float(anchor_metrics["nmi"]),
                        "v54a_anchor_ari_diagnostic": 0.0 if anchor_metrics is None else float(anchor_metrics["ari"]),
                        "v55a_anchor_acc_diagnostic": 0.0 if anchor_metrics is None else float(anchor_metrics["acc"]),
                        "v55a_anchor_nmi_diagnostic": 0.0 if anchor_metrics is None else float(anchor_metrics["nmi"]),
                        "v55a_anchor_ari_diagnostic": 0.0 if anchor_metrics is None else float(anchor_metrics["ari"]),
                        "v56a_anchor_acc_diagnostic": 0.0 if anchor_metrics is None else float(anchor_metrics["acc"]),
                        "v56a_anchor_nmi_diagnostic": 0.0 if anchor_metrics is None else float(anchor_metrics["nmi"]),
                        "v56a_anchor_ari_diagnostic": 0.0 if anchor_metrics is None else float(anchor_metrics["ari"]),
                        "v57a_anchor_acc_diagnostic": 0.0 if anchor_metrics is None else float(anchor_metrics["acc"]),
                        "v57a_anchor_nmi_diagnostic": 0.0 if anchor_metrics is None else float(anchor_metrics["nmi"]),
                        "v57a_anchor_ari_diagnostic": 0.0 if anchor_metrics is None else float(anchor_metrics["ari"]),
                        "v58a_anchor_acc_diagnostic": 0.0 if anchor_metrics is None else float(anchor_metrics["acc"]),
                        "v58a_anchor_nmi_diagnostic": 0.0 if anchor_metrics is None else float(anchor_metrics["nmi"]),
                        "v58a_anchor_ari_diagnostic": 0.0 if anchor_metrics is None else float(anchor_metrics["ari"]),
                        "v59a_anchor_acc_diagnostic": 0.0 if anchor_metrics is None else float(anchor_metrics["acc"]),
                        "v59a_anchor_nmi_diagnostic": 0.0 if anchor_metrics is None else float(anchor_metrics["nmi"]),
                        "v59a_anchor_ari_diagnostic": 0.0 if anchor_metrics is None else float(anchor_metrics["ari"]),
                        "v60a_anchor_acc_diagnostic": 0.0 if anchor_metrics is None else float(anchor_metrics["acc"]),
                        "v60a_anchor_nmi_diagnostic": 0.0 if anchor_metrics is None else float(anchor_metrics["nmi"]),
                        "v60a_anchor_ari_diagnostic": 0.0 if anchor_metrics is None else float(anchor_metrics["ari"]),
                        "v61a_anchor_acc_diagnostic": 0.0 if anchor_metrics is None else float(anchor_metrics["acc"]),
                        "v61a_anchor_nmi_diagnostic": 0.0 if anchor_metrics is None else float(anchor_metrics["nmi"]),
                        "v61a_anchor_ari_diagnostic": 0.0 if anchor_metrics is None else float(anchor_metrics["ari"]),
                        "v62a_anchor_acc_diagnostic": 0.0 if anchor_metrics is None else float(anchor_metrics["acc"]),
                        "v62a_anchor_nmi_diagnostic": 0.0 if anchor_metrics is None else float(anchor_metrics["nmi"]),
                        "v62a_anchor_ari_diagnostic": 0.0 if anchor_metrics is None else float(anchor_metrics["ari"]),
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


def run_legacy_fast_elss_head(
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
    labels = legacy_fast_elss_head(adapter, legacy_cfg, device)
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
    full_tolerance = float(getattr(cfg, "subspace_sil_full_tolerance", 1.0))
    full_ok = best_sil >= sil_full - full_tolerance
    if best_labels is not None and best_z is not None and best_sil > threshold and full_ok:
        return best_labels.astype(np.int64), best_z, {
            "choice": f"subspace_{best_dim}",
            "sil_full": float(sil_full),
            "sil_best_sub": float(best_sil),
            "best_dim": int(best_dim),
            "sil_full_tolerance": float(full_tolerance),
        }
    return lab_full, base.astype(np.float32), {
        "choice": "full_kmeans",
        "sil_full": float(sil_full),
        "sil_best_sub": float(best_sil),
        "best_dim": int(best_dim),
        "sil_full_tolerance": float(full_tolerance),
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


def spectral_anchor_alignment_loss(
    q: torch.Tensor,
    q_anchor: torch.Tensor,
    *,
    q_embed: torch.Tensor | None = None,
    enabled: bool,
    effective_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    zero = q.new_tensor(0.0)
    if (not bool(enabled)) or q_anchor.numel() != q.numel():
        stats = {
            "v50a_anchor_loss": zero,
            "v50a_q_anchor_kl": zero,
            "v50a_q_anchor_agreement": zero,
            "v50a_embedding_anchor_agreement": zero,
            "v50a_anchor_entropy": zero,
            "v50a_anchor_confidence": zero,
            "v50a_anchor_cluster_usage_entropy": zero,
            "v50a_anchor_effective_weight": zero,
        }
        return zero, stats

    anchor = q_anchor.to(device=q.device, dtype=q.dtype).detach().clamp_min(1e-8)
    anchor = anchor / anchor.sum(dim=1, keepdim=True).clamp_min(1e-8)
    q_safe = q.clamp_min(1e-8)
    q_safe = q_safe / q_safe.sum(dim=1, keepdim=True).clamp_min(1e-8)
    loss = F.kl_div(q_safe.log(), anchor, reduction="batchmean")
    k = max(2, int(anchor.shape[1]))
    anchor_entropy = -torch.sum(anchor * anchor.clamp_min(1e-8).log(), dim=1).mean() / math.log(float(k))
    anchor_conf = anchor.max(dim=1).values.mean()
    usage = anchor.mean(dim=0).clamp_min(1e-8)
    usage_entropy = -torch.sum(usage * usage.log()) / math.log(float(k))
    agreement = (q_safe.argmax(dim=1) == anchor.argmax(dim=1)).to(q.dtype).mean()
    if q_embed is not None and q_embed.numel() == q.numel():
        embed_safe = q_embed.to(device=q.device, dtype=q.dtype).clamp_min(1e-8)
        embed_safe = embed_safe / embed_safe.sum(dim=1, keepdim=True).clamp_min(1e-8)
        embed_agreement = (embed_safe.argmax(dim=1) == anchor.argmax(dim=1)).to(q.dtype).mean()
    else:
        embed_agreement = zero
    stats = {
        "v50a_anchor_loss": loss.detach(),
        "v50a_q_anchor_kl": loss.detach(),
        "v50a_q_anchor_agreement": agreement.detach(),
        "v50a_embedding_anchor_agreement": embed_agreement.detach(),
        "v50a_anchor_entropy": anchor_entropy.detach(),
        "v50a_anchor_confidence": anchor_conf.detach(),
        "v50a_anchor_cluster_usage_entropy": usage_entropy.detach(),
        "v50a_anchor_effective_weight": q.new_tensor(float(effective_weight)).detach(),
    }
    return loss, stats


def reliability_gated_spectral_anchor_loss(
    q: torch.Tensor,
    q_anchor: torch.Tensor,
    *,
    q_embed: torch.Tensor | None,
    edge_index: torch.Tensor,
    enabled: bool,
    effective_weight: float,
    reliability_floor: float,
    reliable_threshold: float,
    min_effective_mass: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    zero = q.new_tensor(0.0)
    stats_zero = {
        "v51a_anchor_loss": zero,
        "v51a_weighted_q_anchor_kl": zero,
        "v51a_weighted_q_anchor_agreement": zero,
        "v51a_unweighted_q_anchor_agreement": zero,
        "v51a_embedding_anchor_agreement": zero,
        "v51a_anchor_entropy": zero,
        "v51a_anchor_confidence": zero,
        "v51a_anchor_cluster_usage_entropy": zero,
        "v51a_anchor_effective_weight": zero,
        "v51a_reliability_mean": zero,
        "v51a_reliability_std": zero,
        "v51a_reliability_p10": zero,
        "v51a_reliability_p50": zero,
        "v51a_reliability_p90": zero,
        "v51a_reliable_node_ratio": zero,
        "v51a_effective_anchor_mass": zero,
        "v51a_confidence_component_mean": zero,
        "v51a_q_anchor_component_mean": zero,
        "v51a_embed_anchor_component_mean": zero,
        "v51a_local_component_mean": zero,
    }
    if (not bool(enabled)) or q_anchor.numel() != q.numel():
        return zero, stats_zero

    anchor = q_anchor.to(device=q.device, dtype=q.dtype).detach().clamp_min(1e-8)
    anchor = anchor / anchor.sum(dim=1, keepdim=True).clamp_min(1e-8)
    q_safe = q.clamp_min(1e-8)
    q_safe = q_safe / q_safe.sum(dim=1, keepdim=True).clamp_min(1e-8)
    if q_embed is not None and q_embed.numel() == q.numel():
        embed_safe = q_embed.to(device=q.device, dtype=q.dtype).clamp_min(1e-8)
        embed_safe = embed_safe / embed_safe.sum(dim=1, keepdim=True).clamp_min(1e-8)
    else:
        embed_safe = q_safe.detach()

    n = int(anchor.shape[0])
    k = max(2, int(anchor.shape[1]))
    inv_uniform_gap = float(k) / float(k - 1)
    anchor_det = anchor.detach()
    q_det = q_safe.detach()
    embed_det = embed_safe.detach()

    conf = (anchor_det.max(dim=1).values - (1.0 / float(k))) * inv_uniform_gap
    conf = conf.clamp(0.0, 1.0)
    qa = (q_det * anchor_det).sum(dim=1)
    qa_norm = ((qa - (1.0 / float(k))) * inv_uniform_gap).clamp(0.0, 1.0)
    ea = (embed_det * anchor_det).sum(dim=1)
    ea_norm = ((ea - (1.0 / float(k))) * inv_uniform_gap).clamp(0.0, 1.0)

    if edge_index.numel() > 0:
        src, dst = edge_index[0].to(q.device), edge_index[1].to(q.device)
        sim = (anchor_det[src] * anchor_det[dst]).sum(dim=1)
        local_sum = q.new_zeros(n)
        local_cnt = q.new_zeros(n)
        one = torch.ones_like(sim)
        local_sum.index_add_(0, src, sim)
        local_sum.index_add_(0, dst, sim)
        local_cnt.index_add_(0, src, one)
        local_cnt.index_add_(0, dst, one)
        local = local_sum / local_cnt.clamp_min(1.0)
        local = torch.where(local_cnt > 0.0, local, conf)
    else:
        local = conf
    local_norm = ((local - (1.0 / float(k))) * inv_uniform_gap).clamp(0.0, 1.0)

    reliability = conf * torch.sqrt((qa_norm * ea_norm).clamp_min(0.0)) * torch.sqrt(local_norm.clamp_min(0.0))
    reliability = reliability.clamp(0.0, 1.0).detach()
    per_node_kl = torch.sum(q_safe * (q_safe.clamp_min(1e-8).log() - anchor.clamp_min(1e-8).log()), dim=1)
    min_mass = max(float(reliability_floor), float(min_effective_mass), 1e-8) * float(max(1, n))
    denom = reliability.sum().clamp_min(float(min_mass))
    loss = torch.sum(reliability * per_node_kl) / denom

    usage = anchor.mean(dim=0).clamp_min(1e-8)
    anchor_entropy = -torch.sum(anchor * anchor.clamp_min(1e-8).log(), dim=1).mean() / math.log(float(k))
    anchor_conf = anchor.max(dim=1).values.mean()
    usage_entropy = -torch.sum(usage * usage.log()) / math.log(float(k))
    anchor_label = anchor.argmax(dim=1)
    q_label = q_safe.argmax(dim=1)
    embed_label = embed_safe.argmax(dim=1)
    match = (q_label == anchor_label).to(q.dtype)
    weighted_agreement = torch.sum(reliability * match) / denom
    unweighted_agreement = match.mean()
    embed_agreement = (embed_label == anchor_label).to(q.dtype).mean()
    rel_sorted = torch.sort(reliability).values
    reliable_node_ratio = (reliability >= float(reliable_threshold)).to(q.dtype).mean()
    effective_anchor_mass = reliability.mean()
    stats = {
        "v51a_anchor_loss": loss.detach(),
        "v51a_weighted_q_anchor_kl": loss.detach(),
        "v51a_weighted_q_anchor_agreement": weighted_agreement.detach(),
        "v51a_unweighted_q_anchor_agreement": unweighted_agreement.detach(),
        "v51a_embedding_anchor_agreement": embed_agreement.detach(),
        "v51a_anchor_entropy": anchor_entropy.detach(),
        "v51a_anchor_confidence": anchor_conf.detach(),
        "v51a_anchor_cluster_usage_entropy": usage_entropy.detach(),
        "v51a_anchor_effective_weight": q.new_tensor(float(effective_weight)).detach(),
        "v51a_reliability_mean": reliability.mean().detach(),
        "v51a_reliability_std": reliability.std(unbiased=False).detach(),
        "v51a_reliability_p10": torch.quantile(rel_sorted, 0.10).detach(),
        "v51a_reliability_p50": torch.quantile(rel_sorted, 0.50).detach(),
        "v51a_reliability_p90": torch.quantile(rel_sorted, 0.90).detach(),
        "v51a_reliable_node_ratio": reliable_node_ratio.detach(),
        "v51a_effective_anchor_mass": effective_anchor_mass.detach(),
        "v51a_confidence_component_mean": conf.mean().detach(),
        "v51a_q_anchor_component_mean": qa_norm.mean().detach(),
        "v51a_embed_anchor_component_mean": ea_norm.mean().detach(),
        "v51a_local_component_mean": local_norm.mean().detach(),
    }
    return loss, stats


def curriculum_reliability_spectral_anchor_loss(
    q: torch.Tensor,
    q_anchor: torch.Tensor,
    *,
    q_embed: torch.Tensor | None,
    edge_index: torch.Tensor,
    enabled: bool,
    current_epoch: int,
    effective_weight: float,
    reliability_floor: float,
    reliable_threshold: float,
    min_effective_mass: float,
    warmup_epochs: int,
    ramp_epochs: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    zero = q.new_tensor(0.0)
    stats_zero = {
        "v52a_gamma": zero,
        "v52a_anchor_loss": zero,
        "v52a_weighted_q_anchor_kl": zero,
        "v52a_weighted_q_anchor_agreement": zero,
        "v52a_unweighted_q_anchor_agreement": zero,
        "v52a_embedding_anchor_agreement": zero,
        "v52a_anchor_entropy": zero,
        "v52a_anchor_confidence": zero,
        "v52a_anchor_cluster_usage_entropy": zero,
        "v52a_anchor_effective_weight": zero,
        "v52a_reliability_mean": zero,
        "v52a_reliability_std": zero,
        "v52a_reliability_p10": zero,
        "v52a_reliability_p50": zero,
        "v52a_reliability_p90": zero,
        "v52a_reliable_node_ratio": zero,
        "v52a_effective_anchor_mass": zero,
        "v52a_base_reliability_mean": zero,
        "v52a_agreement_reliability_mean": zero,
        "v52a_confidence_component_mean": zero,
        "v52a_q_anchor_component_mean": zero,
        "v52a_embed_anchor_component_mean": zero,
        "v52a_local_component_mean": zero,
    }
    if (not bool(enabled)) or q_anchor.numel() != q.numel():
        return zero, stats_zero

    anchor = q_anchor.to(device=q.device, dtype=q.dtype).detach().clamp_min(1e-8)
    anchor = anchor / anchor.sum(dim=1, keepdim=True).clamp_min(1e-8)
    q_safe = q.clamp_min(1e-8)
    q_safe = q_safe / q_safe.sum(dim=1, keepdim=True).clamp_min(1e-8)
    if q_embed is not None and q_embed.numel() == q.numel():
        embed_safe = q_embed.to(device=q.device, dtype=q.dtype).clamp_min(1e-8)
        embed_safe = embed_safe / embed_safe.sum(dim=1, keepdim=True).clamp_min(1e-8)
    else:
        embed_safe = q_safe.detach()

    n = int(anchor.shape[0])
    k = max(2, int(anchor.shape[1]))
    inv_uniform_gap = float(k) / float(k - 1)
    anchor_det = anchor.detach()
    q_det = q_safe.detach()
    embed_det = embed_safe.detach()

    conf = ((anchor_det.max(dim=1).values - (1.0 / float(k))) * inv_uniform_gap).clamp(0.0, 1.0)
    qa_norm = (((q_det * anchor_det).sum(dim=1) - (1.0 / float(k))) * inv_uniform_gap).clamp(0.0, 1.0)
    ea_norm = (((embed_det * anchor_det).sum(dim=1) - (1.0 / float(k))) * inv_uniform_gap).clamp(0.0, 1.0)

    if edge_index.numel() > 0:
        src, dst = edge_index[0].to(q.device), edge_index[1].to(q.device)
        sim = (anchor_det[src] * anchor_det[dst]).sum(dim=1)
        local_sum = q.new_zeros(n)
        local_cnt = q.new_zeros(n)
        one = torch.ones_like(sim)
        local_sum.index_add_(0, src, sim)
        local_sum.index_add_(0, dst, sim)
        local_cnt.index_add_(0, src, one)
        local_cnt.index_add_(0, dst, one)
        local = local_sum / local_cnt.clamp_min(1.0)
        local = torch.where(local_cnt > 0.0, local, conf)
    else:
        local = conf
    local_norm = ((local - (1.0 / float(k))) * inv_uniform_gap).clamp(0.0, 1.0)

    warm = max(0, int(warmup_epochs))
    ramp = max(1, int(ramp_epochs))
    gamma_value = float(np.clip((float(current_epoch + 1 - warm) / float(ramp)), 0.0, 1.0))
    gamma = q.new_tensor(gamma_value)
    r_base = (0.5 * conf + 0.5 * local_norm).clamp(0.0, 1.0)
    r_agree = (0.5 * qa_norm + 0.5 * ea_norm).clamp(0.0, 1.0)
    reliability = ((1.0 - gamma) * r_base + gamma * (r_base * r_agree)).clamp(0.0, 1.0).detach()

    per_node_kl = torch.sum(q_safe * (q_safe.clamp_min(1e-8).log() - anchor.clamp_min(1e-8).log()), dim=1)
    min_mass = max(float(reliability_floor), float(min_effective_mass), 1e-8) * float(max(1, n))
    denom = reliability.sum().clamp_min(float(min_mass))
    loss = torch.sum(reliability * per_node_kl) / denom

    usage = anchor.mean(dim=0).clamp_min(1e-8)
    anchor_entropy = -torch.sum(anchor * anchor.clamp_min(1e-8).log(), dim=1).mean() / math.log(float(k))
    anchor_conf = anchor.max(dim=1).values.mean()
    usage_entropy = -torch.sum(usage * usage.log()) / math.log(float(k))
    anchor_label = anchor.argmax(dim=1)
    q_label = q_safe.argmax(dim=1)
    embed_label = embed_safe.argmax(dim=1)
    match = (q_label == anchor_label).to(q.dtype)
    weighted_agreement = torch.sum(reliability * match) / denom
    rel_sorted = torch.sort(reliability).values
    reliable_node_ratio = (reliability >= float(reliable_threshold)).to(q.dtype).mean()
    stats = {
        "v52a_gamma": gamma.detach(),
        "v52a_anchor_loss": loss.detach(),
        "v52a_weighted_q_anchor_kl": loss.detach(),
        "v52a_weighted_q_anchor_agreement": weighted_agreement.detach(),
        "v52a_unweighted_q_anchor_agreement": match.mean().detach(),
        "v52a_embedding_anchor_agreement": (embed_label == anchor_label).to(q.dtype).mean().detach(),
        "v52a_anchor_entropy": anchor_entropy.detach(),
        "v52a_anchor_confidence": anchor_conf.detach(),
        "v52a_anchor_cluster_usage_entropy": usage_entropy.detach(),
        "v52a_anchor_effective_weight": q.new_tensor(float(effective_weight)).detach(),
        "v52a_reliability_mean": reliability.mean().detach(),
        "v52a_reliability_std": reliability.std(unbiased=False).detach(),
        "v52a_reliability_p10": torch.quantile(rel_sorted, 0.10).detach(),
        "v52a_reliability_p50": torch.quantile(rel_sorted, 0.50).detach(),
        "v52a_reliability_p90": torch.quantile(rel_sorted, 0.90).detach(),
        "v52a_reliable_node_ratio": reliable_node_ratio.detach(),
        "v52a_effective_anchor_mass": reliability.mean().detach(),
        "v52a_base_reliability_mean": r_base.mean().detach(),
        "v52a_agreement_reliability_mean": r_agree.mean().detach(),
        "v52a_confidence_component_mean": conf.mean().detach(),
        "v52a_q_anchor_component_mean": qa_norm.mean().detach(),
        "v52a_embed_anchor_component_mean": ea_norm.mean().detach(),
        "v52a_local_component_mean": local_norm.mean().detach(),
    }
    return loss, stats


def residual_curriculum_spectral_anchor_loss(
    q: torch.Tensor,
    q_anchor: torch.Tensor,
    *,
    q_embed: torch.Tensor | None,
    edge_index: torch.Tensor,
    enabled: bool,
    current_epoch: int,
    effective_weight: float,
    reliability_floor: float,
    reliable_threshold: float,
    min_effective_mass: float,
    warmup_epochs: int,
    ramp_epochs: int,
    residual_beta: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    zero = q.new_tensor(0.0)
    stats_zero = {
        "v53a_gamma": zero,
        "v53a_residual_beta": zero,
        "v53a_residual_multiplier_mean": zero,
        "v53a_anchor_loss": zero,
        "v53a_weighted_q_anchor_kl": zero,
        "v53a_weighted_q_anchor_agreement": zero,
        "v53a_unweighted_q_anchor_agreement": zero,
        "v53a_embedding_anchor_agreement": zero,
        "v53a_anchor_entropy": zero,
        "v53a_anchor_confidence": zero,
        "v53a_anchor_cluster_usage_entropy": zero,
        "v53a_anchor_effective_weight": zero,
        "v53a_reliability_mean": zero,
        "v53a_reliability_std": zero,
        "v53a_reliability_p10": zero,
        "v53a_reliability_p50": zero,
        "v53a_reliability_p90": zero,
        "v53a_reliable_node_ratio": zero,
        "v53a_effective_anchor_mass": zero,
        "v53a_base_reliability_mean": zero,
        "v53a_agreement_reliability_mean": zero,
        "v53a_confidence_component_mean": zero,
        "v53a_q_anchor_component_mean": zero,
        "v53a_embed_anchor_component_mean": zero,
        "v53a_local_component_mean": zero,
    }
    if (not bool(enabled)) or q_anchor.numel() != q.numel():
        return zero, stats_zero

    anchor = q_anchor.to(device=q.device, dtype=q.dtype).detach().clamp_min(1e-8)
    anchor = anchor / anchor.sum(dim=1, keepdim=True).clamp_min(1e-8)
    q_safe = q.clamp_min(1e-8)
    q_safe = q_safe / q_safe.sum(dim=1, keepdim=True).clamp_min(1e-8)
    if q_embed is not None and q_embed.numel() == q.numel():
        embed_safe = q_embed.to(device=q.device, dtype=q.dtype).clamp_min(1e-8)
        embed_safe = embed_safe / embed_safe.sum(dim=1, keepdim=True).clamp_min(1e-8)
    else:
        embed_safe = q_safe.detach()

    n = int(anchor.shape[0])
    k = max(2, int(anchor.shape[1]))
    inv_uniform_gap = float(k) / float(k - 1)
    anchor_det = anchor.detach()
    q_det = q_safe.detach()
    embed_det = embed_safe.detach()

    conf = ((anchor_det.max(dim=1).values - (1.0 / float(k))) * inv_uniform_gap).clamp(0.0, 1.0)
    qa_norm = (((q_det * anchor_det).sum(dim=1) - (1.0 / float(k))) * inv_uniform_gap).clamp(0.0, 1.0)
    ea_norm = (((embed_det * anchor_det).sum(dim=1) - (1.0 / float(k))) * inv_uniform_gap).clamp(0.0, 1.0)

    if edge_index.numel() > 0:
        src, dst = edge_index[0].to(q.device), edge_index[1].to(q.device)
        sim = (anchor_det[src] * anchor_det[dst]).sum(dim=1)
        local_sum = q.new_zeros(n)
        local_cnt = q.new_zeros(n)
        one = torch.ones_like(sim)
        local_sum.index_add_(0, src, sim)
        local_sum.index_add_(0, dst, sim)
        local_cnt.index_add_(0, src, one)
        local_cnt.index_add_(0, dst, one)
        local = local_sum / local_cnt.clamp_min(1.0)
        local = torch.where(local_cnt > 0.0, local, conf)
    else:
        local = conf
    local_norm = ((local - (1.0 / float(k))) * inv_uniform_gap).clamp(0.0, 1.0)

    warm = max(0, int(warmup_epochs))
    ramp = max(1, int(ramp_epochs))
    gamma_value = float(np.clip((float(current_epoch + 1 - warm) / float(ramp)), 0.0, 1.0))
    beta_value = float(np.clip(float(residual_beta), 0.0, 1.0))
    gamma = q.new_tensor(gamma_value)
    beta = q.new_tensor(beta_value)
    r_base = (0.5 * conf + 0.5 * local_norm).clamp(0.0, 1.0)
    r_agree = (0.5 * qa_norm + 0.5 * ea_norm).clamp(0.0, 1.0)
    multiplier = ((1.0 - gamma) + gamma * (beta + (1.0 - beta) * r_agree)).clamp(0.0, 1.0)
    reliability = (r_base * multiplier).clamp(0.0, 1.0).detach()

    per_node_kl = torch.sum(q_safe * (q_safe.clamp_min(1e-8).log() - anchor.clamp_min(1e-8).log()), dim=1)
    min_mass = max(float(reliability_floor), float(min_effective_mass), 1e-8) * float(max(1, n))
    denom = reliability.sum().clamp_min(float(min_mass))
    loss = torch.sum(reliability * per_node_kl) / denom

    usage = anchor.mean(dim=0).clamp_min(1e-8)
    anchor_entropy = -torch.sum(anchor * anchor.clamp_min(1e-8).log(), dim=1).mean() / math.log(float(k))
    anchor_conf = anchor.max(dim=1).values.mean()
    usage_entropy = -torch.sum(usage * usage.log()) / math.log(float(k))
    anchor_label = anchor.argmax(dim=1)
    q_label = q_safe.argmax(dim=1)
    embed_label = embed_safe.argmax(dim=1)
    match = (q_label == anchor_label).to(q.dtype)
    weighted_agreement = torch.sum(reliability * match) / denom
    rel_sorted = torch.sort(reliability).values
    reliable_node_ratio = (reliability >= float(reliable_threshold)).to(q.dtype).mean()
    stats = {
        "v53a_gamma": gamma.detach(),
        "v53a_residual_beta": beta.detach(),
        "v53a_residual_multiplier_mean": multiplier.mean().detach(),
        "v53a_anchor_loss": loss.detach(),
        "v53a_weighted_q_anchor_kl": loss.detach(),
        "v53a_weighted_q_anchor_agreement": weighted_agreement.detach(),
        "v53a_unweighted_q_anchor_agreement": match.mean().detach(),
        "v53a_embedding_anchor_agreement": (embed_label == anchor_label).to(q.dtype).mean().detach(),
        "v53a_anchor_entropy": anchor_entropy.detach(),
        "v53a_anchor_confidence": anchor_conf.detach(),
        "v53a_anchor_cluster_usage_entropy": usage_entropy.detach(),
        "v53a_anchor_effective_weight": q.new_tensor(float(effective_weight)).detach(),
        "v53a_reliability_mean": reliability.mean().detach(),
        "v53a_reliability_std": reliability.std(unbiased=False).detach(),
        "v53a_reliability_p10": torch.quantile(rel_sorted, 0.10).detach(),
        "v53a_reliability_p50": torch.quantile(rel_sorted, 0.50).detach(),
        "v53a_reliability_p90": torch.quantile(rel_sorted, 0.90).detach(),
        "v53a_reliable_node_ratio": reliable_node_ratio.detach(),
        "v53a_effective_anchor_mass": reliability.mean().detach(),
        "v53a_base_reliability_mean": r_base.mean().detach(),
        "v53a_agreement_reliability_mean": r_agree.mean().detach(),
        "v53a_confidence_component_mean": conf.mean().detach(),
        "v53a_q_anchor_component_mean": qa_norm.mean().detach(),
        "v53a_embed_anchor_component_mean": ea_norm.mean().detach(),
        "v53a_local_component_mean": local_norm.mean().detach(),
    }
    return loss, stats


def consensus_bounded_residual_spectral_anchor_loss(
    q: torch.Tensor,
    q_anchor: torch.Tensor,
    *,
    q_embed: torch.Tensor | None,
    edge_index: torch.Tensor,
    enabled: bool,
    current_epoch: int,
    effective_weight: float,
    reliability_floor: float,
    reliable_threshold: float,
    min_effective_mass: float,
    warmup_epochs: int,
    ramp_epochs: int,
    beta_min: float,
    beta_max: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    zero = q.new_tensor(0.0)
    stats_zero = {
        "v54a_gamma": zero,
        "v54a_beta_min": zero,
        "v54a_beta_max": zero,
        "v54a_beta_mean": zero,
        "v54a_beta_p10": zero,
        "v54a_beta_p50": zero,
        "v54a_beta_p90": zero,
        "v54a_hard_q_anchor_match_ratio": zero,
        "v54a_hard_embed_anchor_match_ratio": zero,
        "v54a_hard_both_anchor_match_ratio": zero,
        "v54a_residual_multiplier_mean": zero,
        "v54a_anchor_loss": zero,
        "v54a_weighted_q_anchor_kl": zero,
        "v54a_weighted_q_anchor_agreement": zero,
        "v54a_unweighted_q_anchor_agreement": zero,
        "v54a_embedding_anchor_agreement": zero,
        "v54a_anchor_entropy": zero,
        "v54a_anchor_confidence": zero,
        "v54a_anchor_cluster_usage_entropy": zero,
        "v54a_anchor_effective_weight": zero,
        "v54a_reliability_mean": zero,
        "v54a_reliability_std": zero,
        "v54a_reliability_p10": zero,
        "v54a_reliability_p50": zero,
        "v54a_reliability_p90": zero,
        "v54a_reliable_node_ratio": zero,
        "v54a_effective_anchor_mass": zero,
        "v54a_base_reliability_mean": zero,
        "v54a_agreement_reliability_mean": zero,
        "v54a_confidence_component_mean": zero,
        "v54a_q_anchor_component_mean": zero,
        "v54a_embed_anchor_component_mean": zero,
        "v54a_local_component_mean": zero,
    }
    if (not bool(enabled)) or q_anchor.numel() != q.numel():
        return zero, stats_zero

    anchor = q_anchor.to(device=q.device, dtype=q.dtype).detach().clamp_min(1e-8)
    anchor = anchor / anchor.sum(dim=1, keepdim=True).clamp_min(1e-8)
    q_safe = q.clamp_min(1e-8)
    q_safe = q_safe / q_safe.sum(dim=1, keepdim=True).clamp_min(1e-8)
    if q_embed is not None and q_embed.numel() == q.numel():
        embed_safe = q_embed.to(device=q.device, dtype=q.dtype).clamp_min(1e-8)
        embed_safe = embed_safe / embed_safe.sum(dim=1, keepdim=True).clamp_min(1e-8)
    else:
        embed_safe = q_safe.detach()

    n = int(anchor.shape[0])
    k = max(2, int(anchor.shape[1]))
    inv_uniform_gap = float(k) / float(k - 1)
    anchor_det = anchor.detach()
    q_det = q_safe.detach()
    embed_det = embed_safe.detach()

    conf = ((anchor_det.max(dim=1).values - (1.0 / float(k))) * inv_uniform_gap).clamp(0.0, 1.0)
    qa_norm = (((q_det * anchor_det).sum(dim=1) - (1.0 / float(k))) * inv_uniform_gap).clamp(0.0, 1.0)
    ea_norm = (((embed_det * anchor_det).sum(dim=1) - (1.0 / float(k))) * inv_uniform_gap).clamp(0.0, 1.0)

    if edge_index.numel() > 0:
        src, dst = edge_index[0].to(q.device), edge_index[1].to(q.device)
        sim = (anchor_det[src] * anchor_det[dst]).sum(dim=1)
        local_sum = q.new_zeros(n)
        local_cnt = q.new_zeros(n)
        one = torch.ones_like(sim)
        local_sum.index_add_(0, src, sim)
        local_sum.index_add_(0, dst, sim)
        local_cnt.index_add_(0, src, one)
        local_cnt.index_add_(0, dst, one)
        local = local_sum / local_cnt.clamp_min(1.0)
        local = torch.where(local_cnt > 0.0, local, conf)
    else:
        local = conf
    local_norm = ((local - (1.0 / float(k))) * inv_uniform_gap).clamp(0.0, 1.0)

    warm = max(0, int(warmup_epochs))
    ramp = max(1, int(ramp_epochs))
    gamma_value = float(np.clip((float(current_epoch + 1 - warm) / float(ramp)), 0.0, 1.0))
    beta_lo = float(np.clip(float(beta_min), 0.0, 1.0))
    beta_hi = float(np.clip(float(beta_max), 0.0, 1.0))
    if beta_hi < beta_lo:
        beta_lo, beta_hi = beta_hi, beta_lo
    gamma = q.new_tensor(gamma_value)
    beta_min_t = q.new_tensor(beta_lo)
    beta_max_t = q.new_tensor(beta_hi)

    anchor_label = anchor_det.argmax(dim=1)
    q_label = q_det.argmax(dim=1)
    embed_label = embed_det.argmax(dim=1)
    hard_q = (q_label == anchor_label).to(q.dtype).detach()
    hard_embed = (embed_label == anchor_label).to(q.dtype).detach()
    hard_both = (hard_q * hard_embed).detach()
    hard_consensus = (0.5 * hard_q + 0.5 * hard_embed).detach()
    beta = (beta_min_t + (beta_max_t - beta_min_t) * hard_consensus).clamp(beta_lo, beta_hi).detach()

    r_base = (0.5 * conf + 0.5 * local_norm).clamp(0.0, 1.0)
    r_agree = (0.5 * qa_norm + 0.5 * ea_norm).clamp(0.0, 1.0)
    multiplier = ((1.0 - gamma) + gamma * (beta + (1.0 - beta) * r_agree)).clamp(0.0, 1.0).detach()
    reliability = (r_base * multiplier).clamp(0.0, 1.0).detach()

    per_node_kl = torch.sum(q_safe * (q_safe.clamp_min(1e-8).log() - anchor.clamp_min(1e-8).log()), dim=1)
    min_mass = max(float(reliability_floor), float(min_effective_mass), 1e-8) * float(max(1, n))
    denom = reliability.sum().clamp_min(float(min_mass))
    loss = torch.sum(reliability * per_node_kl) / denom

    usage = anchor.mean(dim=0).clamp_min(1e-8)
    anchor_entropy = -torch.sum(anchor * anchor.clamp_min(1e-8).log(), dim=1).mean() / math.log(float(k))
    anchor_conf = anchor.max(dim=1).values.mean()
    usage_entropy = -torch.sum(usage * usage.log()) / math.log(float(k))
    match = (q_label == anchor_label).to(q.dtype)
    weighted_agreement = torch.sum(reliability * match) / denom
    rel_sorted = torch.sort(reliability).values
    beta_sorted = torch.sort(beta).values
    reliable_node_ratio = (reliability >= float(reliable_threshold)).to(q.dtype).mean()
    stats = {
        "v54a_gamma": gamma.detach(),
        "v54a_beta_min": beta_min_t.detach(),
        "v54a_beta_max": beta_max_t.detach(),
        "v54a_beta_mean": beta.mean().detach(),
        "v54a_beta_p10": torch.quantile(beta_sorted, 0.10).detach(),
        "v54a_beta_p50": torch.quantile(beta_sorted, 0.50).detach(),
        "v54a_beta_p90": torch.quantile(beta_sorted, 0.90).detach(),
        "v54a_hard_q_anchor_match_ratio": hard_q.mean().detach(),
        "v54a_hard_embed_anchor_match_ratio": hard_embed.mean().detach(),
        "v54a_hard_both_anchor_match_ratio": hard_both.mean().detach(),
        "v54a_residual_multiplier_mean": multiplier.mean().detach(),
        "v54a_anchor_loss": loss.detach(),
        "v54a_weighted_q_anchor_kl": loss.detach(),
        "v54a_weighted_q_anchor_agreement": weighted_agreement.detach(),
        "v54a_unweighted_q_anchor_agreement": match.mean().detach(),
        "v54a_embedding_anchor_agreement": (embed_label == anchor_label).to(q.dtype).mean().detach(),
        "v54a_anchor_entropy": anchor_entropy.detach(),
        "v54a_anchor_confidence": anchor_conf.detach(),
        "v54a_anchor_cluster_usage_entropy": usage_entropy.detach(),
        "v54a_anchor_effective_weight": q.new_tensor(float(effective_weight)).detach(),
        "v54a_reliability_mean": reliability.mean().detach(),
        "v54a_reliability_std": reliability.std(unbiased=False).detach(),
        "v54a_reliability_p10": torch.quantile(rel_sorted, 0.10).detach(),
        "v54a_reliability_p50": torch.quantile(rel_sorted, 0.50).detach(),
        "v54a_reliability_p90": torch.quantile(rel_sorted, 0.90).detach(),
        "v54a_reliable_node_ratio": reliable_node_ratio.detach(),
        "v54a_effective_anchor_mass": reliability.mean().detach(),
        "v54a_base_reliability_mean": r_base.mean().detach(),
        "v54a_agreement_reliability_mean": r_agree.mean().detach(),
        "v54a_confidence_component_mean": conf.mean().detach(),
        "v54a_q_anchor_component_mean": qa_norm.mean().detach(),
        "v54a_embed_anchor_component_mean": ea_norm.mean().detach(),
        "v54a_local_component_mean": local_norm.mean().detach(),
    }
    return loss, stats


def soft_consensus_bounded_residual_spectral_anchor_loss(
    q: torch.Tensor,
    q_anchor: torch.Tensor,
    *,
    q_embed: torch.Tensor | None,
    edge_index: torch.Tensor,
    enabled: bool,
    current_epoch: int,
    effective_weight: float,
    reliability_floor: float,
    reliable_threshold: float,
    min_effective_mass: float,
    warmup_epochs: int,
    ramp_epochs: int,
    beta_min: float,
    beta_max: float,
    soft_power: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    zero = q.new_tensor(0.0)
    stats_zero = {
        "v55a_gamma": zero,
        "v55a_beta_min": zero,
        "v55a_beta_max": zero,
        "v55a_soft_power": zero,
        "v55a_soft_consensus_mean": zero,
        "v55a_soft_consensus_p10": zero,
        "v55a_soft_consensus_p50": zero,
        "v55a_soft_consensus_p90": zero,
        "v55a_beta_mean": zero,
        "v55a_beta_p10": zero,
        "v55a_beta_p50": zero,
        "v55a_beta_p90": zero,
        "v55a_residual_multiplier_mean": zero,
        "v55a_anchor_loss": zero,
        "v55a_weighted_q_anchor_kl": zero,
        "v55a_weighted_q_anchor_agreement": zero,
        "v55a_unweighted_q_anchor_agreement": zero,
        "v55a_embedding_anchor_agreement": zero,
        "v55a_anchor_entropy": zero,
        "v55a_anchor_confidence": zero,
        "v55a_anchor_cluster_usage_entropy": zero,
        "v55a_anchor_effective_weight": zero,
        "v55a_reliability_mean": zero,
        "v55a_reliability_std": zero,
        "v55a_reliability_p10": zero,
        "v55a_reliability_p50": zero,
        "v55a_reliability_p90": zero,
        "v55a_reliable_node_ratio": zero,
        "v55a_effective_anchor_mass": zero,
        "v55a_base_reliability_mean": zero,
        "v55a_agreement_reliability_mean": zero,
        "v55a_confidence_component_mean": zero,
        "v55a_q_anchor_component_mean": zero,
        "v55a_embed_anchor_component_mean": zero,
        "v55a_local_component_mean": zero,
    }
    if (not bool(enabled)) or q_anchor.numel() != q.numel():
        return zero, stats_zero

    anchor = q_anchor.to(device=q.device, dtype=q.dtype).detach().clamp_min(1e-8)
    anchor = anchor / anchor.sum(dim=1, keepdim=True).clamp_min(1e-8)
    q_safe = q.clamp_min(1e-8)
    q_safe = q_safe / q_safe.sum(dim=1, keepdim=True).clamp_min(1e-8)
    if q_embed is not None and q_embed.numel() == q.numel():
        embed_safe = q_embed.to(device=q.device, dtype=q.dtype).clamp_min(1e-8)
        embed_safe = embed_safe / embed_safe.sum(dim=1, keepdim=True).clamp_min(1e-8)
    else:
        embed_safe = q_safe.detach()

    n = int(anchor.shape[0])
    k = max(2, int(anchor.shape[1]))
    inv_uniform_gap = float(k) / float(k - 1)
    anchor_det = anchor.detach()
    q_det = q_safe.detach()
    embed_det = embed_safe.detach()

    conf = ((anchor_det.max(dim=1).values - (1.0 / float(k))) * inv_uniform_gap).clamp(0.0, 1.0)
    qa_norm = (((q_det * anchor_det).sum(dim=1) - (1.0 / float(k))) * inv_uniform_gap).clamp(0.0, 1.0)
    ea_norm = (((embed_det * anchor_det).sum(dim=1) - (1.0 / float(k))) * inv_uniform_gap).clamp(0.0, 1.0)

    if edge_index.numel() > 0:
        src, dst = edge_index[0].to(q.device), edge_index[1].to(q.device)
        sim = (anchor_det[src] * anchor_det[dst]).sum(dim=1)
        local_sum = q.new_zeros(n)
        local_cnt = q.new_zeros(n)
        one = torch.ones_like(sim)
        local_sum.index_add_(0, src, sim)
        local_sum.index_add_(0, dst, sim)
        local_cnt.index_add_(0, src, one)
        local_cnt.index_add_(0, dst, one)
        local = local_sum / local_cnt.clamp_min(1.0)
        local = torch.where(local_cnt > 0.0, local, conf)
    else:
        local = conf
    local_norm = ((local - (1.0 / float(k))) * inv_uniform_gap).clamp(0.0, 1.0)

    warm = max(0, int(warmup_epochs))
    ramp = max(1, int(ramp_epochs))
    gamma_value = float(np.clip((float(current_epoch + 1 - warm) / float(ramp)), 0.0, 1.0))
    beta_lo = float(np.clip(float(beta_min), 0.0, 1.0))
    beta_hi = float(np.clip(float(beta_max), 0.0, 1.0))
    if beta_hi < beta_lo:
        beta_lo, beta_hi = beta_hi, beta_lo
    power_value = max(1e-8, float(soft_power))
    gamma = q.new_tensor(gamma_value)
    beta_min_t = q.new_tensor(beta_lo)
    beta_max_t = q.new_tensor(beta_hi)
    soft_power_t = q.new_tensor(power_value)

    soft_consensus = (0.5 * qa_norm + 0.5 * ea_norm).clamp(0.0, 1.0).detach()
    lifted_consensus = soft_consensus.clamp_min(0.0).pow(power_value).clamp(0.0, 1.0).detach()
    beta = (beta_min_t + (beta_max_t - beta_min_t) * lifted_consensus).clamp(beta_lo, beta_hi).detach()

    r_base = (0.5 * conf + 0.5 * local_norm).clamp(0.0, 1.0)
    r_agree = (0.5 * qa_norm + 0.5 * ea_norm).clamp(0.0, 1.0)
    multiplier = ((1.0 - gamma) + gamma * (beta + (1.0 - beta) * r_agree)).clamp(0.0, 1.0).detach()
    reliability = (r_base * multiplier).clamp(0.0, 1.0).detach()

    per_node_kl = torch.sum(q_safe * (q_safe.clamp_min(1e-8).log() - anchor.clamp_min(1e-8).log()), dim=1)
    min_mass = max(float(reliability_floor), float(min_effective_mass), 1e-8) * float(max(1, n))
    denom = reliability.sum().clamp_min(float(min_mass))
    loss = torch.sum(reliability * per_node_kl) / denom

    anchor_label = anchor_det.argmax(dim=1)
    q_label = q_det.argmax(dim=1)
    embed_label = embed_det.argmax(dim=1)
    usage = anchor.mean(dim=0).clamp_min(1e-8)
    anchor_entropy = -torch.sum(anchor * anchor.clamp_min(1e-8).log(), dim=1).mean() / math.log(float(k))
    anchor_conf = anchor.max(dim=1).values.mean()
    usage_entropy = -torch.sum(usage * usage.log()) / math.log(float(k))
    match = (q_label == anchor_label).to(q.dtype)
    weighted_agreement = torch.sum(reliability * match) / denom
    rel_sorted = torch.sort(reliability).values
    soft_sorted = torch.sort(soft_consensus).values
    beta_sorted = torch.sort(beta).values
    reliable_node_ratio = (reliability >= float(reliable_threshold)).to(q.dtype).mean()
    stats = {
        "v55a_gamma": gamma.detach(),
        "v55a_beta_min": beta_min_t.detach(),
        "v55a_beta_max": beta_max_t.detach(),
        "v55a_soft_power": soft_power_t.detach(),
        "v55a_soft_consensus_mean": soft_consensus.mean().detach(),
        "v55a_soft_consensus_p10": torch.quantile(soft_sorted, 0.10).detach(),
        "v55a_soft_consensus_p50": torch.quantile(soft_sorted, 0.50).detach(),
        "v55a_soft_consensus_p90": torch.quantile(soft_sorted, 0.90).detach(),
        "v55a_beta_mean": beta.mean().detach(),
        "v55a_beta_p10": torch.quantile(beta_sorted, 0.10).detach(),
        "v55a_beta_p50": torch.quantile(beta_sorted, 0.50).detach(),
        "v55a_beta_p90": torch.quantile(beta_sorted, 0.90).detach(),
        "v55a_residual_multiplier_mean": multiplier.mean().detach(),
        "v55a_anchor_loss": loss.detach(),
        "v55a_weighted_q_anchor_kl": loss.detach(),
        "v55a_weighted_q_anchor_agreement": weighted_agreement.detach(),
        "v55a_unweighted_q_anchor_agreement": match.mean().detach(),
        "v55a_embedding_anchor_agreement": (embed_label == anchor_label).to(q.dtype).mean().detach(),
        "v55a_anchor_entropy": anchor_entropy.detach(),
        "v55a_anchor_confidence": anchor_conf.detach(),
        "v55a_anchor_cluster_usage_entropy": usage_entropy.detach(),
        "v55a_anchor_effective_weight": q.new_tensor(float(effective_weight)).detach(),
        "v55a_reliability_mean": reliability.mean().detach(),
        "v55a_reliability_std": reliability.std(unbiased=False).detach(),
        "v55a_reliability_p10": torch.quantile(rel_sorted, 0.10).detach(),
        "v55a_reliability_p50": torch.quantile(rel_sorted, 0.50).detach(),
        "v55a_reliability_p90": torch.quantile(rel_sorted, 0.90).detach(),
        "v55a_reliable_node_ratio": reliable_node_ratio.detach(),
        "v55a_effective_anchor_mass": reliability.mean().detach(),
        "v55a_base_reliability_mean": r_base.mean().detach(),
        "v55a_agreement_reliability_mean": r_agree.mean().detach(),
        "v55a_confidence_component_mean": conf.mean().detach(),
        "v55a_q_anchor_component_mean": qa_norm.mean().detach(),
        "v55a_embed_anchor_component_mean": ea_norm.mean().detach(),
        "v55a_local_component_mean": local_norm.mean().detach(),
    }
    return loss, stats


def hybrid_consensus_floor_residual_spectral_anchor_loss(
    q: torch.Tensor,
    q_anchor: torch.Tensor,
    *,
    q_embed: torch.Tensor | None,
    edge_index: torch.Tensor,
    enabled: bool,
    current_epoch: int,
    effective_weight: float,
    reliability_floor: float,
    reliable_threshold: float,
    min_effective_mass: float,
    warmup_epochs: int,
    ramp_epochs: int,
    beta_min: float,
    beta_max: float,
    soft_power: float,
    hybrid_compensation: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    zero = q.new_tensor(0.0)
    stats_zero = {
        "v56a_gamma": zero,
        "v56a_beta_min": zero,
        "v56a_beta_max": zero,
        "v56a_soft_power": zero,
        "v56a_hybrid_compensation": zero,
        "v56a_hard_consensus_mean": zero,
        "v56a_soft_consensus_mean": zero,
        "v56a_lifted_soft_consensus_mean": zero,
        "v56a_compensation_mean": zero,
        "v56a_compensation_active_ratio": zero,
        "v56a_hybrid_consensus_mean": zero,
        "v56a_beta_mean": zero,
        "v56a_beta_p10": zero,
        "v56a_beta_p50": zero,
        "v56a_beta_p90": zero,
        "v56a_residual_multiplier_mean": zero,
        "v56a_anchor_loss": zero,
        "v56a_weighted_q_anchor_kl": zero,
        "v56a_weighted_q_anchor_agreement": zero,
        "v56a_unweighted_q_anchor_agreement": zero,
        "v56a_embedding_anchor_agreement": zero,
        "v56a_anchor_entropy": zero,
        "v56a_anchor_confidence": zero,
        "v56a_anchor_cluster_usage_entropy": zero,
        "v56a_anchor_effective_weight": zero,
        "v56a_reliability_mean": zero,
        "v56a_reliability_std": zero,
        "v56a_reliability_p10": zero,
        "v56a_reliability_p50": zero,
        "v56a_reliability_p90": zero,
        "v56a_reliable_node_ratio": zero,
        "v56a_effective_anchor_mass": zero,
        "v56a_base_reliability_mean": zero,
        "v56a_agreement_reliability_mean": zero,
        "v56a_confidence_component_mean": zero,
        "v56a_q_anchor_component_mean": zero,
        "v56a_embed_anchor_component_mean": zero,
        "v56a_local_component_mean": zero,
    }
    if (not bool(enabled)) or q_anchor.numel() != q.numel():
        return zero, stats_zero

    anchor = q_anchor.to(device=q.device, dtype=q.dtype).detach().clamp_min(1e-8)
    anchor = anchor / anchor.sum(dim=1, keepdim=True).clamp_min(1e-8)
    q_safe = q.clamp_min(1e-8)
    q_safe = q_safe / q_safe.sum(dim=1, keepdim=True).clamp_min(1e-8)
    if q_embed is not None and q_embed.numel() == q.numel():
        embed_safe = q_embed.to(device=q.device, dtype=q.dtype).clamp_min(1e-8)
        embed_safe = embed_safe / embed_safe.sum(dim=1, keepdim=True).clamp_min(1e-8)
    else:
        embed_safe = q_safe.detach()

    n = int(anchor.shape[0])
    k = max(2, int(anchor.shape[1]))
    inv_uniform_gap = float(k) / float(k - 1)
    anchor_det = anchor.detach()
    q_det = q_safe.detach()
    embed_det = embed_safe.detach()

    conf = ((anchor_det.max(dim=1).values - (1.0 / float(k))) * inv_uniform_gap).clamp(0.0, 1.0)
    qa_norm = (((q_det * anchor_det).sum(dim=1) - (1.0 / float(k))) * inv_uniform_gap).clamp(0.0, 1.0)
    ea_norm = (((embed_det * anchor_det).sum(dim=1) - (1.0 / float(k))) * inv_uniform_gap).clamp(0.0, 1.0)

    if edge_index.numel() > 0:
        src, dst = edge_index[0].to(q.device), edge_index[1].to(q.device)
        sim = (anchor_det[src] * anchor_det[dst]).sum(dim=1)
        local_sum = q.new_zeros(n)
        local_cnt = q.new_zeros(n)
        one = torch.ones_like(sim)
        local_sum.index_add_(0, src, sim)
        local_sum.index_add_(0, dst, sim)
        local_cnt.index_add_(0, src, one)
        local_cnt.index_add_(0, dst, one)
        local = local_sum / local_cnt.clamp_min(1.0)
        local = torch.where(local_cnt > 0.0, local, conf)
    else:
        local = conf
    local_norm = ((local - (1.0 / float(k))) * inv_uniform_gap).clamp(0.0, 1.0)

    warm = max(0, int(warmup_epochs))
    ramp = max(1, int(ramp_epochs))
    gamma_value = float(np.clip((float(current_epoch + 1 - warm) / float(ramp)), 0.0, 1.0))
    beta_lo = float(np.clip(float(beta_min), 0.0, 1.0))
    beta_hi = float(np.clip(float(beta_max), 0.0, 1.0))
    if beta_hi < beta_lo:
        beta_lo, beta_hi = beta_hi, beta_lo
    power_value = max(1e-8, float(soft_power))
    compensation_value = float(np.clip(float(hybrid_compensation), 0.0, 1.0))
    gamma = q.new_tensor(gamma_value)
    beta_min_t = q.new_tensor(beta_lo)
    beta_max_t = q.new_tensor(beta_hi)
    soft_power_t = q.new_tensor(power_value)
    compensation_t = q.new_tensor(compensation_value)

    anchor_label = anchor_det.argmax(dim=1)
    q_label = q_det.argmax(dim=1)
    embed_label = embed_det.argmax(dim=1)
    hard_q = (q_label == anchor_label).to(q.dtype).detach()
    hard_embed = (embed_label == anchor_label).to(q.dtype).detach()
    hard_consensus = (0.5 * hard_q + 0.5 * hard_embed).detach()
    soft_consensus = (0.5 * qa_norm + 0.5 * ea_norm).clamp(0.0, 1.0).detach()
    lifted_soft = soft_consensus.clamp_min(0.0).pow(power_value).clamp(0.0, 1.0).detach()
    compensation = (compensation_t * F.relu(lifted_soft - hard_consensus)).clamp(0.0, 1.0).detach()
    hybrid_consensus = (hard_consensus + compensation).clamp(0.0, 1.0).detach()
    beta = (beta_min_t + (beta_max_t - beta_min_t) * hybrid_consensus).clamp(beta_lo, beta_hi).detach()

    r_base = (0.5 * conf + 0.5 * local_norm).clamp(0.0, 1.0)
    r_agree = (0.5 * qa_norm + 0.5 * ea_norm).clamp(0.0, 1.0)
    multiplier = ((1.0 - gamma) + gamma * (beta + (1.0 - beta) * r_agree)).clamp(0.0, 1.0).detach()
    reliability = (r_base * multiplier).clamp(0.0, 1.0).detach()

    per_node_kl = torch.sum(q_safe * (q_safe.clamp_min(1e-8).log() - anchor.clamp_min(1e-8).log()), dim=1)
    min_mass = max(float(reliability_floor), float(min_effective_mass), 1e-8) * float(max(1, n))
    denom = reliability.sum().clamp_min(float(min_mass))
    loss = torch.sum(reliability * per_node_kl) / denom

    usage = anchor.mean(dim=0).clamp_min(1e-8)
    anchor_entropy = -torch.sum(anchor * anchor.clamp_min(1e-8).log(), dim=1).mean() / math.log(float(k))
    anchor_conf = anchor.max(dim=1).values.mean()
    usage_entropy = -torch.sum(usage * usage.log()) / math.log(float(k))
    match = (q_label == anchor_label).to(q.dtype)
    weighted_agreement = torch.sum(reliability * match) / denom
    rel_sorted = torch.sort(reliability).values
    beta_sorted = torch.sort(beta).values
    reliable_node_ratio = (reliability >= float(reliable_threshold)).to(q.dtype).mean()
    stats = {
        "v56a_gamma": gamma.detach(),
        "v56a_beta_min": beta_min_t.detach(),
        "v56a_beta_max": beta_max_t.detach(),
        "v56a_soft_power": soft_power_t.detach(),
        "v56a_hybrid_compensation": compensation_t.detach(),
        "v56a_hard_consensus_mean": hard_consensus.mean().detach(),
        "v56a_soft_consensus_mean": soft_consensus.mean().detach(),
        "v56a_lifted_soft_consensus_mean": lifted_soft.mean().detach(),
        "v56a_compensation_mean": compensation.mean().detach(),
        "v56a_compensation_active_ratio": (compensation > 0.0).to(q.dtype).mean().detach(),
        "v56a_hybrid_consensus_mean": hybrid_consensus.mean().detach(),
        "v56a_beta_mean": beta.mean().detach(),
        "v56a_beta_p10": torch.quantile(beta_sorted, 0.10).detach(),
        "v56a_beta_p50": torch.quantile(beta_sorted, 0.50).detach(),
        "v56a_beta_p90": torch.quantile(beta_sorted, 0.90).detach(),
        "v56a_residual_multiplier_mean": multiplier.mean().detach(),
        "v56a_anchor_loss": loss.detach(),
        "v56a_weighted_q_anchor_kl": loss.detach(),
        "v56a_weighted_q_anchor_agreement": weighted_agreement.detach(),
        "v56a_unweighted_q_anchor_agreement": match.mean().detach(),
        "v56a_embedding_anchor_agreement": (embed_label == anchor_label).to(q.dtype).mean().detach(),
        "v56a_anchor_entropy": anchor_entropy.detach(),
        "v56a_anchor_confidence": anchor_conf.detach(),
        "v56a_anchor_cluster_usage_entropy": usage_entropy.detach(),
        "v56a_anchor_effective_weight": q.new_tensor(float(effective_weight)).detach(),
        "v56a_reliability_mean": reliability.mean().detach(),
        "v56a_reliability_std": reliability.std(unbiased=False).detach(),
        "v56a_reliability_p10": torch.quantile(rel_sorted, 0.10).detach(),
        "v56a_reliability_p50": torch.quantile(rel_sorted, 0.50).detach(),
        "v56a_reliability_p90": torch.quantile(rel_sorted, 0.90).detach(),
        "v56a_reliable_node_ratio": reliable_node_ratio.detach(),
        "v56a_effective_anchor_mass": reliability.mean().detach(),
        "v56a_base_reliability_mean": r_base.mean().detach(),
        "v56a_agreement_reliability_mean": r_agree.mean().detach(),
        "v56a_confidence_component_mean": conf.mean().detach(),
        "v56a_q_anchor_component_mean": qa_norm.mean().detach(),
        "v56a_embed_anchor_component_mean": ea_norm.mean().detach(),
        "v56a_local_component_mean": local_norm.mean().detach(),
    }
    return loss, stats


def mass_floor_normalized_residual_spectral_anchor_loss(
    q: torch.Tensor,
    q_anchor: torch.Tensor,
    *,
    q_embed: torch.Tensor | None,
    edge_index: torch.Tensor,
    enabled: bool,
    current_epoch: int,
    effective_weight: float,
    reliability_floor: float,
    reliable_threshold: float,
    min_effective_mass: float,
    warmup_epochs: int,
    ramp_epochs: int,
    beta_min: float,
    beta_max: float,
    soft_power: float,
    hybrid_compensation: float,
    target_mass: float,
    max_mass_scale: float,
    max_reliability_cap: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    zero = q.new_tensor(0.0)
    stats_zero = {
        "v57a_gamma": zero,
        "v57a_beta_min": zero,
        "v57a_beta_max": zero,
        "v57a_soft_power": zero,
        "v57a_hybrid_compensation": zero,
        "v57a_target_mass": zero,
        "v57a_max_mass_scale": zero,
        "v57a_max_reliability_cap": zero,
        "v57a_hard_consensus_mean": zero,
        "v57a_soft_consensus_mean": zero,
        "v57a_lifted_soft_consensus_mean": zero,
        "v57a_compensation_mean": zero,
        "v57a_compensation_active_ratio": zero,
        "v57a_hybrid_consensus_mean": zero,
        "v57a_beta_mean": zero,
        "v57a_raw_reliability_mean": zero,
        "v57a_mass_scale": zero,
        "v57a_scaled_reliability_mean": zero,
        "v57a_residual_multiplier_mean": zero,
        "v57a_anchor_loss": zero,
        "v57a_weighted_q_anchor_kl": zero,
        "v57a_weighted_q_anchor_agreement": zero,
        "v57a_unweighted_q_anchor_agreement": zero,
        "v57a_embedding_anchor_agreement": zero,
        "v57a_anchor_entropy": zero,
        "v57a_anchor_confidence": zero,
        "v57a_anchor_cluster_usage_entropy": zero,
        "v57a_anchor_effective_weight": zero,
        "v57a_reliability_mean": zero,
        "v57a_reliability_std": zero,
        "v57a_reliability_p10": zero,
        "v57a_reliability_p50": zero,
        "v57a_reliability_p90": zero,
        "v57a_reliable_node_ratio": zero,
        "v57a_effective_anchor_mass": zero,
        "v57a_base_reliability_mean": zero,
        "v57a_agreement_reliability_mean": zero,
        "v57a_confidence_component_mean": zero,
        "v57a_q_anchor_component_mean": zero,
        "v57a_embed_anchor_component_mean": zero,
        "v57a_local_component_mean": zero,
    }
    if (not bool(enabled)) or q_anchor.numel() != q.numel():
        return zero, stats_zero

    anchor = q_anchor.to(device=q.device, dtype=q.dtype).detach().clamp_min(1e-8)
    anchor = anchor / anchor.sum(dim=1, keepdim=True).clamp_min(1e-8)
    q_safe = q.clamp_min(1e-8)
    q_safe = q_safe / q_safe.sum(dim=1, keepdim=True).clamp_min(1e-8)
    if q_embed is not None and q_embed.numel() == q.numel():
        embed_safe = q_embed.to(device=q.device, dtype=q.dtype).clamp_min(1e-8)
        embed_safe = embed_safe / embed_safe.sum(dim=1, keepdim=True).clamp_min(1e-8)
    else:
        embed_safe = q_safe.detach()

    n = int(anchor.shape[0])
    k = max(2, int(anchor.shape[1]))
    inv_uniform_gap = float(k) / float(k - 1)
    anchor_det = anchor.detach()
    q_det = q_safe.detach()
    embed_det = embed_safe.detach()

    conf = ((anchor_det.max(dim=1).values - (1.0 / float(k))) * inv_uniform_gap).clamp(0.0, 1.0)
    qa_norm = (((q_det * anchor_det).sum(dim=1) - (1.0 / float(k))) * inv_uniform_gap).clamp(0.0, 1.0)
    ea_norm = (((embed_det * anchor_det).sum(dim=1) - (1.0 / float(k))) * inv_uniform_gap).clamp(0.0, 1.0)

    if edge_index.numel() > 0:
        src, dst = edge_index[0].to(q.device), edge_index[1].to(q.device)
        sim = (anchor_det[src] * anchor_det[dst]).sum(dim=1)
        local_sum = q.new_zeros(n)
        local_cnt = q.new_zeros(n)
        one = torch.ones_like(sim)
        local_sum.index_add_(0, src, sim)
        local_sum.index_add_(0, dst, sim)
        local_cnt.index_add_(0, src, one)
        local_cnt.index_add_(0, dst, one)
        local = local_sum / local_cnt.clamp_min(1.0)
        local = torch.where(local_cnt > 0.0, local, conf)
    else:
        local = conf
    local_norm = ((local - (1.0 / float(k))) * inv_uniform_gap).clamp(0.0, 1.0)

    warm = max(0, int(warmup_epochs))
    ramp = max(1, int(ramp_epochs))
    gamma_value = float(np.clip((float(current_epoch + 1 - warm) / float(ramp)), 0.0, 1.0))
    beta_lo = float(np.clip(float(beta_min), 0.0, 1.0))
    beta_hi = float(np.clip(float(beta_max), 0.0, 1.0))
    if beta_hi < beta_lo:
        beta_lo, beta_hi = beta_hi, beta_lo
    power_value = max(1e-8, float(soft_power))
    compensation_value = float(np.clip(float(hybrid_compensation), 0.0, 1.0))
    target_value = float(np.clip(float(target_mass), 0.0, 1.0))
    max_scale_value = max(1.0, float(max_mass_scale))
    cap_value = float(np.clip(float(max_reliability_cap), 1e-8, 1.0))
    gamma = q.new_tensor(gamma_value)
    beta_min_t = q.new_tensor(beta_lo)
    beta_max_t = q.new_tensor(beta_hi)
    soft_power_t = q.new_tensor(power_value)
    compensation_t = q.new_tensor(compensation_value)
    target_mass_t = q.new_tensor(target_value)
    max_mass_scale_t = q.new_tensor(max_scale_value)
    max_reliability_cap_t = q.new_tensor(cap_value)

    anchor_label = anchor_det.argmax(dim=1)
    q_label = q_det.argmax(dim=1)
    embed_label = embed_det.argmax(dim=1)
    hard_q = (q_label == anchor_label).to(q.dtype).detach()
    hard_embed = (embed_label == anchor_label).to(q.dtype).detach()
    hard_consensus = (0.5 * hard_q + 0.5 * hard_embed).detach()
    soft_consensus = (0.5 * qa_norm + 0.5 * ea_norm).clamp(0.0, 1.0).detach()
    lifted_soft = soft_consensus.clamp_min(0.0).pow(power_value).clamp(0.0, 1.0).detach()
    compensation = (compensation_t * F.relu(lifted_soft - hard_consensus)).clamp(0.0, 1.0).detach()
    hybrid_consensus = (hard_consensus + compensation).clamp(0.0, 1.0).detach()
    beta = (beta_min_t + (beta_max_t - beta_min_t) * hybrid_consensus).clamp(beta_lo, beta_hi).detach()

    r_base = (0.5 * conf + 0.5 * local_norm).clamp(0.0, 1.0)
    r_agree = (0.5 * qa_norm + 0.5 * ea_norm).clamp(0.0, 1.0)
    multiplier = ((1.0 - gamma) + gamma * (beta + (1.0 - beta) * r_agree)).clamp(0.0, 1.0).detach()
    raw_reliability = (r_base * multiplier).clamp(0.0, 1.0).detach()
    raw_mass = raw_reliability.mean().detach()
    scale_value = float(np.clip(target_value / max(float(raw_mass.detach().cpu()), 1e-8), 1.0, max_scale_value))
    mass_scale = q.new_tensor(scale_value)
    reliability = (raw_reliability * mass_scale).clamp(0.0, cap_value).detach()

    per_node_kl = torch.sum(q_safe * (q_safe.clamp_min(1e-8).log() - anchor.clamp_min(1e-8).log()), dim=1)
    min_mass = max(float(reliability_floor), float(min_effective_mass), 1e-8) * float(max(1, n))
    denom = reliability.sum().clamp_min(float(min_mass))
    loss = torch.sum(reliability * per_node_kl) / denom

    usage = anchor.mean(dim=0).clamp_min(1e-8)
    anchor_entropy = -torch.sum(anchor * anchor.clamp_min(1e-8).log(), dim=1).mean() / math.log(float(k))
    anchor_conf = anchor.max(dim=1).values.mean()
    usage_entropy = -torch.sum(usage * usage.log()) / math.log(float(k))
    match = (q_label == anchor_label).to(q.dtype)
    weighted_agreement = torch.sum(reliability * match) / denom
    rel_sorted = torch.sort(reliability).values
    reliable_node_ratio = (reliability >= float(reliable_threshold)).to(q.dtype).mean()
    stats = {
        "v57a_gamma": gamma.detach(),
        "v57a_beta_min": beta_min_t.detach(),
        "v57a_beta_max": beta_max_t.detach(),
        "v57a_soft_power": soft_power_t.detach(),
        "v57a_hybrid_compensation": compensation_t.detach(),
        "v57a_target_mass": target_mass_t.detach(),
        "v57a_max_mass_scale": max_mass_scale_t.detach(),
        "v57a_max_reliability_cap": max_reliability_cap_t.detach(),
        "v57a_hard_consensus_mean": hard_consensus.mean().detach(),
        "v57a_soft_consensus_mean": soft_consensus.mean().detach(),
        "v57a_lifted_soft_consensus_mean": lifted_soft.mean().detach(),
        "v57a_compensation_mean": compensation.mean().detach(),
        "v57a_compensation_active_ratio": (compensation > 0.0).to(q.dtype).mean().detach(),
        "v57a_hybrid_consensus_mean": hybrid_consensus.mean().detach(),
        "v57a_beta_mean": beta.mean().detach(),
        "v57a_raw_reliability_mean": raw_mass.detach(),
        "v57a_mass_scale": mass_scale.detach(),
        "v57a_scaled_reliability_mean": reliability.mean().detach(),
        "v57a_residual_multiplier_mean": multiplier.mean().detach(),
        "v57a_anchor_loss": loss.detach(),
        "v57a_weighted_q_anchor_kl": loss.detach(),
        "v57a_weighted_q_anchor_agreement": weighted_agreement.detach(),
        "v57a_unweighted_q_anchor_agreement": match.mean().detach(),
        "v57a_embedding_anchor_agreement": (embed_label == anchor_label).to(q.dtype).mean().detach(),
        "v57a_anchor_entropy": anchor_entropy.detach(),
        "v57a_anchor_confidence": anchor_conf.detach(),
        "v57a_anchor_cluster_usage_entropy": usage_entropy.detach(),
        "v57a_anchor_effective_weight": q.new_tensor(float(effective_weight)).detach(),
        "v57a_reliability_mean": reliability.mean().detach(),
        "v57a_reliability_std": reliability.std(unbiased=False).detach(),
        "v57a_reliability_p10": torch.quantile(rel_sorted, 0.10).detach(),
        "v57a_reliability_p50": torch.quantile(rel_sorted, 0.50).detach(),
        "v57a_reliability_p90": torch.quantile(rel_sorted, 0.90).detach(),
        "v57a_reliable_node_ratio": reliable_node_ratio.detach(),
        "v57a_effective_anchor_mass": reliability.mean().detach(),
        "v57a_base_reliability_mean": r_base.mean().detach(),
        "v57a_agreement_reliability_mean": r_agree.mean().detach(),
        "v57a_confidence_component_mean": conf.mean().detach(),
        "v57a_q_anchor_component_mean": qa_norm.mean().detach(),
        "v57a_embed_anchor_component_mean": ea_norm.mean().detach(),
        "v57a_local_component_mean": local_norm.mean().detach(),
    }
    return loss, stats


def _v58a_release_gamma_value(
    current_epoch: int,
    *,
    release_warmup_epochs: int,
    release_ramp_epochs: int,
    release_hold_until_epoch: int,
    release_decay_epochs: int,
    release_floor: float,
) -> float:
    epoch_number = float(int(current_epoch) + 1)
    warmup = max(0.0, float(release_warmup_epochs))
    ramp = max(1.0, float(release_ramp_epochs))
    hold_until = max(warmup + ramp, float(release_hold_until_epoch))
    floor = float(np.clip(float(release_floor), 0.0, 1.0))
    decay_epochs = max(1.0, float(release_decay_epochs))
    if epoch_number <= warmup:
        return 0.0
    if epoch_number <= warmup + ramp:
        return float(np.clip((epoch_number - warmup) / ramp, 0.0, 1.0))
    if epoch_number <= hold_until:
        return 1.0
    decay_denominator = max(1.0, decay_epochs / max(1e-8, 1.0 - floor))
    decayed = 1.0 - ((epoch_number - hold_until) / decay_denominator)
    return float(max(floor, decayed))


def anchor_release_residual_spectral_anchor_loss(
    q: torch.Tensor,
    q_anchor: torch.Tensor,
    *,
    q_embed: torch.Tensor | None,
    edge_index: torch.Tensor,
    enabled: bool,
    current_epoch: int,
    effective_weight: float,
    reliability_floor: float,
    reliable_threshold: float,
    min_effective_mass: float,
    warmup_epochs: int,
    ramp_epochs: int,
    beta_min: float,
    beta_max: float,
    soft_power: float,
    hybrid_compensation: float,
    target_mass: float,
    max_mass_scale: float,
    max_reliability_cap: float,
    release_warmup_epochs: int,
    release_ramp_epochs: int,
    release_hold_until_epoch: int,
    release_decay_epochs: int,
    release_floor: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pre_release_loss, base_stats = mass_floor_normalized_residual_spectral_anchor_loss(
        q,
        q_anchor,
        q_embed=q_embed,
        edge_index=edge_index,
        enabled=enabled,
        current_epoch=current_epoch,
        effective_weight=effective_weight,
        reliability_floor=reliability_floor,
        reliable_threshold=reliable_threshold,
        min_effective_mass=min_effective_mass,
        warmup_epochs=warmup_epochs,
        ramp_epochs=ramp_epochs,
        beta_min=beta_min,
        beta_max=beta_max,
        soft_power=soft_power,
        hybrid_compensation=hybrid_compensation,
        target_mass=target_mass,
        max_mass_scale=max_mass_scale,
        max_reliability_cap=max_reliability_cap,
    )
    release_gamma = q.new_tensor(
        _v58a_release_gamma_value(
            current_epoch,
            release_warmup_epochs=release_warmup_epochs,
            release_ramp_epochs=release_ramp_epochs,
            release_hold_until_epoch=release_hold_until_epoch,
            release_decay_epochs=release_decay_epochs,
            release_floor=release_floor,
        )
    )
    if not bool(enabled):
        release_gamma = q.new_tensor(0.0)
    released_loss = release_gamma * pre_release_loss
    stats: dict[str, torch.Tensor] = {
        key.replace("v57a_", "v58a_"): value.detach()
        for key, value in base_stats.items()
        if key.startswith("v57a_")
    }
    stats["v58a_release_gamma"] = release_gamma.detach()
    stats["v58a_release_warmup_epochs"] = q.new_tensor(float(release_warmup_epochs)).detach()
    stats["v58a_release_ramp_epochs"] = q.new_tensor(float(release_ramp_epochs)).detach()
    stats["v58a_release_hold_until_epoch"] = q.new_tensor(float(release_hold_until_epoch)).detach()
    stats["v58a_release_decay_epochs"] = q.new_tensor(float(release_decay_epochs)).detach()
    stats["v58a_release_floor"] = q.new_tensor(float(release_floor)).detach()
    stats["v58a_pre_release_anchor_loss"] = pre_release_loss.detach()
    stats["v58a_pre_release_weighted_q_anchor_kl"] = base_stats["v57a_weighted_q_anchor_kl"].detach()
    stats["v58a_anchor_loss"] = released_loss.detach()
    stats["v58a_weighted_q_anchor_kl"] = (release_gamma * base_stats["v57a_weighted_q_anchor_kl"]).detach()
    return released_loss, stats


def _v59a_release_gamma_value(
    current_epoch: int,
    *,
    release_start_epoch: int,
    release_decay_epochs: int,
    release_floor: float,
) -> float:
    epoch_number = float(int(current_epoch) + 1)
    start_epoch = max(1.0, float(release_start_epoch))
    floor = float(np.clip(float(release_floor), 0.0, 1.0))
    decay_epochs = max(1.0, float(release_decay_epochs))
    if epoch_number <= start_epoch:
        return 1.0
    decay_denominator = max(1.0, decay_epochs / max(1e-8, 1.0 - floor))
    decayed = 1.0 - ((epoch_number - start_epoch) / decay_denominator)
    return float(max(floor, decayed))


def post80_anchor_release_residual_spectral_anchor_loss(
    q: torch.Tensor,
    q_anchor: torch.Tensor,
    *,
    q_embed: torch.Tensor | None,
    edge_index: torch.Tensor,
    enabled: bool,
    current_epoch: int,
    effective_weight: float,
    reliability_floor: float,
    reliable_threshold: float,
    min_effective_mass: float,
    warmup_epochs: int,
    ramp_epochs: int,
    beta_min: float,
    beta_max: float,
    soft_power: float,
    hybrid_compensation: float,
    target_mass: float,
    max_mass_scale: float,
    max_reliability_cap: float,
    release_start_epoch: int,
    release_decay_epochs: int,
    release_floor: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pre_release_loss, base_stats = mass_floor_normalized_residual_spectral_anchor_loss(
        q,
        q_anchor,
        q_embed=q_embed,
        edge_index=edge_index,
        enabled=enabled,
        current_epoch=current_epoch,
        effective_weight=effective_weight,
        reliability_floor=reliability_floor,
        reliable_threshold=reliable_threshold,
        min_effective_mass=min_effective_mass,
        warmup_epochs=warmup_epochs,
        ramp_epochs=ramp_epochs,
        beta_min=beta_min,
        beta_max=beta_max,
        soft_power=soft_power,
        hybrid_compensation=hybrid_compensation,
        target_mass=target_mass,
        max_mass_scale=max_mass_scale,
        max_reliability_cap=max_reliability_cap,
    )
    release_gamma = q.new_tensor(
        _v59a_release_gamma_value(
            current_epoch,
            release_start_epoch=release_start_epoch,
            release_decay_epochs=release_decay_epochs,
            release_floor=release_floor,
        )
    )
    if not bool(enabled):
        release_gamma = q.new_tensor(0.0)
    released_loss = release_gamma * pre_release_loss
    stats: dict[str, torch.Tensor] = {
        key.replace("v57a_", "v59a_"): value.detach()
        for key, value in base_stats.items()
        if key.startswith("v57a_")
    }
    stats["v59a_release_gamma"] = release_gamma.detach()
    stats["v59a_release_start_epoch"] = q.new_tensor(float(release_start_epoch)).detach()
    stats["v59a_release_decay_epochs"] = q.new_tensor(float(release_decay_epochs)).detach()
    stats["v59a_release_floor"] = q.new_tensor(float(release_floor)).detach()
    stats["v59a_pre_release_anchor_loss"] = pre_release_loss.detach()
    stats["v59a_pre_release_weighted_q_anchor_kl"] = base_stats["v57a_weighted_q_anchor_kl"].detach()
    stats["v59a_anchor_loss"] = released_loss.detach()
    stats["v59a_weighted_q_anchor_kl"] = (release_gamma * base_stats["v57a_weighted_q_anchor_kl"]).detach()
    return released_loss, stats


def v60a_post80_anchor_release_residual_spectral_anchor_loss(
    q: torch.Tensor,
    q_anchor: torch.Tensor,
    *,
    q_embed: torch.Tensor | None,
    edge_index: torch.Tensor,
    enabled: bool,
    current_epoch: int,
    effective_weight: float,
    reliability_floor: float,
    reliable_threshold: float,
    min_effective_mass: float,
    warmup_epochs: int,
    ramp_epochs: int,
    beta_min: float,
    beta_max: float,
    soft_power: float,
    hybrid_compensation: float,
    target_mass: float,
    max_mass_scale: float,
    max_reliability_cap: float,
    release_start_epoch: int,
    release_decay_epochs: int,
    release_floor: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss, base_stats = post80_anchor_release_residual_spectral_anchor_loss(
        q,
        q_anchor,
        q_embed=q_embed,
        edge_index=edge_index,
        enabled=enabled,
        current_epoch=current_epoch,
        effective_weight=effective_weight,
        reliability_floor=reliability_floor,
        reliable_threshold=reliable_threshold,
        min_effective_mass=min_effective_mass,
        warmup_epochs=warmup_epochs,
        ramp_epochs=ramp_epochs,
        beta_min=beta_min,
        beta_max=beta_max,
        soft_power=soft_power,
        hybrid_compensation=hybrid_compensation,
        target_mass=target_mass,
        max_mass_scale=max_mass_scale,
        max_reliability_cap=max_reliability_cap,
        release_start_epoch=release_start_epoch,
        release_decay_epochs=release_decay_epochs,
        release_floor=release_floor,
    )
    stats: dict[str, torch.Tensor] = {
        key.replace("v59a_", "v60a_"): value.detach()
        for key, value in base_stats.items()
        if key.startswith("v59a_")
    }
    return loss, stats


def v61a_post80_anchor_release_residual_spectral_anchor_loss(
    q: torch.Tensor,
    q_anchor: torch.Tensor,
    *,
    q_embed: torch.Tensor | None,
    edge_index: torch.Tensor,
    enabled: bool,
    current_epoch: int,
    effective_weight: float,
    reliability_floor: float,
    reliable_threshold: float,
    min_effective_mass: float,
    warmup_epochs: int,
    ramp_epochs: int,
    beta_min: float,
    beta_max: float,
    soft_power: float,
    hybrid_compensation: float,
    target_mass: float,
    max_mass_scale: float,
    max_reliability_cap: float,
    release_start_epoch: int,
    release_decay_epochs: int,
    release_floor: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss, base_stats = post80_anchor_release_residual_spectral_anchor_loss(
        q,
        q_anchor,
        q_embed=q_embed,
        edge_index=edge_index,
        enabled=enabled,
        current_epoch=current_epoch,
        effective_weight=effective_weight,
        reliability_floor=reliability_floor,
        reliable_threshold=reliable_threshold,
        min_effective_mass=min_effective_mass,
        warmup_epochs=warmup_epochs,
        ramp_epochs=ramp_epochs,
        beta_min=beta_min,
        beta_max=beta_max,
        soft_power=soft_power,
        hybrid_compensation=hybrid_compensation,
        target_mass=target_mass,
        max_mass_scale=max_mass_scale,
        max_reliability_cap=max_reliability_cap,
        release_start_epoch=release_start_epoch,
        release_decay_epochs=release_decay_epochs,
        release_floor=release_floor,
    )
    stats: dict[str, torch.Tensor] = {
        key.replace("v59a_", "v61a_"): value.detach()
        for key, value in base_stats.items()
        if key.startswith("v59a_")
    }
    return loss, stats


def v62a_post80_anchor_release_residual_spectral_anchor_loss(
    q: torch.Tensor,
    q_anchor: torch.Tensor,
    *,
    q_embed: torch.Tensor | None,
    edge_index: torch.Tensor,
    enabled: bool,
    current_epoch: int,
    effective_weight: float,
    reliability_floor: float,
    reliable_threshold: float,
    min_effective_mass: float,
    warmup_epochs: int,
    ramp_epochs: int,
    beta_min: float,
    beta_max: float,
    soft_power: float,
    hybrid_compensation: float,
    target_mass: float,
    max_mass_scale: float,
    max_reliability_cap: float,
    release_start_epoch: int,
    release_decay_epochs: int,
    release_floor: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss, base_stats = post80_anchor_release_residual_spectral_anchor_loss(
        q,
        q_anchor,
        q_embed=q_embed,
        edge_index=edge_index,
        enabled=enabled,
        current_epoch=current_epoch,
        effective_weight=effective_weight,
        reliability_floor=reliability_floor,
        reliable_threshold=reliable_threshold,
        min_effective_mass=min_effective_mass,
        warmup_epochs=warmup_epochs,
        ramp_epochs=ramp_epochs,
        beta_min=beta_min,
        beta_max=beta_max,
        soft_power=soft_power,
        hybrid_compensation=hybrid_compensation,
        target_mass=target_mass,
        max_mass_scale=max_mass_scale,
        max_reliability_cap=max_reliability_cap,
        release_start_epoch=release_start_epoch,
        release_decay_epochs=release_decay_epochs,
        release_floor=release_floor,
    )
    stats: dict[str, torch.Tensor] = {
        key.replace("v59a_", "v62a_"): value.detach()
        for key, value in base_stats.items()
        if key.startswith("v59a_")
    }
    return loss, stats


def _v60a_guard_gamma_value(
    current_epoch: int,
    *,
    start_epoch: int,
    ramp_epochs: int,
    max_gamma: float,
) -> float:
    epoch_number = float(int(current_epoch) + 1)
    start = float(max(1, int(start_epoch)))
    ramp = float(max(1, int(ramp_epochs)))
    cap = float(np.clip(float(max_gamma), 0.0, 1.0))
    if epoch_number <= start:
        return 0.0
    return float(np.clip((epoch_number - start) / ramp, 0.0, cap))


def v60a_self_distillation_guard_loss(
    q: torch.Tensor,
    teacher_q: torch.Tensor,
    *,
    enabled: bool,
    teacher_ready: bool,
    teacher_epoch: int,
    current_epoch: int,
    guard_weight: float,
    confidence_threshold: float,
    start_epoch: int,
    ramp_epochs: int,
    max_gamma: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    zero = q.new_tensor(0.0)
    ready = bool(enabled) and bool(teacher_ready) and teacher_q.numel() == q.numel()
    gamma = q.new_tensor(
        _v60a_guard_gamma_value(
            current_epoch,
            start_epoch=start_epoch,
            ramp_epochs=ramp_epochs,
            max_gamma=max_gamma,
        )
        if ready
        else 0.0
    )
    threshold_value = float(np.clip(float(confidence_threshold), 0.0, 1.0))
    guard_weight_t = q.new_tensor(float(guard_weight))
    teacher_epoch_t = q.new_tensor(float(teacher_epoch if ready else -1))
    threshold_t = q.new_tensor(threshold_value)
    if not ready:
        stats_zero = {
            "v60a_guard_enabled": q.new_tensor(bool(enabled)),
            "v60a_teacher_ready": q.new_tensor(False),
            "v60a_teacher_epoch": teacher_epoch_t,
            "v60a_guard_gamma": gamma.detach(),
            "v60a_guard_weight": guard_weight_t.detach(),
            "v60a_confidence_threshold": threshold_t.detach(),
            "v60a_teacher_confidence_mean": zero,
            "v60a_teacher_active_ratio": zero,
            "v60a_guard_kl": zero,
            "v60a_guard_loss": zero,
            "v60a_q_teacher_agreement": zero,
            "v60a_q_teacher_kl": zero,
        }
        return zero, stats_zero

    teacher = teacher_q.to(device=q.device, dtype=q.dtype).detach().clamp_min(1e-8)
    teacher = teacher / teacher.sum(dim=1, keepdim=True).clamp_min(1e-8)
    student = q.clamp_min(1e-8)
    student = student / student.sum(dim=1, keepdim=True).clamp_min(1e-8)
    teacher_conf = teacher.max(dim=1).values.detach()
    active = (teacher_conf >= threshold_value).detach()
    per_node_kl = torch.sum(teacher * (teacher.clamp_min(1e-8).log() - student.clamp_min(1e-8).log()), dim=1)
    if bool(active.any()):
        guard_kl = per_node_kl[active].mean()
    else:
        guard_kl = zero
    guard_loss = gamma * guard_kl
    q_label = student.detach().argmax(dim=1)
    teacher_label = teacher.argmax(dim=1)
    agreement = (q_label == teacher_label).to(q.dtype).mean()
    full_kl = per_node_kl.mean()
    stats = {
        "v60a_guard_enabled": q.new_tensor(bool(enabled)),
        "v60a_teacher_ready": q.new_tensor(True),
        "v60a_teacher_epoch": teacher_epoch_t.detach(),
        "v60a_guard_gamma": gamma.detach(),
        "v60a_guard_weight": guard_weight_t.detach(),
        "v60a_confidence_threshold": threshold_t.detach(),
        "v60a_teacher_confidence_mean": teacher_conf.mean().detach(),
        "v60a_teacher_active_ratio": active.to(q.dtype).mean().detach(),
        "v60a_guard_kl": guard_kl.detach(),
        "v60a_guard_loss": guard_loss.detach(),
        "v60a_q_teacher_agreement": agreement.detach(),
        "v60a_q_teacher_kl": full_kl.detach(),
    }
    return guard_loss, stats


def v61a_quantile_coverage_self_distillation_guard_loss(
    q: torch.Tensor,
    teacher_q: torch.Tensor,
    *,
    enabled: bool,
    teacher_ready: bool,
    teacher_epoch: int,
    current_epoch: int,
    guard_weight: float,
    absolute_floor: float,
    min_teacher_coverage: float,
    start_epoch: int,
    ramp_epochs: int,
    max_gamma: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    zero = q.new_tensor(0.0)
    ready = bool(enabled) and bool(teacher_ready) and teacher_q.numel() == q.numel()
    gamma = q.new_tensor(
        _v60a_guard_gamma_value(
            current_epoch,
            start_epoch=start_epoch,
            ramp_epochs=ramp_epochs,
            max_gamma=max_gamma,
        )
        if ready
        else 0.0
    )
    floor_value = float(np.clip(float(absolute_floor), 0.0, 1.0))
    coverage_value = float(np.clip(float(min_teacher_coverage), 0.0, 1.0))
    guard_weight_t = q.new_tensor(float(guard_weight))
    teacher_epoch_t = q.new_tensor(float(teacher_epoch if ready else -1))
    floor_t = q.new_tensor(floor_value)
    coverage_t = q.new_tensor(coverage_value)
    if not ready:
        stats_zero = {
            "v61a_guard_enabled": q.new_tensor(bool(enabled)),
            "v61a_teacher_ready": q.new_tensor(False),
            "v61a_teacher_epoch": teacher_epoch_t,
            "v61a_guard_gamma": gamma.detach(),
            "v61a_guard_weight": guard_weight_t.detach(),
            "v61a_absolute_floor": floor_t.detach(),
            "v61a_min_teacher_coverage": coverage_t.detach(),
            "v61a_teacher_confidence_mean": zero,
            "v61a_teacher_active_ratio": zero,
            "v61a_teacher_floor_active_ratio": zero,
            "v61a_teacher_topk_active_ratio": zero,
            "v61a_guard_kl": zero,
            "v61a_guard_loss": zero,
            "v61a_q_teacher_agreement": zero,
            "v61a_q_teacher_kl": zero,
        }
        return zero, stats_zero

    teacher = teacher_q.to(device=q.device, dtype=q.dtype).detach().clamp_min(1e-8)
    teacher = teacher / teacher.sum(dim=1, keepdim=True).clamp_min(1e-8)
    student = q.clamp_min(1e-8)
    student = student / student.sum(dim=1, keepdim=True).clamp_min(1e-8)
    teacher_conf = teacher.max(dim=1).values.detach()
    floor_mask = (teacher_conf >= floor_value).detach()
    topk_mask = torch.zeros_like(floor_mask, dtype=torch.bool)
    n_nodes = int(teacher_conf.numel())
    if n_nodes > 0 and coverage_value > 0.0:
        k = int(math.ceil(coverage_value * float(n_nodes)))
        k = max(1, min(n_nodes, k))
        top_idx = torch.topk(teacher_conf, k=k, largest=True, sorted=False).indices
        topk_mask[top_idx] = True
    active = (floor_mask | topk_mask).detach()
    per_node_kl = torch.sum(teacher * (teacher.clamp_min(1e-8).log() - student.clamp_min(1e-8).log()), dim=1)
    if bool(active.any()):
        guard_kl = per_node_kl[active].mean()
    else:
        guard_kl = zero
    guard_loss = gamma * guard_kl
    q_label = student.detach().argmax(dim=1)
    teacher_label = teacher.argmax(dim=1)
    agreement = (q_label == teacher_label).to(q.dtype).mean()
    full_kl = per_node_kl.mean()
    stats = {
        "v61a_guard_enabled": q.new_tensor(bool(enabled)),
        "v61a_teacher_ready": q.new_tensor(True),
        "v61a_teacher_epoch": teacher_epoch_t.detach(),
        "v61a_guard_gamma": gamma.detach(),
        "v61a_guard_weight": guard_weight_t.detach(),
        "v61a_absolute_floor": floor_t.detach(),
        "v61a_min_teacher_coverage": coverage_t.detach(),
        "v61a_teacher_confidence_mean": teacher_conf.mean().detach(),
        "v61a_teacher_active_ratio": active.to(q.dtype).mean().detach(),
        "v61a_teacher_floor_active_ratio": floor_mask.to(q.dtype).mean().detach(),
        "v61a_teacher_topk_active_ratio": topk_mask.to(q.dtype).mean().detach(),
        "v61a_guard_kl": guard_kl.detach(),
        "v61a_guard_loss": guard_loss.detach(),
        "v61a_q_teacher_agreement": agreement.detach(),
        "v61a_q_teacher_kl": full_kl.detach(),
    }
    return guard_loss, stats


def v62a_drift_responsive_self_distillation_guard_loss(
    q: torch.Tensor,
    teacher_q: torch.Tensor,
    *,
    enabled: bool,
    teacher_ready: bool,
    teacher_epoch: int,
    current_epoch: int,
    guard_weight: float,
    absolute_floor: float,
    min_teacher_coverage: float,
    start_epoch: int,
    ramp_epochs: int,
    max_gamma: float,
    drift_start_epoch: int,
    drift_floor: float,
    drift_scale: float,
    drift_boost: float,
    max_effective_guard_multiplier: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    zero = q.new_tensor(0.0)
    one = q.new_tensor(1.0)
    ready = bool(enabled) and bool(teacher_ready) and teacher_q.numel() == q.numel()
    gamma = q.new_tensor(
        _v60a_guard_gamma_value(
            current_epoch,
            start_epoch=start_epoch,
            ramp_epochs=ramp_epochs,
            max_gamma=max_gamma,
        )
        if ready
        else 0.0
    )
    floor_value = float(np.clip(float(absolute_floor), 0.0, 1.0))
    coverage_value = float(np.clip(float(min_teacher_coverage), 0.0, 1.0))
    drift_floor_value = max(0.0, float(drift_floor))
    drift_scale_value = max(1e-8, float(drift_scale))
    drift_boost_value = max(0.0, float(drift_boost))
    max_multiplier_value = max(1.0, float(max_effective_guard_multiplier))
    guard_weight_t = q.new_tensor(float(guard_weight))
    teacher_epoch_t = q.new_tensor(float(teacher_epoch if ready else -1))
    floor_t = q.new_tensor(floor_value)
    coverage_t = q.new_tensor(coverage_value)
    drift_floor_t = q.new_tensor(drift_floor_value)
    drift_scale_t = q.new_tensor(drift_scale_value)
    drift_boost_t = q.new_tensor(drift_boost_value)
    max_multiplier_t = q.new_tensor(max_multiplier_value)
    if not ready:
        stats_zero = {
            "v62a_guard_enabled": q.new_tensor(bool(enabled)),
            "v62a_teacher_ready": q.new_tensor(False),
            "v62a_teacher_epoch": teacher_epoch_t,
            "v62a_guard_gamma": gamma.detach(),
            "v62a_guard_weight": guard_weight_t.detach(),
            "v62a_absolute_floor": floor_t.detach(),
            "v62a_min_teacher_coverage": coverage_t.detach(),
            "v62a_teacher_confidence_mean": zero,
            "v62a_teacher_active_ratio": zero,
            "v62a_teacher_floor_active_ratio": zero,
            "v62a_teacher_topk_active_ratio": zero,
            "v62a_guard_kl": zero,
            "v62a_guard_loss": zero,
            "v62a_q_teacher_agreement": zero,
            "v62a_q_teacher_kl": zero,
            "v62a_drift_score": zero,
            "v62a_drift_gamma": zero,
            "v62a_drift_floor": drift_floor_t.detach(),
            "v62a_drift_scale": drift_scale_t.detach(),
            "v62a_drift_boost": drift_boost_t.detach(),
            "v62a_effective_guard_multiplier": one,
            "v62a_max_effective_guard_multiplier": max_multiplier_t.detach(),
        }
        return zero, stats_zero

    teacher = teacher_q.to(device=q.device, dtype=q.dtype).detach().clamp_min(1e-8)
    teacher = teacher / teacher.sum(dim=1, keepdim=True).clamp_min(1e-8)
    student = q.clamp_min(1e-8)
    student = student / student.sum(dim=1, keepdim=True).clamp_min(1e-8)
    teacher_conf = teacher.max(dim=1).values.detach()
    floor_mask = (teacher_conf >= floor_value).detach()
    topk_mask = torch.zeros_like(floor_mask, dtype=torch.bool)
    n_nodes = int(teacher_conf.numel())
    if n_nodes > 0 and coverage_value > 0.0:
        k = int(math.ceil(coverage_value * float(n_nodes)))
        k = max(1, min(n_nodes, k))
        top_idx = torch.topk(teacher_conf, k=k, largest=True, sorted=False).indices
        topk_mask[top_idx] = True
    active = (floor_mask | topk_mask).detach()
    per_node_kl = torch.sum(teacher * (teacher.clamp_min(1e-8).log() - student.clamp_min(1e-8).log()), dim=1)
    if bool(active.any()):
        guard_kl = per_node_kl[active].mean()
    else:
        guard_kl = zero
    drift_score = guard_kl.detach()
    epoch_number = int(current_epoch) + 1
    if epoch_number <= int(drift_start_epoch):
        drift_gamma = zero
    else:
        drift_gamma = ((drift_score - drift_floor_t) / drift_scale_t).clamp(0.0, 1.0).detach()
    effective_multiplier = torch.minimum(max_multiplier_t, one + drift_boost_t * drift_gamma).detach()
    guard_loss = gamma * effective_multiplier * guard_kl
    q_label = student.detach().argmax(dim=1)
    teacher_label = teacher.argmax(dim=1)
    agreement = (q_label == teacher_label).to(q.dtype).mean()
    full_kl = per_node_kl.mean()
    stats = {
        "v62a_guard_enabled": q.new_tensor(bool(enabled)),
        "v62a_teacher_ready": q.new_tensor(True),
        "v62a_teacher_epoch": teacher_epoch_t.detach(),
        "v62a_guard_gamma": gamma.detach(),
        "v62a_guard_weight": guard_weight_t.detach(),
        "v62a_absolute_floor": floor_t.detach(),
        "v62a_min_teacher_coverage": coverage_t.detach(),
        "v62a_teacher_confidence_mean": teacher_conf.mean().detach(),
        "v62a_teacher_active_ratio": active.to(q.dtype).mean().detach(),
        "v62a_teacher_floor_active_ratio": floor_mask.to(q.dtype).mean().detach(),
        "v62a_teacher_topk_active_ratio": topk_mask.to(q.dtype).mean().detach(),
        "v62a_guard_kl": guard_kl.detach(),
        "v62a_guard_loss": guard_loss.detach(),
        "v62a_q_teacher_agreement": agreement.detach(),
        "v62a_q_teacher_kl": full_kl.detach(),
        "v62a_drift_score": drift_score.detach(),
        "v62a_drift_gamma": drift_gamma.detach(),
        "v62a_drift_floor": drift_floor_t.detach(),
        "v62a_drift_scale": drift_scale_t.detach(),
        "v62a_drift_boost": drift_boost_t.detach(),
        "v62a_effective_guard_multiplier": effective_multiplier.detach(),
        "v62a_max_effective_guard_multiplier": max_multiplier_t.detach(),
    }
    return guard_loss, stats


def build_spectral_subspace_embedding(
    x_np: np.ndarray,
    adj: sp.spmatrix,
    n_clusters: int,
    cfg: E2ESECTCoCoConfig,
) -> np.ndarray:
    n = int(x_np.shape[0])
    k = int(n_clusters)
    if n <= 0 or k <= 0:
        return np.empty((0, 0), dtype=np.float32)
    h = normalize(np.nan_to_num(np.asarray(x_np, dtype=np.float32)), norm="l2", axis=1)
    graph = as_csr(adj)
    graph = (graph + sp.eye(graph.shape[0], dtype=np.float32, format="csr")).tocsr()
    graph.eliminate_zeros()
    row_sum = np.asarray(graph.sum(axis=1)).reshape(-1).astype(np.float32)
    inv = np.zeros_like(row_sum, dtype=np.float32)
    np.divide(1.0, row_sum, out=inv, where=row_sum > 0.0)
    graph = sp.diags(inv, dtype=np.float32, format="csr") @ graph
    for _ in range(max(0, int(getattr(cfg, "v64a_filter_steps", 2)))):
        h = graph @ h
        h = normalize(np.nan_to_num(np.asarray(h, dtype=np.float32)), norm="l2", axis=1)

    rank_mult = max(1e-6, float(getattr(cfg, "v64a_rank_multiplier", 4.0)))
    max_rank = max(1, int(getattr(cfg, "v64a_max_rank", 64)))
    rank = int(math.ceil(float(k) * rank_mult))
    rank = max(1, min(rank, max_rank, max(1, n - 1), max(1, h.shape[1] - 1)))
    if rank < h.shape[1] and rank < n:
        z = TruncatedSVD(n_components=rank, random_state=int(cfg.seed)).fit_transform(h)
    else:
        z = h
    z = normalize(np.nan_to_num(np.asarray(z, dtype=np.float32)), norm="l2", axis=1)
    return z.astype(np.float32)


def build_elss_anchor_subspace_embedding(
    x_np: np.ndarray,
    adj: sp.spmatrix,
    n_clusters: int,
    cfg: E2ESECTCoCoConfig,
    device: torch.device,
) -> np.ndarray:
    n = int(x_np.shape[0])
    k = int(n_clusters)
    if n <= 1 or k <= 0:
        return np.empty((0, 0), dtype=np.float32)
    x = normalize(np.nan_to_num(np.asarray(x_np, dtype=np.float64)), norm="l2", axis=1)
    norm_adj = _elss_row_normalize(_elss_sym_normalize(as_csr(adj), add_self_loops=True), add_self_loops=True)
    n_anchors = int(getattr(cfg, "v66a_elss_n_anchors", 300))
    n_anchors = max(k + 2, min(n_anchors, n - 1))
    k_rank = int(getattr(cfg, "v66a_elss_k_rank", 0))
    if k_rank <= 0:
        k_rank = k + 1
    k_rank = max(k + 1, min(k_rank, n - 1, x.shape[1] + 1))
    head = _AnchorSubspaceHead(
        n_clusters=k,
        k_rank=k_rank,
        n_anchors=n_anchors,
        power=int(getattr(cfg, "v66a_elss_power", 2)),
        d=float(getattr(cfg, "v66a_elss_d", 0.875)),
        alpha2=float(getattr(cfg, "v66a_elss_alpha2", 0.00005)),
        gamma=float(getattr(cfg, "v66a_elss_gamma", 0.003)),
        filter_coef=getattr(cfg, "v66a_elss_filter_coef", None),
        return_k_rank=bool(getattr(cfg, "v66a_elss_return_k_rank", False)),
        seed=int(cfg.seed),
        device=device,
    )
    try:
        with torch.no_grad():
            z_t = head.fit_transform(
                torch.as_tensor(x, dtype=torch.float64, device=device),
                _scipy_to_torch64(norm_adj, device),
            )
            z = z_t.detach().cpu().numpy()
    except RuntimeError as exc:
        if device.type == "cuda" and ("out of memory" in str(exc).lower() or "cuda" in str(exc).lower()):
            torch.cuda.empty_cache()
            cpu = torch.device("cpu")
            head.device = cpu
            with torch.no_grad():
                z_t = head.fit_transform(
                    torch.as_tensor(x, dtype=torch.float64),
                    _scipy_to_torch64(norm_adj, cpu),
                )
                z = z_t.detach().cpu().numpy()
        else:
            raise
    mode = str(getattr(cfg, "v66a_elss_q_norm", "l2")).lower()
    if mode == "l2":
        z = normalize(np.nan_to_num(z), norm="l2", axis=1)
    return np.nan_to_num(np.asarray(z, dtype=np.float32))


def soft_cluster_distribution_from_embedding(
    z: np.ndarray,
    n_clusters: int,
    cfg: E2ESECTCoCoConfig,
    *,
    temperature: float,
) -> np.ndarray:
    n = int(z.shape[0])
    k = int(n_clusters)
    if n <= 0 or k <= 0:
        return np.empty((0, 0), dtype=np.float32)
    z_norm = normalize(np.nan_to_num(np.asarray(z, dtype=np.float32)), norm="l2", axis=1)
    _, centers = cluster_numpy(z_norm, k, cfg)
    centers = normalize(np.nan_to_num(centers.astype(np.float32)), norm="l2", axis=1)
    dist2 = ((z_norm[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    tau = max(1e-4, float(temperature))
    logits = -dist2 / tau
    logits = logits - logits.max(axis=1, keepdims=True)
    q = np.exp(logits).astype(np.float32)
    q = q / np.clip(q.sum(axis=1, keepdims=True), 1e-8, None)
    return q.astype(np.float32)


def build_spectral_compactness_anchor(
    x_np: np.ndarray,
    adj: sp.spmatrix,
    n_clusters: int,
    cfg: E2ESECTCoCoConfig,
) -> np.ndarray:
    n = int(x_np.shape[0])
    k = int(n_clusters)
    if n <= 0 or k <= 0:
        return np.empty((0, 0), dtype=np.float32)
    h = normalize(np.nan_to_num(np.asarray(x_np, dtype=np.float32)), norm="l2", axis=1)
    graph = as_csr(adj)
    graph = (graph + sp.eye(graph.shape[0], dtype=np.float32, format="csr")).tocsr()
    graph.eliminate_zeros()
    row_sum = np.asarray(graph.sum(axis=1)).reshape(-1).astype(np.float32)
    inv = np.zeros_like(row_sum, dtype=np.float32)
    np.divide(1.0, row_sum, out=inv, where=row_sum > 0.0)
    graph = sp.diags(inv, dtype=np.float32, format="csr") @ graph
    for _ in range(max(0, int(getattr(cfg, "v50a_filter_steps", 2)))):
        h = graph @ h
        h = normalize(np.nan_to_num(np.asarray(h, dtype=np.float32)), norm="l2", axis=1)
    rank_mult = max(1e-6, float(getattr(cfg, "v50a_anchor_rank_multiplier", 1.0)))
    rank = int(math.ceil(float(k) * rank_mult))
    rank = max(1, min(rank, max(1, n - 1), max(1, h.shape[1] - 1)))
    if rank < h.shape[1] and rank < n:
        z = TruncatedSVD(n_components=rank, random_state=int(cfg.seed)).fit_transform(h)
    else:
        z = h
    z = normalize(np.nan_to_num(np.asarray(z, dtype=np.float32)), norm="l2", axis=1)
    labels, centers = cluster_numpy(z, k, cfg)
    centers = normalize(np.nan_to_num(centers.astype(np.float32)), norm="l2", axis=1)
    dist2 = ((z[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    temperature = max(1e-4, float(getattr(cfg, "v50a_anchor_temperature", 0.35)))
    logits = -dist2 / temperature
    logits = logits - logits.max(axis=1, keepdims=True)
    q = np.exp(logits).astype(np.float32)
    q = q / np.clip(q.sum(axis=1, keepdims=True), 1e-8, None)
    return q.astype(np.float32)


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
