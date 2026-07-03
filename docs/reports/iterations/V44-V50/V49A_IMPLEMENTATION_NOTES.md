# V49A Implementation Notes

Target variant:

```text
v49a_reparameterized_topology_transition
```

This file records implementation details and code-level sanity checks only.
Formal V49A first-stage smoke results must be recorded separately after the
preregistered 80-epoch run.

## 1. Files Changed

```text
core/e2e/sect_coco_e2e.py
scripts/run_unified_aptc_9datasets.py
```

Supporting implementation-prep documents:

```text
V49A_IMPLEMENTATION_REVIEW.md
V49A_MINIMAL_IMPLEMENTATION_PLAN.md
```

## 2. Core Implementation

Added config fields:

```text
v49a_enabled
v49a_tau_clear
v49a_tau_orient
v49a_snapshot_sample_size
v49a_movement_eps
```

Added module snapshot buffers:

```text
v49a_snapshot_ready
v49a_prev_homo
v49a_prev_hetero
v49a_prev_hard
v49a_prev_score
```

Added topology mapping:

```text
orient = sigmoid(edge_logit / tau_orient)
clear = sigmoid(abs(edge_logit) / tau_clear)

homo = clear * orient
hetero = clear * (1 - orient)
hard = 1 - clear
```

This mapping uses the existing shared `AdaptiveEdgeConfidence` output
`edge_logit`. It does not add labels, dataset IDs, selectors, or posterior
teacher targets.

## 3. Frontend Integration

The default frontend remains unchanged when `v49a_enabled=false`.

When `v49a_enabled=true`, `_frontend_pass` still computes the original
`score`, `edge_logit`, and old low/high thresholds, but replaces
`homo/hetero/hard` with the V49A orientation/clarity simplex before support
weights, diffusion, high-pass routing, APTC, and downstream losses.

Old `low_threshold` and `high_threshold` are retained only for compatibility
diagnostics in this variant.

## 4. Loss Boundary

V49A adds no new loss term.

No `cfg.v49a_*` term is included in `total`.

The V49A runner variant explicitly disables failed or confounding loss families:

```text
v43b_* = 0.0
ideal_* = 0.0
v44_* = 0.0
v44b_pre_hp_corr_weight = 0.0
v45a_* = 0.0
v46a_* = 0.0
v47a_* = 0.0
v48a_enabled = false
partition_spread_weight = 0.0
freq_separation_weight = 0.0
freq_ortho_weight = 0.0
threshold_reg_weight = 0.0
edge_quantile_anchor_weight = 0.0
```

`threshold_reg`, `edge_quantile_anchor`, `partition_spread`,
`frequency_separation`, and `frequency_ortho` may still appear as diagnostic
values because the code computes them before weighted summation. In the V49A
runner variant their weights are zero, so they do not contribute to training.

## 5. Diagnostics

Added V49A diagnostics:

```text
v49a_enabled
v49a_homo_usage
v49a_hetero_usage
v49a_hard_usage
v49a_band_mass
v49a_usage_entropy
v49a_clear_mean
v49a_clear_std
v49a_orient_mean
v49a_orient_std
v49a_has_prev_snapshot
v49a_sample_size
v49a_mean_abs_delta_homo
v49a_mean_abs_delta_hetero
v49a_mean_abs_delta_hard
v49a_mean_abs_delta_score
v49a_hard_mass_delta
v49a_hard_rank_corr_prev
v49a_homo_target_mass
v49a_hetero_target_mass
v49a_defer_target_mass
v49a_raw_homo_target_mass
v49a_raw_hetero_target_mass
v49a_raw_defer_target_mass
v49a_targeted_homo_delta
v49a_targeted_hetero_delta
v49a_targeted_hard_delta
```

Posterior-guided target groups are diagnostics only. They do not create a CE
loss or any other optimization objective.

## 6. Verification

Static compile passed:

```powershell
conda run -n aaai-e2e-subspace python -m py_compile core\e2e\sect_coco_e2e.py scripts\run_unified_aptc_9datasets.py
```

Connectivity check passed:

```powershell
conda run -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v49a_reparameterized_topology_transition --datasets acm --epochs 1 --device cpu --log-level WARNING
```

The 1-epoch check is not a formal V49A smoke result.

Sanity snapshot from the 1-epoch ACM connectivity record:

```text
legacy_head_used=false
v43b/v44/v44b/v45a/v46a/v47a_enabled=false
v48a_enabled=false
v49a_enabled=true
v49a_homo_usage=0.3365
v49a_hetero_usage=0.3295
v49a_hard_usage=0.3340
v49a_usage_entropy=1.0000
v49a_clear_mean=0.6660
v49a_clear_std=0.1248
v49a_orient_mean=0.5180
v49a_orient_std=0.2069
v49a_has_prev_snapshot=false
v49a_sample_size=20000
```

`v49a_has_prev_snapshot=false` and zero movement values are expected for a
single-epoch connectivity run because there is no previous epoch snapshot.

## 7. Current Status

V49A has completed:

```text
implementation review
minimal implementation
py_compile
1-epoch ACM CPU connectivity
```

V49A has not yet run the preregistered 80-epoch ACM/DBLP/Flickr first-stage
smoke.
