# V45A Minimal Implementation Plan

Target variant:

```text
v45a_edge_local_band_guarded_frequency
```

This document is an implementation plan only. It does not change training code,
does not register a variant, does not run experiments, and does not report new
results.

## 1. Preconditions

Implementation may start only after the following documents are treated as
active constraints:

- `CRITICAL_RED_LINES.md`
- `V45A_PREREGISTRATION.md`
- `V45A_IMPLEMENTATION_REVIEW.md`
- `V44B_FIRST_SMOKE_VERDICT.md`

The key implementation correction is now fixed in the preregistration:

```text
band_reference = frozen warmup reference from the same run
```

Do not use ACM/DBLP/Flickr ceiling values inside training. Those ceilings are
evaluation gates only.

## 2. Allowed File Scope

Minimal code implementation should be limited to:

```text
core/e2e/sect_coco_e2e.py
scripts/run_unified_aptc_9datasets.py
```

Optional after implementation:

```text
V45A_IMPLEMENTATION_NOTES.md
```

No result files should be edited manually.

## 3. Config Additions

Add fields with these first-version defaults:

```text
v45a_edge_freq_weight: float = 0.0
v45a_band_guard_weight: float = 0.0
v45a_warmup_epochs: int = 5
v45a_band_gate_k: float = 20.0
v45a_target_edge_gap: float = 0.0
v45a_band_reference_delta: float = 0.0
v45a_corr_eps: float = 1e-8
```

`v45a_enabled` should be true iff either v45a loss weight is greater than zero.

## 4. State Needed For Warmup Reference

The model needs a small persistent buffer/state for:

```text
v45a_band_reference
v45a_warmup_epoch_count
v45a_band_reference_ready
```

Required behavior:

1. During epochs `1..W`, v45a losses are inactive.
2. During warmup, collect `band_mass` from the same unified forward path.
3. At the end of warmup, freeze:

```text
band_reference = stopgrad(mean_or_ema(warmup_band_mass) - band_reference_delta)
```

4. After warmup, do not update the reference.

If the current training loop does not expose epoch cleanly inside the model,
prefer passing epoch/state through the existing training context rather than
creating dataset-specific or runner-specific branches.

## 5. Loss Implementation

### 5.1 Keep V44B Response Diagnostic

Reuse existing pre-normalization response:

```text
pre_hp_response_i = log1p(raw_high_response_i)
```

Keep existing v44b diagnostics, but ensure:

```text
v44b_pre_hp_corr_weight = 0.0
```

### 5.2 Edge-Local Frequency Loss

Use detached topology masks:

```text
edge_response_ij = 0.5 * (pre_hp_response_i + pre_hp_response_j)
boundary_weight_ij = stopgrad((hetero_ij + hard_ij).clamp(0, 1))
safe_homo_weight_ij = stopgrad(homo_ij.clamp(0, 1))
mean_boundary_response = weighted_mean(edge_response, boundary_weight)
mean_safe_response = weighted_mean(edge_response, safe_homo_weight)
edge_response_gap = mean_boundary_response - mean_safe_response
L_edge_freq = ReLU(target_edge_gap - edge_response_gap)^2
```

Weighted means must use an epsilon denominator and must be finite if one mask
has near-zero mass.

### 5.3 Band Guard

After warmup:

```text
band_guard_loss = ReLU(band_mass - band_reference)^2
safe_band_gate = sigmoid(k * (band_reference - band_mass))
effective_edge_freq_loss = safe_band_gate * L_edge_freq
```

During warmup:

```text
band_guard_loss = 0
effective_edge_freq_loss = 0
```

Diagnostics should still be computed during warmup where possible.

## 6. Total Loss Addition

Add only:

```text
total_loss += v45a_band_guard_weight * band_guard_loss
total_loss += v45a_edge_freq_weight * effective_edge_freq_loss
```

Do not revive:

```text
v43b_conflict_margin_weight
v43b_band_conflict_weight
v43b_highpass_energy_weight
ideal_signed_embedding_weight
ideal_band_resolution_weight
ideal_highpass_energy_weight
v44_topology_band_resolution_weight
v44_conflict_highpass_corr_weight
v44b_pre_hp_corr_weight
```

## 7. Diagnostics To Emit

Required:

```text
v45a_enabled
v45a_band_mass
v45a_band_reference
v45a_band_reference_ready
v45a_warmup_epoch_count
v45a_band_guard_loss
v45a_safe_band_gate
v45a_edge_freq_loss
v45a_boundary_response_mean
v45a_safe_homo_response_mean
v45a_edge_response_gap
v45a_edge_response_corr
v45a_boundary_mass
v45a_safe_homo_mass
```

Retain:

```text
v44b_pre_hp_response_mean
v44b_pre_hp_response_std
v44b_pre_hp_response_p10
v44b_pre_hp_response_p90
v44b_conflict_response_corr
v44b_response_gap
v44b_postnorm_hp_energy_mean
v44b_postnorm_hp_energy_std
embedding_posterior_gap
legacy_head_used
v43b_enabled
v44_enabled
v44b_enabled
```

## 8. Runner Variant

Add one variant only:

```text
v45a_edge_local_band_guarded_frequency
```

Required override intent:

```text
v43b_* failed losses = 0.0
ideal_* failed losses = 0.0
v44_* failed losses = 0.0
v44b_pre_hp_corr_weight = 0.0
v45a_edge_freq_weight = nonzero preregistered first value
v45a_band_guard_weight = nonzero preregistered first value
v45a_warmup_epochs = 5
v45a_band_gate_k = 20.0
v45a_target_edge_gap = 0.0
v45a_band_reference_delta = 0.0
```

The first nonzero weights must be chosen once before running any formal smoke.
Do not use a sweep.

## 9. Sanity Checks Before Formal Smoke

After implementation, run only code-level sanity checks first:

```powershell
conda run -n aaai-e2e-subspace python -m py_compile core\e2e\sect_coco_e2e.py scripts\run_unified_aptc_9datasets.py
```

Then run one minimal connectivity check only if needed:

```powershell
conda run -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v45a_edge_local_band_guarded_frequency --datasets acm --epochs 1 --device cpu --log-level WARNING
```

This connectivity check is not a gate result.

Sanity pass requires:

```text
legacy_head_used=false
v43b_enabled=false
v44_enabled=false
v44b_enabled=false
v45a_enabled=true
v45a diagnostics present and finite
v44b diagnostics retained
embedding_posterior_gap finite
```

For a 1-epoch check with `W=5`, `v45a_band_reference_ready=false` is expected.

## 10. Formal First Smoke

Only after sanity passes:

```powershell
conda run -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v45a_edge_local_band_guarded_frequency --datasets "acm,dblp,flickr" --epochs 80 --device cuda --log-level WARNING
```

No second-batch smoke, full run, or sweep is allowed unless all preregistered
first-stage gates pass.

## 11. Rejection Checklist

Reject the implementation if any of these appear:

- dataset name used to set `band_reference`
- dataset name used to set v45a weights, `k`, warmup length, or target edge gap
- per-dataset band ceilings used inside training
- selector-like switch in posterior, head, or assignment
- non-detached topology masks in the first edge-local frequency loss
- revived v43b/v44a/v44b failed losses
- manually edited result files

## 12. No-Fabrication Status

This plan contains no new results. All future v45a values remain `TBD` until a
preregistered run is executed and recorded.
