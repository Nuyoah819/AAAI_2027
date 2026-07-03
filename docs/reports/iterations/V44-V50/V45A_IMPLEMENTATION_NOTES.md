# V45A Implementation Notes

Target variant:

```text
v45a_edge_local_band_guarded_frequency
```

This file records implementation details only. Gate results are recorded in
`V45A_FIRST_SMOKE_VERDICT.md`.

## Files Changed

```text
core/e2e/sect_coco_e2e.py
scripts/run_unified_aptc_9datasets.py
```

## Core Implementation

Added config fields:

```text
v45a_edge_freq_weight
v45a_band_guard_weight
v45a_warmup_epochs
v45a_band_gate_k
v45a_target_edge_gap
v45a_band_reference_delta
v45a_corr_eps
```

Added model buffers:

```text
v45a_band_reference
v45a_warmup_band_sum
v45a_warmup_epoch_count
v45a_last_warmup_epoch
v45a_band_reference_ready
```

Warmup behavior:

- During epochs 1..W, V45A losses are inactive.
- During warmup, `band_mass` is collected from the same forward path.
- At the end of warmup, the same-run reference is frozen:

```text
band_reference = stopgrad(mean(warmup_band_mass) - band_reference_delta)
```

No dataset name, dataset family, historical ceiling table, or test-set metric
is used to set `band_reference`.

## Losses

Added `v45a_edge_local_band_guarded_frequency_regularizer`.

Edge-local response:

```text
edge_response_ij = 0.5 * (pre_hp_response_i + pre_hp_response_j)
boundary_weight_ij = stopgrad((hetero_ij + hard_ij).clamp(0, 1))
safe_homo_weight_ij = stopgrad(homo_ij.clamp(0, 1))
edge_response_gap = mean_boundary_response - mean_safe_response
L_edge_freq = ReLU(target_edge_gap - edge_response_gap)^2
```

Band-guarded coupling:

```text
band_guard_loss = ReLU(band_mass - band_reference)^2
safe_band_gate = sigmoid(k * (band_reference - band_mass))
effective_edge_freq_loss = safe_band_gate * L_edge_freq
```

Total loss additions:

```text
total_loss += v45a_edge_freq_weight * effective_edge_freq_loss
total_loss += v45a_band_guard_weight * band_guard_loss
```

## Runner Variant

Added one variant:

```text
v45a_edge_local_band_guarded_frequency
```

First implementation constants:

```text
v45a_edge_freq_weight = 0.01
v45a_band_guard_weight = 0.01
v45a_warmup_epochs = 5
v45a_band_gate_k = 20.0
v45a_target_edge_gap = 0.0
v45a_band_reference_delta = 0.0
v45a_corr_eps = 1e-8
```

Failed prior losses are explicitly disabled in the variant:

```text
v43b_conflict_margin_weight = 0.0
v43b_band_conflict_weight = 0.0
v43b_highpass_energy_weight = 0.0
ideal_signed_embedding_weight = 0.0
ideal_band_resolution_weight = 0.0
ideal_highpass_energy_weight = 0.0
v44_topology_band_resolution_weight = 0.0
v44_conflict_highpass_corr_weight = 0.0
v44b_pre_hp_corr_weight = 0.0
```

## Verification

Code-level check passed:

```powershell
conda run -n aaai-e2e-subspace python -m py_compile core\e2e\sect_coco_e2e.py scripts\run_unified_aptc_9datasets.py
```

Connectivity check passed:

```powershell
conda run -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v45a_edge_local_band_guarded_frequency --datasets acm --epochs 1 --device cpu --log-level WARNING
```

The 1-epoch check is not a gate result. It remains in the V45A result files as
an implementation connectivity record.
