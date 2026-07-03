# V46A Implementation Notes

Target variant:

```text
v46a_topology_band_calibration
```

This file records implementation details only. Gate results are recorded in
`V46A_FIRST_SMOKE_VERDICT.md`.

## Files Changed

```text
core/e2e/sect_coco_e2e.py
scripts/run_unified_aptc_9datasets.py
V46A_PREREGISTRATION.md
```

The preregistration was corrected before implementation so that V46A uses the
current codebase's band-mass semantics:

```text
band_ij = hard_ij
band_mass = mean(hard_ij)
```

It explicitly rejects `1 - max(homo, hetero, hard)` as the V46A band mass.

## Core Implementation

Added config fields:

```text
v46a_band_cal_weight
v46a_balance_weight
v46a_spread_weight
v46a_entropy_floor
v46a_min_threshold_gap
v46a_corr_eps
```

Added helper:

```text
v46a_topology_band_calibration_regularizer
```

Implemented losses:

```text
band_cal_loss = mean(hard^2)
usage = normalize([mean(homo), mean(hetero), mean(hard)])
usage_entropy = -sum(usage * log(usage)) / log(3)
balance_loss = ReLU(entropy_floor - usage_entropy)^2
threshold_gap = high_threshold - low_threshold
spread_loss = ReLU(min_threshold_gap - threshold_gap)^2
```

Total loss additions:

```text
total_loss += v46a_band_cal_weight * band_cal_loss
total_loss += v46a_balance_weight * balance_loss
total_loss += v46a_spread_weight * spread_loss
```

## Runner Variant

Added one variant:

```text
v46a_topology_band_calibration
```

First implementation constants:

```text
v46a_band_cal_weight = 0.01
v46a_balance_weight = 0.005
v46a_spread_weight = 0.005
v46a_entropy_floor = 0.60
v46a_min_threshold_gap = 0.05
v46a_corr_eps = 1e-8
```

Explicitly disabled failed or confounding losses:

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
v45a_edge_freq_weight = 0.0
v45a_band_guard_weight = 0.0
partition_spread_weight = 0.0
freq_separation_weight = 0.0
freq_ortho_weight = 0.0
```

`partition_spread`, `frequency_separation`, and `frequency_ortho` may still
appear as diagnostics, but their V46A variant weights are zero.

## Verification

Code-level check passed:

```powershell
conda run -n aaai-e2e-subspace python -m py_compile core\e2e\sect_coco_e2e.py scripts\run_unified_aptc_9datasets.py
```

Connectivity check passed:

```powershell
conda run -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v46a_topology_band_calibration --datasets acm --epochs 1 --device cpu --log-level WARNING
```

The 1-epoch check is not a gate result. It remains in the V46A result files as
an implementation connectivity record.
