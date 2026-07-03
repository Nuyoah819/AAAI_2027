# V46A Minimal Implementation Plan

Target variant:

```text
v46a_topology_band_calibration
```

This is an implementation plan only. It does not report results.

## 1. Preconditions

Treat these as active constraints:

- `CRITICAL_RED_LINES.md`
- `V45A_FIRST_SMOKE_VERDICT.md`
- `V46A_ROUTE_DECISION.md`
- `V46A_PREREGISTRATION.md`
- `V46A_IMPLEMENTATION_REVIEW.md`

The implementation-critical correction is:

```text
band_ij = hard_ij
band_mass = mean(hard_ij)
```

Do not use `1 - max(homo, hetero, hard)` as V46A band mass.

## 2. Allowed File Scope

Minimal code implementation should be limited to:

```text
core/e2e/sect_coco_e2e.py
scripts/run_unified_aptc_9datasets.py
```

Optional after implementation:

```text
V46A_IMPLEMENTATION_NOTES.md
```

No result file should be edited manually.

## 3. Config Additions

Add fields:

```text
v46a_band_cal_weight: float = 0.0
v46a_balance_weight: float = 0.0
v46a_spread_weight: float = 0.0
v46a_entropy_floor: float = 0.60
v46a_min_threshold_gap: float = 0.05
v46a_corr_eps: float = 1e-8
```

`v46a_enabled` is true iff any V46A loss weight is greater than zero.

## 4. Loss Implementation

Add a helper:

```text
v46a_topology_band_calibration_regularizer(
    homo, hetero, hard, low_threshold, high_threshold,
    entropy_floor, min_threshold_gap, eps
)
```

Required formulas:

```text
band = hard.clamp(0, 1)
band_cal_loss = mean(band^2)
usage = normalize([mean(homo), mean(hetero), mean(hard)])
usage_entropy = -sum(usage * log(usage)) / log(3)
balance_loss = ReLU(entropy_floor - usage_entropy)^2
threshold_gap = high_threshold - low_threshold
spread_loss = ReLU(min_threshold_gap - threshold_gap)^2
```

## 5. Total Loss Addition

Add only:

```text
total_loss += v46a_band_cal_weight * band_cal_loss
total_loss += v46a_balance_weight * balance_loss
total_loss += v46a_spread_weight * spread_loss
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
v45a_edge_freq_weight
v45a_band_guard_weight
partition_spread_weight
freq_separation_weight
freq_ortho_weight
```

## 6. Diagnostics

Add:

```text
v46a_enabled
v46a_band_cal_loss
v46a_balance_loss
v46a_spread_loss
v46a_band_mass
v46a_homo_usage
v46a_hetero_usage
v46a_hard_usage
v46a_usage_entropy
v46a_threshold_gap
v46a_low_threshold
v46a_high_threshold
```

Retain V44B/V45A diagnostics as diagnostic only.

## 7. Runner Variant

Add exactly one variant:

```text
v46a_topology_band_calibration
```

First constants:

```text
v46a_band_cal_weight = 0.01
v46a_balance_weight = 0.005
v46a_spread_weight = 0.005
v46a_entropy_floor = 0.60
v46a_min_threshold_gap = 0.05
v46a_corr_eps = 1e-8
```

Also set all failed or confounding loss weights to zero, including inherited
`partition_spread_weight`, `freq_separation_weight`, and `freq_ortho_weight`.

## 8. Sanity Checks

After implementation:

```powershell
conda run -n aaai-e2e-subspace python -m py_compile core\e2e\sect_coco_e2e.py scripts\run_unified_aptc_9datasets.py
```

If needed, run one connectivity check:

```powershell
conda run -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v46a_topology_band_calibration --datasets acm --epochs 1 --device cpu --log-level WARNING
```

For 1 epoch, do not evaluate gates.

Sanity requires:

```text
legacy_head_used=false
v43b_enabled=false
v44_enabled=false
v44b_enabled=false
v45a_enabled=false
v46a_enabled=true
v46a diagnostics present and finite
embedding_posterior_gap finite
```

## 9. Formal Smoke

Only after sanity passes:

```powershell
conda run -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v46a_topology_band_calibration --datasets "acm,dblp,flickr" --epochs 80 --device cuda --log-level WARNING
```

No second-batch smoke, full run, or sweep is allowed unless all preregistered
first-stage gates pass.

## 10. Rejection Checklist

Reject if any appear:

- dataset name used by V46A loss
- per-dataset band ceiling used inside training
- `band_ij = 1 - max(homo, hetero, hard)` used as V46A band mass
- frequency response contributes to V46A loss
- revived V43B/V44/V44B/V45A failed losses
- inherited `partition_spread`, `freq_separation`, or `freq_ortho` left active
  in the V46A variant
- manually edited result files

## 11. No-Fabrication Status

No V46A result exists yet. All V46A result values remain `TBD`.
