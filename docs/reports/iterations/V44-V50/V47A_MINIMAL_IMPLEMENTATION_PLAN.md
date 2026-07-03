# V47A Minimal Implementation Plan

This plan translates `V47A_PREREGISTRATION.md` into a scoped implementation.
It does not report results and does not authorize sweeps or full runs.

## 1. Scope

Implement only:

```text
v47a_posterior_guided_band_resolution
```

Do not implement:

- alternative V47 routes
- weight sweeps
- quantile sweeps
- entropy-floor sweeps
- selector or post-processing logic
- new head or dataset-specific branch

## 2. Config Additions

Add fields to `E2ESECTCoCoConfig`:

```python
v47a_resolution_weight: float = 0.0
v47a_usage_guard_weight: float = 0.0
v47a_agree_high_quantile: float = 0.70
v47a_agree_low_quantile: float = 0.30
v47a_uncert_high_quantile: float = 0.70
v47a_usage_entropy_floor: float = 0.60
v47a_eps: float = 1e-8
```

Default weights stay zero so existing variants are unchanged.

## 3. New Regularizer

Add:

```python
def v47a_posterior_guided_band_resolution_regularizer(
    q_posterior,
    edge_index,
    homo,
    hetero,
    hard,
    *,
    agree_high_quantile=0.70,
    agree_low_quantile=0.30,
    uncert_high_quantile=0.70,
    usage_entropy_floor=0.60,
    eps=1e-8,
):
    ...
```

Required behavior:

- use `q = q_posterior.detach()`
- compute posterior agreement and normalized entropy on edges
- compute fixed preregistered quantiles
- construct homo/hetero/defer targets
- compute hard-weighted CE-style resolution loss
- compute usage entropy guard
- return `resolution_loss`, `usage_guard_loss`, and stats

Do not detach `homo`, `hetero`, or `hard` in the loss terms. These masks are the
intended optimization targets.

## 4. Diagnostics

Add at least the preregistered fields:

```text
v47a_enabled
v47a_posterior_agreement_mean
v47a_posterior_agreement_std
v47a_posterior_uncertainty_mean
v47a_agree_high_threshold
v47a_agree_low_threshold
v47a_uncert_high_threshold
v47a_homo_target_mass
v47a_hetero_target_mass
v47a_defer_target_mass
v47a_unassigned_target_mass
v47a_resolution_loss
v47a_usage_guard_loss
v47a_band_mass
v47a_homo_usage
v47a_hetero_usage
v47a_hard_usage
v47a_usage_entropy
```

Also add implementation-safety diagnostics:

```text
v47a_raw_homo_target_mass
v47a_raw_hetero_target_mass
v47a_raw_defer_target_mass
v47a_effective_target_mass
```

Interpretation:

```text
v47a_*_target_mass = hard-weighted effective mass
v47a_raw_*_target_mass = all-edge raw target mass
```

This keeps the gate aligned with the actual hard-band loss.

## 5. Loss Integration

In `EndToEndSECTCoCoModule.loss`, call the V47A regularizer after V46A stats are
computed and before `total` is assembled.

Add to `total`:

```python
+ cfg.v47a_resolution_weight * v47a_resolution_loss
+ cfg.v47a_usage_guard_weight * v47a_usage_guard_loss
```

Add:

```python
v47a_enabled = (
    cfg.v47a_resolution_weight > 0.0
    or cfg.v47a_usage_guard_weight > 0.0
)
```

## 6. Variant Registration

Add to `scripts/run_unified_aptc_9datasets.py`:

```python
EXPERIMENT_VARIANTS["v47a_posterior_guided_band_resolution"] = {
    "output_stem": "unified_aptc_9datasets_v47a_posterior_guided_band_resolution",
    "overrides": {
        **EXPERIMENT_VARIANTS["v28b"]["overrides"],
        "aptc_local_teacher": False,
        "v43b_conflict_margin_weight": 0.0,
        "v43b_band_conflict_weight": 0.0,
        "v43b_highpass_energy_weight": 0.0,
        "ideal_signed_embedding_weight": 0.0,
        "ideal_band_resolution_weight": 0.0,
        "ideal_highpass_energy_weight": 0.0,
        "v44_topology_band_resolution_weight": 0.0,
        "v44_conflict_highpass_corr_weight": 0.0,
        "v44b_pre_hp_corr_weight": 0.0,
        "v45a_edge_freq_weight": 0.0,
        "v45a_band_guard_weight": 0.0,
        "v46a_band_cal_weight": 0.0,
        "v46a_balance_weight": 0.0,
        "v46a_spread_weight": 0.0,
        "partition_spread_weight": 0.0,
        "freq_separation_weight": 0.0,
        "freq_ortho_weight": 0.0,
        "v47a_resolution_weight": 0.01,
        "v47a_usage_guard_weight": 0.005,
        "v47a_agree_high_quantile": 0.70,
        "v47a_agree_low_quantile": 0.30,
        "v47a_uncert_high_quantile": 0.70,
        "v47a_usage_entropy_floor": 0.60,
        "v47a_eps": 1e-8,
    },
}
```

## 7. Sanity Checks Before Smoke

Run only after implementation:

```powershell
conda run -n aaai-e2e-subspace python -m py_compile core\e2e\sect_coco_e2e.py scripts\run_unified_aptc_9datasets.py
```

Then a connectivity check only:

```powershell
conda run -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v47a_posterior_guided_band_resolution --datasets acm --epochs 1 --device cpu --log-level WARNING
```

The 1-epoch run is not a result.

## 8. First-Stage Smoke Command

Only if sanity checks pass:

```powershell
conda run -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v47a_posterior_guided_band_resolution --datasets "acm,dblp,flickr" --epochs 80 --device cuda --log-level WARNING
```

Expected outputs:

```text
results/archive/v40-v50/unified_aptc_9datasets_v47a_posterior_guided_band_resolution.csv
results/archive/v40-v50/unified_aptc_9datasets_v47a_posterior_guided_band_resolution_diagnostics.jsonl
```

## 9. Stop Rule

After the first-stage smoke, stop and write:

```text
V47A_FIRST_SMOKE_VERDICT.md
```

Do not proceed to second-batch or full runs unless all preregistered gates pass.
