# V47A Implementation Notes

Target variant:

```text
v47a_posterior_guided_band_resolution
```

This file records implementation details and code-level sanity checks only.
Gate results must be recorded separately after a preregistered 80-epoch smoke.

## Files Changed

```text
core/e2e/sect_coco_e2e.py
scripts/run_unified_aptc_9datasets.py
```

## Core Implementation

Added config fields:

```text
v47a_resolution_weight
v47a_usage_guard_weight
v47a_agree_high_quantile
v47a_agree_low_quantile
v47a_uncert_high_quantile
v47a_usage_entropy_floor
v47a_eps
```

Added helper:

```text
v47a_posterior_guided_band_resolution_regularizer
```

Posterior targets use:

```text
q = q_posterior.detach()
posterior_agreement_ij = dot(q_i, q_j)
posterior_uncertainty_ij = 0.5 * (entropy(q_i) + entropy(q_j)) / log(K)
```

Topology masks are not detached in the loss terms, so gradients from V47A flow
to `homo`, `hetero`, and `hard`, but not into posterior targets.

## Losses

Implemented:

```text
resolution_loss =
  mean(hard * (
    homo_target   * -log(homo + eps)
    + hetero_target * -log(hetero + eps)
    + defer_target  * -log(hard + eps)
  ))

usage_guard_loss = ReLU(usage_entropy_floor - usage_entropy)^2
```

Total loss additions:

```text
total_loss += v47a_resolution_weight * resolution_loss
total_loss += v47a_usage_guard_weight * usage_guard_loss
```

## Diagnostics

Added preregistered diagnostics:

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

Added implementation-safety diagnostics:

```text
v47a_raw_homo_target_mass
v47a_raw_hetero_target_mass
v47a_raw_defer_target_mass
v47a_raw_unassigned_target_mass
v47a_effective_target_mass
```

The non-raw target mass fields are hard-weighted effective masses.

## Runner Variant

Added one variant:

```text
v47a_posterior_guided_band_resolution
```

Failed or confounding losses are explicitly disabled:

```text
v43b_* = 0.0
ideal_* = 0.0
v44_* = 0.0
v44b_pre_hp_corr_weight = 0.0
v45a_* = 0.0
v46a_* = 0.0
partition_spread_weight = 0.0
freq_separation_weight = 0.0
freq_ortho_weight = 0.0
```

First implementation constants:

```text
v47a_resolution_weight = 0.01
v47a_usage_guard_weight = 0.005
v47a_agree_high_quantile = 0.70
v47a_agree_low_quantile = 0.30
v47a_uncert_high_quantile = 0.70
v47a_usage_entropy_floor = 0.60
v47a_eps = 1e-8
```

## Verification

Code-level check passed:

```powershell
conda run -n aaai-e2e-subspace python -m py_compile core\e2e\sect_coco_e2e.py scripts\run_unified_aptc_9datasets.py
```

Connectivity check passed:

```powershell
conda run -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v47a_posterior_guided_band_resolution --datasets acm --epochs 1 --device cpu --log-level WARNING
```

The 1-epoch check is not a gate result. It remains in the V47A result files as
an implementation connectivity record.

Sanity snapshot from the 1-epoch ACM connectivity check:

```text
legacy_head_used=false
v43b/v44/v44b/v45a/v46a_enabled=false
v47a_enabled=true
v47a_homo_target_mass=0.2939
v47a_hetero_target_mass=0.2539
v47a_defer_target_mass=0.3046
v47a_unassigned_target_mass=0.1476
v47a_effective_target_mass=0.8524
```

No formal V47A smoke has been run yet.
