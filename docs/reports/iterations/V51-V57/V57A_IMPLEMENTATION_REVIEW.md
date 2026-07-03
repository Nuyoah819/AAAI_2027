# V57A Implementation Review

This document reviews whether `v57a_mass_floor_normalized_residual_anchor` may
be implemented after the V56A STOP verdict and V57A preregistration.

Decision:

```text
APPROVE MINIMAL IMPLEMENTATION ONLY.
```

This review does not authorize a full 9-dataset run, a 260-epoch run, a seed
sweep, target-mass sweep, max-mass-scale sweep, reliability-cap sweep,
beta-bound sweep, soft-power sweep, hybrid-compensation sweep, schedule variant,
reliability formula variant, threshold sweep, or V50A anchor hyperparameter
change.

## 1. Evidence Basis

Files reviewed:

```text
CRITICAL_RED_LINES.md
V56A_FIRST_MIXED_STRESS_VERDICT.md
V56A_HYBRID_COMPENSATION_LIMIT_ANALYSIS.md
V57A_PREREGISTRATION.md
core/e2e/sect_coco_e2e.py
scripts/run_unified_aptc_9datasets.py
```

Current state:

```text
V56A is implemented.
V56A connectivity passed.
V56A first mixed-stress stopped before expansion.
V57A is preregistered but not implemented.
```

## 2. Problem Fit

V56A preserves the safest bounded-residual profile so far:

```text
ACM/DBLP/Squirrel floors pass.
Posterior/readout safety passes.
Anchor usefulness passes.
Heterophily stress passes.
DBLP ACC improves to 0.6919.
```

But V56A still fails reliability non-collapse:

```text
DBLP reliability/effective mass = 0.0756.
Texas reliability/effective mass = 0.0561.
```

The V54A-V56A chain has already tested hard consensus, soft consensus, and
hybrid consensus beta. V57A therefore should not change beta again. It should
test the narrower mechanism:

```text
Can the same detached V56A raw reliability ranking be made operationally useful
through a fixed dataset-agnostic mass-floor normalization?
```

## 3. Red-Line Review

| Constraint | Review |
| --- | --- |
| No dataset-specific module, branch, head, loss, selector | Pass if V57A uses only `q_refined`, `q_embed`, `q_spec`, `edge_index`, and `epoch`. |
| Unified pipeline | Pass if the same mass-floor formula and constants apply to every dataset. |
| End-to-end trainability | Pass because the KL updates `q_refined`; reliability scaling is detached by design. |
| Preserve frontend innovations | Pass because V57A does not bypass the existing frontend or `q_refined` path. |
| Final labels remain unified output | Pass if no final-label path is changed and anchor labels remain diagnostic only. |
| No post-hoc selection | Pass if V57A is one named runner variant and disables V50A-V56A losses. |
| No sweep/cherry-pick | Pass only for one connectivity run, then one preregistered mixed-stress run if connectivity passes. |

## 4. Required Code Changes

Allowed changes in `core/e2e/sect_coco_e2e.py`:

```text
Add config fields:
v57a_enabled
v57a_anchor_weight
v57a_reliability_floor
v57a_reliable_threshold
v57a_min_effective_mass
v57a_warmup_epochs
v57a_ramp_epochs
v57a_beta_min
v57a_beta_max
v57a_soft_power
v57a_hybrid_compensation
v57a_target_mass
v57a_max_mass_scale
v57a_max_reliability_cap
```

Add a new helper:

```text
mass_floor_normalized_residual_spectral_anchor_loss
```

The helper may reuse V56A's raw reliability computation but must emit
independent `v57a_*` diagnostics. It must not mutate the V56A helper or change
V56A semantics.

Allowed integration:

```text
Compute V57A loss next to V50A-V56A losses.
Add `cfg.v57a_anchor_weight * v57a_anchor_loss` to the total loss.
Add `v57a_*` diagnostics to the training diagnostic dictionary.
Include V57A in spectral-anchor build conditions and anchor metric diagnostics.
Add V57A snapshot fields for epoch 1/40/80 coupling and mass checks.
```

Allowed changes in `scripts/run_unified_aptc_9datasets.py`:

```text
Add V57A defaults to BASE_OVERRIDES.
Add `v57a_mass_floor_normalized_residual_anchor` as one named variant.
In that variant, disable V50A/V51A/V52A/V53A/V54A/V55A/V56A losses and enable only V57A.
Set fixed preregistered constants.
```

## 5. Formula Lock

V57A must inherit V56A raw reliability exactly:

```text
raw_r_i = V56A hybrid-consensus reliability
```

Then apply only this detached normalization:

```text
raw_mass = mean(raw_r_i)
mass_scale = clamp(target_mass / clamp(raw_mass, 1e-8), 1.0, max_mass_scale)
r_i = detach(clamp(raw_r_i * mass_scale, 0, max_reliability_cap))
```

Fixed constants:

```text
target_mass = 0.08
max_mass_scale = 1.50
max_reliability_cap = 0.90
```

Loss:

```text
KL(q_refined_i || stopgrad(q_spec_i))
weighted by detached r_i
normalized by max(sum_i r_i, N * max(reliability_floor, min_effective_mass))
```

## 6. Detach And Gradient Boundary

Required:

```text
q_spec is detached.
q_refined/q_embed copies used for reliability are detached.
V56A raw reliability is detached.
raw_mass, mass_scale, scaled reliability, and final reliability are detached.
```

Allowed gradient:

```text
The KL term may update the model through q_refined only.
```

Forbidden gradient:

```text
The model must not learn mass_scale, reliability, beta, consensus, or anchor
construction to escape the anchor loss.
```

## 7. Required Diagnostics

The implementation must emit finite zero defaults when disabled and finite
values when enabled:

```text
v57a_enabled
v57a_gamma
v57a_beta_min
v57a_beta_max
v57a_soft_power
v57a_hybrid_compensation
v57a_target_mass
v57a_max_mass_scale
v57a_max_reliability_cap
v57a_hard_consensus_mean
v57a_soft_consensus_mean
v57a_lifted_soft_consensus_mean
v57a_compensation_mean
v57a_compensation_active_ratio
v57a_hybrid_consensus_mean
v57a_beta_mean
v57a_raw_reliability_mean
v57a_mass_scale
v57a_scaled_reliability_mean
v57a_residual_multiplier_mean
v57a_anchor_loss
v57a_weighted_q_anchor_kl
v57a_weighted_q_anchor_agreement
v57a_unweighted_q_anchor_agreement
v57a_embedding_anchor_agreement
v57a_anchor_entropy
v57a_anchor_confidence
v57a_anchor_cluster_usage_entropy
v57a_reliability_mean
v57a_reliability_std
v57a_reliability_p10
v57a_reliability_p50
v57a_reliability_p90
v57a_reliable_node_ratio
v57a_effective_anchor_mass
v57a_base_reliability_mean
v57a_agreement_reliability_mean
v57a_confidence_component_mean
v57a_q_anchor_component_mean
v57a_embed_anchor_component_mean
v57a_local_component_mean
```

Final anchor metrics must include:

```text
v57a_anchor_acc_diagnostic
v57a_anchor_nmi_diagnostic
v57a_anchor_ari_diagnostic
```

## 8. Static Gates Before Experiment

Before any V57A run:

```powershell
conda run -n aaai-e2e-subspace python -m py_compile core\e2e\sect_coco_e2e.py scripts\run_unified_aptc_9datasets.py
```

Static expectations:

```text
No syntax error.
No dataset-name condition in V57A helper.
No V57A post-processing selector.
No change to `build_spectral_compactness_anchor`.
No changed final label path.
```

## 9. Only Authorized Run After Implementation

If static gates pass, only the following connectivity run is authorized:

```powershell
conda run --no-capture-output -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v57a_mass_floor_normalized_residual_anchor --datasets "acm" --epochs 1 --device cuda --log-level WARNING
```

Connectivity pass condition:

```text
status=ok
legacy_head_used=false
v57a_enabled=true
v50a_enabled=false
v51a_enabled=false
v52a_enabled=false
v53a_enabled=false
v54a_enabled=false
v55a_enabled=false
v56a_enabled=false
v57a_anchor_loss finite
v57a_gamma = 0
v57a_target_mass = 0.08
v57a_max_mass_scale = 1.50
v57a_max_reliability_cap = 0.90
v57a_raw_reliability_mean finite
v57a_mass_scale finite and within [1.0, 1.50]
v57a_reliability_mean finite
v57a_effective_anchor_mass finite
```

Connectivity is not a performance result.

## 10. Post-Connectivity Boundary

If connectivity passes, write:

```text
V57A_CONNECTIVITY_VERDICT.md
```

Only then may the preregistered mixed-stress run be launched:

```text
acm,dblp,flickr,texas,squirrel,chameleon
80 epochs
seed 42
device cuda
```

If connectivity fails, stop and write the failure verdict. Do not repair by
tuning target mass, mass scale, reliability cap, beta bounds, soft power,
hybrid compensation, schedule, thresholds, V50A anchor construction, or
dataset-specific behavior.

## 11. Review Verdict

```text
V57A may proceed to minimal implementation.
```

Reason:

```text
The implementation is a narrow, preregistered normalization of V56A detached
raw reliability. It preserves the unified pipeline, final q_refined labels,
V50A anchor construction, and V56A consensus mechanism while directly testing
whether mass allocation is the remaining bottleneck.
```
