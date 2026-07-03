# V55A Implementation Review

This document reviews whether `v55a_soft_consensus_bounded_residual_anchor` may
be implemented after the V54A STOP verdict and the V55A preregistration.

Decision:

```text
APPROVE MINIMAL IMPLEMENTATION ONLY.
```

This review does not authorize a full 9-dataset run, a 260-epoch run, a seed
sweep, beta-bound sweep, soft-power sweep, schedule variant, reliability formula
variant, threshold sweep, or V50A anchor hyperparameter change.

## 1. Evidence Basis

Files reviewed:

```text
CRITICAL_RED_LINES.md
V54A_FIRST_MIXED_STRESS_VERDICT.md
V54A_CONSENSUS_UNDERACTIVATION_ANALYSIS.md
V55A_PREREGISTRATION.md
core/e2e/sect_coco_e2e.py
scripts/run_unified_aptc_9datasets.py
```

Current state:

```text
V54A is implemented.
V54A connectivity passed.
V54A first mixed-stress stopped before expansion.
V55A is preregistered but not implemented.
```

## 2. Problem Fit

V54A fixed the V53A Squirrel overexposure failure but still failed reliability
non-collapse because DBLP and Texas effective anchor mass stayed below 0.08.
The failure is narrow:

```text
V54A safety passes.
ACM/DBLP/Squirrel floors pass.
Anchor usefulness passes.
Heterophily stress passes.
Reliability mass still underactivates on medium-consensus graphs.
```

The V55A mechanism matches this failure because it changes only the consensus
signal used to bound residual anchor exposure:

```text
V54A: hard argmax consensus -> three beta levels.
V55A: detached soft agreement consensus -> continuous beta in [0.35, 0.70].
```

This is a valid rescue mechanism, not a post-hoc selector, because the same
formula is used for every dataset and final labels remain `q_refined`.

## 3. Red-Line Review

| Constraint | Review |
| --- | --- |
| No dataset-specific module, branch, head, loss, selector | Pass if V55A uses only `q_refined`, `q_embed`, `q_spec`, `edge_index`, `epoch`, and `epochs`. |
| Unified pipeline | Pass if V55A is a config-controlled loss path with the same defaults for all datasets. |
| End-to-end trainability | Pass because the loss is differentiable through `q_refined`; reliability and anchor are detached by design. |
| Preserve frontend innovations | Pass because V55A does not bypass the existing frontend, graph filtering, or `q_refined` path. |
| Final labels remain unified output | Pass if no final-label path is changed and no anchor/KMeans/legacy labels are returned. |
| No post-hoc selection | Pass if the runner creates one named V55A variant and disables V50A-V54A losses inside it. |
| No sweep/cherry-pick | Pass only for one connectivity run, then one preregistered mixed-stress run if connectivity passes. |

## 4. Required Code Changes

Allowed changes in `core/e2e/sect_coco_e2e.py`:

```text
Add config fields:
v55a_enabled
v55a_anchor_weight
v55a_reliability_floor
v55a_reliable_threshold
v55a_min_effective_mass
v55a_warmup_epochs
v55a_ramp_epochs
v55a_beta_min
v55a_beta_max
v55a_soft_power
```

Add a new helper:

```text
soft_consensus_bounded_residual_spectral_anchor_loss
```

The helper may reuse the V54A component structure but must emit independent
`v55a_*` diagnostics. It must not mutate the V54A helper or change V54A
semantics.

Allowed integration:

```text
Compute V55A loss next to V50A-V54A losses.
Add `cfg.v55a_anchor_weight * v55a_anchor_loss` to the total loss.
Add `v55a_*` diagnostics to the training diagnostic dictionary.
Include V55A in spectral-anchor build conditions and anchor metric diagnostics.
Add V55A snapshot fields for epoch 1/40/80 coupling checks.
```

Allowed changes in `scripts/run_unified_aptc_9datasets.py`:

```text
Add V55A defaults to BASE_OVERRIDES.
Add `v55a_soft_consensus_bounded_residual_anchor` as one named variant.
In that variant, disable V50A/V51A/V52A/V53A/V54A losses and enable only V55A.
Set fixed preregistered constants.
```

## 5. Formula Lock

The first implementation must use exactly:

```text
soft_i = clamp(0.5 * qa_i_norm + 0.5 * ea_i_norm, 0, 1)
c_i = soft_i ^ 0.50
beta_i = 0.35 + (0.70 - 0.35) * c_i
```

Schedule:

```text
warmup_epochs = 20
ramp_epochs = 40
gamma_t = clamp((epoch + 1 - warmup_epochs) / ramp_epochs, 0, 1)
```

Reliability:

```text
r_base_i = 0.5 * conf_i + 0.5 * local_i_norm
r_agree_i = 0.5 * qa_i_norm + 0.5 * ea_i_norm
r_multiplier_i = (1 - gamma_t)
               + gamma_t * (beta_i + (1 - beta_i) * r_agree_i)
r_i = detach(clamp(r_base_i * r_multiplier_i, 0, 1))
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
conf, qa_norm, ea_norm, local_norm, soft consensus, beta, multiplier, and
reliability are detached before weighting the KL.
```

Allowed gradient:

```text
The KL term may update the model through q_refined only.
```

Forbidden gradient:

```text
The model must not learn to change reliability, beta, soft consensus, or anchor
construction to escape the anchor loss.
```

## 7. Required Diagnostics

The implementation must emit finite zero defaults when disabled and finite
values when enabled:

```text
v55a_enabled
v55a_gamma
v55a_beta_min
v55a_beta_max
v55a_soft_power
v55a_soft_consensus_mean
v55a_soft_consensus_p10
v55a_soft_consensus_p50
v55a_soft_consensus_p90
v55a_beta_mean
v55a_beta_p10
v55a_beta_p50
v55a_beta_p90
v55a_residual_multiplier_mean
v55a_anchor_loss
v55a_weighted_q_anchor_kl
v55a_weighted_q_anchor_agreement
v55a_unweighted_q_anchor_agreement
v55a_embedding_anchor_agreement
v55a_anchor_entropy
v55a_anchor_confidence
v55a_anchor_cluster_usage_entropy
v55a_anchor_effective_weight
v55a_reliability_mean
v55a_reliability_std
v55a_reliability_p10
v55a_reliability_p50
v55a_reliability_p90
v55a_reliable_node_ratio
v55a_effective_anchor_mass
v55a_base_reliability_mean
v55a_agreement_reliability_mean
v55a_confidence_component_mean
v55a_q_anchor_component_mean
v55a_embed_anchor_component_mean
v55a_local_component_mean
```

Final anchor metrics must include:

```text
v55a_anchor_acc_diagnostic
v55a_anchor_nmi_diagnostic
v55a_anchor_ari_diagnostic
```

## 8. Static Gates Before Experiment

Before any V55A run:

```powershell
conda run -n aaai-e2e-subspace python -m py_compile core\e2e\sect_coco_e2e.py scripts\run_unified_aptc_9datasets.py
```

Static expectations:

```text
No syntax error.
No dataset-name condition in V55A helper.
No V55A post-processing selector.
No change to `build_spectral_compactness_anchor`.
No changed final label path.
```

## 9. Only Authorized Run After Implementation

If static gates pass, only the following connectivity run is authorized:

```powershell
conda run --no-capture-output -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v55a_soft_consensus_bounded_residual_anchor --datasets "acm" --epochs 1 --device cuda --log-level WARNING
```

Connectivity pass condition:

```text
status=ok
legacy_head_used=false
v55a_enabled=true
v50a_enabled=false
v51a_enabled=false
v52a_enabled=false
v53a_enabled=false
v54a_enabled=false
v55a_anchor_loss finite
v55a_gamma = 0
v55a_beta_min = 0.35
v55a_beta_max = 0.70
v55a_soft_power = 0.50
v55a_soft_consensus_mean finite
v55a_beta_mean finite and within [0.35, 0.70]
v55a_reliability_mean finite
v55a_effective_anchor_mass finite
```

Connectivity is not a performance result.

## 10. Post-Connectivity Boundary

If connectivity passes, write:

```text
V55A_CONNECTIVITY_VERDICT.md
```

Only then may the preregistered mixed-stress run be launched:

```text
acm,dblp,flickr,texas,squirrel,chameleon
80 epochs
seed 42
device cuda
```

If connectivity fails, stop and write the failure verdict. Do not repair by
tuning beta bounds, soft power, schedule, thresholds, V50A anchor construction,
or dataset-specific behavior.

## 11. Review Verdict

```text
V55A may proceed to minimal implementation.
```

Reason:

```text
The implementation is a narrow, preregistered replacement of V54A's hard
consensus beta with detached soft-consensus beta. It preserves the unified
pipeline, final q_refined labels, V50A anchor construction, and V54A safety
motivation while directly testing the underactivation diagnosis.
```
