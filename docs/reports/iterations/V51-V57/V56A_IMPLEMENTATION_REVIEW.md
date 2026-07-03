# V56A Implementation Review

This document reviews whether
`v56a_hybrid_consensus_floor_residual_anchor` may be implemented after the
V55A STOP verdict and the V56A preregistration. It follows
`ccf-idea-optimizer` standard rescue mode: optimize the mechanism and evidence
boundary before code changes.

Decision:

```text
APPROVE MINIMAL IMPLEMENTATION ONLY.
```

This review does not authorize a full 9-dataset run, a 260-epoch run, a seed
sweep, beta-bound sweep, soft-power sweep, hybrid-compensation sweep, schedule
variant, reliability formula variant, threshold sweep, or V50A anchor
hyperparameter change.

## 1. Evidence Basis

Files reviewed:

```text
CRITICAL_RED_LINES.md
V54A_FIRST_MIXED_STRESS_VERDICT.md
V54A_CONSENSUS_UNDERACTIVATION_ANALYSIS.md
V55A_PREREGISTRATION.md
V55A_IMPLEMENTATION_REVIEW.md
V55A_CONNECTIVITY_VERDICT.md
V55A_FIRST_MIXED_STRESS_VERDICT.md
V55A_SOFT_CONSENSUS_FAILURE_ANALYSIS.md
V56A_PREREGISTRATION.md
core/e2e/sect_coco_e2e.py
scripts/run_unified_aptc_9datasets.py
```

Current state:

```text
V55A is implemented.
V55A connectivity passed.
V55A first mixed-stress stopped before expansion.
V56A is preregistered but not implemented.
```

## 2. Problem Fit

V54A proved that hard argmax consensus protects Squirrel but still underactivates
DBLP/Texas. V55A proved that pure soft consensus is too weak and reduces beta
on the very datasets it was supposed to lift.

The remaining problem is narrow:

```text
Preserve hard-consensus safety while adding only positive soft compensation for
medium-evidence nodes.
```

The V56A mechanism matches this diagnosis:

```text
V54A: hard consensus only.
V55A: soft consensus only.
V56A: hard consensus floor plus fixed half-strength soft compensation.
```

This is a valid rescue mechanism, not a post-hoc selector, because the same
formula is used for every dataset and final labels remain `q_refined`.

## 3. Idea Card

| Field | V56A position |
| --- | --- |
| Target venue family | AAAI / AI-ML, assumed from workspace |
| Task | End-to-end graph clustering with one unified posterior output |
| Gap | Bounded residual anchors need enough mass without weak-anchor overexposure |
| Root challenge | Hard consensus is safe but coarse; soft consensus is continuous but too weak |
| Core insight | Use hard consensus as a floor and soft agreement only as positive compensation |
| Proposed mechanism | Detached hybrid beta in `[0.35, 0.70]` |
| Innovation type | Method mechanism plus diagnostic evidence |
| Evidence | Static checks, ACM 1-epoch connectivity, 6-dataset mixed-stress gates |
| Main risk | Compensation may still fail to lift DBLP/Texas or may over-lift Squirrel |

Novelty status:

```text
needs-search. This review only authorizes local mechanism testing and does not
claim literature novelty.
```

## 4. Red-Line Review

| Constraint | Review |
| --- | --- |
| No dataset-specific module, branch, head, loss, selector | Pass if V56A uses only `q_refined`, `q_embed`, `q_spec`, `edge_index`, `epoch`, and `epochs`. |
| Unified pipeline | Pass if V56A is a config-controlled loss path with the same defaults for all datasets. |
| End-to-end trainability | Pass because the loss is differentiable through `q_refined`; reliability and anchor are detached by design. |
| Preserve frontend innovations | Pass because V56A does not bypass the existing frontend, graph filtering, or `q_refined` path. |
| Final labels remain unified output | Pass if no final-label path is changed and no anchor/KMeans/legacy labels are returned. |
| No post-hoc selection | Pass if the runner creates one named V56A variant and disables V50A-V55A losses inside it. |
| No sweep/cherry-pick | Pass only for one connectivity run, then one preregistered mixed-stress run if connectivity passes. |

## 5. Required Code Changes

Allowed changes in `core/e2e/sect_coco_e2e.py`:

```text
Add config fields:
v56a_enabled
v56a_anchor_weight
v56a_reliability_floor
v56a_reliable_threshold
v56a_min_effective_mass
v56a_warmup_epochs
v56a_ramp_epochs
v56a_beta_min
v56a_beta_max
v56a_soft_power
v56a_hybrid_compensation
```

Add a new helper:

```text
hybrid_consensus_floor_residual_spectral_anchor_loss
```

The helper may reuse V54A/V55A component structure but must emit independent
`v56a_*` diagnostics. It must not mutate V54A or V55A helpers or change their
semantics.

Allowed integration:

```text
Compute V56A loss next to V50A-V55A losses.
Add `cfg.v56a_anchor_weight * v56a_anchor_loss` to the total loss.
Add `v56a_*` diagnostics to the training diagnostic dictionary.
Include V56A in spectral-anchor build conditions and anchor metric diagnostics.
Add V56A snapshot fields for epoch 1/40/80 coupling checks.
```

Allowed changes in `scripts/run_unified_aptc_9datasets.py`:

```text
Add V56A defaults to BASE_OVERRIDES.
Add `v56a_hybrid_consensus_floor_residual_anchor` as one named variant.
In that variant, disable V50A/V51A/V52A/V53A/V54A/V55A losses and enable only V56A.
Set fixed preregistered constants.
```

## 6. Formula Lock

The first implementation must use exactly:

```text
h_q_i = 1[argmax(q_refined_i) == argmax(q_spec_i)]
h_e_i = 1[argmax(q_embed_i) == argmax(q_spec_i)]
h_i = 0.5 * h_q_i + 0.5 * h_e_i

s_i = clamp(0.5 * qa_i_norm + 0.5 * ea_i_norm, 0, 1)
c_i = s_i ^ 0.50

hybrid_i = clamp(h_i + 0.50 * relu(c_i - h_i), 0, 1)
beta_i = 0.35 + (0.70 - 0.35) * hybrid_i
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

## 7. Detach And Gradient Boundary

Required:

```text
q_spec is detached.
q_refined/q_embed copies used for reliability are detached.
conf, qa_norm, ea_norm, local_norm, hard consensus, soft consensus, lifted
soft consensus, compensation, hybrid consensus, beta, multiplier, and
reliability are detached before weighting the KL.
```

Allowed gradient:

```text
The KL term may update the model through q_refined only.
```

Forbidden gradient:

```text
The model must not learn to change reliability, beta, hard/soft consensus, or
anchor construction to escape the anchor loss.
```

## 8. Required Diagnostics

The implementation must emit finite zero defaults when disabled and finite
values when enabled:

```text
v56a_enabled
v56a_gamma
v56a_beta_min
v56a_beta_max
v56a_soft_power
v56a_hybrid_compensation
v56a_hard_consensus_mean
v56a_soft_consensus_mean
v56a_lifted_soft_consensus_mean
v56a_compensation_mean
v56a_compensation_active_ratio
v56a_hybrid_consensus_mean
v56a_beta_mean
v56a_beta_p10
v56a_beta_p50
v56a_beta_p90
v56a_residual_multiplier_mean
v56a_anchor_loss
v56a_weighted_q_anchor_kl
v56a_weighted_q_anchor_agreement
v56a_unweighted_q_anchor_agreement
v56a_embedding_anchor_agreement
v56a_anchor_entropy
v56a_anchor_confidence
v56a_anchor_cluster_usage_entropy
v56a_anchor_effective_weight
v56a_reliability_mean
v56a_reliability_std
v56a_reliability_p10
v56a_reliability_p50
v56a_reliability_p90
v56a_reliable_node_ratio
v56a_effective_anchor_mass
v56a_base_reliability_mean
v56a_agreement_reliability_mean
v56a_confidence_component_mean
v56a_q_anchor_component_mean
v56a_embed_anchor_component_mean
v56a_local_component_mean
```

Final anchor metrics must include:

```text
v56a_anchor_acc_diagnostic
v56a_anchor_nmi_diagnostic
v56a_anchor_ari_diagnostic
```

## 9. Static Gates Before Experiment

Before any V56A run:

```powershell
conda run -n aaai-e2e-subspace python -m py_compile core\e2e\sect_coco_e2e.py scripts\run_unified_aptc_9datasets.py
```

Static expectations:

```text
No syntax error.
No dataset-name condition in V56A helper.
No V56A post-processing selector.
No change to `build_spectral_compactness_anchor`.
No changed final label path.
```

## 10. Only Authorized Run After Implementation

If static gates pass, only the following connectivity run is authorized:

```powershell
conda run --no-capture-output -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v56a_hybrid_consensus_floor_residual_anchor --datasets "acm" --epochs 1 --device cuda --log-level WARNING
```

Connectivity pass condition:

```text
status=ok
legacy_head_used=false
v56a_enabled=true
v50a_enabled=false
v51a_enabled=false
v52a_enabled=false
v53a_enabled=false
v54a_enabled=false
v55a_enabled=false
v56a_anchor_loss finite
v56a_gamma = 0
v56a_beta_min = 0.35
v56a_beta_max = 0.70
v56a_soft_power = 0.50
v56a_hybrid_compensation = 0.50
v56a_hard_consensus_mean finite
v56a_soft_consensus_mean finite
v56a_compensation_mean finite
v56a_beta_mean finite and within [0.35, 0.70]
v56a_reliability_mean finite
v56a_effective_anchor_mass finite
```

Connectivity is not a performance result.

## 11. Post-Connectivity Boundary

If connectivity passes, write:

```text
V56A_CONNECTIVITY_VERDICT.md
```

Only then may the preregistered mixed-stress run be launched:

```text
acm,dblp,flickr,texas,squirrel,chameleon
80 epochs
seed 42
device cuda
```

If connectivity fails, stop and write the failure verdict. Do not repair by
tuning beta bounds, soft power, hybrid compensation, schedule, thresholds,
V50A anchor construction, or dataset-specific behavior.

## 12. Review Verdict

```text
V56A may proceed to minimal implementation.
```

Reason:

```text
The implementation is a narrow, preregistered hybridization of V54A's hard
consensus floor and V55A's soft evidence signal. It preserves the unified
pipeline, final q_refined labels, V50A anchor construction, and the bounded
residual safety gates while directly testing the V55A failure diagnosis.
```
