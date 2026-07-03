# v49a_reparameterized_topology_transition Preregistration

This document preregisters the next mechanism after the diagnostic outcome of
`v48a_topology_dynamics_audit`. It is a design document only. No V49A code has
been implemented and no V49A experiment has been run.

## 1. Motivation

V48A showed that the topology masks are not frozen. They move between epochs on
all three first-stage datasets:

| Dataset | ACC | dHomo | dHetero | dHard | dScore | hard corr | Direction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ACM | 0.6450 | 0.0328 | 0.0695 | 0.0792 | 0.0439 | 0.8594 | FAIL |
| DBLP | 0.6522 | 0.0668 | 0.0489 | 0.1004 | 0.0633 | 0.8426 | PASS |
| Flickr | 0.3401 | 0.0338 | 0.0168 | 0.0428 | 0.0226 | 0.9728 | FAIL |

However, V48A also showed that ACM and Flickr move against the intended
posterior target directions:

| Dataset | Homo Tgt | Hetero Tgt | Defer Tgt | Targeted Homo Delta | Targeted Hetero Delta | Targeted Hard Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ACM | 0.3037 | 0.1994 | 0.3035 | -0.001801 | -0.001988 | 0.003623 |
| DBLP | 0.2852 | 0.1752 | 0.3135 | 0.002328 | 0.004466 | 0.002717 |
| Flickr | 0.1201 | 0.2975 | 0.2541 | -0.000474 | -0.002498 | 0.003840 |

Conclusion:

```text
The current topology contraction is movable but not reliably target-responsive.
```

Therefore V49A changes the topology transition parameterization rather than
adding another external topology loss.

## 2. Version Name

```text
v49a_reparameterized_topology_transition
```

Core hypothesis:

```text
If homo-vs-hetero orientation and resolved-vs-hard clarity are decoupled in the
topology simplex, then hard-to-homo and hard-to-hetero transitions can become
more directionally reliable without another failed external target loss.
```

## 3. Hard Prohibitions

V49A must not use:

- dataset-specific module, branch, head, loss, assigner, threshold, or weight
- legacy head
- adaptive selector or post-processing selector
- embedding cosine margin loss
- edge-level overlap margin loss
- v43b-style selective conflict gate
- post-normalized high-pass energy loss
- global or edge-local pre-HP response pressure loss
- direct `mean(hard^2)` band penalty from V46A
- posterior-guided hard-band CE loss from V47A
- V48A diagnostic outcome as a training target
- label information
- test-set-driven safety correction
- sweep over weights, temperatures, quantiles, or margins

The following weights must remain zero:

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
v46a_band_cal_weight = 0.0
v46a_balance_weight = 0.0
v46a_spread_weight = 0.0
v47a_resolution_weight = 0.0
v47a_usage_guard_weight = 0.0
partition_spread_weight = 0.0
freq_separation_weight = 0.0
freq_ortho_weight = 0.0
```

V48A diagnostics may be retained but must not contribute to loss.

## 4. Allowed Mechanism

### 4.1 Decoupled Topology Coordinates

Replace the current single-score ordered-threshold contraction for this variant
with a unified differentiable topology transition that produces two edge-level
coordinates:

```text
orientation_logit_ij
clarity_logit_ij
```

Interpretation:

```text
orientation: homo side vs hetero side
clarity: resolved edge vs ambiguous/hard edge
```

The first preregistered mask mapping is:

```text
clear_ij = sigmoid(clarity_logit_ij / tau_clear)
orient_ij = sigmoid(orientation_logit_ij / tau_orient)

homo_ij   = clear_ij * orient_ij
hetero_ij = clear_ij * (1 - orient_ij)
hard_ij   = 1 - clear_ij
```

The masks already sum to one:

```text
homo_ij + hetero_ij + hard_ij = 1
```

No post-processing selector is allowed.

### 4.2 Shared Unified Inputs

The new coordinates must be produced from the same edge evidence pathway used
by the current frontend. They must not use labels or dataset identifiers.

Allowed implementation shapes:

```text
edge evidence/features -> shared edge module -> orientation_logit, clarity_logit
```

or a minimally invasive extension of the existing edge confidence module.

### 4.3 Initialization Constraint

The first implementation should avoid a random topology reset. It should
initialize or bias the new mapping so early mask usage is not collapsed.

Allowed safe initialization:

```text
orientation initialized from the existing score direction
clarity initialized so hard usage is nonzero and comparable to the existing
frontend's hard usage
```

The exact code-level initialization must be recorded in
`V49A_IMPLEMENTATION_NOTES.md` before running the first smoke.

### 4.4 Diagnostics Only Targets

V49A may compute V47A/V48A posterior-guided target groups as diagnostics only:

```text
homo_target
hetero_target
defer_target
```

No posterior-guided CE loss is allowed in V49A.

## 5. First Implementation Constants

If implemented, use exactly:

```text
v49a_enabled = true
v49a_tau_clear = 1.0
v49a_tau_orient = 1.0
v49a_snapshot_sample_size = 20000
v49a_movement_eps = 1e-8
```

No sweep is allowed. If the first configuration fails, stop and write a verdict.

## 6. Required Diagnostics

Red-line and safety:

```text
legacy_head_used
v43b_enabled
v44_enabled
v44b_enabled
v45a_enabled
v46a_enabled
v47a_enabled
v48a_enabled
v49a_enabled
embedding_kmeans_acc
final_acc
embedding_posterior_gap
```

Topology usage:

```text
v49a_homo_usage
v49a_hetero_usage
v49a_hard_usage
v49a_usage_entropy
v49a_clear_mean
v49a_clear_std
v49a_orient_mean
v49a_orient_std
```

Movement and direction:

```text
v49a_has_prev_snapshot
v49a_mean_abs_delta_homo
v49a_mean_abs_delta_hetero
v49a_mean_abs_delta_hard
v49a_mean_abs_delta_score
v49a_hard_mass_delta
v49a_hard_rank_corr_prev
v49a_homo_target_mass
v49a_hetero_target_mass
v49a_defer_target_mass
v49a_targeted_homo_delta
v49a_targeted_hetero_delta
v49a_targeted_hard_delta
```

Performance and band:

```text
final_acc
nmi
ari
v49a_band_mass
```

Retained diagnostics, diagnostic only:

```text
v44b_pre_hp_response_std
v45a_edge_response_gap
v45a_edge_response_corr
v46a_band_mass
v47a_homo_target_mass
v48a_mean_abs_delta_hard
```

## 7. First-Stage Experiment

Only after implementation sanity checks:

```text
datasets = acm,dblp,flickr
epochs = 80
seed = 42
device = cuda
```

Command template:

```powershell
conda run -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v49a_reparameterized_topology_transition --datasets "acm,dblp,flickr" --epochs 80 --device cuda --log-level WARNING
```

No second-batch smoke is allowed unless all first-stage gates pass.

## 8. First-Stage Gates

### 8.1 Red-Line Gate

Must pass:

```text
legacy_head_used=false
v43b_enabled=false
v44_enabled=false
v44b_enabled=false
v45a_enabled=false
v46a_enabled=false
v47a_enabled=false
v49a_enabled=true
no selector / no post-processing selector
no failed external topology loss revived
```

`v48a_enabled` may be either false or true depending on whether V49A reuses the
same audit helper for diagnostics, but it must be diagnostic only.

### 8.2 Usage Non-Collapse Gate

Must pass on 3/3:

```text
v49a_usage_entropy >= 0.60
v49a_homo_usage > 0.05
v49a_hetero_usage > 0.05
v49a_hard_usage > 0.05
```

### 8.3 Diagnostic Completeness Gate

Must pass on 3/3:

```text
v49a_has_prev_snapshot=true by final epoch
all movement and direction diagnostics finite
```

### 8.4 Directional Consistency Gate

Primary mechanism gate:

At least 2/3 datasets must satisfy:

```text
v49a_targeted_homo_delta > 0
v49a_targeted_hetero_delta > 0
v49a_targeted_hard_delta >= 0
```

Additional safety requirement:

```text
ACM and Flickr must not both repeat the V48A failure pattern:
targeted_homo_delta < 0 and targeted_hetero_delta < 0
```

If directional consistency fails, stop regardless of ACC.

### 8.5 Posterior/Readout Safety Gate

Must pass on at least 2/3 and must not catastrophically fail on any dataset:

```text
abs(embedding_posterior_gap) <= 0.02 on at least 2/3
abs(embedding_posterior_gap) <= 0.04 on 3/3
```

Rationale: V48A had Flickr `embedding_posterior_gap=0.0263` as an audit-only
run. V49A is a mechanism variant, so posterior/readout safety remains required,
but the first-stage diagnostic should distinguish mild readout drift from
collapse.

### 8.6 Band and Performance Context Gate

Record but do not use alone to expand:

```text
ACM band reference: V48A TBD from diagnostics if needed
DBLP band reference: V48A TBD from diagnostics if needed
Flickr band reference: V48A TBD from diagnostics if needed
```

Performance context thresholds:

```text
ACM ACC >= 0.6450
DBLP ACC >= 0.6522
Flickr ACC >= 0.3401
```

These are V48A audit ACC references and are context only. V49A cannot expand
based on ACC unless directional and safety gates pass.

## 9. Stop Conditions

Stop immediately after first-stage smoke if any occurs:

- any red-line violation
- usage collapse
- missing or non-finite movement diagnostics
- directional consistency gate failure
- posterior/readout catastrophic failure
- ACM/Flickr repeat the same wrong-direction pattern as V48A

Do not run:

- second-batch smoke
- full 9-dataset smoke
- 260-epoch full run
- weight sweep
- temperature sweep
- initialization sweep
- target/quantile sweep

## 10. Result Templates

### 10.1 Direction Gate Table

| Dataset | ACC | Emb Gap | Homo Use | Hetero Use | Hard Use | Homo Delta | Hetero Delta | Hard Delta | Direction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ACM | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| DBLP | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Flickr | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### 10.2 V48A vs V49A Mechanism Comparison

| Dataset | V48A Direction | V49A Direction | V48A ACC | V49A ACC | Verdict |
| --- | --- | --- | ---: | ---: | --- |
| ACM | FAIL | TBD | 0.6450 | TBD | TBD |
| DBLP | PASS | TBD | 0.6522 | TBD | TBD |
| Flickr | FAIL | TBD | 0.3401 | TBD | TBD |

## 11. No-Fabrication Status

All cited V48A values come from `V48A_FIRST_AUDIT_VERDICT.md` and diagnostics.
V49A has not been implemented or run. All V49A results are `TBD`.
