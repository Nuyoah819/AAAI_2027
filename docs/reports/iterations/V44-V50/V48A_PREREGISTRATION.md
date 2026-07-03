# v48a_topology_dynamics_audit Preregistration

This document preregisters a diagnostic audit after the failure of
`v47a_posterior_guided_band_resolution`. It is not a performance-claim variant.
No V48A code has been implemented and no V48A experiment has been run.

## 1. Motivation

V47A produced non-degenerate hard-band targets and nonzero resolution loss, but
band mass and ACC worsened against V46A:

| Dataset | ACC | Emb Gap | Band | Effective Target | Resolution Loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| ACM | 0.6651 | 0.0000 | 0.5215 | 0.7869 | 0.7018 |
| DBLP | 0.6485 | 0.0000 | 0.6853 | 0.7731 | 0.8818 |
| Flickr | 0.3537 | 0.0000 | 0.5100 | 0.6664 | 0.6353 |

This establishes that:

```text
posterior-guided targets were present, but topology masks did not move in the
desired direction.
```

Therefore V48A asks whether the current topology contraction is dynamically
responsive to calibration signals.

## 2. Version Name

```text
v48a_topology_dynamics_audit
```

Core hypothesis:

```text
If topology masks and thresholds show little or misdirected movement under
existing calibration signals, then additional external target losses should
stop until the topology contraction parameterization is redesigned.
```

## 3. Hard Prohibitions

V48A must not introduce a new optimization mechanism beyond audit-safe
instrumentation.

Must remain disabled:

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
```

Also keep disabled in the first audit variant:

```text
partition_spread_weight = 0.0
freq_separation_weight = 0.0
freq_ortho_weight = 0.0
```

V48A must not use:

- dataset-specific branch, threshold, weight, or module
- legacy head
- selector or post-processing selector
- labels
- test-set-driven correction
- sweep

## 4. Allowed Audit Mechanism

V48A may add diagnostics only. It may compute:

```text
homo, hetero, hard
score
low_threshold, high_threshold
q_refined
```

It may retain V47A target diagnostics as diagnostic only, with zero V47A loss.

### 4.1 Epoch-To-Epoch Movement Diagnostics

Preferred first implementation avoids gradient hooks and records movement
between consecutive training epochs:

```text
mean_abs_delta_homo
mean_abs_delta_hetero
mean_abs_delta_hard
mean_abs_delta_score
hard_mass_delta
threshold_delta
hard_rank_corr_prev
```

These require storing previous-epoch detached snapshots or compact summaries.
If full edge-level snapshots are too memory-heavy, store sampled or aggregate
summaries, but the sampling rule must be fixed and not dataset-specific.

### 4.2 Directional Movement Diagnostics

Using V47A posterior-guided targets as diagnostics only:

```text
targeted_hard_delta_homo
targeted_hard_delta_hetero
targeted_hard_delta_hard
```

Interpretation:

- For homo-targeted hard edges, does `homo` rise?
- For hetero-targeted hard edges, does `hetero` rise?
- For defer-targeted edges, does `hard` remain stable?

No V47A loss is active.

### 4.3 Optional Gradient Diagnostics

If feasible without invasive changes, V48A may record gradient norms:

```text
score_grad_norm
low_threshold_grad_norm
high_threshold_grad_norm
```

Gradient diagnostics are optional for the first audit because hook complexity
can distract from the main question.

## 5. First Implementation Constants

Use exactly:

```text
v48a_enabled = true
v48a_snapshot_sample_size = 20000
v48a_movement_eps = 1e-8
```

No sweep is allowed. If edge count is smaller than the sample size, use all
edges. If sampling is used, sample deterministically from edge order, not
randomly.

## 6. Required Diagnostics

Red-line:

```text
legacy_head_used
v43b_enabled
v44_enabled
v44b_enabled
v45a_enabled
v46a_enabled
v47a_enabled
v48a_enabled
embedding_posterior_gap
```

Movement:

```text
v48a_has_prev_snapshot
v48a_mean_abs_delta_homo
v48a_mean_abs_delta_hetero
v48a_mean_abs_delta_hard
v48a_mean_abs_delta_score
v48a_hard_mass_delta
v48a_threshold_delta
v48a_hard_rank_corr_prev
```

Target-direction diagnostics:

```text
v48a_homo_target_mass
v48a_hetero_target_mass
v48a_defer_target_mass
v48a_targeted_homo_delta
v48a_targeted_hetero_delta
v48a_targeted_hard_delta
```

Performance is recorded but not used as an expansion trigger.

## 7. First-Stage Audit Run

First-stage audit:

```text
datasets = acm,dblp,flickr
epochs = 80
seed = 42
device = cuda
```

Command template:

```powershell
conda run -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v48a_topology_dynamics_audit --datasets "acm,dblp,flickr" --epochs 80 --device cuda --log-level WARNING
```

## 8. Audit Gates

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
v48a_enabled=true
```

### 8.2 Diagnostic Completeness Gate

Must pass on 3/3:

```text
v48a_has_prev_snapshot=true by final epoch
all movement diagnostics finite
```

### 8.3 Movement Non-Degeneracy Gate

At least one must be nonzero on 3/3:

```text
v48a_mean_abs_delta_homo > 1e-6
v48a_mean_abs_delta_hetero > 1e-6
v48a_mean_abs_delta_hard > 1e-6
v48a_mean_abs_delta_score > 1e-6
```

If all are near zero, topology contraction is effectively frozen under the
current training path.

### 8.4 Directional Consistency Gate

This is diagnostic, not pass/fail for expansion:

```text
targeted_homo_delta > 0
targeted_hetero_delta > 0
targeted_hard_delta >= 0 for defer targets
```

Record whether each holds. Do not tune based on the outcome.

## 9. Stop Conditions

After the first-stage audit, stop and write:

```text
V48A_FIRST_AUDIT_VERDICT.md
```

Do not run:

- second-batch smoke
- full run
- sweep
- performance-claim run

V48A is diagnostic. It cannot authorize expansion by ACC alone.

## 10. Result Template

| Dataset | ACC | Emb Gap | dHomo | dHetero | dHard | dScore | Direction Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ACM | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| DBLP | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Flickr | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## 11. No-Fabrication Status

All V47A values cited here come from `V47A_FIRST_SMOKE_VERDICT.md` and
diagnostics. V48A has not been implemented or run. All V48A outputs are `TBD`.
