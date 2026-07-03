# V61A Failure Analysis

This file follows `V61A_FULL_RUN_VERDICT.md`.

No V61A rerun, seed sweep, coverage sweep, confidence-floor sweep, guard-weight
sweep, teacher-epoch sweep, dataset-specific branch, final-label selector, or
post-hoc choice between 100e and 260e is authorized.

## 1. Failure Boundary

V61A does not fail because of wiring:

```text
9/9 status=ok
legacy_head_used=false on 9/9
v50a-v60a_enabled=false on 9/9
v61a_enabled=true on 9/9
teacher_ready_epoch_80=true on 9/9
guard_gamma_epoch_80=0.0 on 9/9
guard_gamma_epoch_100=1.0 on 9/9
release_gamma_epoch_260=0.25 on 9/9
```

It also does not fail because V60A's zero-coverage problem remains:

```text
teacher_active_ratio_epoch_80 >= 0.10 on 9/9
teacher_topk_active_ratio_epoch_80 >= 0.10 on 9/9
```

The full-run failure is long-run drift under an active but insufficient frozen
teacher guard:

```text
Squirrel ACC = 0.2103 < 0.2800
Squirrel embedding_posterior_gap = 0.0927 > 0.08
Flickr ACC = 0.3197 < 0.3500
Flickr embedding_posterior_gap = -0.0632
PubMed ACC = 0.5183 < 0.5200
Texas ACC = 0.6995 < 0.7000
```

## 2. What V61A Solved

V61A solved the V60A guard-coverage failure.

Key examples:

| Dataset | V60A Active @80 | V61A Active @80 |
| --- | ---: | ---: |
| Flickr | 0.0000 | 0.1001 |
| Squirrel | 0.0000 | 0.9391 |
| Wiki | not in V60A mixed-stress | 0.1002 |

The 100e mixed-stress pass was real after the anchor wiring fix:

```text
ACM ACC = 0.8972
DBLP ACC = 0.6941
Squirrel ACC = 0.2996
max abs(embedding_posterior_gap) = 0.0035
```

This validates the narrow idea that a dataset-agnostic coverage rule can turn
the teacher guard on for weak-confidence datasets.

## 3. What V61A Did Not Solve

The 260e result shows that coverage is not enough:

| Dataset | V61A 100E ACC | V61A 260E ACC | Change |
| --- | ---: | ---: | ---: |
| ACM | 0.8972 | 0.9160 | +0.0188 |
| DBLP | 0.6941 | 0.7281 | +0.0340 |
| Flickr | 0.4081 | 0.3197 | -0.0883 |
| Texas | 0.7322 | 0.6995 | -0.0328 |
| Squirrel | 0.2996 | 0.2103 | -0.0892 |
| Chameleon | 0.3456 | 0.3412 | -0.0044 |

The teacher guard remains active, but Squirrel returns to the V57A/V59A failed
long-run regime:

```text
V57A 260E Squirrel = 0.2102
V59A 260E Squirrel = 0.2103
V61A 260E Squirrel = 0.2103
```

This means the frozen epoch-80 teacher KL does not prevent the final posterior
or readout geometry from drifting in the exact stress case it was meant to
protect.

## 4. Mechanistic Diagnosis

The V61A mechanism is:

```text
keep V59A post-80 anchor release
store a detached epoch-80 q_refined teacher
force a minimum confidence-based active teacher subset
penalize KL(teacher || q_refined) after epoch 80
```

The observed failure suggests:

```text
The guard anchors selected posterior rows but does not control the late
embedding/readout disagreement mode that reappears after 100 epochs.
```

Two diagnostics support this:

```text
Squirrel teacher_active_ratio_epoch_80 = 0.9391
Squirrel guard_loss_epoch_260 = 0.0599
Squirrel q_teacher_agreement_epoch_260 = 0.8312
Squirrel embedding_posterior_gap = 0.0927
```

The guard is present and nontrivial, but the final readout still separates from
embedding behavior. On Flickr, the gap also becomes unsafe:

```text
Flickr teacher_active_ratio_epoch_80 = 0.1001
Flickr guard_loss_epoch_260 = 0.1151
Flickr q_teacher_agreement_epoch_260 = 0.4425
Flickr embedding_posterior_gap = -0.0632
```

This points to a late drift-control problem rather than a coverage problem.

## 5. What Should Not Be Done

Do not tune V61A against this result:

```text
Do not increase min_teacher_coverage.
Do not lower absolute_floor.
Do not increase guard_weight.
Do not move teacher_epoch.
Do not run a seed sweep.
Do not choose 100e results as the final result.
Do not select V61A only on datasets where it works.
Do not use teacher labels as final labels.
Do not select between q_refined, teacher_q, KMeans, or legacy labels.
```

These are all post-hoc ways to chase the failed 260e result.

## 6. Rescue Insight

The next route should not ask:

```text
How do we make the fixed epoch-80 teacher stronger?
```

It should ask:

```text
Can the guard become drift-aware after epoch 100, increasing pressure only
when q_refined moves away from the frozen teacher or when embedding/posterior
agreement degrades, without using labels, metrics, or dataset names?
```

This reframes the next mechanism from coverage calibration to late-drift
response.

## 7. Candidate V62A Direction

Recommended next route:

```text
v62a_drift_responsive_self_distillation_guard
```

Mechanism:

```text
Keep V61A teacher mask and V59A anchor/release unchanged.
Compute a label-free drift score after epoch 100:
  drift = mean_active KL(teacher_q || q_refined)
Use a fixed, preregistered gate:
  drift_gamma = clamp((drift - drift_floor) / drift_scale, 0, 1)
Apply the guard as:
  total_loss += guard_weight * guard_gamma(epoch) * (1 + drift_boost * drift_gamma) * KL_active
```

This is not a metric-based selector. It uses only the already-defined teacher
and current posterior.

Possible fixed constants to preregister:

```text
drift_floor = 0.02
drift_scale = 0.06
drift_boost = 1.0
max_effective_guard_multiplier = 2.0
drift_start_epoch = 100
```

The constants must be fixed in `V62A_PREREGISTRATION.md` before implementation.

## 8. Required Next Artifact

Before any V62A implementation or experiment, write:

```text
V62A_PREREGISTRATION.md
```

It must specify:

```text
drift score formula
fixed drift floor/scale/boost
guard multiplier cap
diagnostics for drift_gamma and effective guard multiplier
connectivity command
first mixed-stress gates
full-run authorization boundary
hard stop conditions
```

## 9. No-Fabrication Status

All V61A numbers in this analysis come from local V61A verdict artifacts and
diagnostics. V62A is only a proposed preregistration direction; no V62A result
exists.
