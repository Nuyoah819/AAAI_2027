# V60A Failure Analysis

This file analyzes the failed first mixed-stress run of
`v60a_ema_self_distillation_drift_guard`.

It follows `V60A_FIRST_MIXED_STRESS_VERDICT.md`.

No V60A expansion, 260-epoch run, seed sweep, confidence-threshold sweep,
guard-weight sweep, teacher-epoch sweep, EMA update, or post-hoc selector is
authorized by this analysis.

## 1. Failure Boundary

V60A does not fail because the teacher snapshot or guard schedule is miswired:

```text
red-line gate passes
teacher_ready becomes true by epoch 80 on 6/6 datasets
guard_gamma_epoch_80 = 0.0
guard_gamma_epoch_100 = 1.0
V50A-V59A active losses are disabled
```

It fails because the fixed confidence mask provides no guard coverage on the
datasets that most need drift protection:

```text
DBLP teacher_active_ratio_epoch_80 = 0.0000
Flickr teacher_active_ratio_epoch_80 = 0.0000
Squirrel teacher_active_ratio_epoch_80 = 0.0000
```

The hard performance/safety failure is:

```text
Squirrel ACC = 0.2096 < 0.2800
Squirrel abs(embedding_posterior_gap) = 0.0936 > 0.08
```

Therefore V60A must stop before any expansion.

## 2. What V60A Solved

V60A validated the wiring of the intended guard:

```text
teacher snapshot at epoch 80 works
teacher remains detached/frozen
guard starts only after the snapshot
guard loss is zero before teacher readiness
```

It also shows the guard can activate on some datasets:

| Dataset | Active Ratio @80 | Guard Loss @100 |
| --- | ---: | ---: |
| ACM | 0.0585 | 0.0014 |
| Texas | 0.9071 | 0.0078 |
| Chameleon | 0.0031 | 0.0103 |

But this activation is too uneven to protect the stress cases.

## 3. What V60A Did Not Solve

The fixed confidence threshold fails the guard-coverage objective:

| Dataset | Teacher Conf Mean | Active Ratio @80 | Guard Active? |
| --- | ---: | ---: | --- |
| ACM | 0.5763 | 0.0585 | yes |
| DBLP | 0.5164 | 0.0000 | no |
| Flickr | 0.3231 | 0.0000 | no |
| Texas | 0.8260 | 0.9071 | yes |
| Squirrel | 0.4647 | 0.0000 | no |
| Chameleon | 0.4771 | 0.0031 | nearly no |

The preregistered gate required at least 3/6 datasets to have active ratio
above 0.05. Only ACM and Texas pass.

## 4. Mechanistic Diagnosis

The V60A idea was:

```text
Preserve the useful epoch-80 posterior geometry on confident nodes.
```

The implementation exposed a deeper issue:

```text
The weak-anchor and heterophily datasets do not produce enough high-confidence
teacher nodes under a fixed absolute threshold of 0.60.
```

This means V60A applies the guard where the teacher is already very confident
and often safer, but provides no protection where late drift is worst.

Squirrel is the clearest failure:

```text
teacher_active_ratio_epoch_80 = 0.0
guard_loss = 0.0
ACC drops to the same failed regime as V59A 260e by epoch 100
embedding_posterior_gap exceeds the hard safety bound
```

Therefore the failure is not simply that the guard is too weak. On Squirrel,
the guard is absent.

## 5. What Should Not Be Done

Do not tune V60A against this result:

```text
Do not sweep confidence_threshold.
Do not sweep guard_weight.
Do not move teacher_epoch.
Do not add EMA update.
Do not use dataset-specific thresholds.
Do not select teacher labels as final labels.
Do not select V59A for some datasets and V60A for others.
```

A lower absolute threshold may be tempting, but lowering it after seeing this
result would be a threshold sweep. The next route must be preregistered as a
new mechanism with a fixed, dataset-agnostic coverage rule.

## 6. Rescue Insight

The next route should not ask:

```text
What absolute confidence threshold is best?
```

It should ask:

```text
Can a dataset-agnostic minimum-coverage teacher mask provide guard coverage on
weak-confidence datasets without making the teacher all-node or label-like?
```

This reframes the mechanism from absolute confidence to coverage-calibrated
self-distillation.

## 7. Candidate V61A Direction

Recommended next route:

```text
v61a_quantile_coverage_self_distillation_guard
```

Mechanism:

```text
Keep V59A anchor/release unchanged.
Keep epoch-80 detached q_refined teacher.
Replace the absolute confidence-only mask with a fixed dataset-agnostic
coverage rule:
  active if teacher_confidence >= max(absolute_floor, confidence quantile)
  and enforce a fixed minimum top-coverage fraction.
```

This must be preregistered carefully because it changes the teacher mask. It is
not a threshold sweep if the constants and rule are fixed before implementation.

Possible fixed rule to preregister:

```text
absolute_floor = 0.45
min_teacher_coverage = 0.10
active = top max(10%, nodes with confidence >= 0.45) by teacher_confidence
```

The exact rule should be specified in `V61A_PREREGISTRATION.md` before any code
change.

## 8. Required Next Artifact

Before any implementation or experiment, write:

```text
V61A_PREREGISTRATION.md
```

It must specify:

```text
teacher snapshot epoch
coverage mask rule
fixed coverage and confidence constants
loss form
diagnostics for active coverage and weak-dataset coverage
connectivity command
first mixed-stress gates
hard stop conditions
```

## 9. No-Fabrication Status

This analysis uses only local V60A mixed-stress diagnostics and prior local
V59A verdict files. No V61A result exists.
