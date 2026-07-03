# V62A Failure Analysis

This file follows `V62A_FULL_RUN_VERDICT.md`.

No V62A rerun, seed sweep, coverage sweep, confidence-floor sweep, guard-weight
sweep, drift-floor sweep, drift-scale sweep, drift-boost sweep, teacher-epoch
sweep, dataset-specific branch, final-label selector, or post-hoc choice
between 100e and 260e is authorized.

## 1. Failure Boundary

V62A does not fail because of wiring:

```text
9/9 status=ok
legacy_head_used=false on 9/9
v50a-v61a_enabled=false on 9/9
v62a_enabled=true on 9/9
teacher_ready_epoch_80=true on 9/9
guard_gamma_epoch_80=0.0 on 9/9
guard_gamma_epoch_100=1.0 on 9/9
drift_gamma_epoch_100=0.0 on 9/9
effective_guard_multiplier_epoch_100=1.0 on 9/9
release_gamma_epoch_260=0.25 on 9/9
```

It also does not fail because the V62A drift response was inactive:

```text
Flickr drift_gamma_epoch_260 = 0.8679
Squirrel drift_gamma_epoch_260 = 0.5836
Texas drift_gamma_epoch_260 = 1.0000
```

The full-run failure is an accuracy preservation and long-run drift-repair
failure under an active, bounded drift multiplier:

```text
Squirrel ACC = 0.2103 < 0.2800
Flickr ACC = 0.2964 < 0.3500
```

## 2. What V62A Solved

V62A solved one V61A symptom: the final embedding/posterior gap is no longer
the hard failure mode.

| Dataset | V61A 260E Gap | V62A 260E Gap |
| --- | ---: | ---: |
| Flickr | -0.0632 | 0.0203 |
| Squirrel | 0.0927 | 0.0000 |
| Texas | -0.0273 | -0.0055 |

The bounded multiplier also improved two long-run floors:

| Dataset | V61A 260E ACC | V62A 260E ACC | Floor |
| --- | ---: | ---: | ---: |
| PubMed | 0.5183 | 0.5203 | 0.5200 |
| Texas | 0.6995 | 0.7213 | 0.7000 |

This validates the narrow idea that a label-free drift statistic can detect
late teacher/student divergence and increase guard pressure without labels,
metrics, dataset names, or multiplier overflow.

## 3. What V62A Did Not Solve

The same mechanism does not protect the hardest long-run class structure:

| Dataset | V62A 100E ACC | V62A 260E ACC | Change |
| --- | ---: | ---: | ---: |
| ACM | 0.8992 | 0.9131 | +0.0139 |
| DBLP | 0.6788 | 0.7190 | +0.0402 |
| Flickr | 0.4079 | 0.2964 | -0.1115 |
| Texas | 0.7322 | 0.7213 | -0.0109 |
| Squirrel | 0.3013 | 0.2103 | -0.0910 |
| Chameleon | 0.3395 | 0.3390 | -0.0005 |

Squirrel returns to the same failed long-run regime:

```text
V57A 260E Squirrel = 0.2102
V59A 260E Squirrel = 0.2103
V61A 260E Squirrel = 0.2103
V62A 260E Squirrel = 0.2103
```

Flickr regresses below both its 100e value and the long-run drift-repair floor:

```text
V62A Flickr 100E ACC = 0.4079
V62A Flickr 260E ACC = 0.2964
```

## 4. Mechanistic Diagnosis

The V62A mechanism is:

```text
keep V59A post-80 anchor release
store a detached epoch-80 q_refined teacher
use the V61A floor-or-topk active teacher mask
compute active KL(teacher_80 || q_refined)
after epoch 100, multiply the guard by a bounded drift response
```

The observed failure suggests:

```text
Scalar strengthening of KL-to-epoch-80 teacher can suppress readout/posterior
divergence, but it cannot preserve the useful class partition after long
training on Squirrel and Flickr.
```

The strongest evidence is that the intended drift response activates on the
failed datasets:

```text
Flickr drift_score_epoch_260 = 0.0721, multiplier = 1.8679
Squirrel drift_score_epoch_260 = 0.0550, multiplier = 1.5836
Texas drift_score_epoch_260 = 0.1113, multiplier = 2.0000
```

Texas passes after strong drift response, but Squirrel and Flickr do not. This
means the next route should not merely make the same epoch-80 teacher KL
stronger.

## 5. What Should Not Be Done

Do not tune V62A against this result:

```text
Do not increase drift_boost.
Do not lower drift_floor.
Do not change drift_scale.
Do not increase guard_weight.
Do not change min_teacher_coverage.
Do not move teacher_epoch through a sweep.
Do not run a seed sweep.
Do not choose 100e results as final results.
Do not select V62A only on datasets where it works.
Do not use teacher labels as final labels.
Do not select between q_refined, teacher_q, KMeans, or legacy labels.
```

These are post-hoc ways to chase the failed 260e result.

## 6. Rescue Insight

The next route should not ask:

```text
How do we further amplify the epoch-80 frozen teacher?
```

It should ask:

```text
Can the model preserve the useful fully-ramped phase reached around epoch 100
with a fixed label-free phase-lock teacher, instead of relying on a single
earlier epoch-80 teacher and a scalar multiplier?
```

V61A and V62A both pass the 100e mixed-stress gate and fail the 260e Squirrel
long-run gate. The next mechanism should treat epoch 100 as a preregistered
phase boundary because the base guard has fully ramped there, not because
validation/test labels are used.

## 7. Candidate V63A Direction

Recommended next route:

```text
v63a_phase_locked_teacher_guard
```

Mechanism:

```text
Keep V59A anchor/release unchanged.
Keep V61A floor-or-topk teacher mask unchanged.
Keep the epoch-80 teacher for epochs 81-100.
At epoch 100, store a second detached q_refined phase teacher.
After epoch 100, apply the self-distillation guard to the epoch-100 phase
teacher instead of further amplifying the epoch-80 teacher.
Optionally add a small label-free class-mass preservation term against the
epoch-100 teacher's mean posterior.
```

This is not a metric-based selector. It uses only scheduled posterior snapshots
and current `q_refined` during training. Final labels remain `q_refined`.

## 8. Required Next Artifact

Before any V63A implementation or experiment, write:

```text
V63A_PREREGISTRATION.md
```

It must specify:

```text
phase-teacher snapshot epoch
teacher selection schedule
floor-or-topk active mask
optional class-mass term and fixed weight
diagnostics for phase teacher readiness and phase guard loss
connectivity command
first mixed-stress gates
full-run authorization boundary
hard stop conditions
```

## 9. No-Fabrication Status

All V62A numbers in this analysis come from local V62A verdict artifacts and
diagnostics. V63A is only a proposed preregistration direction; no V63A result
exists.
