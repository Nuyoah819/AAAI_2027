# V62A Preregistration

Proposed variant:

```text
v62a_drift_responsive_self_distillation_guard
```

This file follows `V61A_FAILURE_ANALYSIS.md`.

No V62A implementation or result exists at the time this preregistration is
written.

## 1. Research Question

V61A solved V60A's teacher-coverage failure but failed the 260-epoch full run.
The guard was active on all supported datasets, yet Squirrel returned to the
same failed long-run regime as V57A/V59A:

```text
V61A Squirrel 100E ACC = 0.2996
V61A Squirrel 260E ACC = 0.2103
V61A Squirrel 260E embedding_posterior_gap = 0.0927
```

V62A tests a narrower question:

```text
Can a label-free drift-responsive multiplier strengthen the already-active
teacher guard only when q_refined moves away from the epoch-80 teacher, without
using dataset names, labels, validation/test metrics, or final-label selection?
```

This is a late-drift response route. It is not a new anchor, dataset selector,
teacher-label route, confidence/coverage sweep, or seed sweep.

## 2. Inherited Mechanism

V62A keeps V61A unchanged except for a bounded drift-responsive multiplier on
the self-distillation guard:

```text
same V59A anchor/release
same epoch-80 detached q_refined teacher
same V61A floor-or-topk active teacher mask
same teacher-to-student KL direction
same final q_refined labels
same base guard weight and ramp schedule
```

The inherited active mask remains:

```text
teacher_confidence = max_k teacher_q[k]
floor_mask = teacher_confidence >= 0.45
topk_mask = top ceil(0.10 * N) nodes by teacher_confidence
active = floor_mask OR topk_mask
```

## 3. Drift-Responsive Guard

After the teacher is ready, compute a label-free drift score:

```text
active_kl = mean_active KL(teacher_q.detach() || q_refined)
drift_score = active_kl.detach()
```

Use a fixed drift response:

```text
drift_floor = 0.02
drift_scale = 0.06
drift_boost = 1.0
max_effective_guard_multiplier = 2.0
drift_start_epoch = 100
```

Drift multiplier:

```text
if epoch <= 100:
    drift_gamma = 0.0
else:
    drift_gamma = clamp((drift_score - 0.02) / 0.06, 0.0, 1.0)

effective_guard_multiplier =
    min(2.0, 1.0 + 1.0 * drift_gamma)
```

The multiplier is detached from gradients because it is a gate computed from
the current drift statistic, not a learnable objective.

## 4. Loss Form

Use the same detached teacher-to-student KL:

```text
L_guard = mean_active KL(teacher_q.detach() || q_refined)
```

V62A total loss:

```text
total_loss +=
  v62a_guard_weight
  * guard_gamma(epoch)
  * effective_guard_multiplier(epoch)
  * L_guard
```

Fixed constants:

```text
v62a_guard_weight = 0.02
v62a_absolute_floor = 0.45
v62a_min_teacher_coverage = 0.10
v62a_start_epoch = 80
v62a_guard_ramp_epochs = 20
v62a_max_gamma = 1.0
v62a_drift_start_epoch = 100
v62a_drift_floor = 0.02
v62a_drift_scale = 0.06
v62a_drift_boost = 1.0
v62a_max_effective_guard_multiplier = 2.0
```

Base guard schedule remains:

```text
guard_gamma(epoch) =
  0.0                                  if epoch <= 80
  min(1.0, (epoch - 80) / 20)          if epoch > 80
```

## 5. Inherited Anchor Configuration

V62A inherits V59A/V61A unchanged:

```text
anchor_weight=0.04
reliability_floor=0.10
reliable_threshold=0.20
min_effective_mass=0.10
warmup_epochs=20
ramp_epochs=40
beta_min=0.35
beta_max=0.70
soft_power=0.50
hybrid_compensation=0.50
target_mass=0.08
max_mass_scale=1.50
max_reliability_cap=0.90
release_start_epoch=80
release_decay_epochs=60
release_floor=0.25
```

## 6. Hard Prohibitions

V62A must not use:

```text
dataset-specific thresholds, branches, schedules, losses, or heads
validation/test metrics in training
labels in teacher, mask, drift, or multiplier construction
teacher as final labels
selector between teacher_q and q_refined
selector among q_refined, KMeans, teacher labels, anchor labels, or legacy labels
V59A/V60A/V61A fallback selection
seed sweep
confidence-floor sweep
coverage sweep
guard-weight sweep
drift-floor sweep
drift-scale sweep
drift-boost sweep
teacher-epoch sweep
EMA-rate sweep
```

Final labels remain:

```text
q_refined
```

## 7. Required Diagnostics

V62A must expose:

```text
v62a_enabled
v62a_guard_enabled
v62a_teacher_ready
v62a_teacher_epoch
v62a_guard_gamma
v62a_guard_weight
v62a_absolute_floor
v62a_min_teacher_coverage
v62a_teacher_confidence_mean
v62a_teacher_active_ratio
v62a_teacher_floor_active_ratio
v62a_teacher_topk_active_ratio
v62a_guard_kl
v62a_guard_loss
v62a_q_teacher_agreement
v62a_q_teacher_kl
v62a_drift_score
v62a_drift_gamma
v62a_drift_floor
v62a_drift_scale
v62a_drift_boost
v62a_effective_guard_multiplier
v62a_max_effective_guard_multiplier
```

It must also expose inherited anchor/release diagnostics under `v62a_*` names:

```text
v62a_release_gamma
v62a_anchor_loss
v62a_pre_release_anchor_loss
v62a_weighted_q_anchor_kl
v62a_reliability_mean
v62a_mass_scale
v62a_effective_anchor_mass
```

Snapshot diagnostics must include:

```text
v62a_guard_gamma_epoch_1
v62a_guard_gamma_epoch_80
v62a_guard_gamma_epoch_100
v62a_drift_gamma_epoch_100
v62a_effective_guard_multiplier_epoch_100
v62a_teacher_ready_epoch_80
v62a_teacher_active_ratio_epoch_80
v62a_teacher_floor_active_ratio_epoch_80
v62a_teacher_topk_active_ratio_epoch_80
v62a_q_teacher_agreement_epoch_100
v62a_drift_score_epoch_100
v62a_drift_score_epoch_260
v62a_effective_guard_multiplier_epoch_260
```

## 8. Required Implementation Review

Before code changes, write:

```text
V62A_IMPLEMENTATION_REVIEW.md
```

It must confirm:

```text
where teacher_q is stored
how teacher snapshot is taken at epoch 80
how the inherited floor-or-topk active mask is computed
how drift_score is computed without labels, metrics, or dataset names
how drift multiplier is detached and bounded
how loss remains zero before teacher is ready
how V59A anchor/release internals remain unchanged
how V50A-V61A active losses are disabled or wrapped
how final labels remain q_refined
```

Only after this review may minimal implementation proceed.

## 9. Authorized Connectivity Test

After implementation review and code changes, only this connectivity test is
authorized:

```powershell
conda run --no-capture-output -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v62a_drift_responsive_self_distillation_guard --datasets "acm" --epochs 1 --device cuda --log-level WARNING
```

Connectivity pass requires:

```text
status=ok
legacy_head_used=false
v50a-v61a active losses disabled or wrapped as documented
v62a_enabled=true
v62a_teacher_ready=false
v62a_guard_gamma=0.0
v62a_drift_gamma=0.0
v62a_effective_guard_multiplier=1.0
v62a_guard_loss=0.0
inherited anchor branch active
final labels remain q_refined
```

## 10. First Mixed-Stress Test

Only after connectivity passes:

```powershell
conda run --no-capture-output -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v62a_drift_responsive_self_distillation_guard --datasets "acm,dblp,flickr,texas,squirrel,chameleon" --epochs 100 --device cuda --log-level WARNING
```

No V62A 260-epoch run is authorized by this preregistration.

## 11. First Mixed-Stress Gates

Required verdict artifact:

```text
V62A_FIRST_MIXED_STRESS_VERDICT.md
```

Pass requirements:

```text
status=ok on 6/6
red-line pass on 6/6
teacher_ready becomes true by epoch 80
guard_gamma_epoch_80 = 0.0
guard_gamma_epoch_100 = 1.0
drift_gamma_epoch_100 = 0.0
effective_guard_multiplier_epoch_100 = 1.0
teacher_active_ratio_epoch_80 >= 0.10 on 6/6 datasets
teacher_topk_active_ratio_epoch_80 >= 0.10 on 6/6 datasets
guard_loss finite after epoch 80
abs(embedding_posterior_gap) <= 0.04 on 6/6
ACM ACC >= 0.8888
DBLP ACC >= 0.6610
Squirrel ACC >= 0.2800
```

Comparison requirement:

```text
Compare V62A 100e with V61A 100e and V61A 260e where relevant.
Do not claim full-run rescue from a 100e result.
```

## 12. Later Expansion Boundary

If the 100e mixed-stress passes, the next artifact must be:

```text
V62A_EXPANSION_REVIEW.md
```

Only that review may authorize a supported 9-dataset / 260-epoch run.

## 13. Stop Conditions

Stop immediately and write a failure analysis if:

```text
teacher snapshot uses labels or metrics
drift_score uses labels, metrics, or dataset names
teacher is used as final labels
v62a_guard_loss is nonzero before teacher is ready
v62a_teacher_active_ratio_epoch_80 < 0.10 on any mixed-stress dataset
v62a_effective_guard_multiplier exceeds 2.0
Squirrel ACC < 0.2800 in first mixed-stress
ACM ACC < 0.8888 in first mixed-stress
embedding_posterior_gap exceeds 0.08 on any dataset
```

## 14. No-Fabrication Status

This is a preregistration only. It contains no V62A results.
