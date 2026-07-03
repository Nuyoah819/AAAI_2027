# V61A Preregistration

Proposed variant:

```text
v61a_quantile_coverage_self_distillation_guard
```

This file follows `V60A_FAILURE_ANALYSIS.md`.

No V61A implementation or result exists at the time this preregistration is
written.

## 1. Research Question

V60A showed that an epoch-80 detached teacher can be captured and the guard can
activate, but the fixed absolute confidence threshold leaves DBLP, Flickr, and
Squirrel with zero active teacher coverage.

V61A tests a narrower question:

```text
Can a fixed, dataset-agnostic coverage-calibrated teacher mask make the
self-distillation guard active on weak-confidence datasets without turning the
teacher into an all-node final-label selector?
```

This is a teacher-mask rescue route. It is not a new anchor, reliability,
schedule, final-label, or dataset-selector route.

## 2. Mechanism

V61A keeps V60A/V59A unchanged except for the teacher active mask:

```text
same V59A anchor/release
same epoch-80 detached q_refined teacher
same teacher-to-student KL direction
same final q_refined labels
same guard weight and ramp schedule
```

Replace V60A's mask:

```text
active if teacher_confidence >= 0.60
```

with a fixed coverage-calibrated mask:

```text
teacher_confidence = max_k teacher_q[k]
absolute_floor = 0.45
min_teacher_coverage = 0.10

floor_mask = teacher_confidence >= absolute_floor
topk_mask = top ceil(0.10 * N) nodes by teacher_confidence
active = floor_mask OR topk_mask
```

The top-coverage rule is dataset-agnostic because it depends only on the
teacher confidence distribution, not labels, metrics, dataset names, or
validation/test performance.

## 3. Loss Form

Use the same detached teacher-to-student KL:

```text
L_guard = mean_active KL(teacher_q.detach() || q_refined)
```

V61A total loss:

```text
total_loss += v61a_guard_weight * guard_gamma(epoch) * L_guard
```

Fixed constants:

```text
v61a_guard_weight = 0.02
v61a_absolute_floor = 0.45
v61a_min_teacher_coverage = 0.10
v61a_start_epoch = 80
v61a_guard_ramp_epochs = 20
v61a_max_gamma = 1.0
```

Guard schedule:

```text
guard_gamma(epoch) =
  0.0                                  if epoch <= 80
  min(1.0, (epoch - 80) / 20)          if epoch > 80
```

## 4. Inherited Configuration

V61A inherits V59A unchanged:

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

## 5. Hard Prohibitions

V61A must not use:

```text
dataset-specific thresholds, branches, schedules, losses, or heads
validation/test metrics in training
labels in teacher construction or mask construction
teacher as final labels
selector between teacher_q and q_refined
V59A/V60A fallback selection
seed sweep
absolute-floor sweep
coverage sweep
guard-weight sweep
teacher-epoch sweep
EMA-rate sweep
```

Final labels remain:

```text
q_refined
```

## 6. Required Diagnostics

V61A must expose:

```text
v61a_enabled
v61a_guard_enabled
v61a_teacher_ready
v61a_teacher_epoch
v61a_guard_gamma
v61a_guard_weight
v61a_absolute_floor
v61a_min_teacher_coverage
v61a_teacher_confidence_mean
v61a_teacher_active_ratio
v61a_teacher_floor_active_ratio
v61a_teacher_topk_active_ratio
v61a_guard_kl
v61a_guard_loss
v61a_q_teacher_agreement
v61a_q_teacher_kl
```

It must also expose inherited anchor/release diagnostics under `v61a_*` names:

```text
v61a_release_gamma
v61a_anchor_loss
v61a_pre_release_anchor_loss
v61a_weighted_q_anchor_kl
v61a_reliability_mean
v61a_mass_scale
v61a_effective_anchor_mass
```

Snapshot diagnostics must include:

```text
v61a_guard_gamma_epoch_1
v61a_guard_gamma_epoch_80
v61a_guard_gamma_epoch_100
v61a_teacher_ready_epoch_80
v61a_teacher_active_ratio_epoch_80
v61a_teacher_floor_active_ratio_epoch_80
v61a_teacher_topk_active_ratio_epoch_80
v61a_q_teacher_agreement_epoch_100
```

## 7. Required Implementation Review

Before code changes, write:

```text
V61A_IMPLEMENTATION_REVIEW.md
```

It must confirm:

```text
where teacher_q is stored
how teacher snapshot is taken at epoch 80
how the floor-or-topk active mask is computed
how topk avoids labels/metrics/dataset names
how loss remains zero before teacher is ready
how V59A anchor/release internals remain unchanged
how V50A-V60A active losses are disabled or wrapped
how final labels remain q_refined
```

Only after this review may minimal implementation proceed.

## 8. Authorized Connectivity Test

After implementation review and code changes, only this connectivity test is
authorized:

```powershell
conda run --no-capture-output -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v61a_quantile_coverage_self_distillation_guard --datasets "acm" --epochs 1 --device cuda --log-level WARNING
```

Connectivity pass requires:

```text
status=ok
legacy_head_used=false
v50a-v60a active losses disabled or wrapped as documented
v61a_enabled=true
v61a_teacher_ready=false
v61a_guard_gamma=0.0
v61a_guard_loss=0.0
final labels remain q_refined
```

## 9. First Mixed-Stress Test

Only after connectivity passes:

```powershell
conda run --no-capture-output -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v61a_quantile_coverage_self_distillation_guard --datasets "acm,dblp,flickr,texas,squirrel,chameleon" --epochs 100 --device cuda --log-level WARNING
```

No V61A 260-epoch run is authorized by this preregistration.

## 10. First Mixed-Stress Gates

Required verdict artifact:

```text
V61A_FIRST_MIXED_STRESS_VERDICT.md
```

Pass requirements:

```text
status=ok on 6/6
red-line pass on 6/6
teacher_ready becomes true by epoch 80
guard_gamma_epoch_80 = 0.0
guard_gamma_epoch_100 = 1.0
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
Compare V61A 100e with V60A 100e, V59A 80e, and V59A 260e where relevant.
Do not claim full-run rescue from a 100e result.
```

## 11. Later Expansion Boundary

If the 100e mixed-stress passes, the next artifact must be:

```text
V61A_EXPANSION_REVIEW.md
```

Only that review may authorize a supported 9-dataset / 260-epoch run.

## 12. Stop Conditions

Stop immediately and write a failure analysis if:

```text
teacher snapshot uses labels or metrics
teacher is used as final labels
v61a_guard_loss is nonzero before teacher is ready
v61a_teacher_active_ratio_epoch_80 < 0.10 on any mixed-stress dataset
Squirrel ACC < 0.2800 in first mixed-stress
ACM ACC < 0.8888 in first mixed-stress
embedding_posterior_gap exceeds 0.08 on any dataset
```

## 13. No-Fabrication Status

This is a preregistration only. It contains no V61A results.
