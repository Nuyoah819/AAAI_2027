# V63A Preregistration

Proposed variant:

```text
v63a_phase_locked_teacher_guard
```

This file follows `V62A_FAILURE_ANALYSIS.md`.

No V63A implementation or result exists at the time this preregistration is
written.

## 1. Research Question

V61A fixed teacher coverage, and V62A confirmed that a label-free drift
multiplier can activate after epoch 100. But V62A still failed the supported
260-epoch full run:

```text
V62A Squirrel 100E ACC = 0.3013
V62A Squirrel 260E ACC = 0.2103
V62A Flickr 100E ACC = 0.4079
V62A Flickr 260E ACC = 0.2964
```

V63A tests a narrower question:

```text
Can a fixed epoch-100 phase-locked teacher preserve the useful fully-ramped
posterior phase better than an amplified epoch-80 teacher, without using labels,
validation/test metrics, dataset names, or final-label selection?
```

This is a scheduled phase-lock route. It is not a drift-threshold sweep,
teacher-epoch sweep, EMA route, dataset selector, or final-label selector.

## 2. Inherited Mechanism

V63A keeps the following unchanged:

```text
same V59A anchor/release
same V61A floor-or-topk active teacher mask
same teacher-to-student KL direction
same base guard weight and ramp schedule
same final q_refined labels
```

The inherited active mask remains:

```text
teacher_confidence = max_k teacher_q[k]
floor_mask = teacher_confidence >= 0.45
topk_mask = top ceil(0.10 * N) nodes by teacher_confidence
active = floor_mask OR topk_mask
```

## 3. Phase-Locked Teacher

V63A stores two detached scheduled teachers:

```text
teacher80_q = q_refined.detach().clone() at epoch 80
teacher100_q = q_refined.detach().clone() at epoch 100
```

Teacher used for the guard:

```text
epochs <= 100: teacher80_q
epochs > 100: teacher100_q
```

The epoch-100 teacher is allowed because the base guard is fully ramped at
epoch 100, and V61A/V62A both pass the 100e mixed-stress gate. It is not chosen
by validation/test metrics or labels.

## 4. Guard Loss

Use the same detached teacher-to-student KL:

```text
active_kl = mean_active KL(active_teacher_q.detach() || q_refined)
```

Base guard schedule remains:

```text
guard_gamma(epoch) =
  0.0                                  if epoch <= 80
  min(1.0, (epoch - 80) / 20)          if epoch > 80
```

Guard loss:

```text
L_guard = guard_gamma(epoch) * active_kl
total_loss += v63a_guard_weight * L_guard
```

No drift multiplier is used in V63A.

## 5. Optional Class-Mass Preservation

V63A adds a small label-free phase-mass preservation term after epoch 100:

```text
teacher_mass = mean_nodes teacher100_q.detach()
student_mass = mean_nodes q_refined
mass_loss = KL(teacher_mass || student_mass)
```

Schedule:

```text
mass_gamma = 0.0 if epoch <= 100 else 1.0
total_loss += v63a_mass_guard_weight * mass_gamma * mass_loss
```

Fixed constant:

```text
v63a_mass_guard_weight = 0.005
```

This term is global, label-free, and does not use class labels, metrics, or
dataset names. It is included to discourage late collapse or class-mass drift
without selecting final labels.

## 6. Fixed Constants

```text
v63a_anchor_weight = 0.04
v63a_guard_weight = 0.02
v63a_mass_guard_weight = 0.005
v63a_absolute_floor = 0.45
v63a_min_teacher_coverage = 0.10
v63a_start_epoch = 80
v63a_phase_epoch = 100
v63a_guard_ramp_epochs = 20
v63a_max_gamma = 1.0
```

Inherited anchor/release constants:

```text
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

## 7. Hard Prohibitions

V63A must not use:

```text
dataset-specific thresholds, branches, schedules, losses, or heads
validation/test metrics in training
labels in teacher, mask, mass, or phase construction
teacher as final labels
selector between teacher_q and q_refined
selector among q_refined, KMeans, teacher labels, anchor labels, or legacy labels
V59A/V60A/V61A/V62A fallback selection
seed sweep
confidence-floor sweep
coverage sweep
guard-weight sweep
mass-guard-weight sweep
phase-epoch sweep
teacher-epoch sweep
EMA-rate sweep
```

Final labels remain:

```text
q_refined
```

## 8. Required Diagnostics

V63A must expose:

```text
v63a_enabled
v63a_guard_enabled
v63a_teacher80_ready
v63a_teacher80_epoch
v63a_teacher100_ready
v63a_teacher100_epoch
v63a_active_teacher_epoch
v63a_guard_gamma
v63a_guard_weight
v63a_mass_guard_weight
v63a_mass_gamma
v63a_absolute_floor
v63a_min_teacher_coverage
v63a_teacher_confidence_mean
v63a_teacher_active_ratio
v63a_teacher_floor_active_ratio
v63a_teacher_topk_active_ratio
v63a_guard_kl
v63a_guard_loss
v63a_mass_guard_loss
v63a_total_guard_loss
v63a_q_teacher_agreement
v63a_q_teacher_kl
v63a_teacher_mass_kl
v63a_teacher_mass_entropy
v63a_student_mass_entropy
```

It must also expose inherited anchor/release diagnostics under `v63a_*` names:

```text
v63a_release_gamma
v63a_anchor_loss
v63a_pre_release_anchor_loss
v63a_weighted_q_anchor_kl
v63a_reliability_mean
v63a_mass_scale
v63a_effective_anchor_mass
```

Snapshot diagnostics must include:

```text
v63a_guard_gamma_epoch_1
v63a_guard_gamma_epoch_80
v63a_guard_gamma_epoch_100
v63a_teacher80_ready_epoch_80
v63a_teacher100_ready_epoch_100
v63a_active_teacher_epoch_epoch_100
v63a_active_teacher_epoch_epoch_260
v63a_teacher_active_ratio_epoch_80
v63a_teacher_topk_active_ratio_epoch_80
v63a_teacher_active_ratio_epoch_100
v63a_teacher_topk_active_ratio_epoch_100
v63a_q_teacher_agreement_epoch_100
v63a_q_teacher_agreement_epoch_260
v63a_teacher_mass_kl_epoch_100
v63a_teacher_mass_kl_epoch_260
```

## 9. Required Implementation Review

Before code changes, write:

```text
V63A_IMPLEMENTATION_REVIEW.md
```

It must confirm:

```text
where teacher80_q and teacher100_q are stored
how snapshots are taken at epochs 80 and 100
how the active teacher is selected by fixed schedule only
how the floor-or-topk active mask is computed
how phase-mass KL is computed without labels, metrics, or dataset names
how all teacher tensors are detached
how loss remains zero before teacher readiness
how V59A anchor/release internals remain unchanged
how V50A-V62A active losses are disabled or wrapped
how final labels remain q_refined
```

Only after this review may minimal implementation proceed.

## 10. Authorized Connectivity Test

After implementation review and code changes, only this connectivity test is
authorized:

```powershell
conda run --no-capture-output -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v63a_phase_locked_teacher_guard --datasets "acm" --epochs 1 --device cuda --log-level WARNING
```

Connectivity pass requires:

```text
status=ok
legacy_head_used=false
v50a-v62a active losses disabled or wrapped as documented
v63a_enabled=true
v63a_teacher80_ready=false
v63a_teacher100_ready=false
v63a_guard_gamma=0.0
v63a_guard_loss=0.0
v63a_mass_guard_loss=0.0
inherited anchor branch active
final labels remain q_refined
```

## 11. First Mixed-Stress Test

Only after connectivity passes:

```powershell
conda run --no-capture-output -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v63a_phase_locked_teacher_guard --datasets "acm,dblp,flickr,texas,squirrel,chameleon" --epochs 100 --device cuda --log-level WARNING
```

No V63A 260-epoch run is authorized by this preregistration.

## 12. First Mixed-Stress Gates

Required verdict artifact:

```text
V63A_FIRST_MIXED_STRESS_VERDICT.md
```

Pass requirements:

```text
status=ok on 6/6
red-line pass on 6/6
teacher80_ready becomes true by epoch 80
teacher100_ready becomes true by epoch 100
guard_gamma_epoch_80 = 0.0
guard_gamma_epoch_100 = 1.0
active_teacher_epoch_epoch_100 = 80
teacher_active_ratio_epoch_80 >= 0.10 on 6/6 datasets
teacher_topk_active_ratio_epoch_80 >= 0.10 on 6/6 datasets
guard_loss finite after epoch 80
mass_guard_loss finite at epoch 100
abs(embedding_posterior_gap) <= 0.04 on 6/6
ACM ACC >= 0.8888
DBLP ACC >= 0.6610
Squirrel ACC >= 0.2800
```

Comparison requirement:

```text
Compare V63A 100e with V62A 100e and V62A 260e where relevant.
Do not claim full-run rescue from a 100e result.
```

## 13. Later Expansion Boundary

If the 100e mixed-stress passes, the next artifact must be:

```text
V63A_EXPANSION_REVIEW.md
```

Only that review may authorize a supported 9-dataset / 260-epoch run.

## 14. Stop Conditions

Stop immediately and write a failure analysis if:

```text
teacher snapshots use labels or metrics
active teacher selection uses labels, metrics, or dataset names
teacher is used as final labels
v63a_guard_loss is nonzero before teacher80 is ready
v63a_mass_guard_loss is nonzero before teacher100 is ready
v63a_teacher_active_ratio_epoch_80 < 0.10 on any mixed-stress dataset
Squirrel ACC < 0.2800 in first mixed-stress
ACM ACC < 0.8888 in first mixed-stress
embedding_posterior_gap exceeds 0.08 on any dataset
```

## 15. No-Fabrication Status

This is a preregistration only. It contains no V63A results.
