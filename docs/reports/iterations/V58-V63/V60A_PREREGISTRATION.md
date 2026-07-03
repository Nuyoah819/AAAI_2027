# V60A Preregistration

Proposed variant:

```text
v60a_ema_self_distillation_drift_guard
```

This file follows `V59A_FAILURE_ANALYSIS.md`.

No V60A implementation or result exists at the time this preregistration is
written.

## 1. Research Question

V57A solved short-run anchor mass non-collapse but failed at 260 epochs. V59A
kept the validated 80-epoch absorption window and released anchor pressure after
epoch 80, but still failed on Flickr and Squirrel.

V60A tests a narrower diagnosis:

```text
Late drift is not caused only by sustained anchor pressure. The model also needs
a unified, label-free guard that preserves the stable early posterior geometry
after the 80-epoch compactness window.
```

## 2. Mechanism

V60A keeps V59A unchanged and adds one post-80 self-distillation guard:

```text
Use V59A's spectral anchor, reliability, mass normalization, and post-80 release.
At epoch 80, store a detached teacher snapshot of q_refined.
After epoch 80, penalize drift from the teacher only on confident teacher nodes.
```

The final labels remain:

```text
q_refined
```

Teacher labels must never be used as final labels.

## 3. Teacher Definition

Teacher snapshot:

```text
teacher_q = detach(q_refined at epoch 80)
```

No labels, validation metrics, test metrics, or dataset names may be used to
construct the teacher.

If implementation needs a fallback before the snapshot exists:

```text
distillation loss = 0 before epoch 80
```

The teacher is frozen after epoch 80 for the first V60A version. Do not update
it with EMA in V60A-A, because updating the teacher would add another time
constant and blur the first test.

## 4. Confidence Mask

Use a dataset-agnostic confidence mask:

```text
teacher_confidence = max_k teacher_q[k]
active if teacher_confidence >= 0.60
```

If no nodes pass the confidence threshold:

```text
distillation loss = 0
```

No dataset-specific confidence thresholds are allowed.

## 5. Loss Form

Use detached teacher-to-student KL:

```text
L_guard = mean_active KL(teacher_q.detach() || q_refined)
```

Only active teacher-confident nodes contribute.

V60A loss is:

```text
total_loss += v60a_guard_weight * release_guard_gamma(epoch) * L_guard
```

Fixed constants:

```text
v60a_guard_weight = 0.02
v60a_confidence_threshold = 0.60
v60a_start_epoch = 80
v60a_ramp_epochs = 20
v60a_max_gamma = 1.0
```

Guard schedule:

```text
guard_gamma(epoch) =
  0.0                                  if epoch <= 80
  min(1.0, (epoch - 80) / 20)          if epoch > 80
```

The guard starts after the teacher snapshot, then ramps over 20 epochs.

## 6. Inherited V59A Configuration

V60A inherits V59A unchanged:

```text
v59a_anchor_weight=0.04
v59a_reliability_floor=0.10
v59a_reliable_threshold=0.20
v59a_min_effective_mass=0.10
v59a_warmup_epochs=20
v59a_ramp_epochs=40
v59a_beta_min=0.35
v59a_beta_max=0.70
v59a_soft_power=0.50
v59a_hybrid_compensation=0.50
v59a_target_mass=0.08
v59a_max_mass_scale=1.50
v59a_max_reliability_cap=0.90
v59a_release_start_epoch=80
v59a_release_decay_epochs=60
v59a_release_floor=0.25
```

V60A may reuse V59A internals as a wrapper, but all active diagnostics must be
reported as independent `v60a_*` fields.

## 7. Hard Prohibitions

V60A must not use:

```text
dataset-specific branches, thresholds, schedules, losses, or heads
validation/test metrics in training
label-aware teacher construction
teacher as final labels
selector between teacher_q and q_refined
V59A/V57A fallback selection
seed sweep
confidence-threshold sweep
guard-weight sweep
teacher-epoch sweep
EMA-rate sweep
```

## 8. Required Diagnostics

V60A must expose:

```text
v60a_enabled
v60a_guard_enabled
v60a_teacher_ready
v60a_teacher_epoch
v60a_guard_gamma
v60a_guard_weight
v60a_confidence_threshold
v60a_teacher_confidence_mean
v60a_teacher_active_ratio
v60a_guard_kl
v60a_guard_loss
v60a_q_teacher_agreement
v60a_q_teacher_kl
```

It must also expose the inherited V59A anchor diagnostics under `v60a_*` names
or provide a clear `v59a_*`/`v60a_*` mapping in the verdict.

Snapshot diagnostics must include:

```text
v60a_guard_gamma_epoch_1
v60a_guard_gamma_epoch_80
v60a_guard_gamma_epoch_100
v60a_teacher_ready_epoch_80
v60a_teacher_active_ratio_epoch_80
v60a_q_teacher_agreement_epoch_100
```

## 9. Required Implementation Review

Before code changes, write:

```text
V60A_IMPLEMENTATION_REVIEW.md
```

It must confirm:

```text
where teacher_q is stored
how teacher snapshot is taken at epoch 80
how loss remains zero before teacher is ready
how no labels or metrics enter the teacher
how V59A internals remain unchanged
how V50A-V59A active losses are disabled or wrapped
how final labels remain q_refined
```

Only after this review may minimal implementation proceed.

## 10. Authorized Connectivity Test

After implementation review and code changes, only this connectivity test is
authorized:

```powershell
conda run --no-capture-output -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v60a_ema_self_distillation_drift_guard --datasets "acm" --epochs 1 --device cuda --log-level WARNING
```

Connectivity pass requires:

```text
status=ok
legacy_head_used=false
v50a-v59a active losses disabled or wrapped as documented
v60a_enabled=true
v60a_teacher_ready=false
v60a_guard_gamma=0.0
v60a_guard_loss=0.0
final labels remain q_refined
```

## 11. First Mixed-Stress Test

Only after connectivity passes:

```powershell
conda run --no-capture-output -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v60a_ema_self_distillation_drift_guard --datasets "acm,dblp,flickr,texas,squirrel,chameleon" --epochs 100 --device cuda --log-level WARNING
```

Rationale:

```text
V60A's new guard starts after epoch 80 and ramps over 20 epochs. An 80e run
cannot test whether the guard activates correctly. The first mixed-stress run is
therefore 100 epochs, not 80.
```

No V60A 260-epoch run is authorized by this preregistration.

## 12. First Mixed-Stress Gates

Required verdict artifact:

```text
V60A_FIRST_MIXED_STRESS_VERDICT.md
```

Pass requirements:

```text
status=ok on 6/6
red-line pass on 6/6
teacher_ready becomes true by epoch 80
guard_gamma_epoch_80 = 0.0
guard_gamma_epoch_100 = 1.0
teacher_active_ratio_epoch_80 >= 0.05 on at least 3/6 datasets
guard_loss finite after epoch 80
abs(embedding_posterior_gap) <= 0.04 on 6/6
ACM ACC >= 0.8888
DBLP ACC >= 0.6610
Squirrel ACC >= 0.2800
```

Comparison requirement:

```text
Compare V60A 100e with V59A 80e and V59A 260e where relevant. Do not claim
full-run rescue from a 100e result.
```

## 13. Later Expansion Boundary

If the 100e mixed-stress passes, the next artifact must be:

```text
V60A_EXPANSION_REVIEW.md
```

Only that review may authorize a supported 9-dataset / 260-epoch run.

## 14. Stop Conditions

Stop immediately and write a failure analysis if:

```text
teacher snapshot uses labels or metrics
teacher is used as final labels
v60a_guard_loss is nonzero before teacher is ready
v60a_teacher_active_ratio is zero on more than 3/6 datasets at epoch 80
Squirrel ACC < 0.2800 in first mixed-stress
ACM ACC < 0.8888 in first mixed-stress
embedding_posterior_gap exceeds 0.08 on any dataset
```

## 15. No-Fabrication Status

This is a preregistration only. It contains no V60A results.
