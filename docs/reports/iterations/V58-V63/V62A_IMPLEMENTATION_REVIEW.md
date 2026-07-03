# V62A Implementation Review

This review follows `V62A_PREREGISTRATION.md` and must be completed before any
V62A code change or experiment.

## 1. Boundary

Proposed variant:

```text
v62a_drift_responsive_self_distillation_guard
```

Implementation decision:

```text
APPROVE MINIMAL IMPLEMENTATION.
```

V62A is allowed to add only a bounded, detached drift-responsive multiplier to
the existing V61A self-distillation guard:

```text
effective_guard_multiplier =
    min(2.0, 1.0 + drift_boost * drift_gamma)
```

It must not change:

```text
V59A anchor/release
epoch-80 teacher snapshot
V61A floor-or-topk active teacher mask
teacher-to-student KL direction
base guard weight
final q_refined labels
```

## 2. Current Code State

Current code contains:

```text
V61A config fields
V61A teacher buffers
v61a_post80_anchor_release_residual_spectral_anchor_loss
v61a_quantile_coverage_self_distillation_guard_loss
V61A diagnostics and snapshot keys
V61A runner variant
```

No `v62a_*` symbols exist in model code or runner before this implementation.

## 3. Required Insertion Points

### Config

Add to `E2ESECTCoCoConfig`:

```text
v62a_enabled: bool = False
v62a_anchor_weight: float = 0.0
v62a_guard_weight: float = 0.0
v62a_absolute_floor: float = 0.45
v62a_min_teacher_coverage: float = 0.10
v62a_start_epoch: int = 80
v62a_guard_ramp_epochs: int = 20
v62a_max_gamma: float = 1.0
v62a_drift_start_epoch: int = 100
v62a_drift_floor: float = 0.02
v62a_drift_scale: float = 0.06
v62a_drift_boost: float = 1.0
v62a_max_effective_guard_multiplier: float = 2.0
```

Also add inherited V59A/V61A anchor constants under `v62a_*` names so the
variant can run independently while V61A active loss is disabled:

```text
v62a_reliability_floor: float = 0.10
v62a_reliable_threshold: float = 0.20
v62a_min_effective_mass: float = 0.10
v62a_warmup_epochs: int = 20
v62a_ramp_epochs: int = 40
v62a_beta_min: float = 0.35
v62a_beta_max: float = 0.70
v62a_soft_power: float = 0.50
v62a_hybrid_compensation: float = 0.50
v62a_target_mass: float = 0.08
v62a_max_mass_scale: float = 1.50
v62a_max_reliability_cap: float = 0.90
v62a_release_start_epoch: int = 80
v62a_release_decay_epochs: int = 60
v62a_release_floor: float = 0.25
```

### Buffers

Add independent buffers:

```text
v62a_teacher_q
v62a_teacher_ready
v62a_teacher_epoch
```

Do not reuse the mutable `init_teacher` or V60A/V61A teacher buffers.

### Anchor Loss

Add:

```text
v62a_post80_anchor_release_residual_spectral_anchor_loss
```

It should wrap the V61A/V59A anchor helper and remap diagnostics to `v62a_*`.
No anchor or release formula may change.

### Guard Loss

Add:

```text
v62a_drift_responsive_self_distillation_guard_loss
```

It should reuse the V61A floor-or-topk active mask, compute active KL, then
apply a detached bounded drift multiplier after epoch 100.

Required behavior:

```text
if teacher is not ready:
    guard_gamma = 0
    drift_gamma = 0
    effective_guard_multiplier = 1
    guard_loss = 0
else:
    guard_kl = mean_active KL(teacher_q.detach() || q_refined)
    drift_score = guard_kl.detach()
    drift_gamma = 0 if epoch <= 100 else clamp((drift_score - 0.02) / 0.06, 0, 1)
    effective_guard_multiplier = min(2.0, 1.0 + 1.0 * drift_gamma)
    guard_loss = guard_gamma * effective_guard_multiplier.detach() * guard_kl
```

The total loss must multiply this helper output only by `v62a_guard_weight`.

### Teacher Snapshot

Add a V62A training-loop hook matching V61A:

```text
if v62a_enabled and teacher not ready and epoch_number >= 80:
    forward no-grad after optimizer step
    store q_refined.detach().clone()
```

Update the epoch-80 diagnostics after storing so `v62a_teacher_ready_epoch_80`
and active ratios are auditable.

### Runner

Add disabled defaults and:

```text
EXPERIMENT_VARIANTS["v62a_drift_responsive_self_distillation_guard"]
```

The variant must explicitly disable V50A-V61A active losses and enable only:

```text
v62a_enabled=true
v62a_anchor_weight=0.04
v62a_guard_weight=0.02
```

## 4. Required Diagnostics

Add the preregistered diagnostics:

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

Also expose inherited anchor diagnostics under `v62a_*`.

Snapshot keys must include epoch 1/80/100 values and allow final epoch values
to appear in diagnostics for any later expansion verdict.

## 5. Red-Line Review

| Requirement | Review |
| --- | --- |
| Teacher source is label-free | PASS if snapshot uses only `q_refined.detach()`. |
| Mask is V61A floor-or-topk | PASS if floor 0.45 and coverage 0.10 remain fixed. |
| Drift score is label-free | PASS if computed only from active KL. |
| Drift multiplier detached | PASS if `effective_guard_multiplier` is detached before multiplying KL. |
| Multiplier bounded | PASS if never exceeds 2.0. |
| V59A anchor/release unchanged | PASS if helper only wraps existing V61A/V59A anchor logic. |
| V50A-V61A inactive in runner | PASS only if all earlier enabled flags/weights are false/zero. |
| Final labels remain q_refined | PASS if prediction path is unchanged. |
| No V62A 260e authorization | PASS; implementation authorizes only connectivity after compile. |

## 6. Authorized Next Action

After this review, the only authorized action is minimal implementation followed
by static verification:

```powershell
conda run -n aaai-e2e-subspace python -m py_compile core\e2e\sect_coco_e2e.py scripts\run_unified_aptc_9datasets.py
```

If compile passes and no residual training process exists, only the following
connectivity run is authorized:

```powershell
conda run --no-capture-output -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v62a_drift_responsive_self_distillation_guard --datasets "acm" --epochs 1 --device cuda --log-level WARNING
```

No V62A mixed-stress run is authorized until `V62A_CONNECTIVITY_VERDICT.md` is
written.

## 7. No-Fabrication Status

This review contains no V62A results. It only authorizes a minimal
implementation consistent with `V62A_PREREGISTRATION.md`.
