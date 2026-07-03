# V60A Implementation Review

This review follows `V60A_PREREGISTRATION.md` and must be completed before any
V60A code change or experiment.

## 1. Boundary

Proposed variant:

```text
v60a_ema_self_distillation_drift_guard
```

Implementation decision:

```text
APPROVE MINIMAL IMPLEMENTATION.
```

V60A is allowed to add only one post-80, label-free, detached self-distillation
drift guard on top of V59A:

```text
Store detached q_refined at epoch 80 as teacher_q.
After epoch 80, penalize KL(teacher_q || current q_refined) on confident
teacher nodes.
```

Despite the historical variant name containing `ema`, the first V60A version is
not allowed to update the teacher by EMA. The teacher must be frozen after the
epoch-80 snapshot.

## 2. Current Code State

Current code contains V59A:

```text
V59A config fields
post80_anchor_release_residual_spectral_anchor_loss
V59A diagnostics and snapshots
V59A runner variant
```

No `v60a_*` symbols exist in the model or runner before this implementation.

The code already has unrelated earlier teacher mechanisms such as
`init_teacher` and local teacher losses. V60A must not reuse those mutable
teachers because they have different semantics. It needs an independent
`v60a_teacher_q` buffer and ready flag.

## 3. Required Insertion Points

### Config

Add to `E2ESECTCoCoConfig`:

```text
v60a_enabled: bool = False
v60a_anchor_weight: float = 0.0
v60a_guard_weight: float = 0.0
v60a_confidence_threshold: float = 0.60
v60a_start_epoch: int = 80
v60a_guard_ramp_epochs: int = 20
v60a_max_gamma: float = 1.0
```

Also add inherited V59A constants under `v60a_*` names so V60A can run as an
independent variant while disabling V59A active loss:

```text
v60a_reliability_floor: float = 0.10
v60a_reliable_threshold: float = 0.20
v60a_min_effective_mass: float = 0.10
v60a_warmup_epochs: int = 20
v60a_ramp_epochs: int = 40
v60a_beta_min: float = 0.35
v60a_beta_max: float = 0.70
v60a_soft_power: float = 0.50
v60a_hybrid_compensation: float = 0.50
v60a_target_mass: float = 0.08
v60a_max_mass_scale: float = 1.50
v60a_max_reliability_cap: float = 0.90
v60a_release_start_epoch: int = 80
v60a_release_decay_epochs: int = 60
v60a_release_floor: float = 0.25
```

### Buffers

Add independent buffers to `EndToEndSECTCoCoModule.__init__`:

```text
self.register_buffer("v60a_teacher_q", torch.empty(0))
self.register_buffer("v60a_teacher_ready", torch.tensor(False, dtype=torch.bool))
self.register_buffer("v60a_teacher_epoch", torch.tensor(-1, dtype=torch.long))
```

These buffers must not be used as final labels.

### Teacher Snapshot

Add a training-loop hook after the epoch's optimizer update and diagnostics:

```text
if v60a_enabled and not ready and epoch_number >= v60a_start_epoch:
    model.v60a_teacher_q = out["q_refined"].detach().clone()
    model.v60a_teacher_ready = true
    model.v60a_teacher_epoch = epoch_number
```

Use the existing zero-based loop convention:

```text
epoch_number = epoch + 1
```

This makes the snapshot occur at the end of observable epoch 80. It uses only
the model's own `q_refined`, no labels, validation metrics, test metrics, or
dataset name.

### Loss Call

Insert a V60A anchor-loss call immediately after V59A:

```text
v60a_anchor_loss, v60a_anchor_stats =
    v60a_post80_anchor_release_residual_spectral_anchor_loss(...)
```

This helper may wrap `post80_anchor_release_residual_spectral_anchor_loss` or
directly wrap the V57A helper with the same V59A post-80 release schedule. It
must report diagnostics as `v60a_*` and keep V59A internals unchanged.

Add a second helper for the guard:

```text
v60a_guard_loss, v60a_guard_stats =
    v60a_self_distillation_guard_loss(q_refined, teacher_q, ...)
```

### Total Loss

Add only:

```text
+ cfg.v60a_anchor_weight * v60a_anchor_loss
+ cfg.v60a_guard_weight * v60a_guard_loss
```

The guard helper should already include `guard_gamma`, so the total loss should
not multiply by gamma a second time.

### Runner

Add disabled defaults and a new variant:

```text
EXPERIMENT_VARIANTS["v60a_ema_self_distillation_drift_guard"]
```

The V60A variant must explicitly disable V50A-V59A active losses:

```text
v50a_enabled=false
...
v59a_enabled=false
all corresponding anchor weights = 0.0
v60a_enabled=true
v60a_anchor_weight=0.04
v60a_guard_weight=0.02
```

## 4. Guard Loss Definition

Before the teacher is ready:

```text
guard_gamma = 0
guard_loss = 0
```

After teacher is ready:

```text
teacher_confidence = max_k teacher_q[k]
active = teacher_confidence >= 0.60
per_node_kl = KL(teacher_q.detach() || q_refined)
guard_kl = mean(per_node_kl[active])
guard_gamma = min(1.0, max(0, (epoch_number - 80) / 20))
guard_loss = guard_gamma * guard_kl
```

If no active nodes exist:

```text
guard_loss = 0
```

All teacher tensors must be detached. The loss must use current `q_refined` as
student and must not replace final labels.

## 5. Diagnostics

Add diagnostics required by the preregistration:

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

Also expose inherited anchor diagnostics under `v60a_*` names, including:

```text
v60a_release_gamma
v60a_anchor_loss
v60a_pre_release_anchor_loss
v60a_weighted_q_anchor_kl
v60a_pre_release_weighted_q_anchor_kl
v60a_reliability_mean
v60a_mass_scale
v60a_effective_anchor_mass
```

Snapshot list must include:

```text
v60a_guard_gamma
v60a_teacher_ready
v60a_teacher_active_ratio
v60a_q_teacher_agreement
v60a_guard_loss
v60a_release_gamma
v60a_reliability_mean
v60a_mass_scale
```

This is needed to produce:

```text
v60a_guard_gamma_epoch_1
v60a_guard_gamma_epoch_80
v60a_guard_gamma_epoch_100
v60a_teacher_ready_epoch_80
v60a_teacher_active_ratio_epoch_80
v60a_q_teacher_agreement_epoch_100
```

## 6. Anchor Construction And Final Diagnostics

Extend spectral anchor construction with:

```text
or bool(getattr(cfg, "v60a_enabled", False))
```

Do not change `build_spectral_compactness_anchor`.

Extend final anchor metrics with:

```text
v60a_anchor_acc_diagnostic
v60a_anchor_nmi_diagnostic
v60a_anchor_ari_diagnostic
```

## 7. Red-Line Review

| Requirement | Review |
| --- | --- |
| Teacher source is label-free | PASS if snapshot uses only `out["q_refined"].detach()`. |
| Teacher is frozen | PASS if no EMA/update occurs after first ready snapshot. |
| Guard loss zero before ready | PASS if helper returns zero when ready flag is false. |
| V59A internals unchanged | PASS if V60A wraps existing V59A/V57A helpers without edits. |
| V50A-V59A inactive in runner | PASS only if all earlier enabled flags/weights are false/zero. |
| Final labels remain q_refined | PASS if prediction path is unchanged. |
| No dataset-specific branch | PASS if teacher/mask/schedule use only epoch and tensor confidence. |
| No V60A 260e authorization | PASS; implementation authorizes only connectivity after compile. |

## 8. Authorized Next Action

After this review, the only authorized action is minimal implementation followed
by static verification:

```powershell
conda run -n aaai-e2e-subspace python -m py_compile core\e2e\sect_coco_e2e.py scripts\run_unified_aptc_9datasets.py
```

If compile passes and no residual training process exists, only the following
connectivity run is authorized:

```powershell
conda run --no-capture-output -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v60a_ema_self_distillation_drift_guard --datasets "acm" --epochs 1 --device cuda --log-level WARNING
```

No V60A mixed-stress run is authorized until `V60A_CONNECTIVITY_VERDICT.md` is
written.

## 9. No-Fabrication Status

This review contains no V60A results. It only authorizes a minimal
implementation consistent with `V60A_PREREGISTRATION.md`.
