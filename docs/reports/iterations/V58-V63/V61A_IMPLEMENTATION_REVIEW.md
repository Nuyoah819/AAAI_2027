# V61A Implementation Review

This review follows `V61A_PREREGISTRATION.md`.

Decision: minimal V61A implementation is approved.

## 1. Teacher Storage

V61A will store its teacher in dedicated model buffers:

```text
v61a_teacher_q
v61a_teacher_ready
v61a_teacher_epoch
```

It must not reuse `v60a_teacher_q`, because V60A and V61A have different mask
diagnostics and verdict gates.

## 2. Snapshot Timing

The teacher snapshot is taken once at epoch 80, after the optimizer update for
that epoch. The snapshot source is a no-grad forward pass:

```text
snapshot_out["q_refined"].detach().clone()
```

This keeps the teacher frozen, detached, and tied to the post-update epoch-80
posterior rather than labels, validation metrics, or a stale pre-step output.

## 3. Active Mask

The V61A active mask is computed only from teacher confidence:

```text
teacher_confidence = max_k teacher_q[k]
floor_mask = teacher_confidence >= 0.45
topk_mask = top ceil(0.10 * N) teacher_confidence nodes
active = floor_mask OR topk_mask
```

The top-k calculation must use only the current dataset's teacher-confidence
vector. It must not use dataset names, labels, validation/test metrics, or
post-hoc performance.

## 4. Loss Boundary

Before the teacher is ready, or if the teacher tensor shape does not match the
student posterior, V61A guard loss must be exactly zero and report:

```text
v61a_teacher_ready = false
v61a_guard_gamma = 0.0
v61a_guard_loss = 0.0
```

After readiness, the KL direction remains:

```text
KL(teacher_q.detach() || q_refined)
```

The training term is:

```text
v61a_guard_weight * guard_gamma(epoch) * mean_active_KL
```

## 5. Inherited Anchor

V61A reuses the V59A anchor/release formula through the V60A wrapper shape, but
renames diagnostics to `v61a_*`. The inherited constants remain those listed in
the preregistration:

```text
anchor_weight = 0.04
release_start_epoch = 80
release_decay_epochs = 60
release_floor = 0.25
```

No V59A/V60A active loss should run alongside V61A.

## 6. Disabled Prior Variants

The runner variant must explicitly disable active losses from:

```text
V50A, V51A, V52A, V53A, V54A, V55A, V56A, V57A, V58A, V59A, V60A
```

Only `v61a_enabled=true` may activate the V61A anchor and guard path.

## 7. Final Labels

V61A must keep final labels as:

```text
q_refined.argmax(dim=1)
```

The teacher is never used as final labels or as a selector between teacher and
student posteriors.

## 8. Required First Verification

After the minimal implementation, run only the authorized connectivity command:

```powershell
conda run --no-capture-output -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v61a_quantile_coverage_self_distillation_guard --datasets "acm" --epochs 1 --device cuda --log-level WARNING
```

If that passes, the next allowed run is the preregistered 6-dataset / 100-epoch
mixed-stress test.
