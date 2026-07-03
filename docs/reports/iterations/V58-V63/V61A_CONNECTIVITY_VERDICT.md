# V61A Connectivity Verdict

Variant:

```text
v61a_quantile_coverage_self_distillation_guard
```

Command:

```powershell
conda run --no-capture-output -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v61a_quantile_coverage_self_distillation_guard --datasets "acm" --epochs 1 --device cuda --log-level WARNING
```

Verdict: PASS.

## Gate Check

| Gate | Observed | Status |
| --- | --- | --- |
| `status=ok` | `ok` | PASS |
| `legacy_head_used=false` | `False` | PASS |
| V50A-V60A active losses disabled | `v50a_enabled` through `v60a_enabled` are all `False` | PASS |
| `v61a_enabled=true` | `True` | PASS |
| `v61a_teacher_ready=false` | `False` | PASS |
| `v61a_guard_gamma=0.0` | `0.0` | PASS |
| `v61a_guard_loss=0.0` | `0.0` | PASS |
| inherited V59A anchor is active | `v61a_anchor_loss=0.1644`; `v61a_effective_anchor_mass=0.1962` | PASS |
| final labels remain `q_refined` | `legacy_head_used=False`; no teacher final-label path exists | PASS |

## Diagnostic Snapshot

```text
v61a_guard_enabled = True
v61a_release_gamma = 1.0
v61a_anchor_loss = 0.1644
v61a_reliability_mean = 0.1962
v61a_mass_scale = 1.0
v61a_effective_anchor_mass = 0.1962
v61a_teacher_epoch = -1.0
v61a_guard_weight = 0.02
v61a_absolute_floor = 0.45
v61a_min_teacher_coverage = 0.10
v61a_teacher_active_ratio = 0.0
v61a_teacher_floor_active_ratio = 0.0
v61a_teacher_topk_active_ratio = 0.0
v61a_release_gamma_epoch_1 = 1.0
v61a_guard_gamma_epoch_1 = 0.0
v61a_teacher_ready_epoch_1 = 0.0
v61a_guard_loss_epoch_1 = 0.0
```

## Boundary

Connectivity only verifies wiring. It does not support a performance claim.

The next authorized step is the preregistered 6-dataset / 100-epoch
mixed-stress run.
