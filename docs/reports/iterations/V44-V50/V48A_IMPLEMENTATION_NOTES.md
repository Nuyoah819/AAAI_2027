# V48A Implementation Notes

Target variant:

```text
v48a_topology_dynamics_audit
```

This file records implementation details and code-level sanity checks only.
Audit results must be recorded separately after a preregistered 80-epoch audit.

## Files Changed

```text
core/e2e/sect_coco_e2e.py
scripts/run_unified_aptc_9datasets.py
```

## Core Implementation

Added config fields:

```text
v48a_enabled
v48a_snapshot_sample_size
v48a_movement_eps
```

Added module snapshot buffers:

```text
v48a_snapshot_ready
v48a_prev_homo
v48a_prev_hetero
v48a_prev_hard
v48a_prev_score
v48a_prev_low_threshold
v48a_prev_high_threshold
```

Added helper:

```text
_v48a_topology_dynamics_audit
```

The helper stores deterministic edge-prefix snapshots during training only:

```text
sample_idx = arange(min(num_edges, v48a_snapshot_sample_size))
```

It compares current `homo`, `hetero`, `hard`, `score`, and thresholds against
the previous training epoch snapshot.

## Losses

V48A adds no loss term.

The implementation does not add any V48A term to `total`.

V47A losses are also disabled in the V48A runner variant:

```text
v47a_resolution_weight = 0.0
v47a_usage_guard_weight = 0.0
```

## Diagnostics

Added preregistered diagnostics:

```text
v48a_enabled
v48a_has_prev_snapshot
v48a_mean_abs_delta_homo
v48a_mean_abs_delta_hetero
v48a_mean_abs_delta_hard
v48a_mean_abs_delta_score
v48a_hard_mass_delta
v48a_threshold_delta
v48a_hard_rank_corr_prev
v48a_homo_target_mass
v48a_hetero_target_mass
v48a_defer_target_mass
v48a_targeted_homo_delta
v48a_targeted_hetero_delta
v48a_targeted_hard_delta
```

Added implementation-safety diagnostics:

```text
v48a_sample_size
v48a_raw_homo_target_mass
v48a_raw_hetero_target_mass
v48a_raw_defer_target_mass
```

V47A-style posterior targets are diagnostic only. They use detached
`q_refined` and do not create an optimization objective.

## Runner Variant

Added one variant:

```text
v48a_topology_dynamics_audit
```

Failed or confounding losses are explicitly disabled:

```text
v43b_* = 0.0
ideal_* = 0.0
v44_* = 0.0
v44b_pre_hp_corr_weight = 0.0
v45a_* = 0.0
v46a_* = 0.0
v47a_* = 0.0
partition_spread_weight = 0.0
freq_separation_weight = 0.0
freq_ortho_weight = 0.0
```

First implementation constants:

```text
v48a_enabled = true
v48a_snapshot_sample_size = 20000
v48a_movement_eps = 1e-8
```

## Verification

Code-level check passed:

```powershell
conda run -n aaai-e2e-subspace python -m py_compile core\e2e\sect_coco_e2e.py scripts\run_unified_aptc_9datasets.py
```

Connectivity check passed:

```powershell
conda run -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v48a_topology_dynamics_audit --datasets acm --epochs 1 --device cpu --log-level WARNING
```

The 1-epoch check is not an audit result. It remains in the V48A result files
as an implementation connectivity record.

Sanity snapshot from the 1-epoch ACM connectivity check:

```text
legacy_head_used=false
v43b/v44/v44b/v45a/v46a/v47a_enabled=false
v48a_enabled=true
v48a_has_prev_snapshot=false
v48a_sample_size=20000
v48a_homo_target_mass=0.2893
v48a_hetero_target_mass=0.2478
v48a_defer_target_mass=0.3050
```

`v48a_has_prev_snapshot=false` and zero movement values are expected for the
single-epoch connectivity run because no previous epoch exists.

No formal V48A first-stage audit has been run yet.
