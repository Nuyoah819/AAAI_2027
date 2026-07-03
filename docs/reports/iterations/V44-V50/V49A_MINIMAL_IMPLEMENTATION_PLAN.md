# V49A Minimal Implementation Plan

Target:

```text
v49a_reparameterized_topology_transition
```

This plan implements only the preregistered topology transition
reparameterization and diagnostics. It does not authorize a first-stage smoke.

## 1. Code Changes

### 1.1 Config

Add:

```text
v49a_enabled: bool = False
v49a_tau_clear: float = 1.0
v49a_tau_orient: float = 1.0
v49a_snapshot_sample_size: int = 20000
v49a_movement_eps: float = 1e-8
```

### 1.2 Module Buffers

Add deterministic movement snapshot buffers:

```text
v49a_snapshot_ready
v49a_prev_homo
v49a_prev_hetero
v49a_prev_hard
v49a_prev_score
```

No dataset-specific buffer or branch is allowed.

### 1.3 Mask Mapping

Add helper:

```text
_v49a_reparameterized_topology(score, edge_logit)
```

Mapping:

```text
orient = sigmoid(edge_logit / tau_orient)
clear = sigmoid(abs(edge_logit) / tau_clear)
homo = clear * orient
hetero = clear * (1 - orient)
hard = 1 - clear
```

Return `clear` and `orient` for diagnostics.

### 1.4 Frontend Branch

In `_frontend_pass`:

```text
score, alpha, edge_logit = confidence(...)
old_homo, old_hetero, old_hard, low, high = contraction(score)
if v49a_enabled:
    homo, hetero, hard, clear, orient = v49a mapping
else:
    homo, hetero, hard = old masks
```

Keep `low` and `high` for compatibility diagnostics only.

### 1.5 Diagnostics Helper

Add:

```text
_v49a_topology_transition_diagnostics(out)
```

It should compute usage, entropy, clear/orient stats, V47A-style target masses,
and epoch-to-epoch target deltas. It should update snapshots only when:

```text
v49a_enabled and self.training
```

### 1.6 Loss Boundary

Do not add any V49A term to `total`.

### 1.7 Runner Variant

Add:

```text
v49a_reparameterized_topology_transition
```

Use `v28b` as the base and explicitly disable failed or confounding losses,
including:

```text
threshold_reg_weight = 0.0
edge_quantile_anchor_weight = 0.0
v43b/v44/v44b/v45a/v46a/v47a weights = 0.0
partition_spread_weight = 0.0
freq_separation_weight = 0.0
freq_ortho_weight = 0.0
```

Set:

```text
v49a_enabled = true
v49a_tau_clear = 1.0
v49a_tau_orient = 1.0
v49a_snapshot_sample_size = 20000
v49a_movement_eps = 1e-8
```

## 2. Verification

Run only:

```powershell
conda run -n aaai-e2e-subspace python -m py_compile core\e2e\sect_coco_e2e.py scripts\run_unified_aptc_9datasets.py
```

Then:

```powershell
conda run -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v49a_reparameterized_topology_transition --datasets acm --epochs 1 --device cpu --log-level WARNING
```

The 1-epoch run is connectivity only.

## 3. Expected Connectivity Checks

The 1-epoch ACM record should show:

```text
legacy_head_used=false
v43b/v44/v44b/v45a/v46a/v47a_enabled=false
v49a_enabled=true
v49a_has_prev_snapshot=false
v49a_homo_usage > 0
v49a_hetero_usage > 0
v49a_hard_usage > 0
v49a_clear_std finite
v49a_orient_std finite
```

`v49a_has_prev_snapshot=false` is expected for a single epoch.

## 4. Stop Point

After connectivity, write:

```text
V49A_IMPLEMENTATION_NOTES.md
```

Do not run the 80-epoch first-stage smoke in the same step unless explicitly
requested after reviewing implementation notes.
