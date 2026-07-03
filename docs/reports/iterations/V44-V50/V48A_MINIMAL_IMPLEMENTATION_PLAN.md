# V48A Minimal Implementation Plan

This plan translates `V48A_PREREGISTRATION.md` into a scoped diagnostic
implementation. It does not report V48A results and does not authorize sweeps.

## 1. Scope

Implement only:

```text
v48a_topology_dynamics_audit
```

Do not implement:

- new topology loss
- reparameterized topology contraction
- gradient hooks in the first audit
- selector or post-processing logic
- dataset-specific branch
- performance-claim run

## 2. Config Additions

Add fields to `E2ESECTCoCoConfig`:

```python
v48a_enabled: bool = False
v48a_snapshot_sample_size: int = 20000
v48a_movement_eps: float = 1e-8
```

Default `v48a_enabled=False` keeps existing variants unchanged.

## 3. Module Buffers

Add empty/non-active buffers to `EndToEndSECTCoCoModule`:

```text
v48a_snapshot_ready
v48a_prev_homo
v48a_prev_hetero
v48a_prev_hard
v48a_prev_score
v48a_prev_low_threshold
v48a_prev_high_threshold
```

The buffers should be initialized empty and populated only when V48A is enabled
during training.

## 4. Diagnostic Helper

Add a helper method:

```python
def _v48a_topology_dynamics_audit(self, out):
    ...
```

It should:

- select deterministic edge prefix of size `min(E, v48a_snapshot_sample_size)`
- compare current sampled masks/score/thresholds to previous snapshots
- compute V47A-style posterior targets as diagnostics only
- compute directional deltas on sampled target groups
- update previous snapshots after computing current diagnostics

Required diagnostics:

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

Useful additional diagnostics:

```text
v48a_sample_size
v48a_raw_homo_target_mass
v48a_raw_hetero_target_mass
v48a_raw_defer_target_mass
```

## 5. Loss Integration

Call the helper in `EndToEndSECTCoCoModule.loss` after V47A stats are computed.

Do not add V48A terms to `total`.

Add V48A stats to the diagnostics dict.

## 6. Variant Registration

Add to `scripts/run_unified_aptc_9datasets.py`:

```python
EXPERIMENT_VARIANTS["v48a_topology_dynamics_audit"] = {
    "output_stem": "unified_aptc_9datasets_v48a_topology_dynamics_audit",
    "overrides": {
        **EXPERIMENT_VARIANTS["v28b"]["overrides"],
        "aptc_local_teacher": False,
        "v43b_conflict_margin_weight": 0.0,
        "v43b_band_conflict_weight": 0.0,
        "v43b_highpass_energy_weight": 0.0,
        "ideal_signed_embedding_weight": 0.0,
        "ideal_band_resolution_weight": 0.0,
        "ideal_highpass_energy_weight": 0.0,
        "v44_topology_band_resolution_weight": 0.0,
        "v44_conflict_highpass_corr_weight": 0.0,
        "v44b_pre_hp_corr_weight": 0.0,
        "v45a_edge_freq_weight": 0.0,
        "v45a_band_guard_weight": 0.0,
        "v46a_band_cal_weight": 0.0,
        "v46a_balance_weight": 0.0,
        "v46a_spread_weight": 0.0,
        "v47a_resolution_weight": 0.0,
        "v47a_usage_guard_weight": 0.0,
        "partition_spread_weight": 0.0,
        "freq_separation_weight": 0.0,
        "freq_ortho_weight": 0.0,
        "v48a_enabled": True,
        "v48a_snapshot_sample_size": 20000,
        "v48a_movement_eps": 1e-8,
    },
}
```

## 7. Sanity Checks Before Audit

Run only after implementation:

```powershell
conda run -n aaai-e2e-subspace python -m py_compile core\e2e\sect_coco_e2e.py scripts\run_unified_aptc_9datasets.py
```

Then a connectivity check only:

```powershell
conda run -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v48a_topology_dynamics_audit --datasets acm --epochs 1 --device cpu --log-level WARNING
```

The 1-epoch run is not an audit result.

## 8. First-Stage Audit Command

Only if sanity checks pass:

```powershell
conda run -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v48a_topology_dynamics_audit --datasets "acm,dblp,flickr" --epochs 80 --device cuda --log-level WARNING
```

Expected outputs:

```text
results/archive/v40-v50/unified_aptc_9datasets_v48a_topology_dynamics_audit.csv
results/archive/v40-v50/unified_aptc_9datasets_v48a_topology_dynamics_audit_diagnostics.jsonl
```

## 9. Stop Rule

After the first-stage audit, stop and write:

```text
V48A_FIRST_AUDIT_VERDICT.md
```

Do not proceed to second-batch, full run, or sweeps.
