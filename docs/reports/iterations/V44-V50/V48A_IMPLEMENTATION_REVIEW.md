# V48A Implementation-Readiness Review

This file reviews whether `v48a_topology_dynamics_audit` is ready for minimal
diagnostic implementation. It is an implementation-readiness note only. No V48A
code has been implemented and no V48A experiment has been run.

## 1. Reviewed Inputs

Documents reviewed:

- `V48A_ROUTE_DECISION.md`
- `V48A_PREREGISTRATION.md`
- `V47A_FIRST_SMOKE_VERDICT.md`
- `V47A_IMPLEMENTATION_NOTES.md`

Code reviewed:

- `core/e2e/sect_coco_e2e.py`
- `scripts/run_unified_aptc_9datasets.py`

## 2. Current Status

V48A is implementation-ready as a diagnostic-only audit.

The main implementation choice is where to store epoch-to-epoch snapshots.
The safest local choice is inside `EndToEndSECTCoCoModule`, because each call to
`loss()` already has access to:

```text
out["homo"]
out["hetero"]
out["hard"]
out["score"]
out["low_threshold"]
out["high_threshold"]
out["q_refined"]
```

The estimator training loop already copies the final epoch `diag` into
`diagnostics_`, so adding V48A diagnostics to the module loss path is enough for
JSONL output.

## 3. Audit Scope

V48A must add diagnostics only. It must not add a new loss term.

Allowed:

- fixed deterministic edge snapshot
- epoch-to-epoch mask/score/threshold movement
- V47A-style target diagnostics with zero V47A loss
- directional movement by diagnostic target group

Disallowed:

- new optimization objective
- V47A loss reuse
- gradient pressure on topology masks
- dataset-specific sampling or thresholds
- performance-triggered expansion

## 4. Snapshot Design

Use a deterministic edge prefix:

```text
sample_size = min(num_edges, v48a_snapshot_sample_size)
sample_idx = arange(sample_size)
```

This follows the preregistered "sample deterministically from edge order, not
randomly" rule.

For each V48A-enabled training loss call, store detached snapshots of:

```text
homo[sample_idx]
hetero[sample_idx]
hard[sample_idx]
score[sample_idx]
low_threshold
high_threshold
```

The next epoch compares current values against the previous snapshot. Epoch 0
should report `v48a_has_prev_snapshot=false`; by the final epoch of an 80-epoch
run this should be true.

## 5. Directional Diagnostics

V48A can reuse the V47A target construction as diagnostic only:

```text
homo_target
hetero_target
defer_target
```

Targets should be computed from detached `q_refined`, exactly as V47A did, but
no V47A loss should be active. Directional deltas should use the previous
epoch's target group on the sampled edges:

```text
targeted_homo_delta   = mean((homo_now - homo_prev) over homo_target edges)
targeted_hetero_delta = mean((hetero_now - hetero_prev) over hetero_target edges)
targeted_hard_delta   = mean((hard_now - hard_prev) over defer_target edges)
```

If a target group is empty on the sample, report `0.0` for its delta and the
corresponding target mass will explain the absence.

## 6. Rank-Correlation Diagnostic

`v48a_hard_rank_corr_prev` can be computed as a Pearson correlation between the
previous and current sampled hard weights. This is sufficient for the first
audit because the goal is to detect persistence versus movement, not to claim a
statistical rank test.

## 7. Red-Line Compatibility

The V48A runner variant should reuse the V47A/V46A red-line shutdown pattern:

```text
aptc_local_teacher = False
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

V48A should expose:

```text
v48a_enabled = true
```

without making any loss contribution.

## 8. Required Code Touch Points

Minimal code changes should be limited to:

1. Add V48A config fields to `E2ESECTCoCoConfig`.
2. Add snapshot buffers to `EndToEndSECTCoCoModule`.
3. Add a diagnostic helper for V48A movement and target-direction stats.
4. Call the helper in `EndToEndSECTCoCoModule.loss`.
5. Add V48A diagnostics to the diagnostics dict.
6. Register `v48a_topology_dynamics_audit` in the runner.

No data loader, metric, post-processing, or legacy head code needs to change.

## 9. Implementation Risks

### Risk A: Snapshot State Updates During Non-Training Calls

Mitigation:

Only update V48A previous snapshots when `self.training` and
`v48a_enabled=true`.

### Risk B: Snapshot Memory

Mitigation:

Use the preregistered cap `v48a_snapshot_sample_size=20000`. This stores a small
set of one-dimensional edge tensors, not full model states.

### Risk C: Diagnostic Target Groups Empty On Sample

Mitigation:

Report target masses and use safe zero deltas when a group has no sampled
members.

### Risk D: Audit Misread As Performance Variant

Mitigation:

Set all V43B-V47A failed weights to zero and add no V48A loss term.

## 10. Readiness Verdict

Proceed to minimal implementation.

Do not run the first-stage 80-epoch audit until:

- py_compile passes
- 1-epoch CPU connectivity check passes
- `v48a_enabled=true` appears in diagnostics
- V43B-V47A enabled flags are false
- movement diagnostics are finite

V48A remains unimplemented and unrun at the time of this review.
