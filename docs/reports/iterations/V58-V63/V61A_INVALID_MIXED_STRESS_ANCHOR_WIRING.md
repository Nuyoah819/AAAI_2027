# V61A Invalid Mixed-Stress Run: Anchor Wiring

This note invalidates the first local 6-dataset / 100-epoch run of:

```text
v61a_quantile_coverage_self_distillation_guard
```

The run completed with `status=ok` on 6/6 datasets, but it is not a valid V61A
mixed-stress result.

## Reason

The V61A implementation initially added the V61A loss and guard paths but did
not include `v61a_enabled` in the shared spectral-anchor initialization gate.

Observed invalid-run diagnostics:

```text
v61a_reliability_mean = 0.0
v61a_mass_scale = 0.0
v61a_effective_anchor_mass = 0.0
v61a_weighted_q_anchor_agreement_epoch_80 = 0.0
v61a_weighted_q_anchor_agreement_epoch_100 = 0.0
```

This means the inherited V59A anchor/release side was not actually active,
violating `V61A_IMPLEMENTATION_REVIEW.md`.

## Fix

The anchor initialization gate now includes:

```text
v61a_enabled
```

After the fix, the authorized ACM 1-epoch connectivity rerun reports:

```text
v61a_anchor_loss = 0.1644
v61a_reliability_mean = 0.1962
v61a_mass_scale = 1.0
v61a_effective_anchor_mass = 0.1962
```

## Boundary

The invalid run must not be used for V61A performance claims, pass/fail claims,
or mechanism claims.

The next valid step is to rerun the preregistered 6-dataset / 100-epoch
mixed-stress test after the fixed connectivity PASS.
