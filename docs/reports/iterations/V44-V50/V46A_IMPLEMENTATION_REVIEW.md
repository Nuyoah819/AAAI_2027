# V46A Implementation-Readiness Review

Target mechanism: `v46a_topology_band_calibration`

This is a pre-implementation review. It changes no training code, runs no
experiment, and reports no new result.

## 1. Mode

`design / implementation-readiness review`

The goal is to check whether the V46A preregistration can be implemented
without violating the project red lines or accidentally changing the mechanism
through inherited losses.

## 2. Active Evidence

Relevant inputs:

- `V45A_FIRST_SMOKE_VERDICT.md`
- `V46A_ROUTE_DECISION.md`
- `V46A_PREREGISTRATION.md`
- `CRITICAL_RED_LINES.md`
- current `core/e2e/sect_coco_e2e.py`
- current `scripts/run_unified_aptc_9datasets.py`

No V46A run exists yet.

## 3. Main Findings

V46A is implementable, but two preregistration details required tightening
before code changes.

### 3.1 Band-Mass Definition Must Reuse `hard.mean()`

The current `DifferentiableTopologyContraction` returns normalized soft masks:

```text
homo + hetero + hard = 1
```

and existing V44/V45 diagnostics define ambiguous band mass as:

```text
band_mass = mean(hard)
```

Therefore V46A must not use:

```text
band_ij = 1 - max(homo_ij, hetero_ij, hard_ij)
```

That would measure "not the largest topology mask confidence" rather than the
existing ambiguous-band mask. It would make V46A band gates incomparable with
V44/V45 and with the preregistered ACM/DBLP/Flickr ceilings.

Implementation-ready definition:

```text
band_ij = hard_ij
band_mass = mean(hard_ij)
L_band_cal = mean(hard_ij^2)
```

### 3.2 Inherited Generic Band/Frequency Losses Must Be Disabled

The runner's `v28b` base currently includes:

```text
partition_spread_weight = 0.05
freq_separation_weight = 0.10
freq_ortho_weight = 0.20
```

If V46A simply inherits these, the first smoke would mix the new V46A
calibration loss with older band/frequency pressure. That would blur attribution
and partially contradict the post-V45A decision to stop frequency-response
pressure.

The V46A variant must explicitly set:

```text
partition_spread_weight = 0.0
freq_separation_weight = 0.0
freq_ortho_weight = 0.0
```

This does not remove diagnostics; it only prevents inherited pressure losses
from contributing to the first V46A objective.

## 4. Red-Line Check

V46A remains red-line admissible if implemented as follows:

- No dataset name is read by the loss.
- No dataset-specific threshold, band target, head, assigner, or branch is
  introduced.
- Frequency-response diagnostics remain diagnostic only.
- V43B/V44/V44B/V45A failed losses remain disabled.
- The final posterior/readout path is unchanged.
- The loss is a uniform topology calibration applied to every edge graph.

## 5. Minimal Implementation Shape

Add config fields:

```text
v46a_band_cal_weight
v46a_balance_weight
v46a_spread_weight
v46a_entropy_floor
v46a_min_threshold_gap
v46a_corr_eps
```

Add one helper:

```text
v46a_topology_band_calibration_regularizer(
    homo, hetero, hard, low_threshold, high_threshold
)
```

Expected losses:

```text
band_cal_loss = mean(hard^2)
usage_entropy = entropy(normalize([mean(homo), mean(hetero), mean(hard)]))
balance_loss = ReLU(entropy_floor - usage_entropy)^2
spread_loss = ReLU(min_threshold_gap - (high_threshold - low_threshold))^2
```

Expected total-loss addition:

```text
total += v46a_band_cal_weight * band_cal_loss
total += v46a_balance_weight * balance_loss
total += v46a_spread_weight * spread_loss
```

Expected diagnostics:

```text
v46a_enabled
v46a_band_cal_loss
v46a_balance_loss
v46a_spread_loss
v46a_band_mass
v46a_homo_usage
v46a_hetero_usage
v46a_hard_usage
v46a_usage_entropy
v46a_threshold_gap
v46a_low_threshold
v46a_high_threshold
```

## 6. Runner Variant

Add exactly one variant:

```text
v46a_topology_band_calibration
```

It should inherit the stable base but explicitly disable the failed/inherited
pressure losses:

```text
v43b_* = 0.0
ideal_* = 0.0
v44_* = 0.0
v44b_pre_hp_corr_weight = 0.0
v45a_edge_freq_weight = 0.0
v45a_band_guard_weight = 0.0
partition_spread_weight = 0.0
freq_separation_weight = 0.0
freq_ortho_weight = 0.0
```

## 7. Sanity Before Smoke

After implementation, run only:

```powershell
conda run -n aaai-e2e-subspace python -m py_compile core\e2e\sect_coco_e2e.py scripts\run_unified_aptc_9datasets.py
```

Then, if needed, a 1-epoch CPU connectivity check:

```powershell
conda run -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v46a_topology_band_calibration --datasets acm --epochs 1 --device cpu --log-level WARNING
```

The connectivity check is not a gate result.

## 8. Final Decision

V46A is implementation-ready after the preregistration correction that defines:

```text
band_ij = hard_ij
```

and after the runner variant explicitly disables inherited band/frequency
pressure terms. No formal smoke should run before code-level sanity passes.

## 9. No-Fabrication Status

This review reports no V46A result. All V46A outputs remain `TBD`.
