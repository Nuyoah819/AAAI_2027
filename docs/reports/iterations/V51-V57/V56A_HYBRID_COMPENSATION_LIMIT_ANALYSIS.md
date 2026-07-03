# V56A Hybrid Compensation Limit Analysis

This document analyzes why `v56a_hybrid_consensus_floor_residual_anchor` stops
after the first mixed-stress gate. It follows `ccf-idea-optimizer` rescue mode:
diagnose the mechanism before any new implementation.

No new experiment is run in this document. No V56A beta-bound, soft-power,
hybrid-compensation, schedule, threshold, formula, or V50A anchor hyperparameter
sweep is authorized.

## 1. Evidence Basis

Local artifacts:

```text
V54A_FIRST_MIXED_STRESS_VERDICT.md
V55A_FIRST_MIXED_STRESS_VERDICT.md
V55A_SOFT_CONSENSUS_FAILURE_ANALYSIS.md
V56A_PREREGISTRATION.md
V56A_IMPLEMENTATION_REVIEW.md
V56A_CONNECTIVITY_VERDICT.md
V56A_FIRST_MIXED_STRESS_VERDICT.md
results/archive/v51-v57/unified_aptc_9datasets_v56a_hybrid_consensus_floor_residual_anchor_diagnostics.jsonl
```

V56A already stops by preregistered gates. This document extracts the next
rescue question.

## 2. What V56A Preserved

V56A preserved the bounded-residual safety profile:

| Gate | V56A Verdict |
| --- | --- |
| Red-line | PASS |
| Hybrid residual bound | PASS |
| Posterior/readout safety | PASS |
| Anchor usefulness | PASS |
| Heterophily stress | PASS |

It also preserved the key floors:

| Dataset | V56A ACC | Required Floor | Gate |
| --- | ---: | ---: | --- |
| ACM | 0.9005 | 0.8888 | PASS |
| DBLP | 0.6919 | 0.6610 | PASS |
| Squirrel | 0.3011 | 0.2800 | PASS |

Interpretation:

```text
Hard floor plus soft compensation does not reintroduce the V53A Squirrel
overexposure failure. It also gives the best DBLP ACC in the V54A-V56A chain.
```

## 3. What V56A Failed

The reliability non-collapse gate still fails:

| Dataset | Rel Mean | Effective Mass | Reliable Ratio | Gate |
| --- | ---: | ---: | ---: | --- |
| ACM | 0.1393 | 0.1393 | 0.1481 | PASS |
| DBLP | 0.0756 | 0.0756 | 0.0000 | FAIL mass |
| Flickr | 0.0056 | 0.0056 | 0.0000 | weak-anchor non-use |
| Texas | 0.0561 | 0.0561 | 0.0055 | FAIL mass |
| Squirrel | 0.0855 | 0.0855 | 0.0527 | PASS |
| Chameleon | 0.2171 | 0.2171 | 0.4954 | PASS |

The mass gate passes on only 3/6. DBLP is close to the 0.08 floor but still
fails; Texas remains well below it.

## 4. Mechanistic Cause

V56A computes:

```text
hybrid_i = h_i + 0.50 * relu(sqrt(soft_i) - h_i)
beta_i = 0.35 + 0.35 * hybrid_i
```

The compensation was intentionally conservative:

| Dataset | Hard Mean | Lifted Soft | Compensation Mean | Beta Mean |
| --- | ---: | ---: | ---: | ---: |
| ACM | 0.8967 | 0.2791 | 0.0039 | 0.6652 |
| DBLP | 0.4771 | 0.1344 | 0.0022 | 0.5178 |
| Flickr | 0.1008 | 0.0212 | 0.0070 | 0.3877 |
| Texas | 0.3798 | 0.1792 | 0.0198 | 0.4899 |
| Squirrel | 0.0982 | 0.0573 | 0.0054 | 0.3863 |
| Chameleon | 0.2589 | 0.2021 | 0.0318 | 0.4517 |

This shows the limit:

```text
The hybrid compensation is active, but it mostly preserves V54A behavior rather
than changing the mass allocation enough to pass DBLP/Texas.
```

Compared with V54A/V55A:

| Dataset | V54A Rel | V55A Rel | V56A Rel |
| --- | ---: | ---: | ---: |
| ACM | 0.1390 | 0.1004 | 0.1393 |
| DBLP | 0.0749 | 0.0589 | 0.0756 |
| Texas | 0.0562 | 0.0490 | 0.0561 |
| Squirrel | 0.0852 | 0.0820 | 0.0855 |
| Chameleon | 0.2123 | 0.2067 | 0.2171 |

V56A recovers V54A's reliability profile and slightly lifts a few datasets, but
the mass floor does not move enough.

## 5. Why Not Increase Compensation

Do not do:

- increase `hybrid_compensation`;
- sweep `hybrid_compensation`;
- raise `beta_min`;
- add dataset-specific compensation;
- relax the 0.08 reliability mass floor;
- choose V56A for DBLP and another variant elsewhere;
- change V50A anchor construction;
- add a geometry fallback.

Reason:

```text
The next failure is no longer about whether hard or soft consensus should
control beta. V56A already tested a conservative hybrid. The remaining gap is
that reliability mass is not allocated to enough nodes on DBLP/Texas under the
current weighted-KL normalization.
```

## 6. New Bottleneck

DBLP/Texas show a separation between performance and mass:

| Dataset | ACC | Rel Mean | Reliable Ratio | Weighted Agreement Movement |
| --- | ---: | ---: | ---: | ---: |
| DBLP | 0.6919 | 0.0756 | 0.0000 | +0.2120 |
| Texas | 0.7377 | 0.0561 | 0.0055 | -0.0827 |

DBLP especially is informative:

```text
DBLP has useful anchor coupling and strong ACC, but effective mass remains just
below 0.08 and reliable-node ratio remains zero.
```

This suggests the next mechanism should target:

```text
mass allocation and normalization under the reliability floor
```

not:

```text
another beta-source variant.
```

## 7. Rescue Question For V57A

The next question should be:

```text
Can the anchor loss maintain the V56A reliability signal but normalize and
allocate residual anchor mass so medium-consensus datasets cross the effective
mass gate without increasing weak-anchor exposure?
```

Recommended route name:

```text
v57a_mass_floor_normalized_residual_anchor
```

Core design direction:

```text
Keep V56A hybrid reliability unchanged as a raw score.
Use a detached, dataset-agnostic mass-floor normalization to prevent the anchor
loss from becoming ineffective when mean reliability is just below the gate.
Do not change final labels, V50A anchor construction, beta bounds, soft power,
or hybrid compensation.
```

## 8. Candidate Mechanism

Potential V57A direction to preregister:

```text
raw_r_i = V56A reliability
target_mass = min_effective_mass
scale = clamp(target_mass / mean(raw_r_i), 1.0, max_mass_scale)
r_i = detach(clamp(raw_r_i * scale, 0, max_reliability_cap))
```

The first preregistration should use fixed constants:

```text
target_mass = 0.08
max_mass_scale = 1.50
max_reliability_cap = 0.90
```

This is not a threshold relaxation if the gate remains unchanged. It tests
whether the same reliability ranking can be made operationally effective
without dataset-specific routing.

## 9. Expected Failure Mode

V57A may still stop if:

- scaling overexposes Squirrel and drops ACC below 0.2800;
- Texas weighted agreement remains negative and performance falls;
- DBLP mass rises but anchor usefulness or safety fails;
- reliability becomes too dense and violates the >0.90 or posterior/readout
  safety constraints.

These risks must be preregistered before implementation.

## 10. No-Fabrication Status

All numbers in this document come from local V54A/V55A/V56A verdict files and
diagnostics. No V57A code or result exists.
