# V59A Failure Analysis

This file analyzes the failed full run of
`v59a_post80_anchor_release_residual_compactness`.

It follows `V59A_FULL_RUN_VERDICT.md`.

No V59A tuning, rerun, seed sweep, schedule sweep, release-floor sweep,
reliability formula change, or post-hoc selector is authorized by this analysis.

## 1. Failure Boundary

V59A does not fail because the post-80 release wrapper is miswired:

```text
red-line gate passes
release_gamma is 1.0 at epochs 1/40/80
release_gamma is 0.25 at epoch 260
mass/reliability gate passes
posterior/readout safety passes
```

It fails because the release mechanism does not repair the hardest long-run
drift cases:

```text
Squirrel ACC = 0.2103 < 0.2800 preservation floor
Flickr ACC = 0.2630 < 0.3500 drift-repair floor
```

## 2. What V59A Solved

V59A successfully fixed the V58A failure:

```text
The first 80 epochs are V57A-equivalent again.
```

Evidence:

```text
V59A 80e passes all mixed-stress gates.
ACM 80e returns from V58A's 0.7038 to 0.8962.
Squirrel 80e remains above the 0.2800 floor.
```

V59A also partially fixes the V57A full-run drift:

| Dataset | V57A 260E ACC | V59A 260E ACC | Change |
| --- | ---: | ---: | ---: |
| PubMed | 0.4788 | 0.5261 | +0.0473 |
| Texas | 0.6175 | 0.7213 | +0.1038 |

This confirms that post-80 anchor release can help some datasets.

## 3. What V59A Did Not Solve

The two hardest failure cases remain:

| Dataset | V57A 80E | V57A 260E | V59A 80E | V59A 260E | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| Flickr | 0.4133 | 0.2779 | 0.4150 | 0.2630 | Weak-anchor graph drifts even with release. |
| Squirrel | 0.3005 | 0.2102 | 0.3017 | 0.2103 | Heterophily drift is not caused by anchor pressure alone. |

Therefore the core hypothesis is falsified in its broad form:

```text
Long-run drift is not explained solely by sustained static-anchor pressure.
```

## 4. Mechanistic Diagnosis

V59A reduces late anchor pressure to 0.25 after epoch 140. If the V57A full-run
failure were primarily caused by too much anchor pressure, Squirrel and Flickr
should have recovered. They do not.

This suggests a different root cause:

```text
The unified representation/posterior dynamics themselves drift after the early
compactness window, especially when the anchor is weak or heterophily is severe.
```

Dataset-specific hints:

```text
Flickr has extremely low reliability mass and negative/declining anchor
agreement. The anchor is mostly not useful, so releasing it cannot fix the
late collapse.

Squirrel has enough mass to pass the reliability floor, but the late trajectory
still collapses. This points to late q_refined dynamics or topology/attribute
interaction, not anchor pressure.
```

## 5. What Should Not Be Done

Do not keep tuning time schedules:

```text
Do not try release_floor 0.1/0.5.
Do not move release_start to 60/100/120.
Do not add dataset-specific release.
Do not use V57A for ACM/DBLP and V59A for Texas/PubMed.
Do not select 80e for Flickr/Squirrel and 260e for others.
```

The time-allocation family has now produced a clear result:

```text
Post-80 release helps medium drift but does not rescue weak-anchor and
heterophily drift.
```

## 6. Next Rescue Insight

The next route should shift from:

```text
How much anchor pressure should remain late?
```

to:

```text
How do we prevent late q_refined drift after the early compactness signal has
created a useful cluster geometry?
```

The promising mechanism is not more anchor pressure. It is a late consistency
regularizer against the model's own early stable posterior geometry.

## 7. Candidate V60A Direction

Recommended next route:

```text
v60a_ema_self_distillation_drift_guard
```

Mechanism:

```text
Keep V59A's post-80 release and V57A internals unchanged.
At epoch 80, create a detached EMA/teacher snapshot of q_refined or logits.
After epoch 80, apply a small unified consistency loss that discourages
catastrophic late drift on confident nodes while allowing uncertain nodes to
continue adapting.
```

Key constraints:

```text
teacher must be detached
teacher must be created by the same model, same pipeline, no labels
confidence mask must be dataset-agnostic
loss weight must be fixed before implementation
final labels remain q_refined
no dataset-specific early stop
no selector between teacher and current q
```

Why this follows the evidence:

```text
V57A/V59A show the early 80e state is often safer than the 260e state.
V59A shows anchor release alone cannot preserve that safety.
Therefore the next mechanism should preserve the useful early posterior
geometry directly, not only reduce the anchor force.
```

## 8. Required Next Artifact

Before any implementation or experiment, write:

```text
V60A_PREREGISTRATION.md
```

It must specify:

```text
teacher snapshot epoch
whether teacher is q_refined, logits, or sharpened q
confidence rule
loss form and fixed weight
how it combines with V59A release
connectivity command
first mixed-stress gates
whether any 260e run can be authorized later
hard stop conditions
```

## 9. No-Fabrication Status

This analysis uses only local V57A, V58A, and V59A verdict files and diagnostics.
No V60A result exists.
