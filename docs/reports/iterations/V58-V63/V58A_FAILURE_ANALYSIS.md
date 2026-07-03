# V58A Failure Analysis

This file analyzes the failed first mixed-stress run of
`v58a_anchor_release_residual_compactness`.

It follows `V58A_FIRST_MIXED_STRESS_VERDICT.md`.

No V58A tuning, rerun, seed sweep, schedule sweep, or post-hoc selector is
authorized by this analysis.

## 1. Failure Boundary

V58A does not fail because the runner wiring is wrong:

```text
red-line gate passes
release gamma diagnostics match 0.0 / 0.5 / 1.0 at epochs 1 / 40 / 80
V50A-V57A active losses are disabled
V58A is enabled
```

It fails because the first 80-epoch mixed-stress gate no longer preserves the
useful early behavior of V57A:

```text
ACM ACC = 0.7038 < 0.8888
DBLP ACC = 0.6608 < 0.6610
Squirrel abs(embedding_posterior_gap) = 0.0883 > 0.08
reliable-node ratio passes only 2/6
```

Therefore V58A cannot proceed to expansion or any 260-epoch test.

## 2. What The Failure Means

The V57A full-run failure suggested:

```text
Sustained post-80 anchor pressure can cause long-run drift.
```

V58A tested a simple outer release schedule from the beginning:

```text
0 before warmup, ramp to 1.0 by epoch 60, hold through epoch 80, then release.
```

The 80e result shows this is too disruptive:

```text
Even though release_gamma reaches 1.0 by epoch 80, the early trajectory differs
enough from V57A that ACM collapses before the release hypothesis can be
meaningfully tested.
```

In short:

```text
The rescue insight was directionally plausible, but the implementation
perturbed the early absorption window that V57A had already validated.
```

## 3. Evidence

Comparison to the V57A first mixed-stress 80e result:

| Dataset | V57A 80E ACC | V58A 80E ACC | Delta |
| --- | ---: | ---: | ---: |
| ACM | 0.9005 | 0.7038 | -0.1967 |
| DBLP | 0.6919 | 0.6608 | -0.0311 |
| Flickr | 0.3681 | 0.3890 | +0.0209 |
| Texas | 0.7377 | 0.7322 | -0.0055 |
| Squirrel | 0.3011 | 0.3021 | +0.0010 |
| Chameleon | 0.3382 | 0.3263 | -0.0119 |

The failure is not uniform. Flickr slightly improves and Texas/Squirrel are
stable, but the ACM collapse is too large to treat as noise or a harmless
trade-off.

## 4. Mechanistic Diagnosis

V57A's anchor loss already contained an internal reliability ramp:

```text
v57a_gamma: 0 -> 1 over the warmup/ramp window
```

V58A multiplied this with an outer release schedule:

```text
effective pressure before epoch 60 is reduced relative to V57A
```

Because V58A's first mixed-stress run stops at epoch 80, the late release
portion is not yet active. The observed failure therefore cannot support the
claim that late release solves full-run drift. It only shows:

```text
Do not change V57A's 0-80 absorption dynamics.
```

The correct next question is narrower:

```text
Can V57A be kept exactly through the validated 80e window, then released only
after that window to prevent 260e drift?
```

## 5. What Should Not Be Done

Do not try to repair V58A by tuning against this failed run:

```text
Do not change V57A reliability constants.
Do not change target mass, max mass scale, or cap.
Do not change beta bounds or hybrid compensation.
Do not sweep release floor.
Do not sweep release start.
Do not add dataset-specific early stop.
Do not select V57A for ACM and V58A for other datasets.
Do not use q_anchor, q_embed, KMeans, or legacy labels as final output.
```

Those changes would turn the route into post-hoc tuning.

## 6. Rescue Insight

The useful part of V58A is the time-allocation question, not its early
multiplier:

```text
V57A should be treated as the fixed absorption phase.
Only the post-80 continuation should be changed.
```

A viable next route should:

```text
match V57A exactly through epoch 80
apply release only after epoch 80
keep a nonzero late residual anchor floor
test whether Squirrel/Flickr/PubMed/Texas long-run drift is reduced
```

This preserves the evidence that already passed and targets only the failure
window observed in the V57A full run.

## 7. Required Next Artifact

Before any code change or experiment, write:

```text
V59A_PREREGISTRATION.md
```

The preregistration must specify:

```text
V57A-equivalent behavior through epoch 80
post-80 release schedule
fixed release floor
connectivity command
first mixed-stress gates
expansion boundary for any 260e run
hard stop conditions
```

## 8. No-Fabrication Status

This analysis uses only local V57A and V58A verdict files and diagnostics.
No V59A result exists.
