# V57A Full-Run Failure Analysis

This file analyzes the failed 260-epoch full run of
`v57a_mass_floor_normalized_residual_anchor`.

It follows `V57A_FULL_RUN_VERDICT.md`.

No V57A parameter tuning, restart selection, seed sweep, or post-hoc selector is
authorized by this analysis.

## 1. Failure Boundary

V57A does not fail because the mass-floor normalization collapses.

It fails because the fixed 260-epoch training length exposes full-length drift:

```text
The spectral-anchor agreement keeps improving on most datasets, but longer
coupling does not consistently preserve clustering quality.
```

The hard preregistered failure is:

```text
Squirrel ACC = 0.2102 < 0.2800 preservation floor.
```

Additional drift warnings:

```text
Flickr: 0.4133 at 80e -> 0.2779 at 260e
PubMed: 0.5822 at 80e -> 0.4788 at 260e
Texas: 0.7377 at 80e -> 0.6175 at 260e
Squirrel: 0.3005 at 80e -> 0.2102 at 260e
```

## 2. What V57A Actually Solved

V57A successfully solved the local problem inherited from V56A:

```text
Medium-consensus datasets needed enough operational anchor mass without
blindly increasing the anchor weight.
```

Evidence:

```text
The 260e run passes mass-normalization, reliability non-collapse,
anchor-usefulness, posterior/readout safety, and red-line gates.
```

This means the V57A mechanism is not a wiring failure and not an obvious red-line
violation. It is a training-dynamics failure.

## 3. Why The Full Run Fails

The key contradiction is:

```text
Anchor agreement can improve while task ACC deteriorates.
```

Examples:

| Dataset | Agreement @1 | Agreement @260 | ACC @80 | ACC @260 | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| PubMed | 0.2311 | 0.6938 | 0.5822 | 0.4788 | Stronger anchor coupling coincides with worse clustering. |
| Squirrel | 0.1220 | 0.2400 | 0.3005 | 0.2102 | Anchor use increases but heterophily safety floor breaks. |
| Texas | 0.3893 | 0.4671 | 0.7377 | 0.6175 | Longer coupling erodes an initially stable result. |
| Flickr | 0.0243 | 0.0124 | 0.4133 | 0.2779 | Weak-anchor non-use becomes long-run drift. |

The likely mechanism is:

```text
V57A controls how much anchor mass is active, but not when anchor pressure
should stop influencing the representation after the useful compactness signal
has been absorbed.
```

V57A has a warmup/ramp schedule for activating the anchor:

```text
gamma: 0 at epoch 1, 0.5 at epoch 40, 1.0 by epoch 60/80
```

But after gamma reaches 1.0, the anchor pressure remains fully active through
epoch 260. The 80e smoke suggests the rescue signal is useful early. The 260e
run suggests sustained pressure can over-couple the unified posterior to a
static spectral anchor or amplify dataset-specific optimization drift without
using a dataset-specific branch.

## 4. What Should Not Be Done

Do not tune V57A constants against the failed full-run result:

```text
Do not change v57a_target_mass.
Do not change v57a_max_mass_scale.
Do not change v57a_anchor_weight.
Do not change v57a_beta bounds.
Do not change v57a_soft_power or hybrid compensation.
Do not add a dataset-specific early stop.
Do not select 80e for some datasets and 260e for others.
Do not use q_anchor, q_embed, KMeans, or legacy head as a final-label selector.
```

These would convert the rescue route into post-hoc tuning or hidden
dataset-specific behavior.

## 5. Rescue Insight

The salvageable insight is:

```text
The spectral compactness anchor is useful as a transient stabilizer, not as a
constant full-length force.
```

Therefore, the next rescue route should not ask:

```text
How do we make the anchor stronger?
```

It should ask:

```text
How do we retain the early compactness benefit while preventing full-length
anchor over-coupling?
```

This reframes the problem from mass allocation to time allocation.

## 6. Candidate V58A Direction

Recommended next route:

```text
v58a_anchor_release_residual_compactness
```

Mechanism:

```text
Keep V57A's detached mass-floor reliability and anchor construction unchanged,
but replace the always-on full-strength post-ramp anchor with a fixed
release-after-absorption schedule.
```

The schedule should be preregistered before implementation. A conservative form
is:

```text
0 before warmup
ramp to full strength by the early stability window
hold briefly
decay to a small residual anchor floor before late training
```

The residual floor prevents the route from becoming a post-hoc early-stopping
trick, while decay tests the new scientific hypothesis:

```text
The anchor is needed to shape a compact subspace early, but late clustering
should be governed mainly by the unified q_refined dynamics.
```

## 7. Red-Line Safety For V58A

V58A must keep:

```text
same 9-dataset unified pipeline
same final q_refined output
same V57A reliability formula
same V57A mass-floor constants
same spectral anchor construction
same seed and no restart selection
no dataset-specific branches
no post-hoc selector
```

V58A may change only:

```text
the time schedule multiplying the already computed V57A anchor loss
```

The schedule constants must be fixed before seeing V58A results.

## 8. Required Next Artifact

Before any code change or experiment, write:

```text
V58A_PREREGISTRATION.md
```

It must specify:

```text
exact release schedule
whether V57A gamma is replaced or multiplied
anchor residual floor
authorized connectivity command
first mixed-stress gates
full-run gate boundary
hard stop conditions
no tuning after V57A failure
```

## 9. No-Fabrication Status

This analysis uses only local V57A 80e and 260e results already recorded in:

```text
V57A_SUPPORTED_9DATASET_80E_SMOKE_VERDICT.md
V57A_FULL_RUN_VERDICT.md
```

No V58A result exists.
