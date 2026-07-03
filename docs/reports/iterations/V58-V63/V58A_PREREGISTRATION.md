# V58A Preregistration

Proposed variant:

```text
v58a_anchor_release_residual_compactness
```

This file follows `V57A_FULL_RUN_FAILURE_ANALYSIS.md`.

No V58A implementation or result exists at the time this preregistration is
written.

## 1. Research Question

V57A showed that detached mass-floor normalization can prevent short-run anchor
mass collapse, but the 260-epoch full run failed because longer training caused
full-length drift.

V58A tests one narrow question:

```text
Can a fixed release schedule preserve the early compactness benefit of V57A
while preventing sustained late-stage over-coupling to the static spectral
anchor?
```

This is a time-allocation rescue route, not another reliability-formula or mass
allocation variant.

## 2. Mechanism

V58A keeps the V57A anchor construction and reliability mechanism unchanged:

```text
same spectral compactness anchor
same detached reliability ranking
same mass-floor normalization
same target mass 0.08
same max mass scale 1.50
same max reliability cap 0.90
same beta bounds and soft consensus components
same final q_refined labels
```

V58A changes only the time multiplier applied to the V57A anchor loss.

Instead of using V57A's ramp-to-1.0-and-stay schedule, V58A uses:

```text
release_gamma(epoch) =
  0.0                                 if epoch <= 20
  (epoch - 20) / 40                   if 20 < epoch <= 60
  1.0                                 if 60 < epoch <= 80
  max(0.25, 1 - (epoch - 80) / 80)    if 80 < epoch <= 140
  0.25                                if epoch > 140
```

Interpretation:

```text
20 epoch warmup
40 epoch ramp
20 epoch full-strength absorption window
60 epoch release window from 1.0 to 0.25
0.25 late residual floor
```

The residual floor is deliberately nonzero so V58A is not an implicit
dataset-specific early-stop trick.

## 3. Implementation Boundary

V58A must be implemented as a conservative extension of the existing V57A
helper:

```text
v58a_loss = release_gamma(epoch) * v57a_weighted_anchor_kl
```

The V57A internal reliability and mass-normalization calculations remain
unchanged.

Allowed implementation changes:

```text
add v58a config fields
add a v58a helper or wrapper that reuses the V57A loss internals
add v58a diagnostics for release_gamma and anchor loss
add runner variant v58a_anchor_release_residual_compactness
```

Forbidden implementation changes:

```text
no V57A constant changes
no new reliability formula
no new anchor construction
no dataset-specific schedule
no dynamic validation/test-metric schedule
no final-label selector
no legacy head
no KMeans/q_anchor/q_embed final labels
```

## 4. Frozen Constants

V58A must inherit these V57A constants unchanged:

```text
anchor_weight = 0.04
reliability_floor = 0.10
reliable_threshold = 0.20
min_effective_mass = 0.10
beta_min = 0.35
beta_max = 0.70
soft_power = 0.50
hybrid_compensation = 0.50
target_mass = 0.08
max_mass_scale = 1.50
max_reliability_cap = 0.90
```

New V58A constants:

```text
release_warmup_epochs = 20
release_ramp_epochs = 40
release_hold_until_epoch = 80
release_decay_epochs = 60
release_floor = 0.25
```

These constants must not be changed after seeing V58A results.

## 5. Expected Diagnostic Signature

V58A should preserve the useful V57A signature through epoch 80:

```text
mass-normalization gate remains stable
weighted q-anchor agreement improves from epoch 1 to epoch 80
posterior/readout safety remains clean
```

After epoch 80, the desired signature is not unlimited agreement growth.
Instead:

```text
release_gamma decreases
anchor agreement may plateau
ACC should not collapse on Squirrel, Flickr, PubMed, or Texas
embedding_posterior_gap remains small
```

If anchor agreement keeps increasing but ACC still drops, then V58A fails the
time-allocation hypothesis.

## 6. Required Implementation Review

Before code changes, write:

```text
V58A_IMPLEMENTATION_REVIEW.md
```

It must confirm:

```text
exact code insertion points
whether V57A helper is wrapped or duplicated
how release_gamma is computed
which diagnostics are added
that V50A-V57A are disabled in the V58A runner except reused internals
that final labels remain q_refined
that no dataset-specific branch is introduced
```

Only after this review may the minimal implementation be made.

## 7. Authorized First Tests After Implementation

After implementation review and code changes, only this connectivity test is
authorized:

```powershell
conda run --no-capture-output -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v58a_anchor_release_residual_compactness --datasets "acm" --epochs 1 --device cuda --log-level WARNING
```

If connectivity passes, write:

```text
V58A_CONNECTIVITY_VERDICT.md
```

Only then is the first mixed-stress run authorized:

```powershell
conda run --no-capture-output -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v58a_anchor_release_residual_compactness --datasets "acm,dblp,flickr,texas,squirrel,chameleon" --epochs 80 --device cuda --log-level WARNING
```

No 260-epoch V58A run is authorized until a separate expansion review exists.

## 8. First Mixed-Stress Gates

The first mixed-stress verdict must include:

```text
red-line table
release-gamma table
mass-normalization table
reliability table
anchor-usefulness table
posterior/readout safety table
preservation-floor table
comparison to V57A 80e on the same 6 datasets
```

Pass requirements:

```text
status=ok on 6/6
red-line pass on 6/6
release_gamma_epoch_1 = 0
release_gamma_epoch_40 = 0.5
release_gamma_epoch_80 = 1.0
mass pass on at least 4/6
reliable-node ratio pass on at least 3/6
abs(embedding_posterior_gap) <= 0.04 on 6/6
ACM ACC >= 0.8888
DBLP ACC >= 0.6610
Squirrel ACC >= 0.2800
```

Because the first mixed-stress run ends at epoch 80, it mainly verifies that
V58A preserves V57A's useful early behavior and does not break wiring. It does
not prove the release hypothesis.

## 9. Later Expansion Boundary

If the 80e mixed-stress passes, the next artifact must be:

```text
V58A_EXPANSION_REVIEW.md
```

That review may authorize a 260e supported 9-dataset run only if it keeps the
same fixed release schedule and explicitly tests the V57A full-run failure
datasets:

```text
Flickr, PubMed, Squirrel, Texas
```

No 9-dataset full run, seed sweep, schedule sweep, floor sweep, or reliability
formula variant is authorized by this preregistration.

## 10. Stop Conditions

Stop immediately and write a failure analysis if any of the following occur:

```text
connectivity fails
red-line flags show V50A-V57A enabled as active losses in the V58A runner
legacy_head_used=true
final labels are not q_refined
Squirrel falls below 0.2800 at 80e
embedding_posterior_gap exceeds 0.08 on any dataset
release_gamma diagnostics do not match the fixed schedule
```

## 11. No-Fabrication Status

This is a preregistration only. It contains no V58A results.
