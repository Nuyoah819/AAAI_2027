# V59A Preregistration

Proposed variant:

```text
v59a_post80_anchor_release_residual_compactness
```

This file follows `V58A_FAILURE_ANALYSIS.md`.

No V59A implementation or result exists at the time this preregistration is
written.

## 1. Research Question

V57A solved short-run mass non-collapse but failed the 260-epoch full run due to
long-run drift. V58A tried a global release multiplier and failed at 80 epochs
because it perturbed the V57A early absorption window.

V59A tests the narrower question:

```text
Can we keep V57A exactly through the validated 80-epoch absorption window, then
release anchor pressure only after epoch 80 to reduce full-length drift?
```

This is still a time-allocation rescue route. It is not a reliability-formula,
mass-allocation, anchor-construction, final-label, or dataset-selector route.

## 2. Mechanism

V59A must keep V57A internals unchanged:

```text
same spectral compactness anchor
same detached reliability ranking
same mass-floor normalization
same target mass 0.08
same max mass scale 1.50
same max reliability cap 0.90
same beta bounds and hybrid consensus components
same final q_refined labels
```

The only allowed change is the post-80 loss multiplier:

```text
release_gamma(epoch) =
  1.0                                 if epoch <= 80
  max(0.25, 1 - (epoch - 80) / 80)    if 80 < epoch <= 140
  0.25                                if epoch > 140
```

Interpretation:

```text
Epochs 1-80: exactly V57A pressure.
Epochs 81-140: release from 1.0 toward 0.25.
Epochs 141+: keep a 0.25 residual anchor floor.
```

The nonzero residual floor prevents V59A from becoming an implicit early-stop
or dataset-specific shutoff trick.

## 3. Frozen Constants

V59A inherits V57A constants unchanged:

```text
anchor_weight = 0.04
reliability_floor = 0.10
reliable_threshold = 0.20
min_effective_mass = 0.10
warmup_epochs = 20
ramp_epochs = 40
beta_min = 0.35
beta_max = 0.70
soft_power = 0.50
hybrid_compensation = 0.50
target_mass = 0.08
max_mass_scale = 1.50
max_reliability_cap = 0.90
```

New V59A constants:

```text
release_start_epoch = 80
release_decay_epochs = 60
release_floor = 0.25
```

Implementation note:

```text
The preregistered formula uses denominator 80 for the decay expression because
decaying from 1.0 to floor 0.25 over 60 epochs requires dividing by
60 / (1 - 0.25) = 80.
```

No constant may be changed after seeing V59A results.

## 4. Hard Prohibitions

V59A must not use:

```text
dataset-specific branch, schedule, stop rule, selector, or threshold
validation/test metrics in training
label-aware release
adaptive release based on ACC/NMI/ARI
V57A reliability or mass constant changes
V50A anchor hyperparameter changes
post-hoc selection among V57A, V58A, V59A, q_anchor, q_embed, KMeans, or legacy labels
legacy head as final output
q_anchor/q_embed/KMeans final labels
seed sweep
schedule sweep
release-floor sweep
```

Final labels must remain the existing unified `q_refined` output.

## 5. Required Diagnostics

V59A must expose independent `v59a_*` diagnostics:

```text
v59a_enabled
v59a_release_gamma
v59a_release_start_epoch
v59a_release_decay_epochs
v59a_release_floor
v59a_anchor_loss
v59a_pre_release_anchor_loss
v59a_weighted_q_anchor_kl
v59a_pre_release_weighted_q_anchor_kl
v59a_weighted_q_anchor_agreement
v59a_unweighted_q_anchor_agreement
v59a_embedding_anchor_agreement
v59a_raw_reliability_mean
v59a_mass_scale
v59a_scaled_reliability_mean
v59a_reliability_mean
v59a_reliable_node_ratio
v59a_effective_anchor_mass
v59a_target_mass
v59a_max_mass_scale
v59a_max_reliability_cap
```

Snapshot diagnostics must include:

```text
v59a_release_gamma_epoch_1
v59a_release_gamma_epoch_40
v59a_release_gamma_epoch_80
v59a_weighted_q_anchor_agreement_epoch_1
v59a_weighted_q_anchor_agreement_epoch_40
v59a_weighted_q_anchor_agreement_epoch_80
v59a_reliability_mean_epoch_80
v59a_mass_scale_epoch_80
```

For any later 260e expansion, the same diagnostics must also support final
release-gamma and agreement checks at epoch 260.

## 6. Required Implementation Review

Before code changes, write:

```text
V59A_IMPLEMENTATION_REVIEW.md
```

It must confirm:

```text
exact code insertion points
whether the helper wraps V57A internals
that V59A release_gamma is 1.0 through epoch 80
that V57A reliability and mass calculations are unchanged
that V50A-V58A active losses are disabled in the V59A runner
that final labels remain q_refined
that no dataset-specific branch is introduced
```

Only after this review may the minimal implementation be made.

## 7. Authorized Connectivity Test

After implementation review and code changes, only this connectivity test is
authorized:

```powershell
conda run --no-capture-output -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v59a_post80_anchor_release_residual_compactness --datasets "acm" --epochs 1 --device cuda --log-level WARNING
```

If connectivity passes, write:

```text
V59A_CONNECTIVITY_VERDICT.md
```

Connectivity pass requires:

```text
status=ok
legacy_head_used=false
v50a-v58a_enabled=false
v59a_enabled=true
v59a_release_gamma=1.0
v59a_pre_release_anchor_loss finite
v59a_anchor_loss finite and equal to pre-release loss at epoch 1
v59a_target_mass=0.08
v59a_max_mass_scale=1.50
v59a_max_reliability_cap=0.90
```

## 8. Authorized First Mixed-Stress Test

Only after connectivity passes:

```powershell
conda run --no-capture-output -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v59a_post80_anchor_release_residual_compactness --datasets "acm,dblp,flickr,texas,squirrel,chameleon" --epochs 80 --device cuda --log-level WARNING
```

This first mixed-stress run should be V57A-equivalent through 80 epochs except
for naming and diagnostics.

No 260-epoch V59A run is authorized by this preregistration.

## 9. First Mixed-Stress Gates

Required verdict artifact:

```text
V59A_FIRST_MIXED_STRESS_VERDICT.md
```

Pass requirements:

```text
status=ok on 6/6
red-line pass on 6/6
v59a_release_gamma_epoch_1 = 1.0
v59a_release_gamma_epoch_40 = 1.0
v59a_release_gamma_epoch_80 = 1.0
mass pass on at least 4/6
reliable-node ratio pass on at least 3/6
abs(embedding_posterior_gap) <= 0.04 on 6/6
ACM ACC >= 0.8888
DBLP ACC >= 0.6610
Squirrel ACC >= 0.2800
```

Comparison requirement:

```text
Compare V59A 80e to V57A 80e on ACM, DBLP, Flickr, Texas, Squirrel, and
Chameleon. If ACM drops by more than 0.02 from V57A 80e, stop.
```

Because V59A is designed to be V57A-equivalent through epoch 80, a large 80e
deviation is an implementation or dynamics failure.

## 10. Later Expansion Boundary

If the 80e mixed-stress passes, the next artifact must be:

```text
V59A_EXPANSION_REVIEW.md
```

That review may authorize one supported 9-dataset / 260-epoch run only if it
explicitly tests the V57A long-run failure datasets:

```text
Flickr, PubMed, Squirrel, Texas
```

No 260e run, seed sweep, schedule sweep, release-floor sweep, reliability
formula variant, or dataset-specific branch is authorized before that expansion
review.

## 11. Stop Conditions

Stop immediately and write a failure analysis if any occur:

```text
connectivity fails
red-line flags show V50A-V58A enabled as active losses in the V59A runner
legacy_head_used=true
final labels are not q_refined
v59a_release_gamma is not 1.0 during the first 80 epochs
Squirrel falls below 0.2800 at 80e
ACM falls below 0.8888 at 80e
DBLP falls below 0.6610 at 80e
embedding_posterior_gap exceeds 0.08 on any dataset
```

## 12. No-Fabrication Status

This is a preregistration only. It contains no V59A results.
