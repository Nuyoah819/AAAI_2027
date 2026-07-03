# V59A Implementation Review

This review follows `V59A_PREREGISTRATION.md` and must be completed before any
V59A code change or experiment.

## 1. Boundary

Proposed variant:

```text
v59a_post80_anchor_release_residual_compactness
```

Implementation decision:

```text
APPROVE MINIMAL IMPLEMENTATION.
```

V59A is allowed to test only one fixed time-allocation change:

```text
Run V57A-equivalent anchor pressure through epoch 80, then apply a post-80
release multiplier to the same V57A mass-floor normalized anchor loss.
```

It must not change V57A reliability, mass scaling, spectral-anchor construction,
final-label path, dataset routing, or any V57A constants.

## 2. Current Code State

Current code contains:

```text
V57A helper: mass_floor_normalized_residual_spectral_anchor_loss
V58A wrapper: anchor_release_residual_spectral_anchor_loss
V58A runner variant and diagnostics
No V59A symbols in code yet
```

The V58A wrapper is useful structurally but not semantically reusable as-is,
because it changes the pre-80 trajectory. V59A needs a distinct post-80 release
wrapper.

## 3. Required Insertion Points

### Config

Add to `E2ESECTCoCoConfig`:

```text
v59a_enabled: bool = False
v59a_anchor_weight: float = 0.0
v59a_reliability_floor: float = 0.10
v59a_reliable_threshold: float = 0.20
v59a_min_effective_mass: float = 0.10
v59a_warmup_epochs: int = 20
v59a_ramp_epochs: int = 40
v59a_beta_min: float = 0.35
v59a_beta_max: float = 0.70
v59a_soft_power: float = 0.50
v59a_hybrid_compensation: float = 0.50
v59a_target_mass: float = 0.08
v59a_max_mass_scale: float = 1.50
v59a_max_reliability_cap: float = 0.90
v59a_release_start_epoch: int = 80
v59a_release_decay_epochs: int = 60
v59a_release_floor: float = 0.25
```

### Loss Call

Insert the V59A loss call immediately after V58A in
`EndToEndSECTCoCoModule.loss`.

Use a new helper:

```text
post80_anchor_release_residual_spectral_anchor_loss(...)
```

The helper must internally call `mass_floor_normalized_residual_spectral_anchor_loss`
with the V59A inherited constants and then multiply only the returned V57A-style
loss by `v59a_release_gamma`.

### Total Loss

Add only:

```text
+ cfg.v59a_anchor_weight * v59a_anchor_loss
```

The V59A runner must set V50A-V58A enabled flags to false and all their anchor
weights to 0.0.

### Diagnostics

Add independent `v59a_*` diagnostics. Required fields:

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

For audit convenience, copy the V57A reliability, mass, and anchor diagnostic
surface under `v59a_*` names where practical.

### Snapshot Keys

Add these keys to the epoch snapshot list:

```text
v59a_release_gamma
v59a_weighted_q_anchor_agreement
v59a_weighted_q_anchor_kl
v59a_pre_release_weighted_q_anchor_kl
v59a_reliability_mean
v59a_mass_scale
v59a_raw_reliability_mean
v59a_scaled_reliability_mean
```

This must produce:

```text
v59a_release_gamma_epoch_1 = 1.0
v59a_release_gamma_epoch_40 = 1.0
v59a_release_gamma_epoch_80 = 1.0
```

### Anchor Construction Trigger

Extend spectral anchor construction with:

```text
or bool(getattr(cfg, "v59a_enabled", False))
```

The anchor construction itself must remain unchanged.

### Final Anchor Diagnostics

Extend final anchor diagnostic emission to include:

```text
v59a_anchor_acc_diagnostic
v59a_anchor_nmi_diagnostic
v59a_anchor_ari_diagnostic
```

### Runner

In `scripts/run_unified_aptc_9datasets.py`:

```text
add disabled V59A default fields
add EXPERIMENT_VARIANTS["v59a_post80_anchor_release_residual_compactness"]
```

The V59A variant must explicitly disable V50A-V58A active losses:

```text
v50a_enabled=false
v51a_enabled=false
v52a_enabled=false
v53a_enabled=false
v54a_enabled=false
v55a_enabled=false
v56a_enabled=false
v57a_enabled=false
v58a_enabled=false
all corresponding anchor weights = 0.0
v59a_enabled=true
v59a_anchor_weight=0.04
```

## 4. Release Gamma Definition

Use the fixed preregistered schedule:

```text
release_gamma(epoch) =
  1.0                                 if epoch <= 80
  max(0.25, 1 - (epoch - 80) / 80)    if 80 < epoch <= 140
  0.25                                if epoch > 140
```

Implementation should use the existing zero-based `current_epoch` convention:

```text
epoch_number = current_epoch + 1
```

The observable required values are:

```text
epoch_1  -> 1.0
epoch_40 -> 1.0
epoch_80 -> 1.0
```

Use:

```text
decay_denominator = release_decay_epochs / (1 - release_floor)
```

With preregistered constants, this is `60 / 0.75 = 80`, matching the written
formula.

## 5. Helper Strategy

Approved strategy:

```text
Wrap V57A helper output and remap diagnostics from v57a_* to v59a_*.
```

Required behavior:

```text
pre_release_loss, base_stats = mass_floor_normalized_residual_spectral_anchor_loss(...)
release_gamma = post80 fixed schedule
loss = release_gamma * pre_release_loss
```

Diagnostic mapping:

```text
v59a_pre_release_anchor_loss = base_stats["v57a_anchor_loss"]
v59a_anchor_loss = release_gamma * base_stats["v57a_anchor_loss"]
v59a_pre_release_weighted_q_anchor_kl = base_stats["v57a_weighted_q_anchor_kl"]
v59a_weighted_q_anchor_kl = release_gamma * base_stats["v57a_weighted_q_anchor_kl"]
all reliability/mass/agreement diagnostics remapped without value changes
```

Agreement and reliability diagnostics must not be multiplied by release gamma.
Only the anchor-loss pressure is released.

## 6. Red-Line Review

| Requirement | Review |
| --- | --- |
| V57A reliability unchanged | PASS if wrapper calls existing V57A helper without altering internals. |
| V57A 0-80 behavior preserved | PASS if release_gamma is 1.0 through epoch 80. |
| Mass-floor constants unchanged | PASS if V59A runner sets target 0.08, max scale 1.50, cap 0.90. |
| Spectral anchor unchanged | PASS if anchor construction remains `build_spectral_compactness_anchor`. |
| Dataset-agnostic schedule | PASS because release schedule depends only on epoch. |
| No final-label selector | PASS if final output remains existing `q_refined`. |
| No legacy head | PASS if `legacy_head_used=false` remains diagnostic expectation. |
| V50A-V58A inactive in runner | PASS only if all previous enabled flags/weights are false/zero. |
| No V59A 260e authorization | PASS; implementation authorizes only connectivity after compile. |

## 7. Authorized Next Action

After this review, the only authorized action is minimal implementation followed
by static verification:

```powershell
conda run -n aaai-e2e-subspace python -m py_compile core\e2e\sect_coco_e2e.py scripts\run_unified_aptc_9datasets.py
```

If compile passes and no residual training process exists, only the following
connectivity run is authorized:

```powershell
conda run --no-capture-output -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v59a_post80_anchor_release_residual_compactness --datasets "acm" --epochs 1 --device cuda --log-level WARNING
```

No V59A mixed-stress run is authorized until `V59A_CONNECTIVITY_VERDICT.md` is
written.

## 8. No-Fabrication Status

This review contains no V59A results. It only authorizes a minimal
implementation consistent with `V59A_PREREGISTRATION.md`.
