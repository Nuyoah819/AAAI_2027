# V58A Implementation Review

This review follows `V58A_PREREGISTRATION.md` and must be completed before any
V58A code change or experiment.

## 1. Boundary

Proposed variant:

```text
v58a_anchor_release_residual_compactness
```

Implementation decision:

```text
APPROVE MINIMAL IMPLEMENTATION.
```

V58A is allowed to test only a fixed time-allocation change:

```text
Compute the same V57A mass-floor normalized anchor loss internals, then expose
and apply a V58A release multiplier to the anchor loss.
```

It is not allowed to change V57A reliability, mass scaling, spectral-anchor
construction, final-label path, or dataset routing.

## 2. Current Code State

Current code contains V57A but no V58A implementation:

```text
core/e2e/sect_coco_e2e.py contains v57a config, loss call, diagnostics,
snapshot keys, anchor construction trigger, and helper implementation.
scripts/run_unified_aptc_9datasets.py contains the V57A default fields and
variant.
No v58a symbols are present in these files before this implementation.
```

Therefore the next code change should be a narrow additive patch.

## 3. Required Insertion Points

### Config

Add to `SECTCoCoConfig`:

```text
v58a_enabled: bool = False
v58a_anchor_weight: float = 0.0
v58a_reliability_floor: float = 0.10
v58a_reliable_threshold: float = 0.20
v58a_min_effective_mass: float = 0.10
v58a_warmup_epochs: int = 20
v58a_ramp_epochs: int = 40
v58a_beta_min: float = 0.35
v58a_beta_max: float = 0.70
v58a_soft_power: float = 0.50
v58a_hybrid_compensation: float = 0.50
v58a_target_mass: float = 0.08
v58a_max_mass_scale: float = 1.50
v58a_max_reliability_cap: float = 0.90
v58a_release_warmup_epochs: int = 20
v58a_release_ramp_epochs: int = 40
v58a_release_hold_until_epoch: int = 80
v58a_release_decay_epochs: int = 60
v58a_release_floor: float = 0.25
```

The V58A inherited constants intentionally duplicate the V57A values so the
runner can disable V57A while enabling V58A as an independently auditable
variant.

### Loss Call

Insert the V58A loss call immediately after the V57A loss call in
`EndToEndSECTCoCoModule.loss`.

Use a new helper:

```text
anchor_release_residual_spectral_anchor_loss(...)
```

The helper may internally call `mass_floor_normalized_residual_spectral_anchor_loss`
to reuse V57A internals. It must pass the V58A inherited constants unchanged and
then multiply the returned V57A-style anchor loss by `release_gamma`.

### Total Loss

Add only:

```text
+ cfg.v58a_anchor_weight * v58a_anchor_loss
```

Do not change the V57A term. The V58A runner must set `v57a_anchor_weight=0.0`
and `v57a_enabled=false`.

### Diagnostics

Add independent `v58a_*` diagnostics. Required minimum:

```text
v58a_enabled
v58a_release_gamma
v58a_release_warmup_epochs
v58a_release_ramp_epochs
v58a_release_hold_until_epoch
v58a_release_decay_epochs
v58a_release_floor
v58a_anchor_loss
v58a_pre_release_anchor_loss
v58a_weighted_q_anchor_kl
v58a_pre_release_weighted_q_anchor_kl
v58a_weighted_q_anchor_agreement
v58a_unweighted_q_anchor_agreement
v58a_embedding_anchor_agreement
v58a_raw_reliability_mean
v58a_mass_scale
v58a_scaled_reliability_mean
v58a_reliability_mean
v58a_reliable_node_ratio
v58a_effective_anchor_mass
v58a_target_mass
v58a_max_mass_scale
v58a_max_reliability_cap
```

For audit convenience, copy the full V57A reliability/mass/anchor diagnostic
surface under `v58a_*` names where practical.

### Snapshot Keys

Add `v58a_*` keys to the snapshot list so epoch 1/40/80 values are written:

```text
v58a_release_gamma
v58a_weighted_q_anchor_agreement
v58a_weighted_q_anchor_kl
v58a_reliability_mean
v58a_mass_scale
v58a_raw_reliability_mean
v58a_scaled_reliability_mean
```

### Anchor Construction Trigger

Extend the V50A spectral-anchor construction condition with:

```text
or bool(getattr(cfg, "v58a_enabled", False))
```

The construction itself must remain unchanged.

### Final Anchor Diagnostics

Extend final anchor diagnostic emission to include `v58a_anchor_acc_diagnostic`,
`v58a_anchor_nmi_diagnostic`, and `v58a_anchor_ari_diagnostic` when V58A is
enabled.

### Runner

In `scripts/run_unified_aptc_9datasets.py`:

```text
add V58A default fields with disabled defaults
add EXPERIMENT_VARIANTS["v58a_anchor_release_residual_compactness"]
```

The V58A variant must explicitly disable V50A-V57A active losses:

```text
v50a_enabled=false
v51a_enabled=false
v52a_enabled=false
v53a_enabled=false
v54a_enabled=false
v55a_enabled=false
v56a_enabled=false
v57a_enabled=false
all corresponding anchor weights = 0.0
v58a_enabled=true
v58a_anchor_weight=0.04
```

## 4. Release Gamma Definition

Use the fixed preregistered schedule:

```text
release_gamma(epoch) =
  0.0                                 if epoch <= 20
  (epoch - 20) / 40                   if 20 < epoch <= 60
  1.0                                 if 60 < epoch <= 80
  max(0.25, 1 - (epoch - 80) / 80)    if 80 < epoch <= 140
  0.25                                if epoch > 140
```

Implementation should use the existing zero-based `current_epoch` convention
consistently with V57A diagnostics. The observable required values are:

```text
epoch_1  -> 0.0
epoch_40 -> 0.5
epoch_80 -> 1.0
```

To match this, compute schedule time as:

```text
epoch_number = current_epoch + 1
```

The release multiplier must be a tensor on the same device/dtype as the loss.

## 5. Helper Strategy

Approved strategy:

```text
Wrap V57A helper output and remap diagnostics from v57a_* to v58a_*.
```

Rationale:

```text
This minimizes semantic drift. V58A is supposed to reuse V57A internals and
change only the time multiplier.
```

Required behavior:

```text
pre_release_loss, base_stats = mass_floor_normalized_residual_spectral_anchor_loss(...)
release_gamma = fixed_schedule(current_epoch + 1)
loss = release_gamma * pre_release_loss
```

Diagnostic mapping:

```text
v58a_pre_release_anchor_loss = base_stats["v57a_anchor_loss"]
v58a_anchor_loss = release_gamma * base_stats["v57a_anchor_loss"]
v58a_pre_release_weighted_q_anchor_kl = base_stats["v57a_weighted_q_anchor_kl"]
v58a_weighted_q_anchor_kl = release_gamma * base_stats["v57a_weighted_q_anchor_kl"]
all reliability/mass/agreement diagnostics remapped without value changes
```

Agreement and reliability diagnostics should not be multiplied by
`release_gamma`; only the actual anchor-loss pressure should be released.

## 6. Red-Line Review

| Requirement | Review |
| --- | --- |
| V57A reliability unchanged | PASS if wrapper calls the existing V57A helper without altering internals. |
| Mass-floor constants unchanged | PASS if V58A runner sets target 0.08, max scale 1.50, cap 0.90. |
| Spectral anchor unchanged | PASS if anchor construction remains `build_spectral_compactness_anchor`. |
| Dataset-agnostic schedule | PASS because release schedule depends only on epoch. |
| No final-label selector | PASS if final output remains existing `q_refined`. |
| No legacy head | PASS if `legacy_head_used=false` remains diagnostic expectation. |
| V50A-V57A inactive in runner | PASS only if all previous variant enabled flags/weights are false/zero. |
| No V58A 260e authorization | PASS; implementation authorizes only connectivity after compile. |

## 7. Authorized Next Action

After this review, the only authorized action is minimal implementation followed
by static verification:

```powershell
conda run -n aaai-e2e-subspace python -m py_compile core\e2e\sect_coco_e2e.py scripts\run_unified_aptc_9datasets.py
```

If compile passes and no residual training process exists, only the following
connectivity run is authorized:

```powershell
conda run --no-capture-output -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v58a_anchor_release_residual_compactness --datasets "acm" --epochs 1 --device cuda --log-level WARNING
```

No V58A mixed-stress run is authorized until `V58A_CONNECTIVITY_VERDICT.md` is
written.

## 8. No-Fabrication Status

This review contains no V58A results. It only authorizes a minimal
implementation consistent with `V58A_PREREGISTRATION.md`.
