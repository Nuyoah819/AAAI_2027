# V47A Implementation-Readiness Review

This file reviews whether `v47a_posterior_guided_band_resolution` is ready for
minimal implementation. It is an implementation-readiness note only. No V47A
code has been implemented and no V47A experiment has been run.

## 1. Reviewed Inputs

Documents reviewed:

- `V47A_ROUTE_DECISION.md`
- `V47A_PREREGISTRATION.md`
- `V46A_FIRST_SMOKE_VERDICT.md`
- `V45A_FIRST_SMOKE_VERDICT.md`

Code reviewed:

- `core/e2e/sect_coco_e2e.py`
- `scripts/run_unified_aptc_9datasets.py`

## 2. Current Status

V47A is implementation-ready with one required clarification:

```text
Target non-degeneracy should be judged on hard-weighted effective target mass,
not only raw all-edge target mass.
```

Reason: the preregistered loss is multiplied by `hard_ij`. Raw posterior
quantile targets may be nonzero on all candidate edges while contributing
almost nothing inside the current hard band. The first implementation should
therefore report both:

```text
raw target mass      = mean(target)
effective target mass = mean(hard * target) / mean(hard)
```

The preregistered gate fields should use effective hard-weighted mass because
V47A is specifically a hard-band resolution mechanism.

## 3. Posterior Tensor Availability

The required posterior is already available. The best implementation source is:

```text
out["q_refined"]
```

Important detail:

- Inside `AdaptivePosteriorTransportHead.forward`, `q_refined` is produced after
  transport and topology refinement.
- Inside `EndToEndSECTCoCoModule._aptc_pass`, the returned `out["q_refined"]`
  may be the final main posterior after `main_posterior_mode` and optional flow
  blending.
- The raw APTC refined posterior is also retained as:

```text
out["q_aptc_raw"]
```

V47A should use `out["q_refined"]`, matching the preregistration phrase
"existing refined posterior or final differentiable posterior already produced
by the unified pipeline."

## 4. Stop-Gradient Insertion Point

The stop-gradient requirement can be satisfied inside a new regularizer:

```text
q = q_posterior.detach()
```

The new regularizer should receive:

```text
q_posterior=out["q_refined"]
edge_index=self.edge_index
homo=out["homo"]
hetero=out["hetero"]
hard=out["hard"]
```

All posterior-derived quantities must be computed from the detached `q`:

```text
p_i = q.detach()[src]
p_j = q.detach()[dst]
posterior_agreement = (p_i * p_j).sum(dim=1)
posterior_uncertainty = 0.5 * (H(p_i) + H(p_j)) / log(K)
```

Gradients from V47A may flow only into the topology masks
`homo`, `hetero`, and `hard`, not into posterior logits through target
construction.

## 5. Quantile Target Feasibility

The required quantiles can be computed directly with `torch.quantile` over the
candidate-edge tensors:

```text
agree_high = quantile(posterior_agreement, 0.70)
agree_low = quantile(posterior_agreement, 0.30)
uncert_high = quantile(posterior_uncertainty, 0.70)
```

This is graph-adaptive and does not introduce dataset-specific thresholds.

Risk:

```text
agreement and uncertainty can overlap in a way that makes raw targets nonzero
but effective hard-band targets tiny.
```

Required mitigation:

- Keep preregistered raw thresholds unchanged.
- Do not sweep quantiles.
- Add effective hard-weighted diagnostics.
- Stop immediately if effective target masses fail the target gate.

## 6. Loss Feasibility

The preregistered loss can be implemented without a new head or selector:

```text
L_resolve =
  mean(
    hard * (
      homo_target   * -log(homo + eps)
      + hetero_target * -log(hetero + eps)
      + defer_target  * -log(hard + eps)
    )
  )
```

The loss remains inside the differentiable topology frontend because it acts on
the existing masks produced by `DifferentiableTopologyContraction`.

The usage guard is equivalent in shape to V46A's usage entropy guard but does
not include V46A's direct `mean(hard^2)` band penalty.

## 7. Red-Line Compatibility

The existing V46A variant shows the right red-line pattern to reuse:

```text
aptc_local_teacher = False
v43b_* weights = 0.0
ideal_* weights = 0.0
v44_* weights = 0.0
v44b_pre_hp_corr_weight = 0.0
v45a_* weights = 0.0
v46a_* weights = 0.0
partition_spread_weight = 0.0
freq_separation_weight = 0.0
freq_ortho_weight = 0.0
```

V47A should additionally expose:

```text
v47a_enabled = true
```

when either `v47a_resolution_weight` or `v47a_usage_guard_weight` is positive.

## 8. Required Code Touch Points

Minimal code changes should be limited to:

1. Add V47A config fields to `E2ESECTCoCoConfig`.
2. Add `v47a_posterior_guided_band_resolution_regularizer`.
3. Call the regularizer in `EndToEndSECTCoCoModule.loss`.
4. Add V47A diagnostics to the loss diagnostics dict.
5. Add the new experiment variant in `scripts/run_unified_aptc_9datasets.py`.

No data loader, metric, post-processing, or legacy head code needs to change.

## 9. Implementation Risks

### Risk A: Posterior Self-Confirmation

Mitigation:

```text
detach q_refined before every posterior target computation.
```

### Risk B: Target-Mass Mirage

Raw quantile targets can look healthy while hard-weighted active targets are
near zero.

Mitigation:

```text
report both raw and effective hard-weighted target masses;
use effective masses for the non-degeneracy gate.
```

### Risk C: Defer Target Reinforces Hard Band

The defer target explicitly rewards `hard` on uncertain edges. This is intended
as a safety valve, but it can oppose the band gate.

Mitigation:

```text
keep the preregistered weight small and do not sweep it;
judge by first-stage smoke only.
```

### Risk D: Existing Baseline Losses Still Dominate

V47A is a small additive frontend calibration. Existing default losses still
dominate total training.

Mitigation:

```text
do not change baseline weights in V47A;
only compare against the preregistered V46A references.
```

## 10. Readiness Verdict

Proceed to a minimal implementation plan.

Do not run V47A until all of the following are true:

- new variant is registered
- py_compile passes
- 1-epoch CPU connectivity check passes
- red-line fields report failed prior variants disabled
- V47A diagnostics are present in JSONL

V47A remains unimplemented and unrun at the time of this review.
