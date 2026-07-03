# v47a_posterior_guided_band_resolution Preregistration

This document preregisters the next mechanism after the failure of
`v46a_topology_band_calibration`. It is a design document only. No V47A code has
been implemented and no V47A experiment has been run.

## 1. Motivation

V46A showed that directly penalizing the existing ambiguous-band mask
`hard_ij` is insufficient. It did not collapse mask usage or thresholds, but
ACM/Flickr still failed band and performance gates:

| Dataset | ACC | Emb Gap | Band Mass | Usage Entropy | Threshold Gap | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| ACM | 0.6731 | 0.0017 | 0.5201 | 0.9095 | 0.4284 | FAIL |
| DBLP | 0.6603 | 0.0000 | 0.6835 | 0.7620 | 0.5463 | partial |
| Flickr | 0.3593 | 0.0102 | 0.5083 | 0.8989 | 0.4294 | FAIL |

Conclusion:

```text
hard-band mass needs a resolution direction, not only scalar reduction.
```

The only repeatedly stable semantic signal is the posterior/readout path:

```text
embedding_posterior_gap <= 0.02
```

Therefore V47A tests whether stop-gradient posterior agreement can guide
ambiguous topology edges toward homo, hetero, or defer states without becoming
a selector or post-processing head.

## 2. Version Name

```text
v47a_posterior_guided_band_resolution
```

Core hypothesis:

```text
If hard-band edges are resolved using stop-gradient posterior agreement and
posterior uncertainty, then ambiguous topology mass can move toward meaningful
homo/hetero/defer assignments without failed embedding, frequency-response, or
hard-mass scalar pressure losses.
```

## 3. Hard Prohibitions

V47A must not use:

- dataset-specific module, branch, head, loss, assigner, threshold, or weight
- legacy head
- adaptive selector or post-processing selector
- embedding cosine margin loss
- edge-level overlap margin loss
- v43b-style selective conflict gate
- post-normalized high-pass energy loss
- global or edge-local pre-HP response pressure loss
- direct `mean(hard^2)` band penalty from V46A
- label information
- test-set-driven safety correction
- sweep over weights, quantiles, entropy thresholds, or margins

The following weights must remain zero:

```text
v43b_conflict_margin_weight = 0.0
v43b_band_conflict_weight = 0.0
v43b_highpass_energy_weight = 0.0
ideal_signed_embedding_weight = 0.0
ideal_band_resolution_weight = 0.0
ideal_highpass_energy_weight = 0.0
v44_topology_band_resolution_weight = 0.0
v44_conflict_highpass_corr_weight = 0.0
v44b_pre_hp_corr_weight = 0.0
v45a_edge_freq_weight = 0.0
v45a_band_guard_weight = 0.0
v46a_band_cal_weight = 0.0
v46a_balance_weight = 0.0
v46a_spread_weight = 0.0
partition_spread_weight = 0.0
freq_separation_weight = 0.0
freq_ortho_weight = 0.0
```

Diagnostics from earlier variants may be retained, but must not contribute to
loss.

## 4. Allowed Mechanism

### 4.1 Posterior Agreement Signal

Use the existing refined posterior `q_refined` or final differentiable
posterior already produced by the unified pipeline.

The topology calibration loss must use stop-gradient posterior statistics:

```text
p_i = stopgrad(q_i)
p_j = stopgrad(q_j)
posterior_agreement_ij = dot(p_i, p_j)
posterior_uncertainty_i = entropy(p_i) / log(K)
posterior_uncertainty_j = entropy(p_j) / log(K)
posterior_uncertainty_ij = 0.5 * (posterior_uncertainty_i + posterior_uncertainty_j)
```

No gradient from V47A may flow into posterior logits through these targets.

### 4.2 Unified Quantile Targets

Targets must be graph-adaptive but not dataset-specific. Use quantiles computed
within the current graph/run:

```text
agree_high = quantile(posterior_agreement, 0.70)
agree_low = quantile(posterior_agreement, 0.30)
uncert_high = quantile(posterior_uncertainty, 0.70)
```

Preregistered first constants:

```text
agree_high_quantile = 0.70
agree_low_quantile = 0.30
uncert_high_quantile = 0.70
```

These are fixed for the first smoke and must not be swept.

### 4.3 Hard-Band Resolution Targets

Let:

```text
hard_weight_ij = hard_ij
```

Define target masses:

```text
homo_target_ij =
  1[posterior_agreement_ij >= agree_high]
  * 1[posterior_uncertainty_ij < uncert_high]

hetero_target_ij =
  1[posterior_agreement_ij <= agree_low]
  * 1[posterior_uncertainty_ij < uncert_high]

defer_target_ij =
  1[posterior_uncertainty_ij >= uncert_high]
```

If no target is active for an edge, the edge contributes zero V47A resolution
loss. This avoids forcing medium-confidence edges.

### 4.4 Resolution Loss

Use topology masks:

```text
mask_ij = [homo_ij, hetero_ij, hard_ij]
```

Allowed first loss:

```text
L_resolve =
  mean(
    hard_weight_ij * (
      homo_target_ij   * (-log(homo_ij + eps))
      + hetero_target_ij * (-log(hetero_ij + eps))
      + defer_target_ij  * (-log(hard_ij + eps))
    )
  )
```

Rationale:

- The loss only acts strongly where the current model itself marks an edge as
  ambiguous via `hard_ij`.
- Posterior targets are stop-gradient.
- There is no dataset-specific branch.
- Defer target prevents forcing uncertain edges into false confidence.

### 4.5 Collapse Guard

Retain a diagnostic and weak guard against single-mask collapse:

```text
usage = normalize([mean(homo), mean(hetero), mean(hard)])
usage_entropy = entropy(usage) / log(3)
L_usage = ReLU(usage_entropy_floor - usage_entropy)^2
```

Preregistered first constant:

```text
usage_entropy_floor = 0.60
```

This is a safety guard, not a target that favors any dataset.

## 5. First Implementation Constants

If implemented, use exactly:

```text
v47a_resolution_weight = 0.01
v47a_usage_guard_weight = 0.005
v47a_agree_high_quantile = 0.70
v47a_agree_low_quantile = 0.30
v47a_uncert_high_quantile = 0.70
v47a_usage_entropy_floor = 0.60
v47a_eps = 1e-8
```

No sweep is allowed. If this first configuration fails, stop and write a
verdict.

## 6. Required Diagnostics

Red-line and safety:

```text
legacy_head_used
v43b_enabled
v44_enabled
v44b_enabled
v45a_enabled
v46a_enabled
v47a_enabled
embedding_kmeans_acc
final_acc
embedding_posterior_gap
```

Posterior target diagnostics:

```text
v47a_posterior_agreement_mean
v47a_posterior_agreement_std
v47a_posterior_uncertainty_mean
v47a_agree_high_threshold
v47a_agree_low_threshold
v47a_uncert_high_threshold
v47a_homo_target_mass
v47a_hetero_target_mass
v47a_defer_target_mass
v47a_unassigned_target_mass
```

Resolution diagnostics:

```text
v47a_resolution_loss
v47a_usage_guard_loss
v47a_band_mass
v47a_homo_usage
v47a_hetero_usage
v47a_hard_usage
v47a_usage_entropy
```

Retained diagnostics, diagnostic only:

```text
v44b_pre_hp_response_std
v45a_edge_response_gap
v45a_edge_response_corr
v46a_band_mass
```

## 7. First-Stage Experiment

Only after implementation sanity checks:

```text
datasets = acm,dblp,flickr
epochs = 80
seed = 42
device = cuda
```

Command template:

```powershell
conda run -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v47a_posterior_guided_band_resolution --datasets "acm,dblp,flickr" --epochs 80 --device cuda --log-level WARNING
```

No second-batch smoke is allowed unless all first-stage gates pass.

## 8. First-Stage Gates

### 8.1 Red-Line Gate

Must pass:

```text
legacy_head_used=false
v43b_enabled=false
v44_enabled=false
v44b_enabled=false
v45a_enabled=false
v46a_enabled=false
v47a_enabled=true
no selector / no post-processing selector
no frequency-response pressure loss
posterior targets stop-gradient
```

### 8.2 Target Non-Degeneracy Gate

Must pass on 3/3:

```text
v47a_homo_target_mass > 0
v47a_hetero_target_mass > 0
v47a_defer_target_mass > 0
v47a_unassigned_target_mass < 0.80
```

If target masses collapse to one class or mostly unassigned, stop.

### 8.3 Band Gate

Must pass on 3/3:

```text
ACM band_mass <= 0.4991
DBLP band_mass <= 0.6877
Flickr band_mass <= 0.5051
```

At least 2/3 must improve against V46A band mass:

```text
ACM reference = 0.5201
DBLP reference = 0.6835
Flickr reference = 0.5083
```

These references are evaluation gates only and must not be used inside
training.

### 8.4 Posterior/Readout Safety Gate

Must pass on 3/3:

```text
abs(embedding_posterior_gap) <= 0.02
```

### 8.5 Performance Gate

Must pass on 3/3:

```text
ACM ACC >= 0.80
DBLP ACC >= 0.645
Flickr ACC >= 0.45
```

If target and band gates pass but performance fails, record mechanistic partial
pass and stop.

## 9. Stop Conditions

Stop immediately after first-stage smoke if any occurs:

- any red-line violation
- target mass degeneracy
- band gate failure
- `abs(embedding_posterior_gap) > 0.02`
- ACM ACC remains below `0.80`
- DBLP ACC falls below `0.645`
- Flickr ACC remains below `0.45`

Do not run:

- second-batch smoke
- full 9-dataset smoke
- 260-epoch full run
- weight sweep
- quantile sweep
- entropy-floor sweep

## 10. Result Templates

### 10.1 First-Stage Gate Table

| Dataset | ACC | Emb Gap | Band | Homo Tgt | Hetero Tgt | Defer Tgt | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ACM | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| DBLP | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Flickr | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### 10.2 Mechanism Comparison

| Dataset | V46A Band | V47A Band | V46A ACC | V47A ACC | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| ACM | 0.5201 | TBD | 0.6731 | TBD | TBD |
| DBLP | 0.6835 | TBD | 0.6603 | TBD | TBD |
| Flickr | 0.5083 | TBD | 0.3593 | TBD | TBD |

## 11. No-Fabrication Status

All cited V46A values come from `V46A_FIRST_SMOKE_VERDICT.md` and diagnostics.
V47A has not been implemented or run. All V47A results are `TBD`.
