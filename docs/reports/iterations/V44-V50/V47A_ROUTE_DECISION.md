# V47A Route Decision After V46A Failure

This document records the route decision after the preregistered failure of
`v46a_topology_band_calibration`. It is a mechanism-design document only. It
does not change code, run experiments, or report new unrecorded results.

## 1. Evidence Chain

The last four mechanism attempts establish a clear failure boundary:

| Version | Main Target | Key Outcome |
| --- | --- | --- |
| V43B | direct conflict/embedding margin pressure | broad activation, saturated violations, embedding damaged |
| V44B | node-level pre-HP response correlation | measurable response, but ACM anti-aligned |
| V45A | edge-local pre-HP response gap | edge gap/corr negative on 3/3 |
| V46A | direct hard-band mass penalty | no collapse, but ACM/Flickr band and performance failed |

Latest V46A first-smoke result:

| Dataset | ACC | Emb Gap | Band Mass | Usage Entropy | Threshold Gap | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| ACM | 0.6731 | 0.0017 | 0.5201 | 0.9095 | 0.4284 | FAIL |
| DBLP | 0.6603 | 0.0000 | 0.6835 | 0.7620 | 0.5463 | partial ACC/band pass |
| Flickr | 0.3593 | 0.0102 | 0.5083 | 0.8989 | 0.4294 | FAIL |

Stable observations:

- `embedding_posterior_gap` remains safe.
- Pre-HP response remains useful as a diagnostic, not as a loss target.
- The mask system does not collapse under V46A; `usage_entropy` and
  `threshold_gap` pass.
- Direct `mean(hard^2)` pressure does not resolve ACM/Flickr hard band.

## 2. Scientific Conclusion

The unresolved problem is not simply "too much hard mass".

V46A shows that hard-band mass cannot be safely reduced by scalar pressure
alone. The real missing mechanism is an assignment rule for where ambiguous
edges should go:

```text
hard edge -> homo side
hard edge -> hetero side
hard edge -> remain unresolved
```

Earlier variants tried to infer this using embedding margins, frequency
response, or raw band pressure. All failed. Therefore V47A should move from
"penalize ambiguity" to "calibrate ambiguity resolution using semantic
posterior consistency".

## 3. Rejected Continuations

Do not continue:

- V46A second-batch smoke
- V46A full run
- V46A weight sweep
- stronger `mean(hard^2)`
- entropy-floor or threshold-gap sweep
- another response-pressure loss based on `pre_hp_response`
- another edge-local boundary-vs-safe response target

Do not create V46A soft/strict/high/low variants.

## 4. Candidate Routes

### Candidate A: Posterior-Guided Hard-Band Resolution

Name:

```text
v47a_posterior_guided_band_resolution
```

Core idea:

Use posterior agreement as a semantic calibration signal for hard-band edges.
If two endpoints have high posterior agreement, their hard mass should be
resolved toward `homo`; if agreement is low and uncertainty is low, hard mass
may resolve toward `hetero`; if posterior is uncertain, the edge may remain
hard.

This is not a post-processing selector because the topology masks remain in
the unified differentiable frontend and no dataset-specific branch is used.

Why it fits the evidence:

- `embedding_posterior_gap` has repeatedly stayed near zero.
- Direct frequency response pressure is rejected.
- Direct hard penalty lacks a destination for ambiguous mass.

Main risk:

- Posterior self-confirmation. The posterior signal must be stop-gradient and
  must not become a shortcut head.

### Candidate B: Teacher-Free Confidence Calibration

Name:

```text
v47a_confidence_distribution_calibration
```

Core idea:

Calibrate the score distribution and thresholds without using posterior
agreement. Use shape constraints on score entropy, threshold spread, and mask
usage transitions.

Why it fits:

- Avoids posterior circularity.
- Stays entirely inside topology contraction.

Risk:

- May repeat V46A's scalar-pressure failure if it still lacks semantic
  direction.

### Candidate C: Stop Frontend Loss Search

Name:

```text
v47a_diagnostic_recenter
```

Core idea:

Stop adding frontend losses. Retain diagnostics and recenter on the strongest
safe baseline for a broader architectural rethink.

Risk:

- Conservative and may not advance the core novelty enough.

## 5. Recommended Route

Recommended:

```text
v47a_posterior_guided_band_resolution
```

Reason:

V46A proves that ambiguity needs a direction, not only a penalty. The only
stable non-broken semantic signal across V43B-V46A is the posterior/readout
path: it stays aligned with embedding and does not explain the failures.

The next mechanism should therefore ask:

```text
Can stop-gradient posterior agreement tell the topology contraction how to
resolve hard-band mass without becoming a selector or post-processing head?
```

## 6. Required Design Principles For V47A

V47A must:

- use one unified loss for all datasets
- keep posterior signals stop-gradient in the topology calibration loss
- avoid dataset-specific thresholds
- avoid legacy head / selector / post-processing
- keep pre-HP and V45A edge response diagnostics diagnostic-only
- disable V43B/V44/V44B/V45A/V46A failed losses
- not optimize `overlap_gap`
- not use labels

V47A must not directly punish embedding cosine distance or use edge-level
margin separation.

## 7. Proposed First Mechanism Sketch

Let:

```text
posterior_agreement_ij = stopgrad(dot(q_i, q_j))
posterior_uncertainty_ij = stopgrad(0.5 * (entropy(q_i) + entropy(q_j)))
hard_weight_ij = hard_ij
```

Define soft targets:

```text
homo_target_ij = high posterior agreement and low uncertainty
hetero_target_ij = low posterior agreement and low uncertainty
defer_target_ij = high uncertainty
```

Allowed first loss family:

```text
L_hard_resolution =
  hard_weight * (
    homo_target * CE(mask, homo)
    + hetero_target * CE(mask, hetero)
    + defer_target * CE(mask, hard)
  )
```

The exact target formulas and thresholds must be preregistered before
implementation.

## 8. First-Stage Gate Shape

Datasets remain:

```text
ACM, DBLP, Flickr
```

Required:

- red-line pass
- no revived failed losses
- `abs(embedding_posterior_gap) <= 0.02`
- band safety:

```text
ACM band_mass <= 0.4991
DBLP band_mass <= 0.6877
Flickr band_mass <= 0.5051
```

- performance:

```text
ACM ACC >= 0.80
DBLP ACC >= 0.645
Flickr ACC >= 0.45
```

Additional V47A-specific diagnostics should include:

```text
posterior_agreement_mean
posterior_agreement_std
posterior_uncertainty_mean
hard_to_homo_target_mass
hard_to_hetero_target_mass
hard_defer_target_mass
hard_resolution_loss
```

## 9. Next Step

Next owner should be:

```text
ccf-experiment-designer
```

to write a full `V47A_PREREGISTRATION.md` before implementation. Do not
implement V47A from this route note alone.

## 10. No-Fabrication Status

All numbers cited here come from existing V46A verdict and diagnostics. V47A
has not been implemented or run. All V47A results are `TBD`.
