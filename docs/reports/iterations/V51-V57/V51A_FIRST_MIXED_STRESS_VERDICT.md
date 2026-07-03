# v51a_reliability_gated_spectral_anchor First Mixed-Stress Verdict

This file records the preregistered first-stage mixed-stress result for
`v51a_reliability_gated_spectral_anchor`. It follows
`V51A_PREREGISTRATION.md` and `V51A_CONNECTIVITY_VERDICT.md`.

No full 9-dataset smoke, 260-epoch full run, reliability threshold sweep,
formula variant, or V50A anchor hyperparameter sweep is authorized by this
verdict.

## 1. Run

Prerequisite artifacts:

```text
V51A_PREREGISTRATION.md
V51A_IMPLEMENTATION_REVIEW.md
V51A_CONNECTIVITY_VERDICT.md
```

Static check before running:

```powershell
conda run -n aaai-e2e-subspace python -m py_compile core\e2e\sect_coco_e2e.py scripts\run_unified_aptc_9datasets.py
```

Verdict:

```text
PASS
```

Mixed-stress command:

```powershell
conda run --no-capture-output -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v51a_reliability_gated_spectral_anchor --datasets "acm,dblp,flickr,texas,squirrel,chameleon" --epochs 80 --device cuda --log-level WARNING
```

Run status:

```text
6/6 datasets completed with status=ok.
No duplicate mixed-stress run was used.
No V51A threshold, reliability formula, V50A anchor hyperparameter, seed, or
final-label change was made.
```

Output files:

```text
results/archive/v51-v57/unified_aptc_9datasets_v51a_reliability_gated_spectral_anchor.csv
results/archive/v51-v57/unified_aptc_9datasets_v51a_reliability_gated_spectral_anchor_diagnostics.jsonl
```

## 2. Result Summary

Latest complete mixed-stress records:

| Dataset | ACC | NMI | ARI | Emb Gap |
| --- | ---: | ---: | ---: | ---: |
| ACM | 0.6846 | 0.3240 | 0.3505 | -0.0003 |
| DBLP | 0.6520 | 0.3581 | 0.2965 | 0.0000 |
| Flickr | 0.3637 | 0.2133 | 0.1379 | 0.0000 |
| Texas | 0.7322 | 0.4807 | 0.5948 | 0.0055 |
| Squirrel | 0.3021 | 0.0641 | 0.0516 | 0.0000 |
| Chameleon | 0.3303 | 0.1415 | 0.0543 | -0.0312 |

Performance interpretation:

- ACM collapses far below the V50A preservation floor.
- DBLP is slightly below the preregistered preservation floor.
- Squirrel's hard posterior/readout safety failure is removed, but this is not
  enough because the reliability gate effectively turns the anchor off.
- Texas remains strong, consistent with the earlier observation that the base
  model can do well there without trusting the spectral anchor.

## 3. Red-Line Gate

Preregistered requirement:

```text
status=ok
legacy_head_used=false
v43b/v44/v44b/v45a/v46a/v47a/v48a/v49a_enabled=false
v51a_enabled=true
no selector / no post-processing selector
```

Verdict:

```text
PASS on 6/6.
```

| Dataset | Legacy | v43b | v44 | v44b | v45a | v46a | v47a | v48a | v49a | v50a | v51a |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ACM | false | false | false | false | false | false | false | false | false | false | true |
| DBLP | false | false | false | false | false | false | false | false | false | false | true |
| Flickr | false | false | false | false | false | false | false | false | false | false | true |
| Texas | false | false | false | false | false | false | false | false | false | false | true |
| Squirrel | false | false | false | false | false | false | false | false | false | false | true |
| Chameleon | false | false | false | false | false | false | false | false | false | false | true |

V51A did not violate the unified-pipeline or final-label red lines.

## 4. Reliability Non-Collapse Gate

Preregistered requirement:

```text
Pass on at least 5/6:
0.10 <= v51a_reliability_mean <= 0.90
v51a_reliable_node_ratio >= 0.10
v51a_effective_anchor_mass >= 0.10

Hard fail:
v51a_reliability_mean < 0.03 on any dataset
v51a_reliability_mean > 0.97 on any dataset
```

Verdict:

```text
FAIL on 0/6.
Hard fail on 6/6 because v51a_reliability_mean < 0.03.
```

| Dataset | Rel Mean | Rel P50 | Rel P90 | Reliable Ratio | Effective Mass | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| ACM | 0.0054 | 0.0032 | 0.0142 | 0.0000 | 0.0054 | FAIL |
| DBLP | 0.0020 | 0.0000 | 0.0061 | 0.0000 | 0.0020 | FAIL |
| Flickr | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | FAIL |
| Texas | 0.0024 | 0.0004 | 0.0061 | 0.0000 | 0.0024 | FAIL |
| Squirrel | 0.0029 | 0.0000 | 0.0000 | 0.0038 | 0.0029 | FAIL |
| Chameleon | 0.0176 | 0.0000 | 0.0809 | 0.0162 | 0.0176 | FAIL |

Component means show why the collapse happens:

| Dataset | Conf | q-anchor | embed-anchor | local |
| --- | ---: | ---: | ---: | ---: |
| ACM | 0.3011 | 0.0753 | 0.0315 | 0.0913 |
| DBLP | 0.2268 | 0.0392 | 0.0263 | 0.0510 |
| Flickr | 0.0264 | 0.0020 | 0.0000 | 0.0021 |
| Texas | 0.1917 | 0.0667 | 0.0544 | 0.0261 |
| Squirrel | 0.3316 | 0.0155 | 0.0194 | 0.0857 |
| Chameleon | 0.6298 | 0.0479 | 0.0745 | 0.2715 |

The multiplicative reliability formula is too conservative under the current
posterior/readout state. It suppresses the anchor even when the anchor itself is
strong, especially on ACM and DBLP.

## 5. Anchor Safety Gate

Preregistered requirement:

```text
abs(embedding_posterior_gap) <= 0.04 on 6/6
no dataset abs(embedding_posterior_gap) > 0.08
Squirrel must no longer have abs(embedding_posterior_gap) > 0.08
```

Verdict:

```text
PASS on 6/6.
```

| Dataset | Emb Gap | Safety |
| --- | ---: | --- |
| ACM | -0.0003 | PASS |
| DBLP | 0.0000 | PASS |
| Flickr | 0.0000 | PASS |
| Texas | 0.0055 | PASS |
| Squirrel | 0.0000 | PASS |
| Chameleon | -0.0312 | PASS |

Interpretation:

```text
V51A removes the Squirrel hard safety failure, but it does so while nearly
removing anchor supervision. This is safety by anchor avoidance, not a useful
reliable-anchor mechanism.
```

## 6. Anchor Usefulness Gate

Preregistered requirements:

```text
v51a_weighted_q_anchor_agreement_epoch_80 >
v51a_weighted_q_anchor_agreement_epoch_1
Pass on at least 4/6.

ACM ACC >= 0.8888
DBLP ACC >= 0.6610
```

Weighted agreement movement:

| Dataset | Weighted Agreement @1 | @40 | @80 | Movement | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| ACM | 0.0133 | 0.0408 | 0.0536 | +0.0403 | PASS |
| DBLP | 0.0095 | 0.0212 | 0.0193 | +0.0098 | PASS |
| Flickr | 0.0000 | 0.0000 | 0.0000 | 0.0000 | FAIL |
| Texas | 0.0176 | 0.0200 | 0.0216 | +0.0040 | PASS |
| Squirrel | 0.0240 | 0.0284 | 0.0291 | +0.0051 | PASS |
| Chameleon | 0.1487 | 0.1665 | 0.1726 | +0.0239 | PASS |

Agreement movement passes on 5/6, but it is not meaningful because effective
anchor mass is far below 0.10 on every dataset.

Preservation floors:

| Dataset | V51A ACC | Required Floor | Gate |
| --- | ---: | ---: | --- |
| ACM | 0.6846 | 0.8888 | FAIL |
| DBLP | 0.6520 | 0.6610 | FAIL |

Verdict:

```text
FAIL.
```

The ACM failure is decisive. V51A does not preserve the proven V50A rescue
signal.

## 7. Heterophily Stress Gate

Preregistered requirement for Texas, Squirrel, and Chameleon:

```text
At least 2/3 must satisfy:
abs(embedding_posterior_gap) <= 0.04
v51a_reliability_mean within [0.10, 0.90]
v51a_effective_anchor_mass >= 0.10
```

| Dataset | Emb Gap | Rel Mean | Effective Mass | Gate |
| --- | ---: | ---: | ---: | --- |
| Texas | 0.0055 | 0.0024 | 0.0024 | FAIL |
| Squirrel | 0.0000 | 0.0029 | 0.0029 | FAIL |
| Chameleon | -0.0312 | 0.0176 | 0.0176 | FAIL |

Verdict:

```text
FAIL on 0/3.
```

All three pass posterior/readout safety, but none passes reliability
non-collapse. This means V51A does not solve heterophily stress; it avoids the
anchor under stress.

## 8. Gate Summary

| Gate | Verdict |
| --- | --- |
| Red-line | PASS |
| Reliability non-collapse | FAIL, hard |
| Posterior/readout safety | PASS |
| Anchor usefulness | FAIL |
| Heterophily stress | FAIL |

Decision:

```text
STOP.
```

This verdict does not authorize:

- full 9-dataset smoke;
- 260-epoch full run;
- reliability threshold sweep;
- V51A reliability formula variants;
- V50A anchor weight, temperature, rank, filter-step, or refresh sweep.

## 9. Scientific Interpretation

V51A answered the central question negatively in its current form:

```text
The preregistered multiplicative reliability gate can prevent unsafe anchor
imitation, but it is too conservative to transfer the spectral compactness
signal.
```

The important distinction:

```text
V51A did not fail because the spectral anchor collapsed. It failed because the
reliability gate collapsed.
```

This is visible on ACM:

| Signal | Value |
| --- | ---: |
| Anchor ACC | 0.8942 |
| Reliability mean | 0.0054 |
| Reliable node ratio | 0.0000 |
| V51A ACC | 0.6846 |
| Required ACM floor | 0.8888 |

ACM is the strongest evidence that the V51A gate is rejecting useful anchor
signal, not merely filtering unsafe heterophily nodes.

## 10. Rescue Implication

The next route should not tune V51A thresholds or formula constants. The failure
is structural:

```text
Using current q-refined and q-embed agreement as mandatory multiplicative
preconditions makes the anchor unavailable precisely when the anchor is needed
to shape the posterior.
```

Allowed next thinking should be a new preregistered route, not a V51A sweep.

Recommended next analysis artifact:

```text
V51A_RELIABILITY_COLLAPSE_ANALYSIS.md
```

It should decide whether the next rescue should be:

```text
v52a_warm_started_reliability_anchor
```

or a different mechanism with a nonzero anchor curriculum. The key design
question is:

```text
How can the model receive enough early spectral compactness signal to become
anchor-compatible, without blindly trusting the anchor on heterophily-style
nodes?
```

Candidate directions to analyze, not implement yet:

- a preregistered reliability warm-up that excludes q-anchor agreement for the
  first stage and then activates it;
- an additive rather than fully multiplicative reliability model with a hard
  effective-mass guard;
- a two-term objective separating anchor compactness from assignment imitation;
- a stop-gradient curriculum based on anchor confidence and local consistency
  before posterior agreement is required.

These are not authorized implementations. They require a new preregistration.

## 11. No-Fabrication Status

All numbers in this document come from the local V51A mixed-stress run and its
diagnostics:

```text
results/archive/v51-v57/unified_aptc_9datasets_v51a_reliability_gated_spectral_anchor.csv
results/archive/v51-v57/unified_aptc_9datasets_v51a_reliability_gated_spectral_anchor_diagnostics.jsonl
```

No V51A full-run result exists.
