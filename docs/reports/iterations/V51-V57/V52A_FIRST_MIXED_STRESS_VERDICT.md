# v52a_curriculum_reliability_spectral_anchor First Mixed-Stress Verdict

This file records the preregistered first-stage mixed-stress result for
`v52a_curriculum_reliability_spectral_anchor`. It follows
`V52A_PREREGISTRATION.md` and `V52A_CONNECTIVITY_VERDICT.md`.

No full 9-dataset smoke, 260-epoch full run, schedule variant, reliability
formula variant, threshold sweep, or V50A anchor hyperparameter sweep is
authorized by this verdict.

## 1. Run

Prerequisite artifacts:

```text
V52A_PREREGISTRATION.md
V52A_IMPLEMENTATION_REVIEW.md
V52A_CONNECTIVITY_VERDICT.md
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
conda run --no-capture-output -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v52a_curriculum_reliability_spectral_anchor --datasets "acm,dblp,flickr,texas,squirrel,chameleon" --epochs 80 --device cuda --log-level WARNING
```

Run status:

```text
6/6 datasets completed with status=ok.
No duplicate mixed-stress run was used.
No V52A schedule, reliability formula, threshold, V50A anchor hyperparameter,
seed, or final-label change was made.
```

Output files:

```text
results/archive/v51-v57/unified_aptc_9datasets_v52a_curriculum_reliability_spectral_anchor.csv
results/archive/v51-v57/unified_aptc_9datasets_v52a_curriculum_reliability_spectral_anchor_diagnostics.jsonl
```

## 2. Result Summary

Latest complete mixed-stress records:

| Dataset | ACC | NMI | ARI | Emb Gap |
| --- | ---: | ---: | ---: | ---: |
| ACM | 0.9041 | 0.6798 | 0.7362 | 0.0000 |
| DBLP | 0.6867 | 0.4090 | 0.3489 | 0.0002 |
| Flickr | 0.3733 | 0.2154 | 0.1453 | -0.0086 |
| Texas | 0.7377 | 0.4944 | 0.6109 | 0.0000 |
| Squirrel | 0.3003 | 0.0611 | 0.0498 | 0.0015 |
| Chameleon | 0.3329 | 0.1421 | 0.0518 | 0.0127 |

Performance interpretation:

- ACM and DBLP pass the preregistered preservation floors.
- Texas remains strong and slightly improves over V50A/V51A.
- Squirrel's V50A hard posterior/readout failure remains fixed.
- Flickr, Squirrel, and Chameleon remain weak in absolute performance.

## 3. Red-Line Gate

Preregistered requirement:

```text
status=ok
legacy_head_used=false
v43b/v44/v44b/v45a/v46a/v47a/v48a/v49a_enabled=false
v50a_enabled=false
v51a_enabled=false
v52a_enabled=true
no selector / no post-processing selector
```

Verdict:

```text
PASS on 6/6.
```

| Dataset | Legacy | v43b | v44 | v44b | v45a | v46a | v47a | v48a | v49a | v50a | v51a | v52a |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ACM | false | false | false | false | false | false | false | false | false | false | false | true |
| DBLP | false | false | false | false | false | false | false | false | false | false | false | true |
| Flickr | false | false | false | false | false | false | false | false | false | false | false | true |
| Texas | false | false | false | false | false | false | false | false | false | false | false | true |
| Squirrel | false | false | false | false | false | false | false | false | false | false | false | true |
| Chameleon | false | false | false | false | false | false | false | false | false | false | false | true |

V52A did not violate the unified-pipeline or final-label red lines.

## 4. Curriculum Gate

Preregistered requirement:

```text
v52a_gamma_epoch_1 = 0
v52a_gamma_epoch_40 = 0.5
v52a_gamma_epoch_80 = 1
```

Verdict:

```text
PASS on 6/6.
```

| Dataset | Gamma @1 | Gamma @20 | Gamma @40 | Gamma @60 | Gamma @80 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ACM | 0.0000 | 0.0000 | 0.5000 | 1.0000 | 1.0000 |
| DBLP | 0.0000 | 0.0000 | 0.5000 | 1.0000 | 1.0000 |
| Flickr | 0.0000 | 0.0000 | 0.5000 | 1.0000 | 1.0000 |
| Texas | 0.0000 | 0.0000 | 0.5000 | 1.0000 | 1.0000 |
| Squirrel | 0.0000 | 0.0000 | 0.5000 | 1.0000 | 1.0000 |
| Chameleon | 0.0000 | 0.0000 | 0.5000 | 1.0000 | 1.0000 |

The schedule was implemented as preregistered. The later failure is not a
schedule-mismatch bug.

## 5. Reliability Non-Collapse Gate

Preregistered requirement:

```text
Pass on at least 5/6:
0.08 <= v52a_reliability_mean <= 0.90
v52a_reliable_node_ratio >= 0.05
v52a_effective_anchor_mass >= 0.08

Hard fail:
v52a_reliability_mean < 0.03 on ACM, DBLP, Texas, Squirrel, or Chameleon
v52a_reliability_mean > 0.97 on any dataset
```

Verdict:

```text
FAIL on 0/6.
Hard fail on ACM, DBLP, Texas, and Squirrel.
```

| Dataset | Rel Mean | Rel P50 | Rel P90 | Reliable Ratio | Effective Mass | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| ACM | 0.0197 | 0.0178 | 0.0395 | 0.0000 | 0.0197 | HARD FAIL |
| DBLP | 0.0055 | 0.0006 | 0.0150 | 0.0000 | 0.0055 | HARD FAIL |
| Flickr | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | FAIL, weak-anchor case |
| Texas | 0.0075 | 0.0019 | 0.0191 | 0.0000 | 0.0075 | HARD FAIL |
| Squirrel | 0.0068 | 0.0000 | 0.0147 | 0.0060 | 0.0068 | HARD FAIL |
| Chameleon | 0.0376 | 0.0112 | 0.1081 | 0.0531 | 0.0376 | FAIL |

Flickr qualifies as weak-anchor non-use rather than a decisive hard fail:

```text
anchor confidence = 0.0264
local consistency = 0.0021
anchor ACC = 0.3626
```

But the global reliability gate still fails because aligned and heterophily
stress datasets also collapse.

## 6. Base-Vs-Agreement Reliability

| Dataset | Rel @1 | Rel @40 | Rel @80 | Base @1 | Agreement @80 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ACM | 0.1962 | 0.1051 | 0.0197 | 0.1962 | 0.0861 |
| DBLP | 0.1389 | 0.0725 | 0.0055 | 0.1389 | 0.0345 |
| Flickr | 0.0142 | 0.0071 | 0.0000 | 0.0142 | 0.0010 |
| Texas | 0.1089 | 0.0580 | 0.0075 | 0.1089 | 0.0606 |
| Squirrel | 0.2086 | 0.1074 | 0.0068 | 0.2086 | 0.0189 |
| Chameleon | 0.4507 | 0.2422 | 0.0376 | 0.4507 | 0.0809 |

Interpretation:

```text
V52A fixed V51A's early all-off problem, but when gamma reaches 1 the late
agreement gate becomes too suppressive. The failure moved from early reliability
collapse to late reliability collapse.
```

## 7. Anchor Safety Gate

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
| ACM | 0.0000 | PASS |
| DBLP | 0.0002 | PASS |
| Flickr | -0.0086 | PASS |
| Texas | 0.0000 | PASS |
| Squirrel | 0.0015 | PASS |
| Chameleon | 0.0127 | PASS |

The safety benefit from V51A is preserved.

## 8. Anchor Usefulness Gate

Preregistered requirements:

```text
v52a_weighted_q_anchor_agreement_epoch_80 >
v52a_weighted_q_anchor_agreement_epoch_1
Pass on at least 4/6.

ACM ACC >= 0.8888
DBLP ACC >= 0.6610
```

Weighted agreement movement:

| Dataset | Weighted Agreement @1 | @40 | @80 | Movement | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| ACM | 0.4358 | 0.9337 | 0.1957 | -0.2401 | FAIL |
| DBLP | 0.3128 | 0.3948 | 0.0540 | -0.2588 | FAIL |
| Flickr | 0.0162 | 0.0085 | 0.0001 | -0.0161 | FAIL |
| Texas | 0.3893 | 0.2456 | 0.0660 | -0.3233 | FAIL |
| Squirrel | 0.1220 | 0.1776 | 0.0540 | -0.0680 | FAIL |
| Chameleon | 0.1756 | 0.3359 | 0.3261 | +0.1505 | PASS |

Weighted agreement movement passes on only 1/6.

Preservation floors:

| Dataset | V52A ACC | Required Floor | Gate |
| --- | ---: | ---: | --- |
| ACM | 0.9041 | 0.8888 | PASS |
| DBLP | 0.6867 | 0.6610 | PASS |

Verdict:

```text
FAIL overall.
```

V52A preserves ACM/DBLP performance, but it does not preserve a stable
weighted anchor-coupling trajectory.

## 9. Heterophily Stress Gate

Preregistered requirement for Texas, Squirrel, and Chameleon:

```text
At least 2/3 must satisfy:
abs(embedding_posterior_gap) <= 0.04
v52a_reliability_mean within [0.08, 0.90]
v52a_effective_anchor_mass >= 0.08
```

| Dataset | Emb Gap | Rel Mean | Effective Mass | Gate |
| --- | ---: | ---: | ---: | --- |
| Texas | 0.0000 | 0.0075 | 0.0075 | FAIL |
| Squirrel | 0.0015 | 0.0068 | 0.0068 | FAIL |
| Chameleon | 0.0127 | 0.0376 | 0.0376 | FAIL |

Verdict:

```text
FAIL on 0/3.
```

All three are safe by posterior/readout gap, but none passes reliability
non-collapse.

## 10. Gate Summary

| Gate | Verdict |
| --- | --- |
| Red-line | PASS |
| Curriculum schedule | PASS |
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
- V52A schedule variants;
- V52A reliability formula variants;
- reliability threshold sweep;
- V50A anchor weight, temperature, rank, filter-step, or refresh sweep.

## 11. Scientific Interpretation

V52A partially answers the rescue question:

```text
A fixed curriculum can provide nonzero early anchor availability and recover
ACM/DBLP performance while preserving Squirrel safety.
```

But the late agreement gate fails:

```text
When gamma reaches 1, reliability collapses again because posterior/readout
agreement remains too weak to sustain anchor mass.
```

The main lesson is narrower than success:

```text
Early anchor availability is necessary and helpful, but replacing it entirely
with agreement-modulated reliability at the end is too aggressive.
```

This is visible on ACM:

| Signal | Value |
| --- | ---: |
| Rel @1 | 0.1962 |
| Rel @40 | 0.1051 |
| Rel @80 | 0.0197 |
| Weighted Agreement @40 | 0.9337 |
| Weighted Agreement @80 | 0.1957 |
| ACC | 0.9041 |

ACM performance survives, but the final diagnostics show the anchor is no longer
reliably active by the preregistered gate.

## 12. Rescue Implication

The next route should not tune the V52A schedule. The failure is structural:

```text
V52A makes base reliability temporary. Once gamma=1, the method falls back to an
agreement bottleneck that recreates V51A's late-stage anchor avoidance.
```

Allowed next thinking should be a new preregistered route, not a V52A sweep.

Recommended next analysis artifact:

```text
V52A_LATE_RELIABILITY_COLLAPSE_ANALYSIS.md
```

It should decide whether the next rescue should preserve a nonzero base
reliability residual at late training, for example:

```text
v53a_residual_curriculum_spectral_anchor
```

Candidate direction to analyze, not implement yet:

```text
r_i = detach(clamp(beta * r_base_i + (1 - beta) * r_base_i * r_agree_i, 0, 1))
```

with a fixed nonzero residual `beta`, plus strict safety and mass diagnostics.
This is not authorized code and requires a new preregistration.

## 13. No-Fabrication Status

All numbers in this document come from the local V52A mixed-stress run and its
diagnostics:

```text
results/archive/v51-v57/unified_aptc_9datasets_v52a_curriculum_reliability_spectral_anchor.csv
results/archive/v51-v57/unified_aptc_9datasets_v52a_curriculum_reliability_spectral_anchor_diagnostics.jsonl
```

No V52A full-run result exists.
