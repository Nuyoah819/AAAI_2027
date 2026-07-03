# v53a_residual_curriculum_spectral_anchor First Mixed-Stress Verdict

This file records the preregistered first-stage mixed-stress result for
`v53a_residual_curriculum_spectral_anchor`. It follows
`V53A_PREREGISTRATION.md` and `V53A_CONNECTIVITY_VERDICT.md`.

No full 9-dataset smoke, 260-epoch full run, beta sweep, schedule variant,
reliability formula variant, threshold sweep, or V50A anchor hyperparameter
sweep is authorized by this verdict.

## 1. Run

Prerequisite artifacts:

```text
V53A_PREREGISTRATION.md
V53A_IMPLEMENTATION_REVIEW.md
V53A_CONNECTIVITY_VERDICT.md
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
conda run --no-capture-output -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v53a_residual_curriculum_spectral_anchor --datasets "acm,dblp,flickr,texas,squirrel,chameleon" --epochs 80 --device cuda --log-level WARNING
```

Run status:

```text
6/6 datasets completed with status=ok.
No duplicate mixed-stress run was used.
No V53A beta, schedule, reliability formula, threshold, V50A anchor
hyperparameter, seed, or final-label change was made.
```

Output files:

```text
results/archive/v51-v57/unified_aptc_9datasets_v53a_residual_curriculum_spectral_anchor.csv
results/archive/v51-v57/unified_aptc_9datasets_v53a_residual_curriculum_spectral_anchor_diagnostics.jsonl
```

## 2. Result Summary

Latest complete mixed-stress records:

| Dataset | ACC | NMI | ARI | Emb Gap |
| --- | ---: | ---: | ---: | ---: |
| ACM | 0.9005 | 0.6747 | 0.7287 | 0.0000 |
| DBLP | 0.6806 | 0.3981 | 0.3419 | -0.0002 |
| Flickr | 0.3675 | 0.2136 | 0.1424 | 0.0096 |
| Texas | 0.7322 | 0.4779 | 0.5974 | 0.0055 |
| Squirrel | 0.2119 | 0.0149 | 0.0004 | 0.0000 |
| Chameleon | 0.3421 | 0.1577 | 0.0591 | 0.0000 |

Performance interpretation:

- ACM and DBLP pass the preregistered preservation floors.
- Chameleon improves slightly over V52A.
- Squirrel collapses in absolute performance even though posterior/readout
  safety remains clean.
- Texas remains safe but does not pass the reliability gate.

## 3. Red-Line Gate

Preregistered requirement:

```text
status=ok
legacy_head_used=false
v43b-v49a_enabled=false
v50a_enabled=false
v51a_enabled=false
v52a_enabled=false
v53a_enabled=true
no selector / no post-processing selector
```

Verdict:

```text
PASS on 6/6.
```

| Dataset | Legacy | v43b | v44 | v44b | v45a | v46a | v47a | v48a | v49a | v50a | v51a | v52a | v53a |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ACM | false | false | false | false | false | false | false | false | false | false | false | false | true |
| DBLP | false | false | false | false | false | false | false | false | false | false | false | false | true |
| Flickr | false | false | false | false | false | false | false | false | false | false | false | false | true |
| Texas | false | false | false | false | false | false | false | false | false | false | false | false | true |
| Squirrel | false | false | false | false | false | false | false | false | false | false | false | false | true |
| Chameleon | false | false | false | false | false | false | false | false | false | false | false | false | true |

## 4. Residual Schedule Gate

Preregistered requirement:

```text
v53a_gamma_epoch_1 = 0
v53a_gamma_epoch_40 = 0.5
v53a_gamma_epoch_80 = 1
v53a_residual_beta = 0.50
```

Verdict:

```text
PASS on 6/6.
```

| Dataset | Gamma @1 | Gamma @40 | Gamma @80 | Beta | Residual Mult @80 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ACM | 0.0000 | 0.5000 | 1.0000 | 0.5000 | 0.5432 |
| DBLP | 0.0000 | 0.5000 | 1.0000 | 0.5000 | 0.5174 |
| Flickr | 0.0000 | 0.5000 | 1.0000 | 0.5000 | 0.5005 |
| Texas | 0.0000 | 0.5000 | 1.0000 | 0.5000 | 0.5300 |
| Squirrel | 0.0000 | 0.5000 | 1.0000 | 0.5000 | 0.5085 |
| Chameleon | 0.0000 | 0.5000 | 1.0000 | 0.5000 | 0.5404 |

The residual was applied exactly as preregistered.

## 5. Reliability Non-Collapse Gate

Preregistered requirement:

```text
Pass on at least 5/6:
0.08 <= v53a_reliability_mean <= 0.90
v53a_reliable_node_ratio >= 0.05
v53a_effective_anchor_mass >= 0.08

Hard fail:
v53a_reliability_mean < 0.03 on ACM, DBLP, Texas, Squirrel, or Chameleon
v53a_reliability_mean > 0.97 on any dataset
```

Verdict:

```text
FAIL on 2/6.
No hard near-zero failure on ACM/DBLP/Texas/Squirrel/Chameleon except none;
the failure is thresholded mass/ratio insufficiency, not all-off collapse.
```

| Dataset | Rel Mean | Rel P50 | Rel P90 | Reliable Ratio | Effective Mass | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| ACM | 0.1080 | 0.1085 | 0.1641 | 0.0281 | 0.1080 | FAIL, ratio |
| DBLP | 0.0722 | 0.0852 | 0.0976 | 0.0000 | 0.0722 | FAIL |
| Flickr | 0.0071 | 0.0059 | 0.0113 | 0.0000 | 0.0071 | weak-anchor non-use |
| Texas | 0.0582 | 0.0518 | 0.0903 | 0.0000 | 0.0582 | FAIL |
| Squirrel | 0.1074 | 0.0975 | 0.1775 | 0.0692 | 0.1074 | PASS |
| Chameleon | 0.2441 | 0.2492 | 0.3293 | 0.7396 | 0.2441 | PASS |

Compared with V52A, V53A substantially repairs late reliability:

| Dataset | V52A Rel @80 | V53A Rel @80 |
| --- | ---: | ---: |
| ACM | 0.0197 | 0.1080 |
| DBLP | 0.0055 | 0.0722 |
| Texas | 0.0075 | 0.0582 |
| Squirrel | 0.0068 | 0.1074 |
| Chameleon | 0.0376 | 0.2441 |

But it still fails the preregistered non-collapse gate because DBLP/Texas remain
below the 0.08 mass floor and ACM fails the reliable-node ratio threshold.

## 6. Base / Agreement / Residual Contributions

| Dataset | Rel @1 | Rel @40 | Rel @80 | Base @1 | Agreement @80 | Residual Mult @80 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ACM | 0.1962 | 0.1506 | 0.1080 | 0.1962 | 0.0864 | 0.5432 |
| DBLP | 0.1389 | 0.1057 | 0.0722 | 0.1389 | 0.0348 | 0.5174 |
| Flickr | 0.0142 | 0.0107 | 0.0071 | 0.0142 | 0.0010 | 0.5005 |
| Texas | 0.1089 | 0.0834 | 0.0582 | 0.1089 | 0.0600 | 0.5300 |
| Squirrel | 0.2086 | 0.1580 | 0.1074 | 0.2086 | 0.0170 | 0.5085 |
| Chameleon | 0.4507 | 0.3464 | 0.2441 | 0.4507 | 0.0807 | 0.5404 |

Interpretation:

```text
The residual works as designed: reliability no longer collapses to V52A's
near-zero values. However, the fixed residual also increases anchor exposure on
Squirrel, where the anchor is weak and performance drops sharply.
```

## 7. Safety Gate

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
| DBLP | -0.0002 | PASS |
| Flickr | 0.0096 | PASS |
| Texas | 0.0055 | PASS |
| Squirrel | 0.0000 | PASS |
| Chameleon | 0.0000 | PASS |

Posterior/readout safety remains intact.

## 8. Anchor Usefulness Gate

Preregistered requirements:

```text
v53a_weighted_q_anchor_agreement_epoch_80 >
v53a_weighted_q_anchor_agreement_epoch_1
Pass on at least 4/6.

ACM ACC >= 0.8888
DBLP ACC >= 0.6610
```

Weighted agreement movement:

| Dataset | Weighted Agreement @1 | @40 | @80 | Movement | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| ACM | 0.4358 | 0.9274 | 0.9451 | +0.5093 | PASS |
| DBLP | 0.3128 | 0.5228 | 0.3955 | +0.0827 | PASS |
| Flickr | 0.0162 | 0.0127 | 0.0084 | -0.0078 | FAIL |
| Texas | 0.3893 | 0.3417 | 0.2496 | -0.1397 | FAIL |
| Squirrel | 0.1220 | 0.1650 | 0.1733 | +0.0513 | PASS |
| Chameleon | 0.1756 | 0.3038 | 0.3405 | +0.1649 | PASS |

Agreement movement passes on 4/6.

Preservation floors:

| Dataset | V53A ACC | Required Floor | Gate |
| --- | ---: | ---: | --- |
| ACM | 0.9005 | 0.8888 | PASS |
| DBLP | 0.6806 | 0.6610 | PASS |

Verdict:

```text
PASS.
```

## 9. Heterophily Stress Gate

Preregistered requirement for Texas, Squirrel, and Chameleon:

```text
At least 2/3 must satisfy:
abs(embedding_posterior_gap) <= 0.04
v53a_reliability_mean within [0.08, 0.90]
v53a_effective_anchor_mass >= 0.08
```

| Dataset | Emb Gap | Rel Mean | Effective Mass | Gate |
| --- | ---: | ---: | ---: | --- |
| Texas | 0.0055 | 0.0582 | 0.0582 | FAIL |
| Squirrel | 0.0000 | 0.1074 | 0.1074 | PASS |
| Chameleon | 0.0000 | 0.2441 | 0.2441 | PASS |

Verdict:

```text
PASS on 2/3.
```

This is the first rescue variant in the V51-V53 chain to pass the heterophily
stress gate, but it still fails reliability non-collapse overall.

## 10. Gate Summary

| Gate | Verdict |
| --- | --- |
| Red-line | PASS |
| Residual schedule | PASS |
| Reliability non-collapse | FAIL |
| Posterior/readout safety | PASS |
| Anchor usefulness | PASS |
| Heterophily stress | PASS |

Decision:

```text
STOP before full expansion.
```

This verdict does not authorize:

- full 9-dataset smoke;
- 260-epoch full run;
- beta sweep;
- V53A schedule variants;
- V53A reliability formula variants;
- reliability threshold sweep;
- V50A anchor weight, temperature, rank, filter-step, or refresh sweep.

## 11. Scientific Interpretation

V53A gives the strongest rescue signal so far:

```text
The residual reliability mechanism repairs V52A's late collapse enough to pass
ACM/DBLP preservation, safety, anchor usefulness, and heterophily stress.
```

But it is not ready for expansion:

```text
The reliability non-collapse gate still fails, and Squirrel's absolute
performance drops sharply from the V52A/V51A range.
```

The key lesson:

```text
Residual anchor availability is necessary, but a uniform beta residual can
overexpose weak-anchor heterophily nodes while still underactivating reliable
nodes on DBLP/Texas by the strict ratio gate.
```

This suggests the next rescue question is no longer "should there be a
residual?" The answer is yes. The next question is:

```text
Can residual anchor availability be bounded by anchor evidence quality without
using dataset-specific routing or a beta sweep?
```

## 12. Next Rescue Implication

Do not tune `beta=0.50` post hoc. Do not relax the reliable-node threshold after
seeing this result.

Recommended next analysis artifact:

```text
V53A_RESIDUAL_OVEREXPOSURE_ANALYSIS.md
```

It should decide whether the next route should use a fixed evidence-bounded
residual, for example:

```text
v54a_evidence_bounded_residual_anchor
```

Candidate direction to analyze, not implement yet:

```text
beta_i = beta_max * stopgrad(r_base_i)
r_i = r_base_i * ((1 - gamma_t) + gamma_t * (beta_i + (1 - beta_i) * r_agree_i))
```

This would reduce residual exposure where the anchor's own confidence/local
evidence is weak, while preserving nonzero residual where the anchor has
evidence. It is not authorized code and requires a new preregistration.

## 13. No-Fabrication Status

All numbers in this document come from the local V53A mixed-stress run and its
diagnostics:

```text
results/archive/v51-v57/unified_aptc_9datasets_v53a_residual_curriculum_spectral_anchor.csv
results/archive/v51-v57/unified_aptc_9datasets_v53a_residual_curriculum_spectral_anchor_diagnostics.jsonl
```

No V53A full-run result exists.
