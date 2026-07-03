# v54a_consensus_bounded_residual_anchor First Mixed-Stress Verdict

This file records the preregistered first-stage mixed-stress result for
`v54a_consensus_bounded_residual_anchor`. It follows
`V54A_PREREGISTRATION.md`, `V54A_IMPLEMENTATION_REVIEW.md`, and
`V54A_CONNECTIVITY_VERDICT.md`.

No full 9-dataset smoke, 260-epoch full run, beta-bound sweep, schedule
variant, reliability formula variant, threshold sweep, or V50A anchor
hyperparameter sweep is authorized by this verdict.

## 1. Run

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
conda run --no-capture-output -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v54a_consensus_bounded_residual_anchor --datasets "acm,dblp,flickr,texas,squirrel,chameleon" --epochs 80 --device cuda --log-level WARNING
```

Run status:

```text
6/6 datasets completed with status=ok.
No beta-bound, schedule, reliability formula, threshold, V50A anchor
hyperparameter, seed, or final-label change was made.
```

Output files:

```text
results/archive/v51-v57/unified_aptc_9datasets_v54a_consensus_bounded_residual_anchor.csv
results/archive/v51-v57/unified_aptc_9datasets_v54a_consensus_bounded_residual_anchor_diagnostics.jsonl
```

## 2. Result Summary

Latest complete mixed-stress records:

| Dataset | ACC | NMI | ARI | Emb Gap |
| --- | ---: | ---: | ---: | ---: |
| ACM | 0.9051 | 0.6805 | 0.7391 | 0.0000 |
| DBLP | 0.6798 | 0.3983 | 0.3386 | 0.0005 |
| Flickr | 0.3737 | 0.2145 | 0.1459 | -0.0066 |
| Texas | 0.7322 | 0.4807 | 0.5948 | 0.0055 |
| Squirrel | 0.3005 | 0.0608 | 0.0500 | 0.0010 |
| Chameleon | 0.3364 | 0.1536 | 0.0614 | -0.0048 |

Performance interpretation:

- ACM and DBLP pass the preregistered preservation floors.
- Squirrel recovers from the V53A overexposure failure and passes the new
  `ACC >= 0.2800` floor.
- Texas stays stable and safe.
- Chameleon is slightly lower than V53A, but remains safe.
- The central blocker is still reliability mass on DBLP/Texas, not red-line or
  posterior/readout safety.

## 3. Red-Line Gate

Preregistered requirement:

```text
status=ok
legacy_head_used=false
v43b-v49a_enabled=false
v50a_enabled=false
v51a_enabled=false
v52a_enabled=false
v53a_enabled=false
v54a_enabled=true
no selector / no post-processing selector
```

Verdict:

```text
PASS on 6/6.
```

| Dataset | v50a | v51a | v52a | v53a | v54a |
| --- | --- | --- | --- | --- | --- |
| ACM | false | false | false | false | true |
| DBLP | false | false | false | false | true |
| Flickr | false | false | false | false | true |
| Texas | false | false | false | false | true |
| Squirrel | false | false | false | false | true |
| Chameleon | false | false | false | false | true |

All legacy and V43B-V49A red-line flags are false in the diagnostics.

## 4. Residual Bound Gate

Preregistered requirement:

```text
v54a_gamma_epoch_1 = 0
v54a_gamma_epoch_40 = 0.5
v54a_gamma_epoch_80 = 1
v54a_beta_min = 0.35
v54a_beta_max = 0.70
0.35 <= v54a_beta_mean <= 0.70
```

Verdict:

```text
PASS on 6/6.
```

| Dataset | Gamma @1 | Gamma @40 | Gamma @80 | Beta Mean | Beta P10 | Beta P50 | Beta P90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ACM | 0.0000 | 0.5000 | 1.0000 | 0.6636 | 0.5250 | 0.7000 | 0.7000 |
| DBLP | 0.0000 | 0.5000 | 1.0000 | 0.5135 | 0.3500 | 0.5250 | 0.7000 |
| Flickr | 0.0000 | 0.5000 | 1.0000 | 0.3866 | 0.3500 | 0.3500 | 0.5250 |
| Texas | 0.0000 | 0.5000 | 1.0000 | 0.4896 | 0.3500 | 0.3500 | 0.7000 |
| Squirrel | 0.0000 | 0.5000 | 1.0000 | 0.3845 | 0.3500 | 0.3500 | 0.5250 |
| Chameleon | 0.0000 | 0.5000 | 1.0000 | 0.4405 | 0.3500 | 0.3500 | 0.7000 |

The consensus-bounded residual is active: easy aligned datasets receive higher
beta, while weak-anchor datasets receive lower beta.

## 5. Reliability Non-Collapse Gate

Preregistered requirement:

```text
Pass on at least 4/6:
0.08 <= v54a_reliability_mean <= 0.90
v54a_effective_anchor_mass >= 0.08

And at least 3/6:
v54a_reliable_node_ratio >= 0.05
```

Hard fail:

```text
v54a_reliability_mean < 0.03 on ACM, DBLP, Squirrel, or Chameleon
v54a_reliability_mean > 0.97 on any dataset
```

Verdict:

```text
FAIL on mass: 3/6 pass the [0.08, 0.90] and effective-mass gate.
PASS on reliable-node ratio: 3/6 pass.
No hard near-zero failure occurs on ACM, DBLP, Squirrel, or Chameleon.
```

| Dataset | Rel Mean | Reliable Ratio | Effective Mass | Gate |
| --- | ---: | ---: | ---: | --- |
| ACM | 0.1390 | 0.1481 | 0.1390 | PASS |
| DBLP | 0.0749 | 0.0000 | 0.0749 | FAIL, mass |
| Flickr | 0.0056 | 0.0000 | 0.0056 | weak-anchor non-use |
| Texas | 0.0562 | 0.0055 | 0.0562 | FAIL, mass |
| Squirrel | 0.0852 | 0.0525 | 0.0852 | PASS |
| Chameleon | 0.2123 | 0.4668 | 0.2123 | PASS |

Compared with V53A:

| Dataset | V53A Rel @80 | V54A Rel @80 | Direction |
| --- | ---: | ---: | --- |
| ACM | 0.1080 | 0.1390 | up |
| DBLP | 0.0722 | 0.0749 | slight up, still fail |
| Flickr | 0.0071 | 0.0056 | down |
| Texas | 0.0582 | 0.0562 | slight down |
| Squirrel | 0.1074 | 0.0852 | down but still above 0.08 |
| Chameleon | 0.2441 | 0.2123 | down but still above 0.08 |

Interpretation:

```text
Consensus bounding fixes V53A's Squirrel overexposure but does not lift
DBLP/Texas above the preregistered mass floor. The route is safer, but still
not expansion-ready.
```

## 6. Consensus Diagnostics

| Dataset | Hard Q Match | Hard Embed Match | Hard Both Match | Beta Mean |
| --- | ---: | ---: | ---: | ---: |
| ACM | 0.8942 | 0.8975 | 0.8866 | 0.6636 |
| DBLP | 0.4797 | 0.4548 | 0.4318 | 0.5135 |
| Flickr | 0.1071 | 0.1020 | 0.0506 | 0.3866 |
| Texas | 0.3989 | 0.3989 | 0.3934 | 0.4896 |
| Squirrel | 0.1375 | 0.0598 | 0.0386 | 0.3845 |
| Chameleon | 0.3000 | 0.2170 | 0.2060 | 0.4405 |

This table explains the main change from V53A: Squirrel receives low beta
because anchor/posterior/readout consensus is weak, which protects its ACC.

## 7. Safety Gate

Preregistered requirement:

```text
abs(embedding_posterior_gap) <= 0.04 on 6/6
no dataset abs(embedding_posterior_gap) > 0.08
```

Verdict:

```text
PASS on 6/6.
```

| Dataset | Emb Gap | Safety |
| --- | ---: | --- |
| ACM | 0.0000 | PASS |
| DBLP | 0.0005 | PASS |
| Flickr | -0.0066 | PASS |
| Texas | 0.0055 | PASS |
| Squirrel | 0.0010 | PASS |
| Chameleon | -0.0048 | PASS |

Posterior/readout safety remains clean.

## 8. Anchor Usefulness Gate

Preregistered requirements:

```text
v54a_weighted_q_anchor_agreement_epoch_80 >
v54a_weighted_q_anchor_agreement_epoch_1
Pass on at least 4/6.

ACM ACC >= 0.8888
DBLP ACC >= 0.6610
Squirrel ACC >= 0.2800
```

Weighted agreement movement:

| Dataset | Weighted Agreement @1 | @40 | @80 | Movement | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| ACM | 0.4358 | 0.9452 | 0.9669 | +0.5311 | PASS |
| DBLP | 0.3128 | 0.5734 | 0.5105 | +0.1977 | PASS |
| Flickr | 0.0162 | 0.0138 | 0.0107 | -0.0054 | FAIL |
| Texas | 0.3893 | 0.3785 | 0.3314 | -0.0579 | FAIL |
| Squirrel | 0.1220 | 0.1860 | 0.2168 | +0.0948 | PASS |
| Chameleon | 0.1756 | 0.3409 | 0.4597 | +0.2841 | PASS |

Agreement movement passes on 4/6.

Preservation floors:

| Dataset | V54A ACC | Required Floor | Gate |
| --- | ---: | ---: | --- |
| ACM | 0.9051 | 0.8888 | PASS |
| DBLP | 0.6798 | 0.6610 | PASS |
| Squirrel | 0.3005 | 0.2800 | PASS |

Verdict:

```text
PASS.
```

## 9. Heterophily Stress Gate

Preregistered requirement for Texas, Squirrel, and Chameleon:

```text
At least 2/3 must satisfy:
abs(embedding_posterior_gap) <= 0.04
v54a_reliability_mean within [0.08, 0.90]
v54a_effective_anchor_mass >= 0.08

Additionally:
Squirrel ACC >= 0.2800
```

| Dataset | Emb Gap | Rel Mean | Effective Mass | ACC | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Texas | 0.0055 | 0.0562 | 0.0562 | 0.7322 | FAIL, mass |
| Squirrel | 0.0010 | 0.0852 | 0.0852 | 0.3005 | PASS |
| Chameleon | -0.0048 | 0.2123 | 0.2123 | 0.3364 | PASS |

Verdict:

```text
PASS on 2/3, with Squirrel ACC floor satisfied.
```

## 10. Gate Summary

| Gate | Verdict |
| --- | --- |
| Red-line | PASS |
| Residual bound | PASS |
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
- beta-bound sweep;
- V54A schedule variants;
- V54A reliability formula variants;
- reliability threshold sweep;
- V50A anchor weight, temperature, rank, filter-step, or refresh sweep.

## 11. Scientific Interpretation

V54A gives the cleanest rescue signal so far on safety:

```text
Node-level consensus bounding preserves ACM/DBLP, restores Squirrel above the
new safety floor, keeps posterior/readout gap clean, and keeps heterophily
stress passable.
```

But it is not ready for expansion:

```text
DBLP and Texas remain below the 0.08 effective-mass floor. The mechanism now
knows how to avoid weak-anchor overexposure, but it still lacks a way to retain
enough trustworthy anchor mass on medium-consensus graphs.
```

The next rescue question should not be "raise beta." V54A shows that lower beta
is useful on Squirrel and Flickr. The next question is:

```text
Can low-mass datasets retain enough anchor availability by using continuous
consensus strength instead of only hard argmax consensus, without dataset
branches or threshold sweeps?
```

## 12. Next Rescue Implication

Recommended next analysis artifact:

```text
V54A_CONSENSUS_UNDERACTIVATION_ANALYSIS.md
```

Likely next route to preregister, not implement yet:

```text
v55a_soft_consensus_bounded_residual_anchor
```

The candidate mechanism should keep V54A's bounded residual idea but replace
hard `argmax` consensus with a continuous, stop-gradient soft consensus score
derived from already available posterior-anchor and embedding-anchor agreement.
It must not introduce a dataset branch, beta sweep, new final label path, or
geometry fallback.

## 13. No-Fabrication Status

All numbers in this document come from the local V54A mixed-stress run and its
diagnostics:

```text
results/archive/v51-v57/unified_aptc_9datasets_v54a_consensus_bounded_residual_anchor.csv
results/archive/v51-v57/unified_aptc_9datasets_v54a_consensus_bounded_residual_anchor_diagnostics.jsonl
```

No V54A full-run result exists.
