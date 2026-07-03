# v55a_soft_consensus_bounded_residual_anchor First Mixed-Stress Verdict

This file records the preregistered first-stage mixed-stress result for
`v55a_soft_consensus_bounded_residual_anchor`. It follows
`V55A_PREREGISTRATION.md`, `V55A_IMPLEMENTATION_REVIEW.md`, and
`V55A_CONNECTIVITY_VERDICT.md`.

No full 9-dataset smoke, 260-epoch full run, seed sweep, beta-bound sweep,
soft-power sweep, schedule variant, reliability formula variant, threshold
sweep, or V50A anchor hyperparameter sweep is authorized by this verdict.

## 1. Run

Static check before running:

```powershell
conda run -n aaai-e2e-subspace python -m py_compile core\e2e\sect_coco_e2e.py scripts\run_unified_aptc_9datasets.py
```

Verdict:

```text
PASS
```

Connectivity:

```text
PASS in V55A_CONNECTIVITY_VERDICT.md
```

Mixed-stress command:

```powershell
conda run --no-capture-output -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v55a_soft_consensus_bounded_residual_anchor --datasets "acm,dblp,flickr,texas,squirrel,chameleon" --epochs 80 --device cuda --log-level WARNING
```

Run status:

```text
6/6 datasets completed with status=ok.
No beta-bound, soft-power, schedule, reliability formula, threshold, V50A
anchor hyperparameter, seed, final-label, or selector change was made.
```

Output files:

```text
results/archive/v51-v57/unified_aptc_9datasets_v55a_soft_consensus_bounded_residual_anchor.csv
results/archive/v51-v57/unified_aptc_9datasets_v55a_soft_consensus_bounded_residual_anchor_diagnostics.jsonl
```

Note:

```text
The result files also contain the earlier ACM 1-epoch connectivity row. This
verdict uses the latest record per dataset from the 80-epoch mixed-stress run.
```

## 2. Result Summary

| Dataset | ACC | NMI | ARI | Emb Gap |
| --- | ---: | ---: | ---: | ---: |
| ACM | 0.8995 | 0.6732 | 0.7253 | 0.0000 |
| DBLP | 0.6847 | 0.4138 | 0.3492 | 0.0005 |
| Flickr | 0.3724 | 0.2151 | 0.1460 | 0.0000 |
| Texas | 0.7377 | 0.4944 | 0.6109 | 0.0000 |
| Squirrel | 0.3021 | 0.0628 | 0.0516 | -0.0012 |
| Chameleon | 0.3417 | 0.1490 | 0.0545 | -0.0088 |

Performance interpretation:

```text
ACM, DBLP, and Squirrel pass the preregistered preservation floors.
Squirrel remains protected above 0.2800.
Texas remains strong.
The central blocker is still reliability mass and reliable-node activation.
```

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
v54a_enabled=false
v55a_enabled=true
no selector / no post-processing selector
```

Verdict:

```text
PASS on 6/6.
```

| Dataset | V43B-V49A | V50A | V51A | V52A | V53A | V54A | V55A | Legacy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ACM | false | false | false | false | false | false | true | false |
| DBLP | false | false | false | false | false | false | true | false |
| Flickr | false | false | false | false | false | false | true | false |
| Texas | false | false | false | false | false | false | true | false |
| Squirrel | false | false | false | false | false | false | true | false |
| Chameleon | false | false | false | false | false | false | true | false |

## 4. Soft-Residual Bound Gate

Preregistered requirement:

```text
v55a_gamma_epoch_1 = 0
v55a_gamma_epoch_40 = 0.5
v55a_gamma_epoch_80 = 1
v55a_beta_min = 0.35
v55a_beta_max = 0.70
v55a_soft_power = 0.50
0.35 <= v55a_beta_mean <= 0.70
0.00 <= v55a_soft_consensus_mean <= 1.00
```

Verdict:

```text
PASS on 6/6.
```

| Dataset | Gamma @1 | Gamma @40 | Gamma @80 | Soft Mean | Beta Mean | Beta P10 | Beta P50 | Beta P90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ACM | 0.0000 | 0.5000 | 1.0000 | 0.0873 | 0.4481 | 0.4037 | 0.4552 | 0.4823 |
| DBLP | 0.0000 | 0.5000 | 1.0000 | 0.0351 | 0.3970 | 0.3500 | 0.3916 | 0.4542 |
| Flickr | 0.0000 | 0.5000 | 1.0000 | 0.0010 | 0.3575 | 0.3500 | 0.3536 | 0.3691 |
| Texas | 0.0000 | 0.5000 | 1.0000 | 0.0603 | 0.4131 | 0.3500 | 0.4082 | 0.4862 |
| Squirrel | 0.0000 | 0.5000 | 1.0000 | 0.0172 | 0.3697 | 0.3500 | 0.3500 | 0.4385 |
| Chameleon | 0.0000 | 0.5000 | 1.0000 | 0.0798 | 0.4194 | 0.3500 | 0.4064 | 0.5052 |

## 5. Reliability Non-Collapse Gate

Preregistered requirement:

```text
Pass on at least 4/6:
0.08 <= v55a_reliability_mean <= 0.90
v55a_effective_anchor_mass >= 0.08

And at least 3/6:
v55a_reliable_node_ratio >= 0.05
```

Hard fail:

```text
v55a_reliability_mean < 0.03 on ACM, DBLP, Squirrel, or Chameleon
v55a_reliability_mean > 0.97 on any dataset
```

Verdict:

```text
FAIL.
Mass passes on only 3/6.
Reliable-node ratio passes on only 1/6.
No hard near-zero failure occurs on ACM, DBLP, Squirrel, or Chameleon.
```

| Dataset | Rel Mean | Reliable Ratio | Effective Mass | Gate |
| --- | ---: | ---: | ---: | --- |
| ACM | 0.1004 | 0.0241 | 0.1004 | PASS mass, FAIL ratio |
| DBLP | 0.0589 | 0.0000 | 0.0589 | FAIL mass |
| Flickr | 0.0051 | 0.0000 | 0.0051 | weak-anchor non-use |
| Texas | 0.0490 | 0.0000 | 0.0490 | FAIL mass |
| Squirrel | 0.0820 | 0.0458 | 0.0820 | PASS mass, FAIL ratio |
| Chameleon | 0.2067 | 0.5081 | 0.2067 | PASS |

Compared with V54A:

| Dataset | V54A Rel @80 | V55A Rel @80 | Direction |
| --- | ---: | ---: | --- |
| ACM | 0.1390 | 0.1004 | down |
| DBLP | 0.0749 | 0.0589 | down |
| Flickr | 0.0056 | 0.0051 | flat/down |
| Texas | 0.0562 | 0.0490 | down |
| Squirrel | 0.0852 | 0.0820 | slight down, still above floor |
| Chameleon | 0.2123 | 0.2067 | slight down |

Interpretation:

```text
V55A's continuous soft consensus is too weak to replace V54A's hard consensus.
It preserves safety, but it worsens the underactivation problem it was meant to
repair.
```

## 6. Weighted Coupling

Preregistered requirement:

```text
v55a_weighted_q_anchor_agreement_epoch_80 >
v55a_weighted_q_anchor_agreement_epoch_1
Pass on at least 4/6.
```

Verdict:

```text
PASS on 4/6.
```

| Dataset | Weighted Agreement @1 | @40 | @80 | Movement | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| ACM | 0.4358 | 0.9310 | 0.9501 | +0.5143 | PASS |
| DBLP | 0.3128 | 0.5408 | 0.3625 | +0.0497 | PASS |
| Flickr | 0.0162 | 0.0117 | 0.0064 | -0.0098 | FAIL |
| Texas | 0.3893 | 0.3331 | 0.2460 | -0.1433 | FAIL |
| Squirrel | 0.1220 | 0.1779 | 0.1823 | +0.0602 | PASS |
| Chameleon | 0.1756 | 0.3189 | 0.3978 | +0.2222 | PASS |

Preservation floors:

| Dataset | V55A ACC | Required Floor | Gate |
| --- | ---: | ---: | --- |
| ACM | 0.8995 | 0.8888 | PASS |
| DBLP | 0.6847 | 0.6610 | PASS |
| Squirrel | 0.3021 | 0.2800 | PASS |

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
| Flickr | 0.0000 | PASS |
| Texas | 0.0000 | PASS |
| Squirrel | -0.0012 | PASS |
| Chameleon | -0.0088 | PASS |

## 8. Heterophily Stress Gate

Preregistered requirement for Texas, Squirrel, and Chameleon:

```text
At least 2/3 must satisfy:
abs(embedding_posterior_gap) <= 0.04
v55a_reliability_mean within [0.08, 0.90]
v55a_effective_anchor_mass >= 0.08

Additionally:
Squirrel ACC >= 0.2800
```

Verdict:

```text
PASS on 2/3, with Squirrel ACC floor satisfied.
```

| Dataset | Emb Gap | Rel Mean | Effective Mass | ACC | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Texas | 0.0000 | 0.0490 | 0.0490 | 0.7377 | FAIL, mass |
| Squirrel | -0.0012 | 0.0820 | 0.0820 | 0.3021 | PASS |
| Chameleon | -0.0088 | 0.2067 | 0.2067 | 0.3417 | PASS |

## 9. Gate Summary

| Gate | Verdict |
| --- | --- |
| Red-line | PASS |
| Soft-residual bound | PASS |
| Reliability non-collapse | FAIL |
| Posterior/readout safety | PASS |
| Anchor usefulness | PASS |
| Heterophily stress | PASS |

Decision:

```text
STOP before full expansion.
```

This verdict does not authorize:

```text
full 9-dataset smoke
260-epoch full run
seed sweep
beta-bound sweep
soft-power sweep
schedule variant
reliability formula variant
threshold sweep
V50A anchor hyperparameter sweep
```

## 10. Scientific Interpretation

V55A tested a clean hypothesis:

```text
Replacing V54A hard argmax consensus with continuous soft agreement should
retain more anchor mass on DBLP/Texas.
```

The hypothesis is not supported:

```text
Soft agreement is numerically too small. Even with a fixed sqrt lift, beta
means drop below V54A on all datasets, including DBLP and Texas. Therefore
V55A does not repair medium-consensus underactivation.
```

However, the route teaches a useful boundary:

```text
V54A's hard consensus was not merely too coarse; it also carried a useful
argmax-alignment signal that soft dot-product agreement failed to preserve.
The next rescue should not replace hard consensus. It should preserve the hard
safety path and add a narrow, detached compensation only for medium-evidence
nodes.
```

Recommended next analysis artifact:

```text
V55A_SOFT_CONSENSUS_FAILURE_ANALYSIS.md
```

Likely next route to preregister, not implement yet:

```text
v56a_hybrid_consensus_floor_residual_anchor
```

## 11. No-Fabrication Status

All numbers in this document come from the local V55A mixed-stress run and its
diagnostics:

```text
results/archive/v51-v57/unified_aptc_9datasets_v55a_soft_consensus_bounded_residual_anchor.csv
results/archive/v51-v57/unified_aptc_9datasets_v55a_soft_consensus_bounded_residual_anchor_diagnostics.jsonl
```

No V55A full-run result exists.
