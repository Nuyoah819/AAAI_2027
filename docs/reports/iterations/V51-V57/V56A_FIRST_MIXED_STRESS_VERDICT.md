# v56a_hybrid_consensus_floor_residual_anchor First Mixed-Stress Verdict

This file records the preregistered first-stage mixed-stress result for
`v56a_hybrid_consensus_floor_residual_anchor`. It follows
`V56A_PREREGISTRATION.md`, `V56A_IMPLEMENTATION_REVIEW.md`, and
`V56A_CONNECTIVITY_VERDICT.md`.

No full 9-dataset smoke, 260-epoch full run, seed sweep, beta-bound sweep,
soft-power sweep, hybrid-compensation sweep, schedule variant, reliability
formula variant, threshold sweep, or V50A anchor hyperparameter sweep is
authorized by this verdict.

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
PASS in V56A_CONNECTIVITY_VERDICT.md
```

Mixed-stress command:

```powershell
conda run --no-capture-output -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v56a_hybrid_consensus_floor_residual_anchor --datasets "acm,dblp,flickr,texas,squirrel,chameleon" --epochs 80 --device cuda --log-level WARNING
```

Run status:

```text
6/6 datasets completed with status=ok.
No beta-bound, soft-power, hybrid-compensation, schedule, reliability formula,
threshold, V50A anchor hyperparameter, seed, final-label, or selector change was
made.
```

Output files:

```text
results/archive/v51-v57/unified_aptc_9datasets_v56a_hybrid_consensus_floor_residual_anchor.csv
results/archive/v51-v57/unified_aptc_9datasets_v56a_hybrid_consensus_floor_residual_anchor_diagnostics.jsonl
```

Note:

```text
The result files also contain the earlier ACM 1-epoch connectivity row. This
verdict uses the latest record per dataset from the 80-epoch mixed-stress run.
```

## 2. Result Summary

| Dataset | ACC | NMI | ARI | Emb Gap |
| --- | ---: | ---: | ---: | ---: |
| ACM | 0.9005 | 0.6710 | 0.7279 | 0.0000 |
| DBLP | 0.6919 | 0.4120 | 0.3606 | 0.0000 |
| Flickr | 0.3681 | 0.2126 | 0.1423 | 0.0050 |
| Texas | 0.7377 | 0.4944 | 0.6109 | 0.0000 |
| Squirrel | 0.3011 | 0.0619 | 0.0508 | 0.0012 |
| Chameleon | 0.3382 | 0.1508 | 0.0592 | 0.0048 |

Performance interpretation:

```text
ACM, DBLP, and Squirrel pass the preregistered floors. DBLP improves over
V54A/V55A, Texas remains strong, and Squirrel stays above 0.2800. The central
blocker remains reliability mass on DBLP/Texas, not red-line or safety.
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
v55a_enabled=false
v56a_enabled=true
no selector / no post-processing selector
```

Verdict:

```text
PASS on 6/6.
```

| Dataset | V50A | V51A | V52A | V53A | V54A | V55A | V56A |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ACM | false | false | false | false | false | false | true |
| DBLP | false | false | false | false | false | false | true |
| Flickr | false | false | false | false | false | false | true |
| Texas | false | false | false | false | false | false | true |
| Squirrel | false | false | false | false | false | false | true |
| Chameleon | false | false | false | false | false | false | true |

All legacy and V43B-V49A red-line flags are false in the diagnostics.

## 4. Hybrid Residual Bound Gate

Preregistered requirement:

```text
v56a_gamma_epoch_1 = 0
v56a_gamma_epoch_40 = 0.5
v56a_gamma_epoch_80 = 1
v56a_beta_min = 0.35
v56a_beta_max = 0.70
v56a_soft_power = 0.50
v56a_hybrid_compensation = 0.50
0.35 <= v56a_beta_mean <= 0.70
v56a_compensation_mean >= 0
0 <= v56a_compensation_active_ratio <= 1
```

Verdict:

```text
PASS on 6/6.
```

| Dataset | Gamma @1 | Gamma @40 | Gamma @80 | Beta Mean | Comp Mean | Comp Active |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ACM | 0.0000 | 0.5000 | 1.0000 | 0.6652 | 0.0039 | 0.0661 |
| DBLP | 0.0000 | 0.5000 | 1.0000 | 0.5178 | 0.0022 | 0.0569 |
| Flickr | 0.0000 | 0.5000 | 1.0000 | 0.3877 | 0.0070 | 0.3929 |
| Texas | 0.0000 | 0.5000 | 1.0000 | 0.4899 | 0.0198 | 0.2404 |
| Squirrel | 0.0000 | 0.5000 | 1.0000 | 0.3863 | 0.0054 | 0.1036 |
| Chameleon | 0.0000 | 0.5000 | 1.0000 | 0.4517 | 0.0318 | 0.3663 |

## 5. Hybrid Consensus Diagnostics

| Dataset | Hard Mean | Soft Mean | Lifted Soft | Hybrid Mean | Beta Mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| ACM | 0.8967 | 0.0861 | 0.2791 | 0.9006 | 0.6652 |
| DBLP | 0.4771 | 0.0350 | 0.1344 | 0.4793 | 0.5178 |
| Flickr | 0.1008 | 0.0010 | 0.0212 | 0.1078 | 0.3877 |
| Texas | 0.3798 | 0.0596 | 0.1792 | 0.3996 | 0.4899 |
| Squirrel | 0.0982 | 0.0177 | 0.0573 | 0.1036 | 0.3863 |
| Chameleon | 0.2589 | 0.0802 | 0.2021 | 0.2906 | 0.4517 |

Interpretation:

```text
The compensation is active but small. It modestly lifts Texas/Chameleon and
keeps Squirrel close to V54A safety levels. It does not lift DBLP/Texas
reliability mass enough to pass the preregistered gate.
```

## 6. Reliability Non-Collapse Gate

Preregistered requirement:

```text
Pass on at least 4/6:
0.08 <= v56a_reliability_mean <= 0.90
v56a_effective_anchor_mass >= 0.08

And at least 3/6:
v56a_reliable_node_ratio >= 0.05
```

Hard fail:

```text
v56a_reliability_mean < 0.03 on ACM, DBLP, Squirrel, or Chameleon
v56a_reliability_mean > 0.97 on any dataset
```

Verdict:

```text
FAIL.
Mass passes on only 3/6.
Reliable-node ratio passes on only 3/6.
No hard near-zero failure occurs on ACM, DBLP, Squirrel, or Chameleon.
```

| Dataset | Rel Mean | Reliable Ratio | Effective Mass | Gate |
| --- | ---: | ---: | ---: | --- |
| ACM | 0.1393 | 0.1481 | 0.1393 | PASS |
| DBLP | 0.0756 | 0.0000 | 0.0756 | FAIL mass |
| Flickr | 0.0056 | 0.0000 | 0.0056 | weak-anchor non-use |
| Texas | 0.0561 | 0.0055 | 0.0561 | FAIL mass |
| Squirrel | 0.0855 | 0.0527 | 0.0855 | PASS |
| Chameleon | 0.2171 | 0.4954 | 0.2171 | PASS |

Compared with V54A and V55A:

| Dataset | V54A Rel @80 | V55A Rel @80 | V56A Rel @80 |
| --- | ---: | ---: | ---: |
| ACM | 0.1390 | 0.1004 | 0.1393 |
| DBLP | 0.0749 | 0.0589 | 0.0756 |
| Flickr | 0.0056 | 0.0051 | 0.0056 |
| Texas | 0.0562 | 0.0490 | 0.0561 |
| Squirrel | 0.0852 | 0.0820 | 0.0855 |
| Chameleon | 0.2123 | 0.2067 | 0.2171 |

Interpretation:

```text
V56A recovers the V54A reliability profile and slightly improves DBLP/Squirrel/
Chameleon, but it still does not solve the DBLP/Texas mass floor failure.
```

## 7. Weighted Coupling

Preregistered requirement:

```text
v56a_weighted_q_anchor_agreement_epoch_80 >
v56a_weighted_q_anchor_agreement_epoch_1
Pass on at least 4/6.
```

Verdict:

```text
PASS on 4/6.
```

| Dataset | Weighted Agreement @1 | @40 | @80 | Movement | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| ACM | 0.4358 | 0.9444 | 0.9649 | +0.5291 | PASS |
| DBLP | 0.3128 | 0.5796 | 0.5248 | +0.2120 | PASS |
| Flickr | 0.0162 | 0.0139 | 0.0102 | -0.0060 | FAIL |
| Texas | 0.3893 | 0.3786 | 0.3065 | -0.0827 | FAIL |
| Squirrel | 0.1220 | 0.1846 | 0.2172 | +0.0952 | PASS |
| Chameleon | 0.1756 | 0.3336 | 0.4464 | +0.2708 | PASS |

Preservation floors:

| Dataset | V56A ACC | Required Floor | Gate |
| --- | ---: | ---: | --- |
| ACM | 0.9005 | 0.8888 | PASS |
| DBLP | 0.6919 | 0.6610 | PASS |
| Squirrel | 0.3011 | 0.2800 | PASS |

## 8. Safety Gate

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
| DBLP | 0.0000 | PASS |
| Flickr | 0.0050 | PASS |
| Texas | 0.0000 | PASS |
| Squirrel | 0.0012 | PASS |
| Chameleon | 0.0048 | PASS |

## 9. Heterophily Stress Gate

Preregistered requirement for Texas, Squirrel, and Chameleon:

```text
At least 2/3 must satisfy:
abs(embedding_posterior_gap) <= 0.04
v56a_reliability_mean within [0.08, 0.90]
v56a_effective_anchor_mass >= 0.08

Additionally:
Squirrel ACC >= 0.2800
```

Verdict:

```text
PASS on 2/3, with Squirrel ACC floor satisfied.
```

| Dataset | Emb Gap | Rel Mean | Effective Mass | ACC | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Texas | 0.0000 | 0.0561 | 0.0561 | 0.7377 | FAIL mass |
| Squirrel | 0.0012 | 0.0855 | 0.0855 | 0.3011 | PASS |
| Chameleon | 0.0048 | 0.2171 | 0.2171 | 0.3382 | PASS |

## 10. Gate Summary

| Gate | Verdict |
| --- | --- |
| Red-line | PASS |
| Hybrid residual bound | PASS |
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
hybrid-compensation sweep
schedule variant
reliability formula variant
threshold sweep
V50A anchor hyperparameter sweep
```

## 11. Scientific Interpretation

V56A tested a clean hypothesis:

```text
Hard consensus should remain the safety floor, and soft evidence should only
add fixed positive compensation for medium-evidence nodes.
```

The result is partially useful but not sufficient:

```text
V56A recovers V54A-like safety and improves DBLP ACC to 0.6919, but reliability
mass still stops below 0.08 on DBLP and Texas. The added compensation is too
small to change the core gate outcome, while stronger compensation would be a
new mechanism or sweep and is not authorized by this verdict.
```

Recommended next analysis artifact:

```text
V56A_HYBRID_COMPENSATION_LIMIT_ANALYSIS.md
```

Likely next route to preregister, not implement yet:

```text
v57a_mass_floor_normalized_residual_anchor
```

The next route should not simply increase hybrid compensation. The current
evidence suggests the bottleneck may be loss normalization and mass allocation:
DBLP/Texas can preserve performance while reliability mass remains just below
the gate. A future mechanism would need to target mass allocation explicitly
without dataset-specific routing or threshold relaxation.

## 12. No-Fabrication Status

All numbers in this document come from the local V56A mixed-stress run and its
diagnostics:

```text
results/archive/v51-v57/unified_aptc_9datasets_v56a_hybrid_consensus_floor_residual_anchor.csv
results/archive/v51-v57/unified_aptc_9datasets_v56a_hybrid_consensus_floor_residual_anchor_diagnostics.jsonl
```

No V56A full-run result exists.
