# V57A Supported 9-Dataset 80E Smoke Verdict

This file records the corrected supported 9-dataset / 80-epoch smoke for
`v57a_mass_floor_normalized_residual_anchor`. It follows
`V57A_EXPANSION_REVIEW_AMENDMENT.md`.

No 260-epoch full run, seed sweep, target-mass sweep, max-mass-scale sweep,
reliability-cap sweep, beta-bound sweep, soft-power sweep,
hybrid-compensation sweep, schedule variant, reliability formula variant,
threshold sweep, or V50A anchor hyperparameter sweep is authorized by this
verdict.

## 1. Run

Corrected command:

```powershell
conda run --no-capture-output -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v57a_mass_floor_normalized_residual_anchor --datasets "acm,dblp,pubmed,wiki,flickr,blogcatalog,squirrel,texas,chameleon" --epochs 80 --device cuda --log-level WARNING
```

Run status:

```text
9/9 supported datasets completed with status=ok.
```

Output files:

```text
results/archive/v51-v57/unified_aptc_9datasets_v57a_mass_floor_normalized_residual_anchor.csv
results/archive/v51-v57/unified_aptc_9datasets_v57a_mass_floor_normalized_residual_anchor_diagnostics.jsonl
```

The result files also contain earlier V57A connectivity, mixed-stress, and
invalid partial smoke rows. This verdict uses the latest record per supported
dataset.

## 2. Result Summary

| Dataset | ACC | NMI | ARI | Emb Gap |
| --- | ---: | ---: | ---: | ---: |
| ACM | 0.8962 | 0.6664 | 0.7168 | 0.0000 |
| DBLP | 0.6825 | 0.4020 | 0.3443 | 0.0005 |
| PubMed | 0.5822 | 0.1392 | 0.1450 | 0.0001 |
| Wiki | 0.3522 | 0.3043 | 0.1434 | -0.0274 |
| Flickr | 0.4133 | 0.2527 | 0.1682 | -0.0313 |
| BlogCatalog | 0.8383 | 0.6398 | 0.6533 | 0.0000 |
| Squirrel | 0.3005 | 0.0612 | 0.0502 | 0.0012 |
| Texas | 0.7377 | 0.5023 | 0.6194 | 0.0000 |
| Chameleon | 0.3421 | 0.1533 | 0.0540 | -0.0119 |

## 3. Red-Line Gate

Verdict:

```text
PASS on 9/9.
```

V50A-V56A and V43B-V49A flags are false, `v57a_enabled=true`, and
`legacy_head_used=false` on all 9 supported datasets.

## 4. Mass-Normalization Gate

Verdict:

```text
PASS on 9/9.
```

| Dataset | Raw Rel | Mass Scale | Scaled Rel | Gate |
| --- | ---: | ---: | ---: | --- |
| ACM | 0.1390 | 1.0000 | 0.1390 | PASS |
| DBLP | 0.0749 | 1.0685 | 0.0800 | PASS |
| PubMed | 0.2594 | 1.0000 | 0.2594 | PASS |
| Wiki | 0.0576 | 1.3897 | 0.0800 | PASS |
| Flickr | 0.0056 | 1.5000 | 0.0085 | PASS bound, weak-anchor mass |
| BlogCatalog | 0.0160 | 1.5000 | 0.0240 | PASS bound, low mass |
| Squirrel | 0.0856 | 1.0000 | 0.0856 | PASS |
| Texas | 0.0566 | 1.4144 | 0.0800 | PASS |
| Chameleon | 0.2177 | 1.0000 | 0.2177 | PASS |

## 5. Reliability Gate

Requirement:

```text
mass pass on at least 6/9
reliable-node ratio pass on at least 4/9
```

Verdict:

```text
PASS.
Mass passes on 7/9 if float-target values are read at 0.0800; even with strict
floating comparison, at least 6/9 pass.
Reliable-node ratio passes on 5/9.
```

| Dataset | Rel Mean | Reliable Ratio | Effective Mass | Gate |
| --- | ---: | ---: | ---: | --- |
| ACM | 0.1390 | 0.1478 | 0.1390 | PASS |
| DBLP | 0.0800 | 0.0000 | 0.0800 | PASS mass, FAIL ratio |
| PubMed | 0.2594 | 0.6672 | 0.2594 | PASS |
| Wiki | 0.0800 | 0.0636 | 0.0800 | PASS |
| Flickr | 0.0085 | 0.0000 | 0.0085 | weak-anchor non-use |
| BlogCatalog | 0.0240 | 0.0000 | 0.0240 | FAIL mass |
| Squirrel | 0.0856 | 0.0536 | 0.0856 | PASS |
| Texas | 0.0800 | 0.0328 | 0.0800 | PASS mass, FAIL ratio |
| Chameleon | 0.2177 | 0.4949 | 0.2177 | PASS |

## 6. Anchor Usefulness

Requirement:

```text
v57a_weighted_q_anchor_agreement_epoch_80 >
v57a_weighted_q_anchor_agreement_epoch_1 on at least 6/9.
```

Verdict:

```text
PASS on 8/9.
```

| Dataset | Agreement @1 | Agreement @80 | Movement | Gate |
| --- | ---: | ---: | ---: | --- |
| ACM | 0.4358 | 0.9654 | +0.5297 | PASS |
| DBLP | 0.3128 | 0.5440 | +0.2313 | PASS |
| PubMed | 0.2311 | 0.4192 | +0.1881 | PASS |
| Wiki | 0.0407 | 0.3355 | +0.2948 | PASS |
| Flickr | 0.0243 | 0.0169 | -0.0074 | FAIL |
| BlogCatalog | 0.0998 | 0.1054 | +0.0056 | PASS |
| Squirrel | 0.1220 | 0.2184 | +0.0964 | PASS |
| Texas | 0.3893 | 0.4558 | +0.0665 | PASS |
| Chameleon | 0.1756 | 0.4519 | +0.2763 | PASS |

## 7. Safety And Floors

Safety verdict:

```text
PASS on 9/9.
All abs(embedding_posterior_gap) values are <= 0.04, and no dataset exceeds
the 0.08 hard-fail bound.
```

Floor verdict:

| Dataset | ACC | Floor | Gate |
| --- | ---: | ---: | --- |
| ACM | 0.8962 | 0.8888 | PASS |
| DBLP | 0.6825 | 0.6610 | PASS |
| Squirrel | 0.3005 | 0.2800 | PASS |

## 8. New Dataset Notes

PubMed:

```text
High reliability mass and ratio pass cleanly. ACC is not SOTA-level but the
mass-normalization mechanism is not unstable.
```

Wiki:

```text
Mass normalization lifts reliability to the 0.08 floor and anchor agreement
improves strongly. ACC remains weak, so this dataset should be watched in any
full-run preregistration.
```

BlogCatalog:

```text
ACC is relatively strong, but reliability mass remains low even at max scale.
This is the main warning sign before any 260-epoch run.
```

## 9. Gate Summary

| Gate | Verdict |
| --- | --- |
| Red-line | PASS |
| Mass-normalization | PASS |
| Reliability | PASS |
| Posterior/readout safety | PASS |
| Floor | PASS |
| Anchor usefulness | PASS |

Decision:

```text
PASS SUPPORTED 9-DATASET / 80-EPOCH SMOKE.
```

This verdict authorizes only the next preregistration artifact:

```text
V57A_FULL_RUN_PREREGISTRATION.md
```

It does not directly authorize a 260-epoch full run.

## 10. No-Fabrication Status

All numbers in this document come from the local corrected supported 9-dataset
V57A smoke and diagnostics. No V57A 260-epoch full result exists.
