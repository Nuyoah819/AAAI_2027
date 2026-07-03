# V61A Full-Run Verdict

This file records the authorized supported 9-dataset / 260-epoch full run for:

```text
v61a_quantile_coverage_self_distillation_guard
```

It follows `V61A_EXPANSION_REVIEW.md`.

Verdict:

```text
STOP V61A FULL RUN.
```

## 1. Pre-Run Checks

Static check:

```powershell
python -m py_compile core\e2e\sect_coco_e2e.py scripts\run_unified_aptc_9datasets.py
```

Verdict:

```text
PASS
```

Process cleanliness:

```text
No residual training process was present before launch. The only process caught
during the pre-run check was the transient check command itself.
```

## 2. Run

Authorized command:

```powershell
conda run --no-capture-output -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v61a_quantile_coverage_self_distillation_guard --datasets "acm,dblp,pubmed,wiki,flickr,blogcatalog,squirrel,texas,chameleon" --epochs 260 --device cuda --log-level WARNING
```

Run status:

```text
9/9 supported datasets completed with status=ok.
```

Output files:

```text
results/archive/v58-v63/unified_aptc_9datasets_v61a_quantile_coverage_self_distillation_guard.csv
results/archive/v58-v63/unified_aptc_9datasets_v61a_quantile_coverage_self_distillation_guard_diagnostics.jsonl
```

## 3. Result Summary

| Dataset | ACC | NMI | ARI | Emb Gap |
| --- | ---: | ---: | ---: | ---: |
| ACM | 0.9160 | 0.7143 | 0.7676 | 0.0000 |
| DBLP | 0.7281 | 0.4791 | 0.4449 | 0.0000 |
| PubMed | 0.5183 | 0.0857 | 0.0803 | 0.0000 |
| Wiki | 0.3252 | 0.2942 | 0.1321 | 0.0000 |
| Flickr | 0.3197 | 0.1772 | 0.0927 | -0.0632 |
| BlogCatalog | 0.8562 | 0.6691 | 0.6886 | 0.0004 |
| Squirrel | 0.2103 | 0.0130 | 0.0002 | 0.0927 |
| Texas | 0.6995 | 0.4608 | 0.5269 | -0.0273 |
| Chameleon | 0.3412 | 0.1635 | 0.0739 | 0.0044 |

## 4. Red-Line Gate

Requirement:

```text
legacy_head_used=false on 9/9
v50a-v60a_enabled=false on 9/9
v61a_enabled=true on 9/9
```

Verdict:

```text
PASS on 9/9.
```

## 5. Teacher And Guard Gate

Requirement:

```text
teacher ready by epoch 80
guard gamma = 0.0 at epoch 80
guard gamma = 1.0 at epoch 100 and final
teacher active/top-k active ratio @80 >= 0.10
guard loss finite
```

Verdict:

```text
PASS on 9/9.
```

| Dataset | Ready @80 | Guard @80 | Guard @100 | Active @80 | Top-k @80 | Guard Loss @260 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ACM | true | 0.0000 | 1.0000 | 0.9997 | 0.1002 | 0.0624 |
| DBLP | true | 0.0000 | 1.0000 | 0.9919 | 0.1001 | 0.0226 |
| PubMed | true | 0.0000 | 1.0000 | 0.9985 | 0.1000 | 0.0359 |
| Wiki | true | 0.0000 | 1.0000 | 0.1002 | 0.1002 | 0.0049 |
| Flickr | true | 0.0000 | 1.0000 | 0.1001 | 0.1001 | 0.1151 |
| BlogCatalog | true | 0.0000 | 1.0000 | 0.5219 | 0.1001 | 0.0037 |
| Squirrel | true | 0.0000 | 1.0000 | 0.9391 | 0.1002 | 0.0599 |
| Texas | true | 0.0000 | 1.0000 | 0.9781 | 0.1038 | 0.1391 |
| Chameleon | true | 0.0000 | 1.0000 | 0.9592 | 0.1001 | 0.0272 |

## 6. Release Schedule Gate

Requirement:

```text
release gamma @1 = 1.0
release gamma @80 = 1.0
release gamma @260 = 0.25
```

Verdict:

```text
PASS on 9/9.
```

## 7. Mass And Reliability Gate

Verdict:

```text
PASS.
Effective anchor mass passes on 7/9. The preregistered non-weak-anchor
reliability lower bound passes on ACM, DBLP, PubMed, Wiki, Squirrel, Texas,
and Chameleon.
```

| Dataset | Raw Rel | Rel Mean | Mass Scale | Effective Mass | Anchor Agreement @260 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ACM | 0.1497 | 0.1497 | 1.0000 | 0.1497 | 0.9756 |
| DBLP | 0.0783 | 0.0800 | 1.0221 | 0.0800 | 0.5832 |
| PubMed | 0.3048 | 0.3048 | 1.0000 | 0.3048 | 0.5740 |
| Wiki | 0.0647 | 0.0800 | 1.2360 | 0.0800 | 0.4494 |
| Flickr | 0.0054 | 0.0082 | 1.5000 | 0.0082 | 0.0109 |
| BlogCatalog | 0.0160 | 0.0240 | 1.5000 | 0.0240 | 0.1088 |
| Squirrel | 0.0845 | 0.0845 | 1.0000 | 0.0845 | 0.2182 |
| Texas | 0.0571 | 0.0800 | 1.4008 | 0.0800 | 0.4575 |
| Chameleon | 0.2216 | 0.2216 | 1.0000 | 0.2216 | 0.4906 |

## 8. Posterior/Readout Safety Gate

Requirement:

```text
abs(embedding_posterior_gap) <= 0.04 on at least 8/9
no dataset has abs(embedding_posterior_gap) > 0.08
```

Verdict:

```text
FAIL.
Only 7/9 satisfy the 0.04 bound, and Squirrel violates the 0.08 hard bound.
```

| Dataset | Emb Gap | Gate |
| --- | ---: | --- |
| ACM | 0.0000 | PASS |
| DBLP | 0.0000 | PASS |
| PubMed | 0.0000 | PASS |
| Wiki | 0.0000 | PASS |
| Flickr | -0.0632 | FAIL soft bound |
| BlogCatalog | 0.0004 | PASS |
| Squirrel | 0.0927 | FAIL hard bound |
| Texas | -0.0273 | PASS |
| Chameleon | 0.0044 | PASS |

## 9. Preservation Floors

Requirement:

```text
ACM ACC >= 0.8888
DBLP ACC >= 0.6610
Squirrel ACC >= 0.2800
```

Verdict:

```text
FAIL.
Squirrel violates the preregistered preservation floor.
```

| Dataset | ACC | Floor | Gate |
| --- | ---: | ---: | --- |
| ACM | 0.9160 | 0.8888 | PASS |
| DBLP | 0.7281 | 0.6610 | PASS |
| Squirrel | 0.2103 | 0.2800 | FAIL |

## 10. Drift-Repair Gate

Requirement:

```text
PubMed ACC >= 0.5200
Flickr ACC >= 0.3500
Squirrel ACC >= 0.2800
Texas ACC >= 0.7000
```

Verdict:

```text
FAIL.
All four long-run drift-repair floors miss, with Texas missing narrowly and
Squirrel returning to the V57A/V59A failed regime.
```

| Dataset | V57A 260E | V59A 260E | V61A 260E | Floor | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| PubMed | 0.4788 | 0.5261 | 0.5183 | 0.5200 | FAIL |
| Flickr | 0.2779 | 0.2630 | 0.3197 | 0.3500 | FAIL |
| Squirrel | 0.2102 | 0.2103 | 0.2103 | 0.2800 | FAIL |
| Texas | 0.6175 | 0.7213 | 0.6995 | 0.7000 | FAIL |

## 11. Gate Summary

| Gate | Verdict |
| --- | --- |
| Execution | PASS |
| Red-line | PASS |
| Teacher/guard coverage | PASS |
| Release schedule | PASS |
| Mass/reliability | PASS |
| Posterior/readout safety | FAIL |
| Preservation floors | FAIL |
| Drift repair | FAIL |

Decision:

```text
STOP V61A FULL RUN.
```

No V61A tuning, rerun, seed sweep, confidence-floor sweep, coverage sweep,
guard-weight sweep, teacher-epoch sweep, dataset-specific branch, final-label
selector, or 100e/260e cherry-pick is authorized.

The next artifact must be:

```text
V61A_FAILURE_ANALYSIS.md
```

## 12. No-Fabrication Status

All V61A values in this verdict come from the local V61A 260-epoch run and its
diagnostics. V57A and V59A comparison values come from local full-run verdict
artifacts.
