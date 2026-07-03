# V62A Full-Run Verdict

This file records the authorized supported 9-dataset / 260-epoch full run for:

```text
v62a_drift_responsive_self_distillation_guard
```

It follows `V62A_EXPANSION_REVIEW.md`.

Verdict:

```text
STOP V62A FULL RUN.
```

## 1. Pre-Run Checks

Static check:

```powershell
conda run -n aaai-e2e-subspace python -m py_compile core\e2e\sect_coco_e2e.py scripts\run_unified_aptc_9datasets.py
```

Verdict:

```text
PASS
```

Process cleanliness:

```text
No residual training process was present before launch. The process check
captured only the transient compile/check commands themselves.
```

## 2. Run

Authorized command:

```powershell
conda run --no-capture-output -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v62a_drift_responsive_self_distillation_guard --datasets "acm,dblp,pubmed,wiki,flickr,blogcatalog,squirrel,texas,chameleon" --epochs 260 --device cuda --log-level WARNING
```

Run status:

```text
9/9 supported datasets completed with status=ok.
```

Output files:

```text
results/archive/v58-v63/unified_aptc_9datasets_v62a_drift_responsive_self_distillation_guard.csv
results/archive/v58-v63/unified_aptc_9datasets_v62a_drift_responsive_self_distillation_guard_diagnostics.jsonl
```

## 3. Result Summary

| Dataset | ACC | NMI | ARI | Emb Gap |
| --- | ---: | ---: | ---: | ---: |
| ACM | 0.9131 | 0.7054 | 0.7606 | 0.0000 |
| DBLP | 0.7190 | 0.4778 | 0.4391 | 0.0000 |
| PubMed | 0.5203 | 0.0848 | 0.0815 | 0.0000 |
| Wiki | 0.3277 | 0.2886 | 0.1340 | 0.0112 |
| Flickr | 0.2964 | 0.1740 | 0.0894 | 0.0203 |
| BlogCatalog | 0.8537 | 0.6687 | 0.6830 | 0.0000 |
| Squirrel | 0.2103 | 0.0133 | 0.0003 | 0.0000 |
| Texas | 0.7213 | 0.4602 | 0.5741 | -0.0055 |
| Chameleon | 0.3390 | 0.1637 | 0.0754 | -0.0022 |

## 4. Red-Line Gate

Requirement:

```text
legacy_head_used=false on 9/9
v50a-v61a_enabled=false on 9/9
v62a_enabled=true on 9/9
```

Verdict:

```text
PASS on 9/9.
```

## 5. Teacher, Guard, And Drift Gate

Requirement:

```text
teacher ready by epoch 80
guard gamma = 0.0 at epoch 80
guard gamma = 1.0 at epoch 100 and final
teacher active/top-k active ratio @80 >= 0.10
drift gamma = 0.0 at epoch 100
effective multiplier = 1.0 at epoch 100
effective multiplier <= 2.0 at final
guard loss finite
```

Verdict:

```text
PASS on 9/9.
```

| Dataset | Active @80 | Top-k @80 | Drift Score @260 | Drift Gamma @260 | Mult @260 | Guard Loss @260 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ACM | 0.9997 | 0.1002 | 0.0600 | 0.6663 | 1.6663 | 0.0999 |
| DBLP | 0.9911 | 0.1001 | 0.0283 | 0.1383 | 1.1383 | 0.0322 |
| PubMed | 0.9984 | 0.1000 | 0.0338 | 0.2294 | 1.2294 | 0.0415 |
| Wiki | 0.1002 | 0.1002 | 0.0054 | 0.0000 | 1.0000 | 0.0054 |
| Flickr | 0.1001 | 0.1001 | 0.0721 | 0.8679 | 1.8679 | 0.1346 |
| BlogCatalog | 0.5175 | 0.1001 | 0.0040 | 0.0000 | 1.0000 | 0.0040 |
| Squirrel | 0.9335 | 0.1002 | 0.0550 | 0.5836 | 1.5836 | 0.0871 |
| Texas | 0.9672 | 0.1038 | 0.1113 | 1.0000 | 2.0000 | 0.2226 |
| Chameleon | 0.9539 | 0.1001 | 0.0327 | 0.2119 | 1.2119 | 0.0396 |

The intended V62A mechanism was tested: historically drifting datasets
activated a nonzero drift response at epoch 260. The response remained bounded
but did not repair the long-run Squirrel/Flickr failure.

## 6. Release Schedule Gate

Requirement:

```text
release gamma @1 = 1.0
release gamma @80 = 1.0
release gamma at final epoch = 0.25
```

Verdict:

```text
PASS on 9/9.
```

## 7. Mass And Reliability Gate

Verdict:

```text
PASS.
Effective anchor mass is >= 0.08 on 7/9 datasets. Flickr and BlogCatalog
remain weak-anchor exceptions.
```

| Dataset | Rel Mean | Mass Scale | Effective Mass | Anchor Agreement @260 |
| --- | ---: | ---: | ---: | ---: |
| ACM | 0.1496 | 1.0000 | 0.1496 | 0.9755 |
| DBLP | 0.0800 | 1.0165 | 0.0800 | 0.5893 |
| PubMed | 0.3057 | 1.0000 | 0.3057 | 0.5777 |
| Wiki | 0.0800 | 1.2740 | 0.0800 | 0.4190 |
| Flickr | 0.0082 | 1.5000 | 0.0082 | 0.0112 |
| BlogCatalog | 0.0241 | 1.5000 | 0.0241 | 0.1109 |
| Squirrel | 0.0847 | 1.0000 | 0.0847 | 0.2157 |
| Texas | 0.0800 | 1.3979 | 0.0800 | 0.4638 |
| Chameleon | 0.2191 | 1.0000 | 0.2191 | 0.4902 |

## 8. Posterior/Readout Safety Gate

Requirement:

```text
abs(embedding_posterior_gap) <= 0.04 on at least 8/9
no dataset has abs(embedding_posterior_gap) > 0.08
```

Verdict:

```text
PASS.
```

| Dataset | Emb Gap | Gate |
| --- | ---: | --- |
| ACM | 0.0000 | PASS |
| DBLP | 0.0000 | PASS |
| PubMed | 0.0000 | PASS |
| Wiki | 0.0112 | PASS |
| Flickr | 0.0203 | PASS |
| BlogCatalog | 0.0000 | PASS |
| Squirrel | 0.0000 | PASS |
| Texas | -0.0055 | PASS |
| Chameleon | -0.0022 | PASS |

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
| ACM | 0.9131 | 0.8888 | PASS |
| DBLP | 0.7190 | 0.6610 | PASS |
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
PubMed and Texas pass, but Flickr and Squirrel fail. Squirrel remains in the
same long-run failed regime despite nonzero drift response.
```

| Dataset | V61A 260E | V62A 100E | V62A 260E | Floor | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| PubMed | 0.5183 | n/a | 0.5203 | 0.5200 | PASS |
| Flickr | 0.3197 | 0.4079 | 0.2964 | 0.3500 | FAIL |
| Squirrel | 0.2103 | 0.3013 | 0.2103 | 0.2800 | FAIL |
| Texas | 0.6995 | 0.7322 | 0.7213 | 0.7000 | PASS |

## 11. Gate Summary

| Gate | Verdict |
| --- | --- |
| Execution | PASS |
| Red-line | PASS |
| Teacher/base guard | PASS |
| Drift response | PASS wiring, FAIL repair |
| Release schedule | PASS |
| Mass/reliability | PASS |
| Posterior/readout safety | PASS |
| Preservation floors | FAIL |
| Drift repair | FAIL |

Decision:

```text
STOP V62A FULL RUN.
```

No V62A tuning, rerun, seed sweep, confidence-floor sweep, coverage sweep,
guard-weight sweep, drift-floor sweep, drift-scale sweep, drift-boost sweep,
teacher-epoch sweep, dataset-specific branch, final-label selector, or
100e/260e cherry-pick is authorized.

The next artifact must be:

```text
V62A_FAILURE_ANALYSIS.md
```

## 12. No-Fabrication Status

All V62A values in this verdict come from the local V62A 260-epoch run and its
diagnostics. V61A comparison values come from local V61A verdict artifacts.
