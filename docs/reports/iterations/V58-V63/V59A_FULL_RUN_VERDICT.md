# V59A Full-Run Verdict

This file records the authorized supported 9-dataset / 260-epoch full run for
`v59a_post80_anchor_release_residual_compactness`.

It follows `V59A_EXPANSION_REVIEW.md`.

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
No residual training process was present before launch. The only process caught
during the pre-run check was the transient py_compile conda process, which
exited before the full run was launched.
```

## 2. Run

Authorized command:

```powershell
conda run --no-capture-output -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v59a_post80_anchor_release_residual_compactness --datasets "acm,dblp,pubmed,wiki,flickr,blogcatalog,squirrel,texas,chameleon" --epochs 260 --device cuda --log-level WARNING
```

Run status:

```text
9/9 supported datasets completed with status=ok.
```

Output files:

```text
results/archive/v58-v63/unified_aptc_9datasets_v59a_post80_anchor_release_residual_compactness.csv
results/archive/v58-v63/unified_aptc_9datasets_v59a_post80_anchor_release_residual_compactness_diagnostics.jsonl
```

## 3. Result Summary

| Dataset | ACC | NMI | ARI | Emb Gap |
| --- | ---: | ---: | ---: | ---: |
| ACM | 0.9180 | 0.7238 | 0.7732 | 0.0000 |
| DBLP | 0.7321 | 0.4893 | 0.4459 | 0.0002 |
| PubMed | 0.5261 | 0.0882 | 0.0849 | 0.0000 |
| Wiki | 0.3301 | 0.2901 | 0.1321 | -0.0071 |
| Flickr | 0.2630 | 0.1523 | 0.0808 | 0.0277 |
| BlogCatalog | 0.8584 | 0.6733 | 0.6936 | 0.0000 |
| Squirrel | 0.2103 | 0.0130 | 0.0002 | 0.0000 |
| Texas | 0.7213 | 0.4851 | 0.5947 | 0.0000 |
| Chameleon | 0.3412 | 0.1651 | 0.0753 | 0.0000 |

## 4. Execution Gate

Requirement:

```text
9/9 supported datasets complete with status=ok
```

Verdict:

```text
PASS.
```

## 5. Red-Line Gate

Requirement:

```text
legacy_head_used=false
v50a-v58a_enabled=false
v59a_enabled=true
```

Verdict:

```text
PASS on 9/9.
```

| Dataset | Legacy | V50A | V51A | V52A | V53A | V54A | V55A | V56A | V57A | V58A | V59A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ACM | false | false | false | false | false | false | false | false | false | false | true |
| DBLP | false | false | false | false | false | false | false | false | false | false | true |
| PubMed | false | false | false | false | false | false | false | false | false | false | true |
| Wiki | false | false | false | false | false | false | false | false | false | false | true |
| Flickr | false | false | false | false | false | false | false | false | false | false | true |
| BlogCatalog | false | false | false | false | false | false | false | false | false | false | true |
| Squirrel | false | false | false | false | false | false | false | false | false | false | true |
| Texas | false | false | false | false | false | false | false | false | false | false | true |
| Chameleon | false | false | false | false | false | false | false | false | false | false | true |

## 6. Release Schedule Gate

Requirement:

```text
v59a_release_gamma_epoch_1 = 1.0
v59a_release_gamma_epoch_40 = 1.0
v59a_release_gamma_epoch_80 = 1.0
v59a_release_gamma at final epoch = 0.25
```

Verdict:

```text
PASS on 9/9.
```

| Dataset | Release @1 | Release @40 | Release @80 | Release @260 |
| --- | ---: | ---: | ---: | ---: |
| ACM | 1.0000 | 1.0000 | 1.0000 | 0.2500 |
| DBLP | 1.0000 | 1.0000 | 1.0000 | 0.2500 |
| PubMed | 1.0000 | 1.0000 | 1.0000 | 0.2500 |
| Wiki | 1.0000 | 1.0000 | 1.0000 | 0.2500 |
| Flickr | 1.0000 | 1.0000 | 1.0000 | 0.2500 |
| BlogCatalog | 1.0000 | 1.0000 | 1.0000 | 0.2500 |
| Squirrel | 1.0000 | 1.0000 | 1.0000 | 0.2500 |
| Texas | 1.0000 | 1.0000 | 1.0000 | 0.2500 |
| Chameleon | 1.0000 | 1.0000 | 1.0000 | 0.2500 |

## 7. Mass And Reliability Gate

Requirement:

```text
effective anchor mass >= 0.08 on at least 6/9 datasets
reliable-node ratio >= 0.05 on at least 4/9 datasets
```

Verdict:

```text
PASS.
Effective anchor mass passes on 7/9. Reliable-node ratio passes on 5/9.
```

| Dataset | Raw Rel | Mass Scale | Rel Mean | Reliable Ratio | Effective Mass | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| ACM | 0.1499 | 1.0000 | 0.1499 | 0.2060 | 0.1499 | PASS |
| DBLP | 0.0788 | 1.0154 | 0.0800 | 0.0000 | 0.0800 | PASS mass, FAIL ratio |
| PubMed | 0.3040 | 1.0000 | 0.3040 | 0.6849 | 0.3040 | PASS |
| Wiki | 0.0636 | 1.2576 | 0.0800 | 0.0582 | 0.0800 | PASS |
| Flickr | 0.0054 | 1.5000 | 0.0081 | 0.0000 | 0.0081 | weak-anchor |
| BlogCatalog | 0.0160 | 1.5000 | 0.0240 | 0.0000 | 0.0240 | weak-anchor |
| Squirrel | 0.0841 | 1.0000 | 0.0841 | 0.0523 | 0.0841 | PASS |
| Texas | 0.0567 | 1.4105 | 0.0800 | 0.0328 | 0.0800 | PASS mass, FAIL ratio |
| Chameleon | 0.2215 | 1.0000 | 0.2215 | 0.5253 | 0.2215 | PASS |

## 8. Posterior/Readout Safety Gate

Requirement:

```text
abs(embedding_posterior_gap) <= 0.04 on at least 8/9 datasets
no dataset has abs(embedding_posterior_gap) > 0.08
```

Verdict:

```text
PASS on 9/9.
```

| Dataset | Emb Gap | Gate |
| --- | ---: | --- |
| ACM | 0.0000 | PASS |
| DBLP | 0.0002 | PASS |
| PubMed | 0.0000 | PASS |
| Wiki | -0.0071 | PASS |
| Flickr | 0.0277 | PASS |
| BlogCatalog | 0.0000 | PASS |
| Squirrel | 0.0000 | PASS |
| Texas | 0.0000 | PASS |
| Chameleon | 0.0000 | PASS |

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
| ACM | 0.9180 | 0.8888 | PASS |
| DBLP | 0.7321 | 0.6610 | PASS |
| Squirrel | 0.2103 | 0.2800 | FAIL |

## 10. Drift-Repair Gate

Requirement:

```text
Squirrel ACC >= 0.2800
Texas ACC >= 0.7000
Flickr ACC >= 0.3500
PubMed ACC >= 0.5200
```

Verdict:

```text
FAIL.
PubMed and Texas improve enough to pass, but Flickr and Squirrel still fail.
```

| Dataset | V57A 260E ACC | V59A 260E ACC | Repair Floor | Gate |
| --- | ---: | ---: | ---: | --- |
| PubMed | 0.4788 | 0.5261 | 0.5200 | PASS |
| Flickr | 0.2779 | 0.2630 | 0.3500 | FAIL |
| Squirrel | 0.2102 | 0.2103 | 0.2800 | FAIL |
| Texas | 0.6175 | 0.7213 | 0.7000 | PASS |

Interpretation:

```text
Post-80 release helps Texas and modestly helps PubMed, but it does not repair
the two hardest long-run failures. Squirrel is essentially unchanged from V57A
260e, and Flickr becomes slightly worse.
```

## 11. Gate Summary

| Gate | Verdict |
| --- | --- |
| Execution | PASS |
| Red-line | PASS |
| Release schedule | PASS |
| Mass/reliability | PASS |
| Posterior/readout safety | PASS |
| Preservation floors | FAIL |
| Drift-repair | FAIL |

Decision:

```text
STOP V59A FULL RUN.
```

V59A is not authorized for paper-facing main results. No V59A tuning, rerun,
seed sweep, release-floor sweep, schedule sweep, reliability formula change,
dataset-specific branch, or post-hoc final-label selector is authorized.

The next artifact must be:

```text
V59A_FAILURE_ANALYSIS.md
```

## 12. No-Fabrication Status

All V59A numbers in this verdict come from the local V59A 260-epoch run and its
diagnostics. V57A comparison values come from `V57A_FULL_RUN_VERDICT.md`.
