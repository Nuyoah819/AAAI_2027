# V57A Full-Run Verdict

This file records the preregistered supported 9-dataset / 260-epoch full run
for `v57a_mass_floor_normalized_residual_anchor`.

It follows `V57A_FULL_RUN_PREREGISTRATION.md`.

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
No residual training process was present before launch. The only transient
process detected during the pre-run check was the py_compile conda process,
which exited before the full run was launched.
```

## 2. Run

Authorized command:

```powershell
conda run --no-capture-output -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v57a_mass_floor_normalized_residual_anchor --datasets "acm,dblp,pubmed,wiki,flickr,blogcatalog,squirrel,texas,chameleon" --epochs 260 --device cuda --log-level WARNING
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

The result files contain earlier V57A connectivity, 80-epoch mixed-stress,
corrected 9-dataset 80-epoch smoke, and the 260-epoch records. This verdict uses
the latest record per supported dataset.

## 3. Result Summary

| Dataset | ACC | NMI | ARI | Emb Gap |
| --- | ---: | ---: | ---: | ---: |
| ACM | 0.9177 | 0.7130 | 0.7705 | 0.0000 |
| DBLP | 0.7271 | 0.4911 | 0.4459 | -0.0002 |
| PubMed | 0.4788 | 0.0708 | 0.0581 | 0.0000 |
| Wiki | 0.3501 | 0.3227 | 0.1647 | -0.0042 |
| Flickr | 0.2779 | 0.1597 | 0.0865 | -0.0057 |
| BlogCatalog | 0.8585 | 0.6769 | 0.6920 | 0.0000 |
| Squirrel | 0.2102 | 0.0129 | 0.0002 | 0.0000 |
| Texas | 0.6175 | 0.3329 | 0.3114 | 0.0055 |
| Chameleon | 0.3421 | 0.1699 | 0.0804 | 0.0105 |

## 4. Execution Gate

Requirement:

```text
9/9 datasets complete with status=ok.
```

Verdict:

```text
PASS.
```

## 5. Red-Line Gate

Requirement:

```text
legacy_head_used=false
v50a_enabled=false
v51a_enabled=false
v52a_enabled=false
v53a_enabled=false
v54a_enabled=false
v55a_enabled=false
v56a_enabled=false
v57a_enabled=true
```

Verdict:

```text
PASS on 9/9.
```

| Dataset | Legacy | V50A | V51A | V52A | V53A | V54A | V55A | V56A | V57A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ACM | false | false | false | false | false | false | false | false | true |
| DBLP | false | false | false | false | false | false | false | false | true |
| PubMed | false | false | false | false | false | false | false | false | true |
| Wiki | false | false | false | false | false | false | false | false | true |
| Flickr | false | false | false | false | false | false | false | false | true |
| BlogCatalog | false | false | false | false | false | false | false | false | true |
| Squirrel | false | false | false | false | false | false | false | false | true |
| Texas | false | false | false | false | false | false | false | false | true |
| Chameleon | false | false | false | false | false | false | false | false | true |

## 6. Mass-Normalization Gate

Requirement:

```text
v57a_target_mass = 0.08
v57a_max_mass_scale = 1.50
v57a_max_reliability_cap = 0.90
1.0 <= v57a_mass_scale <= 1.50
0.0 <= v57a_reliability_mean <= 0.90
```

Verdict:

```text
PASS on 9/9.
```

| Dataset | Raw Rel | Mass Scale | Scaled Rel | Target | Max Scale | Max Cap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ACM | 0.1492 | 1.0000 | 0.1492 | 0.0800 | 1.5000 | 0.9000 |
| DBLP | 0.0785 | 1.0194 | 0.0800 | 0.0800 | 1.5000 | 0.9000 |
| PubMed | 0.3437 | 1.0000 | 0.3437 | 0.0800 | 1.5000 | 0.9000 |
| Wiki | 0.0651 | 1.2293 | 0.0800 | 0.0800 | 1.5000 | 0.9000 |
| Flickr | 0.0055 | 1.5000 | 0.0082 | 0.0800 | 1.5000 | 0.9000 |
| BlogCatalog | 0.0161 | 1.5000 | 0.0241 | 0.0800 | 1.5000 | 0.9000 |
| Squirrel | 0.0860 | 1.0000 | 0.0860 | 0.0800 | 1.5000 | 0.9000 |
| Texas | 0.0574 | 1.3949 | 0.0800 | 0.0800 | 1.5000 | 0.9000 |
| Chameleon | 0.2261 | 1.0000 | 0.2261 | 0.0800 | 1.5000 | 0.9000 |

## 7. Reliability Non-Collapse Gate

Requirement:

```text
effective anchor mass >= 0.08 on at least 6/9 datasets
reliable-node ratio >= 0.05 on at least 4/9 datasets
```

Verdict:

```text
PASS.
Effective anchor mass passes on 7/9. Reliable-node ratio passes on 5/9.
Flickr and BlogCatalog remain weak-anchor exceptions.
```

| Dataset | Rel Mean | Reliable Ratio | Effective Mass | Gate |
| --- | ---: | ---: | ---: | --- |
| ACM | 0.1492 | 0.2043 | 0.1492 | PASS |
| DBLP | 0.0800 | 0.0000 | 0.0800 | PASS mass, FAIL ratio |
| PubMed | 0.3437 | 0.7052 | 0.3437 | PASS |
| Wiki | 0.0800 | 0.0636 | 0.0800 | PASS |
| Flickr | 0.0082 | 0.0000 | 0.0082 | weak-anchor non-use |
| BlogCatalog | 0.0241 | 0.0000 | 0.0241 | weak-anchor low mass |
| Squirrel | 0.0860 | 0.0596 | 0.0860 | PASS |
| Texas | 0.0800 | 0.0328 | 0.0800 | PASS mass, FAIL ratio |
| Chameleon | 0.2261 | 0.5512 | 0.2261 | PASS |

No hard near-zero reliability failure occurs on ACM, DBLP, PubMed, Wiki,
Squirrel, Texas, or Chameleon. No dataset exceeds the max reliability cap.

## 8. Anchor-Usefulness Gate

Requirement:

```text
v57a_weighted_q_anchor_agreement_epoch_260 >
v57a_weighted_q_anchor_agreement_epoch_1 on at least 6/9 datasets
```

Note:

```text
The diagnostics schema records the final 260-epoch value as
v57a_weighted_q_anchor_agreement. The epoch_1 snapshot is explicit.
```

Verdict:

```text
PASS on 8/9.
```

| Dataset | Agreement @1 | Agreement @260 | Movement | Gate |
| --- | ---: | ---: | ---: | --- |
| ACM | 0.4358 | 0.9730 | +0.5372 | PASS |
| DBLP | 0.3128 | 0.5877 | +0.2749 | PASS |
| PubMed | 0.2311 | 0.6938 | +0.4627 | PASS |
| Wiki | 0.0407 | 0.4589 | +0.4182 | PASS |
| Flickr | 0.0243 | 0.0124 | -0.0119 | FAIL |
| BlogCatalog | 0.0998 | 0.1097 | +0.0099 | PASS |
| Squirrel | 0.1220 | 0.2400 | +0.1180 | PASS |
| Texas | 0.3893 | 0.4671 | +0.0778 | PASS |
| Chameleon | 0.1756 | 0.5164 | +0.3408 | PASS |

## 9. Posterior/Readout Safety Gate

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
| DBLP | -0.0002 | PASS |
| PubMed | 0.0000 | PASS |
| Wiki | -0.0042 | PASS |
| Flickr | -0.0057 | PASS |
| BlogCatalog | 0.0000 | PASS |
| Squirrel | 0.0000 | PASS |
| Texas | 0.0055 | PASS |
| Chameleon | 0.0105 | PASS |

## 10. Preservation Floors

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
| ACM | 0.9177 | 0.8888 | PASS |
| DBLP | 0.7271 | 0.6610 | PASS |
| Squirrel | 0.2102 | 0.2800 | FAIL |

## 11. Full-Length Drift Watch

Flickr:

```text
Weak-anchor non-use is no longer harmless. ACC drops from 0.4133 at the
corrected 80e smoke to 0.2779 at 260e, while agreement movement remains
negative. This is a full-length drift warning, not an anchor-mass gate failure.
```

BlogCatalog:

```text
Reliability mass remains low, but ACC rises from 0.8383 at 80e to 0.8585 at
260e. This weak-anchor exception is tolerable by the preregistered diagnostic
boundary.
```

Wiki:

```text
Wiki stays weak in ACC: 0.3522 at 80e and 0.3501 at 260e. NMI/ARI improve
slightly, but the central ACC weakness is not solved by longer training.
```

Chameleon:

```text
Chameleon stays essentially stable in ACC: 0.3421 at both 80e and 260e, with
NMI/ARI improving modestly. It is not the failure driver.
```

Additional drift:

```text
PubMed drops from 0.5822 at 80e to 0.4788 at 260e. Texas drops from 0.7377 to
0.6175. Squirrel drops from 0.3005 to 0.2102 and triggers the hard preservation
failure.
```

## 12. Gate Summary

| Gate | Verdict |
| --- | --- |
| Execution | PASS |
| Red-line | PASS |
| Mass-normalization | PASS |
| Reliability non-collapse | PASS |
| Anchor usefulness | PASS |
| Posterior/readout safety | PASS |
| Preservation floors | FAIL |

Decision:

```text
STOP V57A FULL RUN.
```

V57A is not authorized for paper-facing main results. The next artifact must be
a failure analysis. No V57A constant tuning, seed sweep, restart selection, or
post-hoc final-label selector is authorized.

## 13. No-Fabrication Status

All numbers in this document come from the local V57A 260-epoch run and its
diagnostics:

```text
results/archive/v51-v57/unified_aptc_9datasets_v57a_mass_floor_normalized_residual_anchor.csv
results/archive/v51-v57/unified_aptc_9datasets_v57a_mass_floor_normalized_residual_anchor_diagnostics.jsonl
```
