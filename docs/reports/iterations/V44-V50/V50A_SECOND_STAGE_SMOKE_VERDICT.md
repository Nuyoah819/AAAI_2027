# v50a_spectral_compactness_anchor Second-Stage Smoke Verdict

This file records the second-stage smoke result for
`v50a_spectral_compactness_anchor`. It follows
`V50A_SECOND_STAGE_PREREGISTRATION.md` and does not authorize a full run,
260-epoch run, or hyperparameter sweep.

## 1. Run

Preregistration file:

```text
V50A_SECOND_STAGE_PREREGISTRATION.md
```

Static check before running:

```powershell
conda run -n aaai-e2e-subspace python -m py_compile core\e2e\sect_coco_e2e.py scripts\run_unified_aptc_9datasets.py
```

Second-stage smoke command:

```powershell
conda run --no-capture-output -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v50a_spectral_compactness_anchor --datasets "pubmed,wiki,blogcatalog,texas,squirrel,chameleon" --epochs 80 --device cuda --log-level WARNING
```

Run status:

```text
6/6 datasets completed with status=ok.
No duplicate second-stage run was used.
No weight, rank, temperature, filter-step, refresh, seed, or final-label change
was made.
```

Output files:

```text
results/archive/v40-v50/unified_aptc_9datasets_v50a_spectral_compactness_anchor.csv
results/archive/v40-v50/unified_aptc_9datasets_v50a_spectral_compactness_anchor_diagnostics.jsonl
```

## 2. Result Summary

Latest complete second-stage records:

| Dataset | ACC | NMI | ARI | Emb Gap |
| --- | ---: | ---: | ---: | ---: |
| PubMed | 0.6215 | 0.2287 | 0.2166 | 0.0000 |
| Wiki | 0.3655 | 0.3253 | 0.1604 | 0.0000 |
| BlogCatalog | 0.8460 | 0.6523 | 0.6696 | 0.0002 |
| Texas | 0.7322 | 0.4807 | 0.5948 | 0.0109 |
| Squirrel | 0.3019 | 0.0632 | 0.0519 | -0.0877 |
| Chameleon | 0.3377 | 0.1597 | 0.0673 | 0.0000 |

Performance interpretation:

- BlogCatalog and Texas remain strong enough to show that V50A is not only an
  ACM/DBLP artifact.
- Wiki, Squirrel, and Chameleon expose clear limits.
- Squirrel is the critical failure because the posterior/readout safety gap
  exceeds the preregistered hard ceiling.

## 3. Red-Line Gate

PASS on 6/6.

| Dataset | Legacy | v43b | v44 | v44b | v45a | v46a | v47a | v48a | v49a | v50a |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PubMed | false | false | false | false | false | false | false | false | false | true |
| Wiki | false | false | false | false | false | false | false | false | false | true |
| BlogCatalog | false | false | false | false | false | false | false | false | false | true |
| Texas | false | false | false | false | false | false | false | false | false | true |
| Squirrel | false | false | false | false | false | false | false | false | false | true |
| Chameleon | false | false | false | false | false | false | false | false | false | true |

V50A did not violate the unified-pipeline red lines.

## 4. Anchor Non-Degeneracy Gate

PASS on 6/6.

Preregistered requirement:

```text
Pass on at least 5/6:
v50a_anchor_cluster_usage_entropy >= 0.60
v50a_anchor_entropy finite
v50a_q_anchor_kl finite
v50a_anchor_loss finite
```

| Dataset | Anchor ACC | Anchor NMI | Anchor ARI | Anchor Entropy | Anchor Conf | Usage Entropy | KL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PubMed | 0.6581 | 0.2734 | 0.2629 | 0.5392 | 0.7733 | 0.9802 | 0.7947 |
| Wiki | 0.4844 | 0.4857 | 0.2728 | 0.8777 | 0.2441 | 0.9883 | 0.5217 |
| BlogCatalog | 0.4355 | 0.2962 | 0.1762 | 0.9855 | 0.2161 | 0.9922 | 0.1744 |
| Texas | 0.3880 | 0.0936 | 0.0661 | 0.9354 | 0.3534 | 0.9952 | 1.2274 |
| Squirrel | 0.2448 | 0.0136 | 0.0088 | 0.8296 | 0.4653 | 0.9632 | 0.5335 |
| Chameleon | 0.3193 | 0.0781 | 0.0693 | 0.5002 | 0.7038 | 0.9602 | 0.9212 |

The anchor is non-collapsed, but its label-aligned quality varies sharply. This
is an important failure-analysis signal, not a reason to tune the anchor.

## 5. Coupling Gate

PASS exactly at the preregistered boundary: 4/6.

Preregistered primary requirement:

```text
v50a_q_anchor_agreement_epoch_80 > v50a_q_anchor_agreement_epoch_1
Pass on at least 4/6.
```

| Dataset | Agreement @1 | Agreement @40 | Agreement @80 | Movement | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| PubMed | 0.2440 | 0.1473 | 0.1579 | -0.0861 | FAIL |
| Wiki | 0.0599 | 0.1351 | 0.1430 | +0.0831 | PASS |
| BlogCatalog | 0.1830 | 0.2273 | 0.2288 | +0.0458 | PASS |
| Texas | 0.3880 | 0.3770 | 0.3716 | -0.0164 | FAIL |
| Squirrel | 0.1165 | 0.1342 | 0.1325 | +0.0160 | PASS |
| Chameleon | 0.1866 | 0.2626 | 0.2644 | +0.0778 | PASS |

Weak-coupling warning:

```text
v50a_q_anchor_agreement_epoch_80 < 0.15
```

Triggered on 2/6 datasets:

```text
Wiki = 0.1430
Squirrel = 0.1325
```

This does not trip the preregistered stop condition of 3 or more weak-coupling
datasets, but the pass is narrow.

## 6. Posterior/Readout Safety Gate

FAIL due to Squirrel hard safety failure.

Preregistered requirement:

```text
abs(embedding_posterior_gap) <= 0.02 on at least 4/6
abs(embedding_posterior_gap) <= 0.04 on at least 5/6
Any dataset with abs(embedding_posterior_gap) > 0.08 is a hard safety failure.
```

| Dataset | Emb Gap | Safety |
| --- | ---: | --- |
| PubMed | 0.0000 | PASS |
| Wiki | 0.0000 | PASS |
| BlogCatalog | 0.0002 | PASS |
| Texas | 0.0109 | PASS |
| Squirrel | -0.0877 | HARD FAIL |
| Chameleon | 0.0000 | PASS |

Although the aggregate 0.04 gate passes on 5/6, Squirrel violates the hard
ceiling. This blocks full-run expansion.

## 7. Heterophily Stress Gate

FAIL.

Preregistered requirement for Texas, Squirrel, and Chameleon:

```text
At least 2/3 must satisfy:
v50a_q_anchor_agreement_epoch_80 > v50a_q_anchor_agreement_epoch_1
abs(embedding_posterior_gap) <= 0.04
```

| Dataset | Coupling Movement | Emb Gap | Gate |
| --- | ---: | ---: | --- |
| Texas | -0.0164 | 0.0109 | FAIL |
| Squirrel | +0.0160 | -0.0877 | FAIL |
| Chameleon | +0.0778 | 0.0000 | PASS |

Only 1/3 passes. This is the clearest second-stage failure boundary.

## 8. Verdict

V50A second-stage smoke does not authorize full expansion.

Gate summary:

| Gate | Verdict |
| --- | --- |
| Red-line | PASS |
| Anchor non-degeneracy | PASS |
| Coupling | PASS, narrow |
| Posterior/readout safety | FAIL |
| Heterophily stress | FAIL |

Decision:

```text
STOP before full 9-dataset smoke, 260-epoch full run, or any V50A hyperparameter
sweep.
```

Scientific interpretation:

```text
V50A remains a valid rescue direction for spectral/low-rank compactness, but
the fixed graph-smoothed spectral teacher is not uniformly safe on heterophily-
style graphs. The failure is not anchor collapse or red-line leakage; it is a
domain-limit problem where anchor coupling can coexist with posterior/readout
instability, especially on Squirrel.
```

## 9. Next Allowed Move

The next step should be failure analysis, not implementation or tuning.

Recommended next artifact:

```text
V50A_HETEROPHILY_FAILURE_ANALYSIS.md
```

It should answer:

- why Squirrel has a hard posterior/readout gap while Chameleon does not;
- why Texas loses q-anchor agreement despite strong final ACC;
- whether the spectral anchor is acting as a useful teacher, a weak diagnostic,
  or a harmful prior on heterophily-style graphs;
- whether the next mechanism should be a confidence-gated anchor reliability
  model, a local/global anchor consistency model, or a non-teacher compactness
  regularizer.

No parameter sweep is authorized by this verdict.
