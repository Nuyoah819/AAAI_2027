# V58A First Mixed-Stress Verdict

This file records the authorized 6-dataset / 80-epoch mixed-stress run for
`v58a_anchor_release_residual_compactness`.

It follows `V58A_CONNECTIVITY_VERDICT.md`.

## 1. Run

Authorized command:

```powershell
conda run --no-capture-output -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v58a_anchor_release_residual_compactness --datasets "acm,dblp,flickr,texas,squirrel,chameleon" --epochs 80 --device cuda --log-level WARNING
```

Run status:

```text
6/6 datasets completed with status=ok.
```

Output files:

```text
results/archive/v58-v63/unified_aptc_9datasets_v58a_anchor_release_residual_compactness.csv
results/archive/v58-v63/unified_aptc_9datasets_v58a_anchor_release_residual_compactness_diagnostics.jsonl
```

## 2. Result Summary

| Dataset | ACC | NMI | ARI | Emb Gap |
| --- | ---: | ---: | ---: | ---: |
| ACM | 0.7038 | 0.3527 | 0.3808 | 0.0000 |
| DBLP | 0.6608 | 0.3695 | 0.3107 | 0.0000 |
| Flickr | 0.3890 | 0.2202 | 0.1508 | -0.0144 |
| Texas | 0.7322 | 0.4779 | 0.5974 | 0.0000 |
| Squirrel | 0.3021 | 0.0631 | 0.0518 | -0.0883 |
| Chameleon | 0.3263 | 0.1391 | 0.0518 | 0.0018 |

## 3. Red-Line Gate

Requirement:

```text
status=ok
legacy_head_used=false
v50a-v57a_enabled=false
v58a_enabled=true
```

Verdict:

```text
PASS on 6/6.
```

| Dataset | Status | Legacy | V50A | V51A | V52A | V53A | V54A | V55A | V56A | V57A | V58A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ACM | ok | false | false | false | false | false | false | false | false | false | true |
| DBLP | ok | false | false | false | false | false | false | false | false | false | true |
| Flickr | ok | false | false | false | false | false | false | false | false | false | true |
| Texas | ok | false | false | false | false | false | false | false | false | false | true |
| Squirrel | ok | false | false | false | false | false | false | false | false | false | true |
| Chameleon | ok | false | false | false | false | false | false | false | false | false | true |

## 4. Release-Gamma Gate

Requirement:

```text
v58a_release_gamma_epoch_1 = 0
v58a_release_gamma_epoch_40 = 0.5
v58a_release_gamma_epoch_80 = 1.0
```

Verdict:

```text
PASS on 6/6.
```

| Dataset | Gamma @1 | Gamma @40 | Gamma @80 | Gate |
| --- | ---: | ---: | ---: | --- |
| ACM | 0.0000 | 0.5000 | 1.0000 | PASS |
| DBLP | 0.0000 | 0.5000 | 1.0000 | PASS |
| Flickr | 0.0000 | 0.5000 | 1.0000 | PASS |
| Texas | 0.0000 | 0.5000 | 1.0000 | PASS |
| Squirrel | 0.0000 | 0.5000 | 1.0000 | PASS |
| Chameleon | 0.0000 | 0.5000 | 1.0000 | PASS |

## 5. Mass-Normalization Gate

Requirement:

```text
mass pass on at least 4/6
reliable-node ratio pass on at least 3/6
```

Verdict:

```text
FAIL.
Mass passes on 5/6, but reliable-node ratio passes on only 2/6.
```

| Dataset | Raw Rel | Mass Scale | Rel Mean | Reliable Ratio | Mass Gate | Ratio Gate |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| ACM | 0.1267 | 1.0000 | 0.1267 | 0.1177 | PASS | PASS |
| DBLP | 0.0740 | 1.0817 | 0.0800 | 0.0000 | PASS | FAIL |
| Flickr | 0.0056 | 1.5000 | 0.0083 | 0.0000 | weak-anchor | FAIL |
| Texas | 0.0563 | 1.4199 | 0.0800 | 0.0328 | PASS | FAIL |
| Squirrel | 0.0843 | 1.0000 | 0.0843 | 0.0473 | PASS | FAIL |
| Chameleon | 0.2037 | 1.0000 | 0.2037 | 0.4269 | PASS | PASS |

## 6. Anchor-Usefulness Gate

Agreement movement from epoch 1 to epoch 80:

| Dataset | Agreement @1 | Agreement @80 | Movement | Gate |
| --- | ---: | ---: | ---: | --- |
| ACM | 0.4358 | 0.8581 | +0.4223 | PASS |
| DBLP | 0.3128 | 0.5309 | +0.2181 | PASS |
| Flickr | 0.0243 | 0.0140 | -0.0103 | FAIL |
| Texas | 0.3893 | 0.4554 | +0.0661 | PASS |
| Squirrel | 0.1220 | 0.1766 | +0.0546 | PASS |
| Chameleon | 0.1756 | 0.3332 | +0.1575 | PASS |

Verdict:

```text
PASS on 5/6, but this does not rescue the failed preservation and safety gates.
```

## 7. Posterior/Readout Safety Gate

Requirement:

```text
abs(embedding_posterior_gap) <= 0.04 on 6/6
hard fail if any abs(embedding_posterior_gap) > 0.08
```

Verdict:

```text
FAIL.
Squirrel has abs(embedding_posterior_gap)=0.0883, exceeding the hard 0.08 bound.
```

| Dataset | Emb Gap | Gate |
| --- | ---: | --- |
| ACM | 0.0000 | PASS |
| DBLP | 0.0000 | PASS |
| Flickr | -0.0144 | PASS |
| Texas | 0.0000 | PASS |
| Squirrel | -0.0883 | FAIL |
| Chameleon | 0.0018 | PASS |

## 8. Preservation Floors

Requirement:

```text
ACM ACC >= 0.8888
DBLP ACC >= 0.6610
Squirrel ACC >= 0.2800
```

Verdict:

```text
FAIL.
ACM fails badly and DBLP is below the fixed floor.
```

| Dataset | ACC | Floor | Gate |
| --- | ---: | ---: | --- |
| ACM | 0.7038 | 0.8888 | FAIL |
| DBLP | 0.6608 | 0.6610 | FAIL |
| Squirrel | 0.3021 | 0.2800 | PASS |

## 9. Comparison To V57A 80E

| Dataset | V57A 80E ACC | V58A 80E ACC | Delta |
| --- | ---: | ---: | ---: |
| ACM | 0.9005 | 0.7038 | -0.1967 |
| DBLP | 0.6919 | 0.6608 | -0.0311 |
| Flickr | 0.3681 | 0.3890 | +0.0209 |
| Texas | 0.7377 | 0.7322 | -0.0055 |
| Squirrel | 0.3011 | 0.3021 | +0.0010 |
| Chameleon | 0.3382 | 0.3263 | -0.0119 |

The V58A first mixed-stress run does not preserve the useful V57A early
behavior. The largest failure is ACM, where the outer release multiplier
substantially weakens the early compactness pressure relative to V57A.

## 10. Gate Summary

| Gate | Verdict |
| --- | --- |
| Execution | PASS |
| Red-line | PASS |
| Release schedule | PASS |
| Mass-normalization | FAIL |
| Anchor usefulness | PASS |
| Posterior/readout safety | FAIL |
| Preservation floors | FAIL |

Decision:

```text
STOP V58A FIRST MIXED-STRESS.
```

No V58A 260-epoch run, 9-dataset expansion, seed sweep, schedule sweep,
release-floor sweep, reliability change, or post-hoc final-label selector is
authorized.

The next artifact must be a V58A failure analysis. If a new rescue route is
proposed, it must be preregistered before implementation.

## 11. No-Fabrication Status

All numbers in this verdict come from the local V58A mixed-stress run and its
diagnostics.
