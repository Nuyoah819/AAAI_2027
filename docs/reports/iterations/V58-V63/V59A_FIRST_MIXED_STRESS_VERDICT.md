# V59A First Mixed-Stress Verdict

This file records the authorized 6-dataset / 80-epoch mixed-stress run for
`v59a_post80_anchor_release_residual_compactness`.

It follows `V59A_CONNECTIVITY_VERDICT.md`.

No V59A 260-epoch run, 9-dataset expansion, seed sweep, schedule sweep,
release-floor sweep, reliability change, or post-hoc final-label selector is
authorized by this verdict.

## 1. Run

Authorized command:

```powershell
conda run --no-capture-output -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v59a_post80_anchor_release_residual_compactness --datasets "acm,dblp,flickr,texas,squirrel,chameleon" --epochs 80 --device cuda --log-level WARNING
```

Run status:

```text
6/6 datasets completed with status=ok.
```

Output files:

```text
results/archive/v58-v63/unified_aptc_9datasets_v59a_post80_anchor_release_residual_compactness.csv
results/archive/v58-v63/unified_aptc_9datasets_v59a_post80_anchor_release_residual_compactness_diagnostics.jsonl
```

## 2. Result Summary

| Dataset | ACC | NMI | ARI | Emb Gap |
| --- | ---: | ---: | ---: | ---: |
| ACM | 0.8962 | 0.6640 | 0.7172 | 0.0000 |
| DBLP | 0.6884 | 0.4057 | 0.3505 | 0.0000 |
| Flickr | 0.4150 | 0.2536 | 0.1683 | 0.0000 |
| Texas | 0.7377 | 0.4944 | 0.6109 | 0.0000 |
| Squirrel | 0.3017 | 0.0621 | 0.0511 | 0.0000 |
| Chameleon | 0.3325 | 0.1437 | 0.0498 | 0.0000 |

## 3. Red-Line Gate

Requirement:

```text
status=ok
legacy_head_used=false
v50a-v58a_enabled=false
v59a_enabled=true
```

Verdict:

```text
PASS on 6/6.
```

| Dataset | Status | Legacy | V50A | V51A | V52A | V53A | V54A | V55A | V56A | V57A | V58A | V59A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ACM | ok | false | false | false | false | false | false | false | false | false | false | true |
| DBLP | ok | false | false | false | false | false | false | false | false | false | false | true |
| Flickr | ok | false | false | false | false | false | false | false | false | false | false | true |
| Texas | ok | false | false | false | false | false | false | false | false | false | false | true |
| Squirrel | ok | false | false | false | false | false | false | false | false | false | false | true |
| Chameleon | ok | false | false | false | false | false | false | false | false | false | false | true |

## 4. Release-Gamma Gate

Requirement:

```text
v59a_release_gamma_epoch_1 = 1.0
v59a_release_gamma_epoch_40 = 1.0
v59a_release_gamma_epoch_80 = 1.0
```

Verdict:

```text
PASS on 6/6.
```

| Dataset | Release @1 | Release @40 | Release @80 | V57A Gamma @1 | V57A Gamma @40 | V57A Gamma @80 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ACM | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.5000 | 1.0000 |
| DBLP | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.5000 | 1.0000 |
| Flickr | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.5000 | 1.0000 |
| Texas | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.5000 | 1.0000 |
| Squirrel | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.5000 | 1.0000 |
| Chameleon | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.5000 | 1.0000 |

Interpretation:

```text
The outer V59A release wrapper does not weaken the first 80 epochs. The
underlying V57A internal gamma remains unchanged.
```

## 5. Mass And Reliability Gate

Requirement:

```text
mass pass on at least 4/6
reliable-node ratio pass on at least 3/6
```

Verdict:

```text
PASS.
Mass passes on 5/6. Reliable-node ratio passes on 3/6.
```

| Dataset | Raw Rel | Mass Scale | Rel Mean | Reliable Ratio | Mass Gate | Ratio Gate |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| ACM | 0.1388 | 1.0000 | 0.1388 | 0.1478 | PASS | PASS |
| DBLP | 0.0754 | 1.0615 | 0.0800 | 0.0000 | PASS | FAIL |
| Flickr | 0.0056 | 1.5000 | 0.0084 | 0.0000 | weak-anchor | FAIL |
| Texas | 0.0565 | 1.4149 | 0.0800 | 0.0328 | PASS | FAIL |
| Squirrel | 0.0856 | 1.0000 | 0.0856 | 0.0529 | PASS | PASS |
| Chameleon | 0.2169 | 1.0000 | 0.2169 | 0.4985 | PASS | PASS |

## 6. Anchor-Usefulness Gate

Agreement movement from epoch 1 to epoch 80:

| Dataset | Agreement @1 | Agreement @40 | Agreement @80 | Movement | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| ACM | 0.4358 | 0.9421 | 0.9604 | +0.5246 | PASS |
| DBLP | 0.3128 | 0.5773 | 0.5553 | +0.2426 | PASS |
| Flickr | 0.0243 | 0.0223 | 0.0166 | -0.0077 | FAIL |
| Texas | 0.3893 | 0.3733 | 0.4554 | +0.0661 | PASS |
| Squirrel | 0.1220 | 0.1874 | 0.2198 | +0.0978 | PASS |
| Chameleon | 0.1756 | 0.3384 | 0.4516 | +0.2760 | PASS |

Verdict:

```text
PASS on 5/6.
```

Flickr remains the expected weak-anchor exception.

## 7. Posterior/Readout Safety Gate

Requirement:

```text
abs(embedding_posterior_gap) <= 0.04 on 6/6
hard fail if any abs(embedding_posterior_gap) > 0.08
```

Verdict:

```text
PASS on 6/6.
```

| Dataset | Emb Gap | Gate |
| --- | ---: | --- |
| ACM | 0.0000 | PASS |
| DBLP | 0.0000 | PASS |
| Flickr | 0.0000 | PASS |
| Texas | 0.0000 | PASS |
| Squirrel | 0.0000 | PASS |
| Chameleon | 0.0000 | PASS |

## 8. Preservation Floors

Requirement:

```text
ACM ACC >= 0.8888
DBLP ACC >= 0.6610
Squirrel ACC >= 0.2800
```

Verdict:

```text
PASS.
```

| Dataset | ACC | Floor | Gate |
| --- | ---: | ---: | --- |
| ACM | 0.8962 | 0.8888 | PASS |
| DBLP | 0.6884 | 0.6610 | PASS |
| Squirrel | 0.3017 | 0.2800 | PASS |

## 9. Comparison To V57A 80E

Reference: `V57A_SUPPORTED_9DATASET_80E_SMOKE_VERDICT.md`.

| Dataset | V57A 80E ACC | V59A 80E ACC | Delta |
| --- | ---: | ---: | ---: |
| ACM | 0.8962 | 0.8962 | +0.0000 |
| DBLP | 0.6825 | 0.6884 | +0.0059 |
| Flickr | 0.4133 | 0.4150 | +0.0017 |
| Texas | 0.7377 | 0.7377 | +0.0000 |
| Squirrel | 0.3005 | 0.3017 | +0.0012 |
| Chameleon | 0.3421 | 0.3325 | -0.0096 |

Preregistered stop condition:

```text
If ACM drops by more than 0.02 from V57A 80e, stop.
```

Verdict:

```text
PASS. ACM does not drop.
```

Interpretation:

```text
V59A restores the V57A early absorption window that V58A disrupted. This does
not prove that post-80 release fixes the V57A 260e drift; it only authorizes the
next expansion review.
```

## 10. Gate Summary

| Gate | Verdict |
| --- | --- |
| Execution | PASS |
| Red-line | PASS |
| Release schedule | PASS |
| Mass/reliability | PASS |
| Anchor usefulness | PASS |
| Posterior/readout safety | PASS |
| Preservation floors | PASS |
| V57A 80e comparison | PASS |

Decision:

```text
PASS FIRST MIXED-STRESS.
```

This authorizes only the next artifact:

```text
V59A_EXPANSION_REVIEW.md
```

No V59A 260-epoch run, supported 9-dataset expansion, seed sweep, schedule
sweep, release-floor sweep, reliability formula change, or dataset-specific
branch is authorized until that review exists.

## 11. No-Fabrication Status

All V59A numbers in this verdict come from the local V59A mixed-stress run and
its diagnostics. The V57A comparison values come from
`V57A_SUPPORTED_9DATASET_80E_SMOKE_VERDICT.md`.
