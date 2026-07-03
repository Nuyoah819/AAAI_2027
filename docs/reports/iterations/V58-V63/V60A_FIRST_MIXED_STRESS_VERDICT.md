# V60A First Mixed-Stress Verdict

This file records the authorized 6-dataset / 100-epoch mixed-stress run for
`v60a_ema_self_distillation_drift_guard`.

It follows `V60A_CONNECTIVITY_VERDICT.md`.

## 1. Run

Authorized command:

```powershell
conda run --no-capture-output -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v60a_ema_self_distillation_drift_guard --datasets "acm,dblp,flickr,texas,squirrel,chameleon" --epochs 100 --device cuda --log-level WARNING
```

Run status:

```text
6/6 datasets completed with status=ok.
```

Output files:

```text
results/archive/v58-v63/unified_aptc_9datasets_v60a_ema_self_distillation_drift_guard.csv
results/archive/v58-v63/unified_aptc_9datasets_v60a_ema_self_distillation_drift_guard_diagnostics.jsonl
```

## 2. Result Summary

| Dataset | ACC | NMI | ARI | Emb Gap |
| --- | ---: | ---: | ---: | ---: |
| ACM | 0.9025 | 0.6777 | 0.7336 | 0.0000 |
| DBLP | 0.6818 | 0.4031 | 0.3466 | -0.0002 |
| Flickr | 0.4094 | 0.2518 | 0.1714 | 0.0000 |
| Texas | 0.7322 | 0.4807 | 0.5948 | 0.0055 |
| Squirrel | 0.2096 | 0.0133 | 0.0002 | 0.0936 |
| Chameleon | 0.3408 | 0.1603 | 0.0642 | 0.0000 |

## 3. Red-Line Gate

Requirement:

```text
status=ok
legacy_head_used=false
v50a-v59a_enabled=false
v60a_enabled=true
```

Verdict:

```text
PASS on 6/6.
```

| Dataset | Legacy | V50A | V51A | V52A | V53A | V54A | V55A | V56A | V57A | V58A | V59A | V60A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ACM | false | false | false | false | false | false | false | false | false | false | false | true |
| DBLP | false | false | false | false | false | false | false | false | false | false | false | true |
| Flickr | false | false | false | false | false | false | false | false | false | false | false | true |
| Texas | false | false | false | false | false | false | false | false | false | false | false | true |
| Squirrel | false | false | false | false | false | false | false | false | false | false | false | true |
| Chameleon | false | false | false | false | false | false | false | false | false | false | false | true |

## 4. Teacher And Guard Gate

Requirements:

```text
teacher_ready becomes true by epoch 80
guard_gamma_epoch_80 = 0.0
guard_gamma_epoch_100 = 1.0
teacher_active_ratio_epoch_80 >= 0.05 on at least 3/6 datasets
guard_loss finite after epoch 80
```

Verdict:

```text
FAIL.
The teacher snapshot and guard schedule work, but active teacher coverage passes
only 2/6 datasets.
```

| Dataset | Ready @80 | Gamma @80 | Gamma @100 | Active @80 | Guard Loss @100 | Gate |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| ACM | true | 0.0000 | 1.0000 | 0.0585 | 0.0014 | PASS |
| DBLP | true | 0.0000 | 1.0000 | 0.0000 | 0.0000 | FAIL active |
| Flickr | true | 0.0000 | 1.0000 | 0.0000 | 0.0000 | FAIL active |
| Texas | true | 0.0000 | 1.0000 | 0.9071 | 0.0078 | PASS |
| Squirrel | true | 0.0000 | 1.0000 | 0.0000 | 0.0000 | FAIL active |
| Chameleon | true | 0.0000 | 1.0000 | 0.0031 | 0.0103 | FAIL active |

Only ACM and Texas meet the active-ratio floor. DBLP, Flickr, and Squirrel have
zero active teacher nodes under the fixed 0.60 confidence threshold.

## 5. Anchor Release Gate

| Dataset | Release @1 | Release @80 | Release @100 |
| --- | ---: | ---: | ---: |
| ACM | 1.0000 | 1.0000 | 0.7500 |
| DBLP | 1.0000 | 1.0000 | 0.7500 |
| Flickr | 1.0000 | 1.0000 | 0.7500 |
| Texas | 1.0000 | 1.0000 | 0.7500 |
| Squirrel | 1.0000 | 1.0000 | 0.7500 |
| Chameleon | 1.0000 | 1.0000 | 0.7500 |

Verdict:

```text
PASS.
```

The inherited V59A post-80 release is wired as expected.

## 6. Posterior/Readout Safety Gate

Requirement:

```text
abs(embedding_posterior_gap) <= 0.04 on 6/6
hard fail if any abs(embedding_posterior_gap) > 0.08
```

Verdict:

```text
FAIL.
Squirrel has abs(embedding_posterior_gap)=0.0936.
```

| Dataset | Emb Gap | Gate |
| --- | ---: | --- |
| ACM | 0.0000 | PASS |
| DBLP | -0.0002 | PASS |
| Flickr | 0.0000 | PASS |
| Texas | 0.0055 | PASS |
| Squirrel | 0.0936 | FAIL |
| Chameleon | 0.0000 | PASS |

## 7. Preservation Floors

Requirement:

```text
ACM ACC >= 0.8888
DBLP ACC >= 0.6610
Squirrel ACC >= 0.2800
```

Verdict:

```text
FAIL.
Squirrel violates the preservation floor.
```

| Dataset | ACC | Floor | Gate |
| --- | ---: | ---: | --- |
| ACM | 0.9025 | 0.8888 | PASS |
| DBLP | 0.6818 | 0.6610 | PASS |
| Squirrel | 0.2096 | 0.2800 | FAIL |

## 8. Comparison Context

| Dataset | V59A 80E ACC | V60A 100E ACC | V59A 260E ACC |
| --- | ---: | ---: | ---: |
| ACM | 0.8962 | 0.9025 | 0.9180 |
| DBLP | 0.6825 | 0.6818 | 0.7321 |
| Flickr | 0.4150 | 0.4094 | 0.2630 |
| Texas | 0.7377 | 0.7322 | 0.7213 |
| Squirrel | 0.3017 | 0.2096 | 0.2103 |
| Chameleon | 0.3421 | 0.3408 | 0.3412 |

Interpretation:

```text
V60A does not merely fail to prove full-run rescue. It already reproduces the
Squirrel late-failure state by epoch 100.
```

The key mechanism issue is visible in the diagnostics:

```text
Squirrel teacher_active_ratio_epoch_80 = 0.0, so the intended guard is inactive
on the hardest failure case.
```

## 9. Gate Summary

| Gate | Verdict |
| --- | --- |
| Execution | PASS |
| Red-line | PASS |
| Teacher readiness | PASS |
| Guard activation coverage | FAIL |
| Anchor release schedule | PASS |
| Posterior/readout safety | FAIL |
| Preservation floors | FAIL |

Decision:

```text
STOP V60A FIRST MIXED-STRESS.
```

No V60A expansion, 260-epoch run, seed sweep, confidence-threshold sweep,
guard-weight sweep, teacher-epoch sweep, EMA update, or post-hoc selector is
authorized.

The next artifact must be a failure analysis. Any new rescue route must be
preregistered before implementation.

## 10. No-Fabrication Status

All V60A numbers in this verdict come from the local V60A 100-epoch mixed-stress
run and its diagnostics. V59A comparison values come from local V59A verdict
files.
