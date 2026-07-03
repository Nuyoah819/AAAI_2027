# V62A First Mixed-Stress Verdict

Variant:

```text
v62a_drift_responsive_self_distillation_guard
```

Command:

```powershell
conda run --no-capture-output -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v62a_drift_responsive_self_distillation_guard --datasets "acm,dblp,flickr,texas,squirrel,chameleon" --epochs 100 --device cuda --log-level WARNING
```

Verdict:

```text
PASS
```

This is a 100-epoch mixed-stress verdict only. It does not claim full-run
rescue.

## 1. Result Table

| Dataset | Status | ACC | NMI | ARI | Emb-Post Gap |
| --- | --- | ---: | ---: | ---: | ---: |
| ACM | ok | 0.8992 | 0.6687 | 0.7252 | 0.0000 |
| DBLP | ok | 0.6788 | 0.3998 | 0.3416 | 0.0000 |
| Flickr | ok | 0.4079 | 0.2509 | 0.1712 | -0.0015 |
| Texas | ok | 0.7322 | 0.4807 | 0.5948 | 0.0000 |
| Squirrel | ok | 0.3013 | 0.0630 | 0.0517 | 0.0015 |
| Chameleon | ok | 0.3395 | 0.1572 | 0.0630 | 0.0000 |

## 2. Gate Check

| Gate | Observed | Status |
| --- | --- | --- |
| `status=ok` on 6/6 | 6/6 | PASS |
| red-line pass on 6/6 | no hard violation | PASS |
| `teacher_ready` by epoch 80 | 6/6 true | PASS |
| `guard_gamma_epoch_80=0.0` | 6/6 | PASS |
| `guard_gamma_epoch_100=1.0` | 6/6 | PASS |
| `drift_gamma_epoch_100=0.0` | 6/6 | PASS |
| `effective_guard_multiplier_epoch_100=1.0` | 6/6 | PASS |
| `teacher_active_ratio_epoch_80 >= 0.10` | min 0.1001 | PASS |
| `teacher_topk_active_ratio_epoch_80 >= 0.10` | min 0.1001 | PASS |
| guard loss finite after epoch 80 | 6/6 finite | PASS |
| `abs(embedding_posterior_gap) <= 0.04` | max 0.0015 | PASS |
| ACM ACC >= 0.8888 | 0.8992 | PASS |
| DBLP ACC >= 0.6610 | 0.6788 | PASS |
| Squirrel ACC >= 0.2800 | 0.3013 | PASS |

## 3. Teacher, Guard, And Drift Diagnostics

| Dataset | Active @80 | Floor @80 | Top-k @80 | Guard @100 | Drift @100 | Mult @100 | Drift Score @100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ACM | 0.9990 | 0.9990 | 0.1002 | 1.0000 | 0.0000 | 1.0000 | 0.0067 |
| DBLP | 0.9936 | 0.9936 | 0.1001 | 1.0000 | 0.0000 | 1.0000 | 0.0115 |
| Flickr | 0.1001 | 0.0000 | 0.1001 | 1.0000 | 0.0000 | 1.0000 | 0.0054 |
| Texas | 0.9727 | 0.9727 | 0.1038 | 1.0000 | 0.0000 | 1.0000 | 0.0230 |
| Squirrel | 0.7724 | 0.7724 | 0.1002 | 1.0000 | 0.0000 | 1.0000 | 0.0077 |
| Chameleon | 0.9605 | 0.9605 | 0.1001 | 1.0000 | 0.0000 | 1.0000 | 0.0104 |

The V62A drift branch is intentionally inactive at epoch 100 because
`drift_start_epoch=100` and the preregistered rule uses `epoch <= 100` as the
zero-drift region.

## 4. Anchor Wiring

The inherited V59A anchor/release branch is active under `v62a_*` diagnostics:

| Dataset | Anchor Loss | Effective Mass | Reliability Mean |
| --- | ---: | ---: | ---: |
| ACM | 0.0163 | 0.1411 | 0.1411 |
| DBLP | 0.0801 | 0.0800 | 0.0800 |
| Flickr | 0.0078 | 0.0084 | 0.0084 |
| Texas | 0.4865 | 0.0800 | 0.0800 |
| Squirrel | 0.3957 | 0.0858 | 0.0858 |
| Chameleon | 0.8229 | 0.2143 | 0.2143 |

## 5. Comparison Note

V62A 100e compared with V61A 100e:

| Dataset | V61A 100e ACC | V62A 100e ACC | Delta |
| --- | ---: | ---: | ---: |
| ACM | 0.8972 | 0.8992 | +0.0020 |
| DBLP | 0.6941 | 0.6788 | -0.0153 |
| Flickr | 0.4081 | 0.4079 | -0.0002 |
| Texas | 0.7322 | 0.7322 | 0.0000 |
| Squirrel | 0.2996 | 0.3013 | +0.0017 |
| Chameleon | 0.3456 | 0.3395 | -0.0061 |

V61A 260e failed mainly on long-run Squirrel/Flickr drift. V62A's drift
multiplier is not tested by this 100e verdict beyond confirming it remains
off through epoch 100 as preregistered. Therefore this result does not prove
the long-run failure has been fixed.

## 6. Decision

V62A passes the preregistered first mixed-stress gates.

The next required artifact is:

```text
V62A_EXPANSION_REVIEW.md
```

No V62A 260-epoch run is authorized until that review is written.
