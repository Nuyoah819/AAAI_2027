# V61A First Mixed-Stress Verdict

Variant:

```text
v61a_quantile_coverage_self_distillation_guard
```

Command:

```powershell
conda run --no-capture-output -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v61a_quantile_coverage_self_distillation_guard --datasets "acm,dblp,flickr,texas,squirrel,chameleon" --epochs 100 --device cuda --log-level WARNING
```

Verdict: PASS.

This verdict uses the rerun after `V61A_INVALID_MIXED_STRESS_ANCHOR_WIRING.md`.
The earlier 100-epoch run is invalid and is not used here.

## 1. Result Table

| Dataset | Status | ACC | NMI | ARI | Emb-Post Gap |
| --- | --- | ---: | ---: | ---: | ---: |
| ACM | ok | 0.8972 | 0.6705 | 0.7193 | 0.0000 |
| DBLP | ok | 0.6941 | 0.4177 | 0.3667 | 0.0000 |
| Flickr | ok | 0.4081 | 0.2578 | 0.1782 | 0.0000 |
| Texas | ok | 0.7322 | 0.4807 | 0.5948 | 0.0000 |
| Squirrel | ok | 0.2996 | 0.0618 | 0.0491 | 0.0025 |
| Chameleon | ok | 0.3456 | 0.1676 | 0.0722 | 0.0035 |

## 2. Gate Check

| Gate | Observed | Status |
| --- | --- | --- |
| `status=ok` on 6/6 | 6/6 | PASS |
| red-line pass on 6/6 | no hard violation | PASS |
| `teacher_ready` by epoch 80 | 6/6 true | PASS |
| `guard_gamma_epoch_80=0.0` | 6/6 | PASS |
| `guard_gamma_epoch_100=1.0` | 6/6 | PASS |
| `teacher_active_ratio_epoch_80 >= 0.10` | min 0.1001 | PASS |
| `teacher_topk_active_ratio_epoch_80 >= 0.10` | min 0.1001 | PASS |
| guard loss finite after epoch 80 | 6/6 finite | PASS |
| `abs(embedding_posterior_gap) <= 0.04` | max 0.0035 | PASS |
| ACM ACC >= 0.8888 | 0.8972 | PASS |
| DBLP ACC >= 0.6610 | 0.6941 | PASS |
| Squirrel ACC >= 0.2800 | 0.2996 | PASS |

## 3. Teacher Coverage

| Dataset | Conf Mean | Active @80 | Floor @80 | Top-k @80 | Guard Loss @100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ACM | 0.5768 | 0.9997 | 0.9997 | 0.1002 | 0.0107 |
| DBLP | 0.5159 | 0.9926 | 0.9926 | 0.1001 | 0.0106 |
| Flickr | 0.3230 | 0.1001 | 0.0000 | 0.1001 | 0.0055 |
| Texas | 0.8293 | 0.9781 | 0.9781 | 0.1038 | 0.0162 |
| Squirrel | 0.4612 | 0.7418 | 0.7418 | 0.1002 | 0.0064 |
| Chameleon | 0.4788 | 0.9587 | 0.9587 | 0.1001 | 0.0082 |

The intended V61A rescue is active: Flickr receives nonzero guard coverage
through the top-k rule even when the absolute floor contributes zero coverage.

## 4. Anchor Wiring

The inherited V59A anchor/release branch is active under `v61a_*` diagnostics:

| Dataset | Reliability Mean | Mass Scale | Effective Mass | Anchor Agreement @100 |
| --- | ---: | ---: | ---: | ---: |
| ACM | 0.1410 | 1.0000 | 0.1410 | 0.9662 |
| DBLP | 0.0800 | 1.0550 | 0.0800 | 0.5593 |
| Flickr | 0.0084 | 1.5000 | 0.0084 | 0.0160 |
| Texas | 0.0800 | 1.4174 | 0.0800 | 0.4507 |
| Squirrel | 0.0858 | 1.0000 | 0.0858 | 0.2225 |
| Chameleon | 0.2146 | 1.0000 | 0.2146 | 0.4568 |

## 5. Comparison Note

Relative to V60A, V61A fixes the zero-coverage failure mode:

```text
V60A Flickr active ratio @80 = 0.0000
V60A Squirrel active ratio @80 = 0.0000
V61A Flickr active ratio @80 = 0.1001
V61A Squirrel active ratio @80 = 0.7418
```

Relative to the V60A first mixed-stress stop condition:

```text
V60A Squirrel ACC = 0.2096, gap = 0.0936
V61A Squirrel ACC = 0.2996, gap = 0.0025
```

This is not a full-run rescue claim. It is only a 100-epoch mixed-stress pass.

## 6. Decision

V61A passes the preregistered first mixed-stress gates.

The next required artifact is:

```text
V61A_EXPANSION_REVIEW.md
```

No 260-epoch run is authorized until that review is written and approved.
