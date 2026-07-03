# v54a_consensus_bounded_residual_anchor Connectivity Verdict

This file records the one-dataset, one-epoch connectivity result for
`v54a_consensus_bounded_residual_anchor`. It follows
`V54A_PREREGISTRATION.md` and `V54A_IMPLEMENTATION_REVIEW.md`.

This is not a performance result and must not be used for model selection.

## 1. Prerequisites

Required artifacts:

```text
V54A_PREREGISTRATION.md
V54A_IMPLEMENTATION_REVIEW.md
```

Static check:

```powershell
conda run -n aaai-e2e-subspace python -m py_compile core\e2e\sect_coco_e2e.py scripts\run_unified_aptc_9datasets.py
```

Verdict:

```text
PASS
```

Manual static checks:

```text
rg -n "v54a|consensus_bounded_residual" core/e2e/sect_coco_e2e.py scripts/run_unified_aptc_9datasets.py
rg -n "v50a_enabled|v51a_enabled|v52a_enabled|v53a_enabled|v54a_enabled" scripts/run_unified_aptc_9datasets.py
```

The V54A variant explicitly disables V50A, V51A, V52A, and V53A anchor losses.

## 2. Connectivity Run

Command:

```powershell
conda run --no-capture-output -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v54a_consensus_bounded_residual_anchor --datasets "acm" --epochs 1 --device cuda --log-level WARNING
```

Output files:

```text
results/archive/v51-v57/unified_aptc_9datasets_v54a_consensus_bounded_residual_anchor.csv
results/archive/v51-v57/unified_aptc_9datasets_v54a_consensus_bounded_residual_anchor_diagnostics.jsonl
```

Run status:

```text
status=ok
```

The observed one-epoch ACC is not interpreted because this run only checks
connectivity, field emission, and red-line compliance.

## 3. Red-Line Gate

| Field | Value | Gate |
| --- | ---: | --- |
| `legacy_head_used` | false | PASS |
| `v43b_enabled` | false | PASS |
| `v44_enabled` | false | PASS |
| `v44b_enabled` | false | PASS |
| `v45a_enabled` | false | PASS |
| `v46a_enabled` | false | PASS |
| `v47a_enabled` | false | PASS |
| `v48a_enabled` | false | PASS |
| `v49a_enabled` | false | PASS |
| `v50a_enabled` | false | PASS |
| `v51a_enabled` | false | PASS |
| `v52a_enabled` | false | PASS |
| `v53a_enabled` | false | PASS |
| `v54a_enabled` | true | PASS |

Verdict:

```text
PASS
```

## 4. V54A Mechanism Fields

| Field | Value | Gate |
| --- | ---: | --- |
| `v54a_gamma` | 0.0000 | PASS |
| `v54a_beta_min` | 0.3500 | PASS |
| `v54a_beta_max` | 0.7000 | PASS |
| `v54a_beta_mean` | 0.5045 | PASS |
| `v54a_beta_p10` | 0.3500 | PASS |
| `v54a_beta_p50` | 0.5250 | PASS |
| `v54a_beta_p90` | 0.7000 | PASS |
| `v54a_anchor_loss` | 0.1644 | PASS |
| `v54a_reliability_mean` | 0.1962 | PASS |
| `v54a_effective_anchor_mass` | 0.1962 | PASS |
| `v54a_residual_multiplier_mean` | 1.0000 | PASS |

Interpretation:

```text
At epoch 1, gamma=0, so the residual multiplier is 1.0 and the run checks
whether the V54A fields and bounded beta computation are connected. Late-stage
behavior is not tested by this connectivity run.
```

## 5. Consensus Diagnostics

| Field | Value |
| --- | ---: |
| `v54a_hard_q_anchor_match_ratio` | 0.4301 |
| `v54a_hard_embed_anchor_match_ratio` | 0.4526 |
| `v54a_hard_both_anchor_match_ratio` | 0.1990 |
| `embedding_posterior_gap` | 0.0026 |

These diagnostics confirm that node-level consensus fields are emitted and
finite. They do not establish mixed-stress safety.

## 6. Connectivity Decision

Decision:

```text
PASS connectivity.
```

This verdict authorizes only the preregistered first-stage mixed-stress run:

```powershell
conda run --no-capture-output -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v54a_consensus_bounded_residual_anchor --datasets "acm,dblp,flickr,texas,squirrel,chameleon" --epochs 80 --device cuda --log-level WARNING
```

Do not run:

- full 9-dataset smoke;
- 260-epoch full run;
- beta bound sweep;
- schedule variants;
- reliability formula variants;
- reliability threshold sweep;
- V50A anchor hyperparameter sweep.

## 7. No-Fabrication Status

All values above come from the local V54A connectivity run and diagnostics.
No V54A mixed-stress or full-run result exists yet.
