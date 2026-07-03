# v56a_hybrid_consensus_floor_residual_anchor Connectivity Verdict

This file records the one-dataset, one-epoch connectivity result for
`v56a_hybrid_consensus_floor_residual_anchor`. It follows
`V56A_PREREGISTRATION.md` and `V56A_IMPLEMENTATION_REVIEW.md`.

This is not a performance result and must not be used for model selection.

## 1. Prerequisites

Required artifacts:

```text
V56A_PREREGISTRATION.md
V56A_IMPLEMENTATION_REVIEW.md
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
rg -n "v56a|hybrid_consensus_floor" core/e2e/sect_coco_e2e.py scripts/run_unified_aptc_9datasets.py
rg -n "v50a_enabled|v51a_enabled|v52a_enabled|v53a_enabled|v54a_enabled|v55a_enabled|v56a_enabled" scripts/run_unified_aptc_9datasets.py
```

The V56A variant explicitly disables V50A, V51A, V52A, V53A, V54A, and V55A
anchor losses.

## 2. Connectivity Run

Command:

```powershell
conda run --no-capture-output -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v56a_hybrid_consensus_floor_residual_anchor --datasets "acm" --epochs 1 --device cuda --log-level WARNING
```

Output files:

```text
results/archive/v51-v57/unified_aptc_9datasets_v56a_hybrid_consensus_floor_residual_anchor.csv
results/archive/v51-v57/unified_aptc_9datasets_v56a_hybrid_consensus_floor_residual_anchor_diagnostics.jsonl
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
| `v54a_enabled` | false | PASS |
| `v55a_enabled` | false | PASS |
| `v56a_enabled` | true | PASS |

Verdict:

```text
PASS
```

## 4. V56A Mechanism Fields

| Field | Value | Gate |
| --- | ---: | --- |
| `v56a_gamma` | 0.0000 | PASS |
| `v56a_beta_min` | 0.3500 | PASS |
| `v56a_beta_max` | 0.7000 | PASS |
| `v56a_soft_power` | 0.5000 | PASS |
| `v56a_hybrid_compensation` | 0.5000 | PASS |
| `v56a_hard_consensus_mean` | 0.4413 | PASS |
| `v56a_soft_consensus_mean` | 0.0263 | PASS |
| `v56a_lifted_soft_consensus_mean` | 0.1317 | PASS |
| `v56a_compensation_mean` | 0.0040 | PASS |
| `v56a_compensation_active_ratio` | 0.1170 | PASS |
| `v56a_hybrid_consensus_mean` | 0.4453 | PASS |
| `v56a_beta_mean` | 0.5058 | PASS |
| `v56a_anchor_loss` | 0.1644 | PASS |
| `v56a_reliability_mean` | 0.1962 | PASS |
| `v56a_effective_anchor_mass` | 0.1962 | PASS |

Interpretation:

```text
At epoch 1, gamma=0, so the residual multiplier is 1.0 and the run checks
whether the V56A fields and hybrid beta computation are connected. Late-stage
behavior is not tested by this connectivity run.
```

## 5. Connectivity Decision

Decision:

```text
PASS connectivity.
```

This verdict authorizes only the preregistered first-stage mixed-stress run:

```powershell
conda run --no-capture-output -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v56a_hybrid_consensus_floor_residual_anchor --datasets "acm,dblp,flickr,texas,squirrel,chameleon" --epochs 80 --device cuda --log-level WARNING
```

Do not run:

- full 9-dataset smoke;
- 260-epoch full run;
- beta-bound sweep;
- soft-power sweep;
- hybrid-compensation sweep;
- schedule variants;
- reliability formula variants;
- reliability threshold sweep;
- V50A anchor hyperparameter sweep.

## 6. No-Fabrication Status

All values above come from the local V56A connectivity run and diagnostics.
No V56A mixed-stress or full-run result exists yet.
