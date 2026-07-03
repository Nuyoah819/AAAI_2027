# v57a_mass_floor_normalized_residual_anchor Connectivity Verdict

This file records the preregistered 1-epoch connectivity check for
`v57a_mass_floor_normalized_residual_anchor`. It follows
`V57A_PREREGISTRATION.md` and `V57A_IMPLEMENTATION_REVIEW.md`.

Connectivity is not a performance result.

## 1. Implementation Status

Implemented files:

```text
core/e2e/sect_coco_e2e.py
scripts/run_unified_aptc_9datasets.py
```

New route:

```text
v57a_mass_floor_normalized_residual_anchor
```

Mechanism:

```text
Use V56A detached raw reliability, then apply fixed detached mass-floor
normalization with target_mass=0.08, max_mass_scale=1.50, and
max_reliability_cap=0.90.
```

## 2. Static Check

Command:

```powershell
conda run -n aaai-e2e-subspace python -m py_compile core\e2e\sect_coco_e2e.py scripts\run_unified_aptc_9datasets.py
```

Verdict:

```text
PASS
```

## 3. Connectivity Run

Command:

```powershell
conda run --no-capture-output -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v57a_mass_floor_normalized_residual_anchor --datasets "acm" --epochs 1 --device cuda --log-level WARNING
```

Output files:

```text
results/archive/v51-v57/unified_aptc_9datasets_v57a_mass_floor_normalized_residual_anchor.csv
results/archive/v51-v57/unified_aptc_9datasets_v57a_mass_floor_normalized_residual_anchor_diagnostics.jsonl
```

Run status:

```text
status=ok
```

The 1-epoch ACC/NMI/ARI values are ignored for model selection and are not a
performance claim.

## 4. Connectivity Gate

| Field | Value | Gate |
| --- | ---: | --- |
| legacy_head_used | false | PASS |
| v50a_enabled | false | PASS |
| v51a_enabled | false | PASS |
| v52a_enabled | false | PASS |
| v53a_enabled | false | PASS |
| v54a_enabled | false | PASS |
| v55a_enabled | false | PASS |
| v56a_enabled | false | PASS |
| v57a_enabled | true | PASS |
| v57a_gamma | 0.0000 | PASS |
| v57a_target_mass | 0.0800 | PASS |
| v57a_max_mass_scale | 1.5000 | PASS |
| v57a_max_reliability_cap | 0.9000 | PASS |
| v57a_raw_reliability_mean | 0.1962 | PASS |
| v57a_mass_scale | 1.0000 | PASS |
| v57a_reliability_mean | 0.1962 | PASS |
| v57a_effective_anchor_mass | 0.1962 | PASS |
| v57a_anchor_loss | 0.1644 | PASS |
| embedding_posterior_gap | 0.0026 | PASS |

`v57a_mass_scale=1.0` is expected on this 1-epoch ACM check because raw mass is
already above the fixed `target_mass=0.08`.

## 5. Hybrid And Mass Diagnostics

| Field | Value |
| --- | ---: |
| v57a_hard_consensus_mean | 0.4413 |
| v57a_soft_consensus_mean | 0.0263 |
| v57a_lifted_soft_consensus_mean | 0.1317 |
| v57a_compensation_mean | 0.0040 |
| v57a_compensation_active_ratio | 0.1170 |
| v57a_hybrid_consensus_mean | 0.4453 |
| v57a_beta_mean | 0.5058 |
| v57a_scaled_reliability_mean | 0.1962 |
| v57a_weighted_q_anchor_agreement | 0.4358 |
| v57a_weighted_q_anchor_kl | 0.1644 |

## 6. Anchor Diagnostic

The anchor diagnostic uses the same frozen V50A spectral compactness anchor.
It is not a final label path.

| Metric | Value |
| --- | ---: |
| v57a_anchor_acc_diagnostic | 0.8942 |
| v57a_anchor_nmi_diagnostic | 0.6630 |
| v57a_anchor_ari_diagnostic | 0.7144 |

## 7. Verdict

```text
PASS CONNECTIVITY.
```

The only next authorized action is the preregistered first-stage mixed-stress
run:

```powershell
conda run --no-capture-output -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v57a_mass_floor_normalized_residual_anchor --datasets "acm,dblp,flickr,texas,squirrel,chameleon" --epochs 80 --device cuda --log-level WARNING
```

Still not authorized:

```text
full 9-dataset run
260-epoch run
seed sweep
target-mass sweep
max-mass-scale sweep
reliability-cap sweep
beta-bound sweep
soft-power sweep
hybrid-compensation sweep
schedule variant
reliability formula variant
threshold sweep
V50A anchor hyperparameter sweep
```
