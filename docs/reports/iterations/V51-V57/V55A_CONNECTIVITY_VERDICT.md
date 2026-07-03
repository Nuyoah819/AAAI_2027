# v55a_soft_consensus_bounded_residual_anchor Connectivity Verdict

This file records the preregistered 1-epoch connectivity check for
`v55a_soft_consensus_bounded_residual_anchor`. It follows
`V55A_PREREGISTRATION.md` and `V55A_IMPLEMENTATION_REVIEW.md`.

Connectivity is not a performance result.

## 1. Implementation Status

Implemented files:

```text
core/e2e/sect_coco_e2e.py
scripts/run_unified_aptc_9datasets.py
```

New route:

```text
v55a_soft_consensus_bounded_residual_anchor
```

Mechanism:

```text
Detached soft posterior-anchor and embedding-anchor agreement controls a
bounded residual beta in [0.35, 0.70] with fixed soft_power=0.50.
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
conda run --no-capture-output -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v55a_soft_consensus_bounded_residual_anchor --datasets "acm" --epochs 1 --device cuda --log-level WARNING
```

Output files:

```text
results/archive/v51-v57/unified_aptc_9datasets_v55a_soft_consensus_bounded_residual_anchor.csv
results/archive/v51-v57/unified_aptc_9datasets_v55a_soft_consensus_bounded_residual_anchor_diagnostics.jsonl
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
| v55a_enabled | true | PASS |
| v55a_gamma | 0.0000 | PASS |
| v55a_beta_min | 0.3500 | PASS |
| v55a_beta_max | 0.7000 | PASS |
| v55a_soft_power | 0.5000 | PASS |
| v55a_soft_consensus_mean | 0.0263 | PASS |
| v55a_beta_mean | 0.3961 | PASS |
| v55a_anchor_loss | 0.1644 | PASS |
| v55a_reliability_mean | 0.1962 | PASS |
| v55a_effective_anchor_mass | 0.1962 | PASS |
| embedding_posterior_gap | 0.0026 | PASS |

## 5. Soft-Consensus Diagnostics

| Field | Value |
| --- | ---: |
| v55a_soft_consensus_p10 | 0.0000 |
| v55a_soft_consensus_p50 | 0.0172 |
| v55a_soft_consensus_p90 | 0.0651 |
| v55a_beta_p10 | 0.3500 |
| v55a_beta_p50 | 0.3959 |
| v55a_beta_p90 | 0.4393 |
| v55a_weighted_q_anchor_agreement | 0.4358 |
| v55a_weighted_q_anchor_kl | 0.1644 |
| v55a_base_reliability_mean | 0.1962 |
| v55a_agreement_reliability_mean | 0.0263 |

## 6. Anchor Diagnostic

The anchor diagnostic uses the same frozen V50A spectral compactness anchor.
It is not a final label path.

| Metric | Value |
| --- | ---: |
| v55a_anchor_acc_diagnostic | 0.8942 |
| v55a_anchor_nmi_diagnostic | 0.6630 |
| v55a_anchor_ari_diagnostic | 0.7144 |

## 7. Verdict

```text
PASS CONNECTIVITY.
```

The only next authorized action is the preregistered first-stage mixed-stress
run:

```powershell
conda run --no-capture-output -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v55a_soft_consensus_bounded_residual_anchor --datasets "acm,dblp,flickr,texas,squirrel,chameleon" --epochs 80 --device cuda --log-level WARNING
```

Still not authorized:

```text
full 9-dataset run
260-epoch run
seed sweep
beta-bound sweep
soft-power sweep
schedule variant
reliability formula variant
threshold sweep
V50A anchor hyperparameter sweep
```
