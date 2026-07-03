# v53a_residual_curriculum_spectral_anchor Connectivity Verdict

This file records the minimal implementation and one-epoch connectivity check
for `v53a_residual_curriculum_spectral_anchor`. It is not a mixed-stress result
and must not be used for model selection.

## 1. Implemented Scope

Implemented the V53A minimum described in `V53A_IMPLEMENTATION_REVIEW.md`:

```text
v53a_enabled
v53a_anchor_weight
v53a_reliability_floor
v53a_reliable_threshold
v53a_min_effective_mass
v53a_warmup_epochs
v53a_ramp_epochs
v53a_residual_beta
residual_curriculum_spectral_anchor_loss
v53a diagnostics
v53a runner variant
```

The V50A spectral anchor builder is reused unchanged. V53A builds the same
fixed anchor when `v53a_enabled=true`, but the V53A variant disables V50A, V51A,
and V52A anchor losses:

```text
v50a_enabled=false
v50a_anchor_weight=0.0
v51a_enabled=false
v51a_anchor_weight=0.0
v52a_enabled=false
v52a_anchor_weight=0.0
v53a_enabled=true
v53a_anchor_weight=0.04
v53a_residual_beta=0.50
```

No new head, selector, post-processing label path, dataset-specific branch,
signed topology-mask anchor, or geometry fallback was added.

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

Authorized command:

```powershell
conda run --no-capture-output -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v53a_residual_curriculum_spectral_anchor --datasets "acm" --epochs 1 --device cuda --log-level WARNING
```

Run output:

| Dataset | Status | ACC | NMI | ARI |
| --- | --- | ---: | ---: | ---: |
| ACM | ok | 0.7379 | 0.3199 | 0.3712 |

This one-epoch ACC/NMI/ARI is only a connectivity byproduct.

## 4. Red-Line Connectivity Check

Latest ACM diagnostic record:

| Field | Value |
| --- | --- |
| `legacy_head_used` | false |
| `v43b_enabled` | false |
| `v44_enabled` | false |
| `v44b_enabled` | false |
| `v45a_enabled` | false |
| `v46a_enabled` | false |
| `v47a_enabled` | false |
| `v48a_enabled` | false |
| `v49a_enabled` | false |
| `v50a_enabled` | false |
| `v51a_enabled` | false |
| `v52a_enabled` | false |
| `v53a_enabled` | true |

Verdict:

```text
PASS
```

## 5. V53A Diagnostic Connectivity

| Diagnostic | Value |
| --- | ---: |
| `v53a_gamma` | 0.0000 |
| `v53a_residual_beta` | 0.5000 |
| `v53a_residual_multiplier_mean` | 1.0000 |
| `v53a_anchor_loss` | 0.1644 |
| `v53a_weighted_q_anchor_kl` | 0.1644 |
| `v53a_weighted_q_anchor_agreement` | 0.4358 |
| `v53a_unweighted_q_anchor_agreement` | 0.4301 |
| `v53a_embedding_anchor_agreement` | 0.4526 |
| `v53a_reliability_mean` | 0.1962 |
| `v53a_reliability_std` | 0.0746 |
| `v53a_reliable_node_ratio` | 0.5041 |
| `v53a_effective_anchor_mass` | 0.1962 |
| `v53a_base_reliability_mean` | 0.1962 |
| `v53a_agreement_reliability_mean` | 0.0263 |
| `v53a_confidence_component_mean` | 0.3011 |
| `v53a_q_anchor_component_mean` | 0.0377 |
| `v53a_embed_anchor_component_mean` | 0.0149 |
| `v53a_local_component_mean` | 0.0913 |
| `v53a_anchor_acc_diagnostic` | 0.8942 |
| `embedding_posterior_gap` | 0.0026 |

Connectivity pass condition:

```text
status=ok
legacy_head_used=false
v53a_enabled=true
v50a_enabled=false
v51a_enabled=false
v52a_enabled=false
v53a_anchor_loss finite
v53a_gamma = 0
v53a_residual_beta = 0.50
v53a_reliability_mean finite
v53a_effective_anchor_mass finite
```

Verdict:

```text
PASS
```

## 6. Connectivity Interpretation

V53A preserves the V52A early availability behavior:

| Signal | V52A ACM 1-epoch | V53A ACM 1-epoch |
| --- | ---: | ---: |
| Gamma | 0.0000 | 0.0000 |
| Reliability mean | 0.1962 | 0.1962 |
| Reliable node ratio | 0.5041 | 0.5041 |
| Effective anchor mass | 0.1962 | 0.1962 |

This is expected because at `gamma=0`, V53A equals the V52A base-reliability
stage. The residual mechanism will only be tested at later epochs.

## 7. Decision

V53A minimal implementation and connectivity are complete.

Allowed next step:

```text
Run the preregistered first-stage mixed stress exactly once:
datasets = acm,dblp,flickr,texas,squirrel,chameleon
epochs = 80
```

Not allowed before that verdict:

- full 9-dataset smoke;
- 260-epoch full run;
- beta sweep;
- V53A schedule variants;
- V53A reliability formula variants;
- reliability threshold sweep;
- V50A anchor hyperparameter sweep.

## 8. No-Fabrication Status

All numbers in this file come from the logged ACM one-epoch connectivity run in
`results/archive/v51-v57/unified_aptc_9datasets_v53a_residual_curriculum_spectral_anchor_diagnostics.jsonl`.
No V53A mixed-stress or full-run result exists yet.
