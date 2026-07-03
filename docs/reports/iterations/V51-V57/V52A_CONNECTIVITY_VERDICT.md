# v52a_curriculum_reliability_spectral_anchor Connectivity Verdict

This file records the minimal implementation and one-epoch connectivity check
for `v52a_curriculum_reliability_spectral_anchor`. It is not a mixed-stress
result and must not be used for model selection.

## 1. Implemented Scope

Implemented the V52A minimum described in `V52A_IMPLEMENTATION_REVIEW.md`:

```text
v52a_enabled
v52a_anchor_weight
v52a_reliability_floor
v52a_reliable_threshold
v52a_min_effective_mass
v52a_warmup_epochs
v52a_ramp_epochs
curriculum_reliability_spectral_anchor_loss
v52a diagnostics
v52a runner variant
```

The V50A spectral anchor builder is reused unchanged. V52A builds the same
fixed anchor when `v52a_enabled=true`, but the V52A variant disables V50A and
V51A anchor losses:

```text
v50a_enabled=false
v50a_anchor_weight=0.0
v51a_enabled=false
v51a_anchor_weight=0.0
v52a_enabled=true
v52a_anchor_weight=0.04
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
conda run --no-capture-output -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v52a_curriculum_reliability_spectral_anchor --datasets "acm" --epochs 1 --device cuda --log-level WARNING
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
| `v52a_enabled` | true |

Verdict:

```text
PASS
```

## 5. V52A Diagnostic Connectivity

| Diagnostic | Value |
| --- | ---: |
| `v52a_gamma` | 0.0000 |
| `v52a_anchor_loss` | 0.1644 |
| `v52a_weighted_q_anchor_kl` | 0.1644 |
| `v52a_weighted_q_anchor_agreement` | 0.4358 |
| `v52a_unweighted_q_anchor_agreement` | 0.4301 |
| `v52a_embedding_anchor_agreement` | 0.4526 |
| `v52a_reliability_mean` | 0.1962 |
| `v52a_reliability_std` | 0.0746 |
| `v52a_reliable_node_ratio` | 0.5041 |
| `v52a_effective_anchor_mass` | 0.1962 |
| `v52a_base_reliability_mean` | 0.1962 |
| `v52a_agreement_reliability_mean` | 0.0263 |
| `v52a_confidence_component_mean` | 0.3011 |
| `v52a_q_anchor_component_mean` | 0.0377 |
| `v52a_embed_anchor_component_mean` | 0.0149 |
| `v52a_local_component_mean` | 0.0913 |
| `v52a_anchor_acc_diagnostic` | 0.8942 |
| `embedding_posterior_gap` | 0.0026 |

Connectivity pass condition:

```text
status=ok
legacy_head_used=false
v52a_enabled=true
v50a_enabled=false
v51a_enabled=false
v52a_anchor_loss finite
v52a_gamma = 0
v52a_reliability_mean finite
v52a_base_reliability_mean finite
v52a_effective_anchor_mass finite
```

Verdict:

```text
PASS
```

## 6. Connectivity Interpretation

V52A fixes the immediate V51A connectivity warning:

| Signal | V51A ACM 1-epoch | V52A ACM 1-epoch |
| --- | ---: | ---: |
| Reliability mean | 0.0014 | 0.1962 |
| Reliable node ratio | 0.0000 | 0.5041 |
| Effective anchor mass | 0.0014 | 0.1962 |

This does not prove the mechanism works at 80 epochs, but it shows that the
curriculum availability path is wired correctly: at `gamma=0`, V52A uses base
anchor evidence instead of requiring posterior/readout agreement first.

## 7. Decision

V52A minimal implementation and connectivity are complete.

Allowed next step:

```text
Run the preregistered first-stage mixed stress exactly once:
datasets = acm,dblp,flickr,texas,squirrel,chameleon
epochs = 80
```

Not allowed before that verdict:

- full 9-dataset smoke;
- 260-epoch full run;
- V52A schedule variants;
- V52A reliability formula variants;
- reliability threshold sweep;
- V50A anchor hyperparameter sweep.

## 8. No-Fabrication Status

All numbers in this file come from the logged ACM one-epoch connectivity run in
`results/archive/v51-v57/unified_aptc_9datasets_v52a_curriculum_reliability_spectral_anchor_diagnostics.jsonl`.
No V52A mixed-stress or full-run result exists yet.
