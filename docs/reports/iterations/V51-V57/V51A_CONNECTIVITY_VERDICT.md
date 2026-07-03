# v51a_reliability_gated_spectral_anchor Connectivity Verdict

This file records the minimal implementation and one-epoch connectivity check
for `v51a_reliability_gated_spectral_anchor`. It is not a mixed-stress result
and must not be used for model selection.

## 1. Implemented Scope

Implemented the V51A minimum described in `V51A_IMPLEMENTATION_REVIEW.md`:

```text
v51a_enabled
v51a_anchor_weight
v51a_reliability_floor
v51a_reliable_threshold
v51a_min_effective_mass
reliability_gated_spectral_anchor_loss
v51a diagnostics
v51a runner variant
```

The V50A spectral anchor builder is reused unchanged. V51A builds the same
fixed anchor when `v51a_enabled=true`, but the V51A variant disables the
unweighted V50A KL:

```text
v50a_enabled=false
v50a_anchor_weight=0.0
v51a_enabled=true
v51a_anchor_weight=0.04
```

No new head, selector, post-processing label path, dataset-specific branch, or
geometry fallback was added.

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
conda run --no-capture-output -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v51a_reliability_gated_spectral_anchor --datasets "acm" --epochs 1 --device cuda --log-level WARNING
```

Run output:

| Dataset | Status | ACC | NMI | ARI |
| --- | --- | ---: | ---: | ---: |
| ACM | ok | 0.7355 | 0.3145 | 0.3666 |

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
| `v51a_enabled` | true |

Verdict:

```text
PASS
```

## 5. V51A Diagnostic Connectivity

| Diagnostic | Value |
| --- | ---: |
| `v50a_anchor_loss` | 0.0000 |
| `v51a_anchor_loss` | 0.0004 |
| `v51a_weighted_q_anchor_kl` | 0.0004 |
| `v51a_weighted_q_anchor_agreement` | 0.0133 |
| `v51a_unweighted_q_anchor_agreement` | 0.4301 |
| `v51a_embedding_anchor_agreement` | 0.4526 |
| `v51a_reliability_mean` | 0.0014 |
| `v51a_reliability_std` | 0.0034 |
| `v51a_reliability_p10` | 0.0000 |
| `v51a_reliability_p50` | 0.0000 |
| `v51a_reliability_p90` | 0.0052 |
| `v51a_reliable_node_ratio` | 0.0000 |
| `v51a_effective_anchor_mass` | 0.0014 |
| `v51a_confidence_component_mean` | 0.3011 |
| `v51a_q_anchor_component_mean` | 0.0377 |
| `v51a_embed_anchor_component_mean` | 0.0149 |
| `v51a_local_component_mean` | 0.0913 |
| `v51a_anchor_acc_diagnostic` | 0.8942 |
| `embedding_posterior_gap` | -0.0013 |

Connectivity pass condition:

```text
status=ok
legacy_head_used=false
v51a_enabled=true
v50a_enabled=false
v51a_anchor_loss finite
v51a_reliability_mean finite
v51a_effective_anchor_mass finite
v51a_reliable_node_ratio finite
```

Verdict:

```text
PASS
```

## 6. Important Warning

The connection works, but the first epoch exposes an early reliability-collapse
risk:

```text
v51a_reliability_mean = 0.0014
v51a_reliable_node_ratio = 0.0000
v51a_effective_anchor_mass = 0.0014
```

This does not fail the connectivity check, because the connectivity check only
requires finite diagnostics. It does mean the preregistered mixed-stress run
must pay close attention to the reliability non-collapse gate:

```text
0.10 <= v51a_reliability_mean <= 0.90
v51a_reliable_node_ratio >= 0.10
v51a_effective_anchor_mass >= 0.10
```

If these remain near zero after 80 epochs, V51A should stop as an
anchor-avoidance failure rather than be tuned by threshold or formula sweep.

## 7. Decision

V51A minimal implementation and connectivity are complete.

Allowed next step:

```text
Run the preregistered first-stage mixed stress exactly once:
datasets = acm,dblp,flickr,texas,squirrel,chameleon
epochs = 80
```

Not allowed before that verdict:

- full 9-dataset smoke;
- 260-epoch full run;
- reliability threshold sweep;
- V51A formula variants;
- V50A anchor hyperparameter sweep.

## 8. No-Fabrication Status

All numbers in this file come from the logged ACM one-epoch connectivity run in
`results/archive/v51-v57/unified_aptc_9datasets_v51a_reliability_gated_spectral_anchor_diagnostics.jsonl`.
No V51A mixed-stress or full-run result exists yet.
