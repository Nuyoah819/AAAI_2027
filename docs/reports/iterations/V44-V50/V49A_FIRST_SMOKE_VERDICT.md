# v49a_reparameterized_topology_transition First Smoke Verdict

This file records the preregistered first-stage smoke result for
`v49a_reparameterized_topology_transition`. It is a post-run verdict, not a
modification of the preregistration.

## Run

Implementation sanity checks:

```powershell
conda run -n aaai-e2e-subspace python -m py_compile core\e2e\sect_coco_e2e.py scripts\run_unified_aptc_9datasets.py
conda run -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v49a_reparameterized_topology_transition --datasets acm --epochs 1 --device cpu --log-level WARNING
```

The 1-epoch ACM run was a connectivity check only. It is not a gate result.

First-stage smoke command:

```powershell
conda run -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v49a_reparameterized_topology_transition --datasets "acm,dblp,flickr" --epochs 80 --device cuda --log-level WARNING
```

Output files:

```text
results/archive/v40-v50/unified_aptc_9datasets_v49a_reparameterized_topology_transition.csv
results/archive/v40-v50/unified_aptc_9datasets_v49a_reparameterized_topology_transition_diagnostics.jsonl
```

The diagnostics file also contains the prior 1-epoch ACM connectivity check.
The verdict below uses the latest 80-epoch record per dataset.

## Implementation Summary

V49A replaced the old threshold-derived mask geometry with the preregistered
orientation/clarity simplex:

```text
orient = sigmoid(edge_logit / tau_orient)
clear  = sigmoid(abs(edge_logit) / tau_clear)

homo   = clear * orient
hetero = clear * (1 - orient)
hard   = 1 - clear
```

Implementation boundary:

- V49A uses the existing shared edge confidence output `edge_logit`.
- V49A adds no new loss term.
- V49A posterior target groups are diagnostic only.
- V43B/V44/V44B/V45A/V46A/V47A failed losses are disabled.
- V48A audit is disabled in the runner variant.
- `threshold_reg_weight`, `edge_quantile_anchor_weight`,
  `partition_spread_weight`, `freq_separation_weight`, and
  `freq_ortho_weight` are zero in the V49A variant.
- Old low/high thresholds remain only as compatibility diagnostics.

Registered first implementation constants:

| Field | Value |
| --- | ---: |
| `v49a_enabled` | true |
| `v49a_tau_clear` | 1.0 |
| `v49a_tau_orient` | 1.0 |
| `v49a_snapshot_sample_size` | 20000 |
| `v49a_movement_eps` | 1e-8 |

## Result Summary

| Dataset | ACC | NMI | ARI | Emb Gap | Band | Homo Use | Hetero Use | Hard Use |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ACM | 0.6208 | 0.2480 | 0.2536 | -0.0003 | 0.2976 | 0.3515 | 0.3509 | 0.2976 |
| DBLP | 0.6571 | 0.3603 | 0.2967 | 0.0000 | 0.3162 | 0.3405 | 0.3433 | 0.3162 |
| Flickr | 0.3376 | 0.1946 | 0.1218 | 0.0000 | 0.3199 | 0.3383 | 0.3417 | 0.3199 |

Topology-coordinate diagnostics:

| Dataset | Usage Entropy | Clear Mean | Clear Std | Orient Mean | Orient Std |
| --- | ---: | ---: | ---: | ---: | ---: |
| ACM | 0.9973 | 0.7024 | 0.1127 | 0.4957 | 0.2316 |
| DBLP | 0.9994 | 0.6838 | 0.1189 | 0.4962 | 0.2189 |
| Flickr | 0.9996 | 0.6801 | 0.1208 | 0.4936 | 0.2167 |

Movement diagnostics:

| Dataset | dHomo | dHetero | dHard | dScore | Hard Delta | Hard Corr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ACM | 0.0244 | 0.0516 | 0.0426 | 0.0460 | 0.0008 | 0.8637 |
| DBLP | 0.0600 | 0.0624 | 0.0651 | 0.0755 | 0.0008 | 0.7500 |
| Flickr | 0.0295 | 0.0253 | 0.0318 | 0.0338 | 0.0020 | 0.9362 |

## Gate Verdict

### Red-Line Gate

PASS:

| Dataset | Legacy Head | v43b | v44 | v44b | v45a | v46a | v47a | v48a | v49a |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ACM | false | false | false | false | false | false | false | false | true |
| DBLP | false | false | false | false | false | false | false | false | true |
| Flickr | false | false | false | false | false | false | false | false | true |

Interpretation: V49A did not revive failed prior losses, did not run V48A as an
active audit branch, and did not use the legacy head.

### Usage Non-Collapse Gate

PASS:

Required on 3/3:

```text
v49a_usage_entropy >= 0.60
v49a_homo_usage > 0.05
v49a_hetero_usage > 0.05
v49a_hard_usage > 0.05
```

Observed:

| Dataset | Usage Entropy | Homo Use | Hetero Use | Hard Use | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| ACM | 0.9973 | 0.3515 | 0.3509 | 0.2976 | PASS |
| DBLP | 0.9994 | 0.3405 | 0.3433 | 0.3162 | PASS |
| Flickr | 0.9996 | 0.3383 | 0.3417 | 0.3199 | PASS |

Interpretation: the orientation/clarity simplex did not collapse mask usage.

### Diagnostic Completeness Gate

PASS:

| Dataset | Has Prev Snapshot | Sample Size | Movement Diagnostics | Verdict |
| --- | --- | ---: | --- | --- |
| ACM | true | 20000 | finite | PASS |
| DBLP | true | 20000 | finite | PASS |
| Flickr | true | 20000 | finite | PASS |

### Directional Consistency Gate

FAIL:

Required primary mechanism gate:

```text
At least 2/3 datasets:
v49a_targeted_homo_delta > 0
v49a_targeted_hetero_delta > 0
v49a_targeted_hard_delta >= 0
```

Additional safety requirement:

```text
ACM and Flickr must not both repeat:
targeted_homo_delta < 0 and targeted_hetero_delta < 0
```

Observed:

| Dataset | Homo Tgt | Hetero Tgt | Defer Tgt | Homo Delta | Hetero Delta | Hard Delta | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ACM | 0.3064 | 0.1990 | 0.3034 | -0.000468 | -0.001294 | 0.000748 | FAIL |
| DBLP | 0.2963 | 0.1628 | 0.3093 | 0.002721 | 0.001359 | 0.003384 | PASS |
| Flickr | 0.1518 | 0.2269 | 0.2965 | -0.000897 | -0.000105 | 0.002291 | FAIL |

Directional consistency passes on only 1/3. ACM and Flickr both repeat the V48A
wrong-direction pattern on homo and hetero target groups. This directly triggers
the preregistered stop condition.

### Posterior/Readout Safety Gate

PASS:

Required:

```text
abs(embedding_posterior_gap) <= 0.02 on at least 2/3
abs(embedding_posterior_gap) <= 0.04 on 3/3
```

Observed:

| Dataset | Emb-Post Gap | Verdict |
| --- | ---: | --- |
| ACM | -0.0003 | PASS |
| DBLP | 0.0000 | PASS |
| Flickr | 0.0000 | PASS |

### Band and Performance Context

These are recorded for context only and do not authorize expansion.

V49A vs V48A audit ACC:

| Dataset | V48A ACC | V49A ACC | Context |
| --- | ---: | ---: | --- |
| ACM | 0.6450 | 0.6208 | worse |
| DBLP | 0.6522 | 0.6571 | slightly better |
| Flickr | 0.3401 | 0.3376 | slightly worse |

V49A produces much lower hard usage than V48A-style threshold-band variants, but
this does not translate into directional consistency or performance recovery.

## Mechanism Interpretation

`v49a_reparameterized_topology_transition` should stop after the first-stage
smoke.

The result is diagnostically useful:

- The new simplex avoids usage collapse.
- Posterior/readout safety remains intact.
- Hard usage is lower than in V46A/V47A/V48A-style hard-band diagnostics.
- However, the central directionality failure remains on ACM and Flickr.

This means that simply decoupling orientation and clarity from the same
`edge_logit` is not sufficient. The new parameterization changed mask mass and
preserved safety, but it did not make hard-to-homo or hard-to-hetero transitions
semantically reliable on the datasets that previously failed.

Most likely interpretation:

```text
The failure is not only the algebra of the simplex. The orientation signal
itself is still inherited from the same scalar edge logit, so semantic direction
remains under-specified.
```

## Decision

Do not run:

- second-batch smoke
- full 9-dataset smoke
- 260-epoch full run
- V49A weight sweep
- V49A temperature sweep
- V49A initialization sweep
- V49A target/quantile sweep
- another variant that only remaps the same `edge_logit`

Do not report the 1-epoch connectivity check as a result.

## Next Direction Constraints

Any next variant must be preregistered before execution and must not be a
simple V49A temperature or initialization sweep.

The next route must address the source of orientation, not only the mapping from
orientation to masks. Valid next directions include:

1. A stronger diagnostic recenter: stop topology mechanism search and summarize
   V43B-V49A as negative evidence.
2. A new orientation-source design that uses independent edge evidence for
   orientation and clarity while preserving unified, non-dataset-specific
   training.
3. A formal ablation package comparing the failed mechanism families to justify
   why the paper should pivot away from topology-transition claims.

## No-Fabrication Status

All V49A numbers in this verdict come from the completed 80-epoch smoke
diagnostics and CSV output. No unrun datasets, second-batch results, full-run
results, sweeps, or SOTA claims are reported here.
