# v45a_edge_local_band_guarded_frequency First Smoke Verdict

This file records the preregistered first-stage smoke result for
`v45a_edge_local_band_guarded_frequency`. It is a post-run verdict, not a
modification of the preregistration.

## Run

Implementation sanity checks:

```powershell
conda run -n aaai-e2e-subspace python -m py_compile core\e2e\sect_coco_e2e.py scripts\run_unified_aptc_9datasets.py
conda run -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v45a_edge_local_band_guarded_frequency --datasets acm --epochs 1 --device cpu --log-level WARNING
```

The 1-epoch ACM run was a connectivity check only. It is not a gate result.

First-stage smoke command:

```powershell
conda run -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v45a_edge_local_band_guarded_frequency --datasets "acm,dblp,flickr" --epochs 80 --device cuda --log-level WARNING
```

Output files:

```text
results/archive/v40-v50/unified_aptc_9datasets_v45a_edge_local_band_guarded_frequency.csv
results/archive/v40-v50/unified_aptc_9datasets_v45a_edge_local_band_guarded_frequency_diagnostics.jsonl
```

The diagnostics file also contains the prior 1-epoch ACM connectivity check.
The verdict below uses the latest 80-epoch record per dataset.

## Implementation Summary

The implementation follows the tightened preregistration:

- `band_reference` is a frozen warmup reference from the same run.
- V45A losses are inactive during warmup epochs 1..5.
- Edge-local frequency masks are detached:
  `boundary_weight=(hetero+hard).detach().clamp(0,1)` and
  `safe_homo_weight=homo.detach().clamp(0,1)`.
- `v44b_pre_hp_corr_weight=0.0`; V44B global correlation is retained only as a
  diagnostic.
- No ACM/DBLP/Flickr ceiling value is used inside the training objective.

Registered first implementation constants:

| Field | Value |
| --- | ---: |
| `v45a_edge_freq_weight` | 0.01 |
| `v45a_band_guard_weight` | 0.01 |
| `v45a_warmup_epochs` | 5 |
| `v45a_band_gate_k` | 20.0 |
| `v45a_target_edge_gap` | 0.0 |
| `v45a_band_reference_delta` | 0.0 |

## Result Summary

| Dataset | ACC | NMI | ARI | Emb-Post Gap | Pre-HP Std | Edge Gap | Edge Corr | Band Mass | Band Ref | Safe Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ACM | 0.7084 | 0.3484 | 0.3815 | 0.0000 | 0.0257 | -0.0057 | -0.2068 | 0.4998 | 0.5205 | 0.6024 |
| DBLP | 0.6596 | 0.3644 | 0.3079 | 0.0000 | 0.0109 | -0.0028 | -0.1853 | 0.6859 | 0.6710 | 0.4263 |
| Flickr | 0.3694 | 0.2154 | 0.1376 | 0.0001 | 0.0472 | -0.0050 | -0.0314 | 0.5037 | 0.6018 | 0.8767 |

## Gate Verdict

### Red-Line Gate

PASS:

| Dataset | Legacy Head | v43b | v44 | v44b | v45a | Reference Ready | Warmup Count |
| --- | --- | --- | --- | --- | --- | --- | ---: |
| ACM | false | false | false | false | true | true | 5 |
| DBLP | false | false | false | false | true | true | 5 |
| Flickr | false | false | false | false | true | true | 5 |

Interpretation: the implementation did not revive the failed v43b/v44/v44b
losses, did not use the legacy head, and did finalize the same-run warmup
reference.

### Pre-HP Diagnostic Gate

PASS on 3/3:

| Dataset | Pre-HP Std | P10 | P90 | Verdict |
| --- | ---: | ---: | ---: | --- |
| ACM | 0.0257 | 0.0019 | 0.0598 | PASS |
| DBLP | 0.0109 | 0.0027 | 0.0300 | PASS |
| Flickr | 0.0472 | 0.0003 | 0.0347 | PASS |

The pre-normalization response remains non-degenerate.

### Edge-Local Frequency Gate

FAIL:

Required at least 2/3:

```text
v45a_edge_response_gap > 0
v45a_edge_response_corr >= 0.05
```

ACM safety requirement:

```text
ACM v45a_edge_response_gap >= 0
ACM v45a_edge_response_corr >= 0
```

Observed:

| Dataset | Edge Gap | Edge Corr | Verdict |
| --- | ---: | ---: | --- |
| ACM | -0.0057 | -0.2068 | FAIL |
| DBLP | -0.0028 | -0.1853 | FAIL |
| Flickr | -0.0050 | -0.0314 | FAIL |

Interpretation: the edge-local objective did not fix the coupling issue. It
became anti-aligned on all three first-stage datasets.

### Band Safety Gate

FAIL:

| Dataset | Ceiling | Band Mass | Warmup Ref | Verdict |
| --- | ---: | ---: | ---: | --- |
| ACM | 0.4991 | 0.4998 | 0.5205 | FAIL |
| DBLP | 0.6877 | 0.6859 | 0.6710 | PASS |
| Flickr | 0.5051 | 0.5037 | 0.6018 | PASS |

The warmup reference was correctly frozen, but ACM still slightly exceeded its
preregistered evaluation ceiling. The ceiling was not used during training.

### Posterior/Readout Safety Gate

PASS:

| Dataset | Emb-Post Gap | Verdict |
| --- | ---: | --- |
| ACM | 0.0000 | PASS |
| DBLP | 0.0000 | PASS |
| Flickr | 0.0001 | PASS |

All satisfy:

```text
abs(embedding_posterior_gap) <= 0.02
```

### Performance Gate

FAIL:

| Dataset | Required ACC | Actual ACC | Verdict |
| --- | ---: | ---: | --- |
| ACM | >= 0.8000 | 0.7084 | FAIL |
| DBLP | >= 0.6450 | 0.6596 | PASS |
| Flickr | >= 0.4500 | 0.3694 | FAIL |

## Mechanism Interpretation

`v45a_edge_local_band_guarded_frequency` should stop after the first-stage
smoke.

The implementation achieved the intended safety plumbing: same-run frozen
warmup reference, inactive warmup losses, detached edge-local masks, retained
V44B pre-HP diagnostics, and no revived failed losses. However, the mechanism
failed the central scientific test:

- Edge-local response did not become positive on any first-stage dataset.
- ACM remained anti-aligned and failed its edge-local safety requirement.
- ACM still slightly exceeded the preregistered band ceiling.
- ACM and Flickr remained far below the preregistered ACC thresholds.

This is an implementation/red-line sanity pass but a mechanism failure.

Key interpretation:

- V44B's pre-HP response remains measurable.
- Changing from global node correlation to this first edge-local gap objective
  is not sufficient.
- The band guard can freeze and report a graph-adaptive reference, but with the
  current loss strength/form it does not close ACM band safety or performance.
- The failure is not due to posterior/readout detachment.

## Decision

Do not run:

- second-batch smoke on Wiki/BlogCatalog/Texas
- full 9-dataset smoke
- 260-epoch full run
- V45A weight sweep
- `k`, warmup length, target-gap, or band-reference-delta sweep

Do not report the 1-epoch connectivity check as a result. Do not treat the V45A
first smoke as a success because the red-line gate passed.

## Next Direction Constraints

Any next variant must be preregistered before execution and must not be a
simple V45A weight or gate sweep. A valid next step must change the mechanism
or explicitly reject frequency-pressure optimization as a loss target.

Possible research-level conclusions to carry forward:

1. Pre-HP response is still a useful diagnostic signal.
2. Edge-local boundary-vs-safe-homophily response, in this form, is not a
   reliable optimization target.
3. Band safety should remain evaluated, but the same-run warmup reference alone
   is not enough to ensure ACM ceiling compliance.

## No-Fabrication Status

All V45A numbers in this verdict come from the completed 80-epoch smoke
diagnostics and CSV output. No unrun datasets, second-batch results, full-run
results, or SOTA claims are reported here.
