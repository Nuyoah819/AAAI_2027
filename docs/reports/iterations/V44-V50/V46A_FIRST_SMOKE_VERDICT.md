# v46a_topology_band_calibration First Smoke Verdict

This file records the preregistered first-stage smoke result for
`v46a_topology_band_calibration`. It is a post-run verdict, not a modification
of the preregistration.

## Run

Implementation sanity checks:

```powershell
conda run -n aaai-e2e-subspace python -m py_compile core\e2e\sect_coco_e2e.py scripts\run_unified_aptc_9datasets.py
conda run -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v46a_topology_band_calibration --datasets acm --epochs 1 --device cpu --log-level WARNING
```

The 1-epoch ACM run was a connectivity check only. It is not a gate result.

First-stage smoke command:

```powershell
conda run -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v46a_topology_band_calibration --datasets "acm,dblp,flickr" --epochs 80 --device cuda --log-level WARNING
```

Output files:

```text
results/archive/v40-v50/unified_aptc_9datasets_v46a_topology_band_calibration.csv
results/archive/v40-v50/unified_aptc_9datasets_v46a_topology_band_calibration_diagnostics.jsonl
```

The diagnostics file also contains the prior 1-epoch ACM connectivity check.
The verdict below uses the latest 80-epoch record per dataset.

## Implementation Summary

The implementation follows the corrected preregistration:

- V46A band mass uses the existing topology-contraction `hard` mask:
  `band_mass=mean(hard)`.
- V44B/V45A frequency-response diagnostics remain diagnostic only.
- Failed V43B/V44/V44B/V45A loss weights are explicitly zero.
- Inherited `partition_spread_weight`, `freq_separation_weight`, and
  `freq_ortho_weight` are explicitly zero in the V46A variant.
- No ACM/DBLP/Flickr ceiling is used inside training.

Registered first implementation constants:

| Field | Value |
| --- | ---: |
| `v46a_band_cal_weight` | 0.01 |
| `v46a_balance_weight` | 0.005 |
| `v46a_spread_weight` | 0.005 |
| `v46a_entropy_floor` | 0.60 |
| `v46a_min_threshold_gap` | 0.05 |

## Result Summary

| Dataset | ACC | NMI | ARI | Emb Gap | Band Mass | Usage Entropy | Threshold Gap | Pre-HP Std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ACM | 0.6731 | 0.3107 | 0.3353 | 0.0017 | 0.5201 | 0.9095 | 0.4284 | 0.0235 |
| DBLP | 0.6603 | 0.3725 | 0.3074 | 0.0000 | 0.6835 | 0.7620 | 0.5463 | 0.0067 |
| Flickr | 0.3593 | 0.2123 | 0.1367 | 0.0102 | 0.5083 | 0.8989 | 0.4294 | 0.0480 |

## Gate Verdict

### Red-Line Gate

PASS:

| Dataset | Legacy Head | v43b | v44 | v44b | v45a | v46a |
| --- | --- | --- | --- | --- | --- | --- |
| ACM | false | false | false | false | false | true |
| DBLP | false | false | false | false | false | true |
| Flickr | false | false | false | false | false | true |

Interpretation: V46A did not revive failed prior losses and did not use the
legacy head.

### Topology Band Gate

FAIL:

Required on 3/3:

```text
ACM band_mass <= 0.4991
DBLP band_mass <= 0.6877
Flickr band_mass <= 0.5051
```

Observed:

| Dataset | Ceiling | V45A Band Ref | V46A Band | Verdict |
| --- | ---: | ---: | ---: | --- |
| ACM | 0.4991 | 0.4998 | 0.5201 | FAIL |
| DBLP | 0.6877 | 0.6859 | 0.6835 | PASS |
| Flickr | 0.5051 | 0.5037 | 0.5083 | FAIL |

Additionally, the preregistered requirement that at least 2/3 improve against
V45A observed band mass fails. Only DBLP improves.

### Collapse Safety Gate

PASS:

| Dataset | Usage Entropy | Threshold Gap | Verdict |
| --- | ---: | ---: | --- |
| ACM | 0.9095 | 0.4284 | PASS |
| DBLP | 0.7620 | 0.5463 | PASS |
| Flickr | 0.8989 | 0.4294 | PASS |

All satisfy:

```text
v46a_usage_entropy >= 0.60
v46a_threshold_gap >= 0.05
```

Interpretation: V46A did not collapse the mask usage or thresholds. The failure
is not a trivial all-one-mask collapse.

### Posterior/Readout Safety Gate

PASS:

| Dataset | Emb-Post Gap | Verdict |
| --- | ---: | --- |
| ACM | 0.0017 | PASS |
| DBLP | 0.0000 | PASS |
| Flickr | 0.0102 | PASS |

All satisfy:

```text
abs(embedding_posterior_gap) <= 0.02
```

### Performance Gate

FAIL:

| Dataset | Required ACC | Actual ACC | Verdict |
| --- | ---: | ---: | --- |
| ACM | >= 0.8000 | 0.6731 | FAIL |
| DBLP | >= 0.6450 | 0.6603 | PASS |
| Flickr | >= 0.4500 | 0.3593 | FAIL |

## Mechanism Interpretation

`v46a_topology_band_calibration` should stop after the first-stage smoke.

The implementation passed red-line, collapse-safety, and posterior/readout
safety checks. However, the central topology-band claim failed: ACM and Flickr
band mass exceeded their preregistered ceilings, and V46A improved band mass
against V45A on only DBLP. Performance also failed on ACM and Flickr.

This is an implementation/red-line sanity pass but a mechanism failure.

Key interpretation:

- Direct `mean(hard^2)` band calibration at the preregistered strength does
  not reduce the unsafe ambiguous band on ACM/Flickr.
- The collapse guard was not active (`balance_loss=0`) because usage entropy
  stayed above the floor.
- The threshold-spread guard was not active (`spread_loss=0`) because threshold
  gaps stayed wide.
- The failure is not due to posterior/readout detachment.
- The frequency-response pressure family remains rejected; V46A confirms that
  simply penalizing the existing hard mask is also insufficient.

## Decision

Do not run:

- second-batch smoke on Wiki/BlogCatalog/Texas
- full 9-dataset smoke
- 260-epoch full run
- V46A weight sweep
- entropy-floor sweep
- threshold-gap sweep
- stronger `mean(hard^2)` variant under the same mechanism

Do not report the 1-epoch connectivity check as a result.

## Next Direction Constraints

Any next variant must be preregistered before execution and must not be a
simple V46A weight or floor sweep. A valid next step must change the topology
calibration target, not merely strengthen `mean(hard^2)`.

Possible research-level conclusions to carry forward:

1. Reducing the existing soft `hard` mask directly is not enough.
2. Collapse/threshold safety can be preserved, but preserving them does not
   guarantee band reduction or performance recovery.
3. The unresolved issue is likely not the amount of regularization, but the
   definition of what should leave the ambiguous band and where it should go.

## No-Fabrication Status

All V46A numbers in this verdict come from the completed 80-epoch smoke
diagnostics and CSV output. No unrun datasets, second-batch results, full-run
results, or SOTA claims are reported here.
