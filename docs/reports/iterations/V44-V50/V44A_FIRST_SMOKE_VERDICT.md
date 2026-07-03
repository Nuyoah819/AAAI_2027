# v44a_conflict_coupled_topology First Smoke Verdict

This file records the preregistered first-stage smoke result for `v44a_conflict_coupled_topology`.
It is a post-run verdict, not a modification of the preregistration.

## Run

Command:

```powershell
conda run -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v44a_conflict_coupled_topology --datasets "acm,dblp,flickr" --epochs 80 --device cuda --log-level WARNING
```

Note: an earlier unquoted `--datasets acm,dblp,flickr` attempt failed at argument parsing and did not start training. The quoted command above is the valid run.

Output files:

```text
results/archive/v40-v50/unified_aptc_9datasets_v44a_conflict_coupled_topology.csv
results/archive/v40-v50/unified_aptc_9datasets_v44a_conflict_coupled_topology_diagnostics.jsonl
```

The diagnostics file also contains a prior 1-epoch ACM connectivity check. The verdict below uses the latest record per dataset.

## Result Summary

| Dataset | ACC | NMI | ARI | Emb-Post Gap | v44 Band | v44 Corr | HP Mean | HP Std | Energy Gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ACM | 0.7197 | 0.3594 | 0.3961 | 0.0000 | 0.4944 | 0.0000 | 0.5000 | 0.0000 | 0.0000 |
| DBLP | 0.6529 | 0.3542 | 0.2953 | 0.0000 | 0.6844 | 0.0000 | 0.5000 | 0.0000 | 0.0000 |
| Flickr | 0.3622 | 0.1972 | 0.1265 | 0.0000 | 0.5051 | 0.0000 | 0.5000 | 0.0000 | 0.0000 |

## Gate Verdict

### Red-Line Gate

PASS:

- `legacy_head_used=false` on ACM/DBLP/Flickr.
- `v44_enabled=true` on ACM/DBLP/Flickr.
- `v43b_enabled=false` on ACM/DBLP/Flickr.
- The runner variant explicitly sets v43b and ideal embedding-pressure weights to 0.

### Posterior/Readout Safety Gate

PASS:

- `embedding_posterior_gap=0.0` on all three datasets.

Interpretation: failure is not caused by posterior/readout detachment.

### Performance Gate

FAIL:

| Dataset | Required ACC | Actual ACC | Verdict |
| --- | ---: | ---: | --- |
| ACM | >= 0.8000 | 0.7197 | FAIL |
| DBLP | >= 0.6450 | 0.6529 | PASS |
| Flickr | >= 0.4500 | 0.3622 | FAIL |

### Topology Gate

FAIL:

| Dataset | v43b Band Reference | v44 Band | Delta | Verdict |
| --- | ---: | ---: | ---: | --- |
| ACM | 0.4991 | 0.4944 | -0.0047 | non-increase only |
| DBLP | 0.6877 | 0.6844 | -0.0033 | non-increase only |
| Flickr | 0.4927 | 0.5051 | +0.0124 | FAIL |

The preregistered requirement was non-increase on all three and clear decrease by at least `0.03` on at least two datasets. This was not met.

### High-Pass Mechanism Gate

FAIL:

- `v44_conflict_energy_corr=0.0` on all three datasets.
- `v44_highpass_energy_std` is effectively zero on all three datasets.
- `v44_energy_gap` is zero or numerically negligible on all three datasets.

This reproduces the original high-pass degeneracy: high-pass energy remains a constant placeholder signal rather than a conflict-coupled signal.

## Mechanism Interpretation

`v44a_conflict_coupled_topology` should stop after the first-stage smoke.

The implementation passed red-line and readout-safety checks, and DBLP passed the ACC threshold. However, ACM and Flickr failed performance gates, topology band resolution did not produce the preregistered clear decrease, and the high-pass coupling mechanism completely failed to activate.

The key technical finding is that the current high-pass energy diagnostic is structurally degenerate under the existing normalized views: `low_view` and `hetero_view` are both L2-normalized, so

```text
||Z_high||^2 / (||Z_low||^2 + ||Z_high||^2)
```

is approximately `0.5` for every node. Therefore the correlation objective has no meaningful energy variance to align with topology conflict.

## Decision

Do not run:

- second-batch smoke on Wiki/BlogCatalog/Texas
- full 9-dataset smoke
- 260-epoch full run
- v44a weight sweep

The next mechanism must first redesign the high-pass conflict signal so it is not based on post-normalization vector norms. A valid next attempt should use a pre-normalization high-pass response, residual magnitude, signed smoothing residual, or edge-level frequency response diagnostic before considering another smoke run.
