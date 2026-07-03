# v47a_posterior_guided_band_resolution First Smoke Verdict

This file records the preregistered first-stage smoke result for
`v47a_posterior_guided_band_resolution`. It is a post-run verdict, not a
modification of the preregistration.

## Run

Implementation sanity checks:

```powershell
conda run -n aaai-e2e-subspace python -m py_compile core\e2e\sect_coco_e2e.py scripts\run_unified_aptc_9datasets.py
conda run -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v47a_posterior_guided_band_resolution --datasets acm --epochs 1 --device cpu --log-level WARNING
```

The 1-epoch ACM run was a connectivity check only. It is not a gate result.

First-stage smoke command:

```powershell
conda run -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v47a_posterior_guided_band_resolution --datasets "acm,dblp,flickr" --epochs 80 --device cuda --log-level WARNING
```

Output files:

```text
results/archive/v40-v50/unified_aptc_9datasets_v47a_posterior_guided_band_resolution.csv
results/archive/v40-v50/unified_aptc_9datasets_v47a_posterior_guided_band_resolution_diagnostics.jsonl
```

The diagnostics file also contains the prior 1-epoch ACM connectivity check.
The verdict below uses the latest 80-epoch record per dataset.

## Implementation Summary

The implementation follows the preregistered V47A mechanism:

- Posterior targets use `out["q_refined"]`.
- Posterior target construction uses `q_posterior.detach()`.
- V47A gradients flow to topology masks `homo`, `hetero`, and `hard`, not into
  posterior targets.
- Target mass diagnostics report hard-weighted effective mass, with raw
  all-edge mass retained separately.
- Failed V43B/V44/V44B/V45A/V46A loss weights are explicitly zero.
- Inherited `partition_spread_weight`, `freq_separation_weight`, and
  `freq_ortho_weight` are explicitly zero in the V47A variant.
- No ACM/DBLP/Flickr ceiling is used inside training.

Registered first implementation constants:

| Field | Value |
| --- | ---: |
| `v47a_resolution_weight` | 0.01 |
| `v47a_usage_guard_weight` | 0.005 |
| `v47a_agree_high_quantile` | 0.70 |
| `v47a_agree_low_quantile` | 0.30 |
| `v47a_uncert_high_quantile` | 0.70 |
| `v47a_usage_entropy_floor` | 0.60 |
| `v47a_eps` | 1e-8 |

## Result Summary

| Dataset | ACC | NMI | ARI | Emb Gap | Band Mass | Homo Tgt | Hetero Tgt | Defer Tgt | Unassigned |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ACM | 0.6651 | 0.2916 | 0.3144 | 0.0000 | 0.5215 | 0.2822 | 0.1957 | 0.3089 | 0.2131 |
| DBLP | 0.6485 | 0.3510 | 0.2854 | 0.0000 | 0.6853 | 0.2756 | 0.1753 | 0.3223 | 0.2269 |
| Flickr | 0.3537 | 0.2031 | 0.1326 | 0.0000 | 0.5100 | 0.1368 | 0.2843 | 0.2453 | 0.3336 |

Additional diagnostics:

| Dataset | Effective Target | Resolution Loss | Usage Guard | Usage Entropy |
| --- | ---: | ---: | ---: | ---: |
| ACM | 0.7869 | 0.7018 | 0.0000 | 0.9085 |
| DBLP | 0.7731 | 0.8818 | 0.0000 | 0.7597 |
| Flickr | 0.6664 | 0.6353 | 0.0000 | 0.8978 |

## Gate Verdict

### Red-Line Gate

PASS:

| Dataset | Legacy Head | v43b | v44 | v44b | v45a | v46a | v47a |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ACM | false | false | false | false | false | false | true |
| DBLP | false | false | false | false | false | false | true |
| Flickr | false | false | false | false | false | false | true |

Interpretation: V47A did not revive failed prior losses and did not use the
legacy head.

### Target Non-Degeneracy Gate

PASS:

Required on 3/3:

```text
v47a_homo_target_mass > 0
v47a_hetero_target_mass > 0
v47a_defer_target_mass > 0
v47a_unassigned_target_mass < 0.80
```

Observed:

| Dataset | Homo Tgt | Hetero Tgt | Defer Tgt | Unassigned | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| ACM | 0.2822 | 0.1957 | 0.3089 | 0.2131 | PASS |
| DBLP | 0.2756 | 0.1753 | 0.3223 | 0.2269 | PASS |
| Flickr | 0.1368 | 0.2843 | 0.2453 | 0.3336 | PASS |

Interpretation: V47A did not fail from target-mass degeneracy. The
posterior-guided targets were present inside the hard band.

### Band Gate

FAIL:

Required on 3/3:

```text
ACM band_mass <= 0.4991
DBLP band_mass <= 0.6877
Flickr band_mass <= 0.5051
```

Observed:

| Dataset | Ceiling | V46A Band Ref | V47A Band | Verdict |
| --- | ---: | ---: | ---: | --- |
| ACM | 0.4991 | 0.5201 | 0.5215 | FAIL |
| DBLP | 0.6877 | 0.6835 | 0.6853 | PASS ceiling, FAIL vs V46A |
| Flickr | 0.5051 | 0.5083 | 0.5100 | FAIL |

The preregistered requirement that at least 2/3 improve against V46A band mass
also fails. V47A improves band mass on 0/3 datasets.

### Posterior/Readout Safety Gate

PASS:

| Dataset | Emb-Post Gap | Verdict |
| --- | ---: | --- |
| ACM | 0.0000 | PASS |
| DBLP | 0.0000 | PASS |
| Flickr | 0.0000 | PASS |

All satisfy:

```text
abs(embedding_posterior_gap) <= 0.02
```

### Performance Gate

FAIL:

| Dataset | Required ACC | Actual ACC | Verdict |
| --- | ---: | ---: | --- |
| ACM | >= 0.8000 | 0.6651 | FAIL |
| DBLP | >= 0.6450 | 0.6485 | PASS |
| Flickr | >= 0.4500 | 0.3537 | FAIL |

## Mechanism Comparison Against V46A

| Dataset | V46A Band | V47A Band | V46A ACC | V47A ACC | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| ACM | 0.5201 | 0.5215 | 0.6731 | 0.6651 | Worse band and ACC |
| DBLP | 0.6835 | 0.6853 | 0.6603 | 0.6485 | Worse band and ACC |
| Flickr | 0.5083 | 0.5100 | 0.3593 | 0.3537 | Worse band and ACC |

## Mechanism Interpretation

`v47a_posterior_guided_band_resolution` should stop after the first-stage
smoke.

The implementation passed red-line, target non-degeneracy, and
posterior/readout safety checks. This is useful diagnostically: the mechanism
did not fail because posterior targets were absent, because prior failed losses
were revived, or because the posterior/readout path detached from the
embedding.

However, V47A failed the central topology-band and performance gates:

- ACM and Flickr exceeded their preregistered band ceilings.
- Band mass worsened against V46A on all 3 first-stage datasets.
- ACM and Flickr remained below required ACC thresholds.
- DBLP barely passed the ACC threshold but also worsened against V46A.

Key interpretation:

- Stop-gradient posterior agreement produced non-degenerate hard-band targets,
  but those targets did not move the topology masks in the desired direction.
- The defer target did not collapse usage, but it also did not protect the band
  gate.
- The failure is not a target-availability or red-line plumbing failure; it is
  a mechanism failure.

## Decision

Do not run:

- second-batch smoke on Wiki/BlogCatalog/Texas
- full 9-dataset smoke
- 260-epoch full run
- V47A weight sweep
- quantile sweep
- entropy-floor sweep
- stricter/lower defer-target variant under the same mechanism

Do not report the 1-epoch connectivity check as a result.

## Next Direction Constraints

Any next variant must be preregistered before execution and must not be a
simple V47A weight, quantile, or defer-threshold sweep. A valid next step must
change the mechanism, not merely strengthen posterior-guided CE pressure.

Possible research-level conclusions to carry forward:

1. Hard-band posterior targets can be made non-degenerate without breaking
   posterior/readout safety.
2. Non-degenerate posterior-guided hard-band targets are not sufficient to
   improve band mass or performance.
3. The current topology mask parameterization may resist target assignment
   even when a semantic target exists.

## No-Fabrication Status

All V47A numbers in this verdict come from the completed 80-epoch smoke
diagnostics and CSV output. No unrun datasets, second-batch results, full-run
results, sweeps, or SOTA claims are reported here.
