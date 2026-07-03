# v57a_mass_floor_normalized_residual_anchor First Mixed-Stress Verdict

This file records the preregistered first-stage mixed-stress result for
`v57a_mass_floor_normalized_residual_anchor`. It follows
`V57A_PREREGISTRATION.md`, `V57A_IMPLEMENTATION_REVIEW.md`, and
`V57A_CONNECTIVITY_VERDICT.md`.

No full 9-dataset smoke, 260-epoch full run, seed sweep, target-mass sweep,
max-mass-scale sweep, reliability-cap sweep, beta-bound sweep, soft-power sweep,
hybrid-compensation sweep, schedule variant, reliability formula variant,
threshold sweep, or V50A anchor hyperparameter sweep is authorized by this
verdict.

## 1. Run

Static check before running:

```powershell
conda run -n aaai-e2e-subspace python -m py_compile core\e2e\sect_coco_e2e.py scripts\run_unified_aptc_9datasets.py
```

Verdict:

```text
PASS
```

Connectivity:

```text
PASS in V57A_CONNECTIVITY_VERDICT.md
```

Mixed-stress command:

```powershell
conda run --no-capture-output -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v57a_mass_floor_normalized_residual_anchor --datasets "acm,dblp,flickr,texas,squirrel,chameleon" --epochs 80 --device cuda --log-level WARNING
```

Run status:

```text
6/6 datasets completed with status=ok.
No target-mass, max-mass-scale, reliability-cap, beta-bound, soft-power,
hybrid-compensation, schedule, reliability formula, threshold, V50A anchor
hyperparameter, seed, final-label, or selector change was made.
```

Output files:

```text
results/archive/v51-v57/unified_aptc_9datasets_v57a_mass_floor_normalized_residual_anchor.csv
results/archive/v51-v57/unified_aptc_9datasets_v57a_mass_floor_normalized_residual_anchor_diagnostics.jsonl
```

Note:

```text
The result files also contain the earlier ACM 1-epoch connectivity row. This
verdict uses the latest record per dataset from the 80-epoch mixed-stress run.
```

## 2. Result Summary

| Dataset | ACC | NMI | ARI | Emb Gap |
| --- | ---: | ---: | ---: | ---: |
| ACM | 0.9025 | 0.6727 | 0.7323 | 0.0000 |
| DBLP | 0.6867 | 0.4064 | 0.3493 | 0.0002 |
| Flickr | 0.4186 | 0.2536 | 0.1723 | -0.0339 |
| Texas | 0.7377 | 0.4944 | 0.6109 | 0.0000 |
| Squirrel | 0.3013 | 0.0616 | 0.0505 | 0.0006 |
| Chameleon | 0.3329 | 0.1387 | 0.0470 | 0.0000 |

Performance interpretation:

```text
ACM, DBLP, and Squirrel pass the preregistered floors. Texas remains stable,
Squirrel remains above the safety floor, and Flickr improves substantially
relative to V54A-V56A. Chameleon drops but remains safety-clean.
```

## 3. Red-Line Gate

Preregistered requirement:

```text
status=ok
legacy_head_used=false
v43b-v49a_enabled=false
v50a_enabled=false
v51a_enabled=false
v52a_enabled=false
v53a_enabled=false
v54a_enabled=false
v55a_enabled=false
v56a_enabled=false
v57a_enabled=true
no selector / no post-processing selector
```

Verdict:

```text
PASS on 6/6.
```

| Dataset | V50A | V51A | V52A | V53A | V54A | V55A | V56A | V57A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ACM | false | false | false | false | false | false | false | true |
| DBLP | false | false | false | false | false | false | false | true |
| Flickr | false | false | false | false | false | false | false | true |
| Texas | false | false | false | false | false | false | false | true |
| Squirrel | false | false | false | false | false | false | false | true |
| Chameleon | false | false | false | false | false | false | false | true |

All legacy and V43B-V49A red-line flags are false in the diagnostics.

## 4. Mass-Normalization Gate

Preregistered requirement:

```text
v57a_gamma_epoch_1 = 0
v57a_gamma_epoch_40 = 0.5
v57a_gamma_epoch_80 = 1
v57a_target_mass = 0.08
v57a_max_mass_scale = 1.50
v57a_max_reliability_cap = 0.90
1.0 <= v57a_mass_scale <= 1.50
0.0 <= v57a_reliability_mean <= 0.90
```

Verdict:

```text
PASS on 6/6.
```

| Dataset | Gamma @1 | Gamma @40 | Gamma @80 | Raw Rel | Mass Scale | Scaled Rel |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ACM | 0.0000 | 0.5000 | 1.0000 | 0.1394 | 1.0000 | 0.1394 |
| DBLP | 0.0000 | 0.5000 | 1.0000 | 0.0752 | 1.0634 | 0.0800 |
| Flickr | 0.0000 | 0.5000 | 1.0000 | 0.0056 | 1.5000 | 0.0085 |
| Texas | 0.0000 | 0.5000 | 1.0000 | 0.0566 | 1.4123 | 0.0800 |
| Squirrel | 0.0000 | 0.5000 | 1.0000 | 0.0855 | 1.0000 | 0.0855 |
| Chameleon | 0.0000 | 0.5000 | 1.0000 | 0.2161 | 1.0000 | 0.2161 |

Interpretation:

```text
The fixed mass-floor normalization works exactly as intended: it leaves already
sufficient datasets unchanged, lifts DBLP/Texas to the 0.08 floor, and caps the
Flickr weak-anchor case at the fixed max scale rather than forcing it to pass.
```

## 5. Raw Vs Scaled Reliability

| Dataset | Raw Rel | Scaled Rel | Rel P10 | Rel P50 | Rel P90 | Reliable Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ACM | 0.1394 | 0.1394 | 0.0524 | 0.1444 | 0.2143 | 0.1484 |
| DBLP | 0.0752 | 0.0800 | 0.0260 | 0.0728 | 0.1329 | 0.0000 |
| Flickr | 0.0056 | 0.0085 | 0.0046 | 0.0067 | 0.0139 | 0.0000 |
| Texas | 0.0566 | 0.0800 | 0.0304 | 0.0702 | 0.1396 | 0.0328 |
| Squirrel | 0.0855 | 0.0855 | 0.0314 | 0.0721 | 0.1436 | 0.0529 |
| Chameleon | 0.2161 | 0.2161 | 0.1100 | 0.1985 | 0.3812 | 0.4901 |

## 6. Reliability Non-Collapse Gate

Preregistered requirement:

```text
Pass on at least 4/6:
0.08 <= v57a_reliability_mean <= 0.90
v57a_effective_anchor_mass >= 0.08

And at least 3/6:
v57a_reliable_node_ratio >= 0.05
```

Hard fail:

```text
v57a_reliability_mean < 0.03 on ACM, DBLP, Squirrel, or Chameleon
v57a_reliability_mean > 0.97 on any dataset
```

Verdict:

```text
PASS.
Mass passes on 5/6.
Reliable-node ratio passes on 3/6.
No hard near-zero failure occurs on ACM, DBLP, Squirrel, or Chameleon.
```

| Dataset | Rel Mean | Reliable Ratio | Effective Mass | Gate |
| --- | ---: | ---: | ---: | --- |
| ACM | 0.1394 | 0.1484 | 0.1394 | PASS |
| DBLP | 0.0800 | 0.0000 | 0.0800 | PASS mass, FAIL ratio |
| Flickr | 0.0085 | 0.0000 | 0.0085 | weak-anchor non-use |
| Texas | 0.0800 | 0.0328 | 0.0800 | PASS mass, FAIL ratio |
| Squirrel | 0.0855 | 0.0529 | 0.0855 | PASS |
| Chameleon | 0.2161 | 0.4901 | 0.2161 | PASS |

This is the first rescue variant in the V54A-V57A chain to pass the
preregistered reliability non-collapse gate.

## 7. Weighted Coupling

Preregistered requirement:

```text
v57a_weighted_q_anchor_agreement_epoch_80 >
v57a_weighted_q_anchor_agreement_epoch_1
Pass on at least 4/6.
```

Verdict:

```text
PASS on 5/6.
```

| Dataset | Weighted Agreement @1 | @40 | @80 | Movement | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| ACM | 0.4358 | 0.9427 | 0.9666 | +0.5308 | PASS |
| DBLP | 0.3128 | 0.5759 | 0.5506 | +0.2378 | PASS |
| Flickr | 0.0243 | 0.0225 | 0.0169 | -0.0074 | FAIL |
| Texas | 0.3893 | 0.3787 | 0.4620 | +0.0727 | PASS |
| Squirrel | 0.1220 | 0.1853 | 0.2137 | +0.0917 | PASS |
| Chameleon | 0.1756 | 0.3339 | 0.4493 | +0.2736 | PASS |

Preservation floors:

| Dataset | V57A ACC | Required Floor | Gate |
| --- | ---: | ---: | --- |
| ACM | 0.9025 | 0.8888 | PASS |
| DBLP | 0.6867 | 0.6610 | PASS |
| Squirrel | 0.3013 | 0.2800 | PASS |

## 8. Safety Gate

Preregistered requirement:

```text
abs(embedding_posterior_gap) <= 0.04 on 6/6
no dataset abs(embedding_posterior_gap) > 0.08
```

Verdict:

```text
PASS on 6/6.
```

| Dataset | Emb Gap | Safety |
| --- | ---: | --- |
| ACM | 0.0000 | PASS |
| DBLP | 0.0002 | PASS |
| Flickr | -0.0339 | PASS |
| Texas | 0.0000 | PASS |
| Squirrel | 0.0006 | PASS |
| Chameleon | 0.0000 | PASS |

## 9. Heterophily Stress Gate

Preregistered requirement for Texas, Squirrel, and Chameleon:

```text
At least 2/3 must satisfy:
abs(embedding_posterior_gap) <= 0.04
v57a_reliability_mean within [0.08, 0.90]
v57a_effective_anchor_mass >= 0.08

Additionally:
Squirrel ACC >= 0.2800
```

Verdict:

```text
PASS on 3/3, with Squirrel ACC floor satisfied.
```

| Dataset | Emb Gap | Rel Mean | Effective Mass | ACC | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Texas | 0.0000 | 0.0800 | 0.0800 | 0.7377 | PASS |
| Squirrel | 0.0006 | 0.0855 | 0.0855 | 0.3013 | PASS |
| Chameleon | 0.0000 | 0.2161 | 0.2161 | 0.3329 | PASS |

## 10. Gate Summary

| Gate | Verdict |
| --- | --- |
| Red-line | PASS |
| Mass-normalization | PASS |
| Reliability non-collapse | PASS |
| Posterior/readout safety | PASS |
| Anchor usefulness | PASS |
| Heterophily stress | PASS |

Decision:

```text
PASS FIRST-STAGE MIXED STRESS.
```

This verdict does not automatically authorize:

```text
260-epoch full run
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

It authorizes only the next review artifact:

```text
V57A_EXPANSION_REVIEW.md
```

That review must decide whether a limited full 9-dataset 80-epoch smoke is
scientifically justified before any expansion beyond the preregistered
mixed-stress setting.

## 11. Scientific Interpretation

V57A tests a clean mechanism:

```text
Keep V56A's reliability ranking, but normalize detached mass to a fixed floor
so medium-consensus datasets receive enough operational anchor weight.
```

The hypothesis is supported at first-stage scale:

```text
DBLP and Texas cross the 0.08 effective-mass floor without violating Squirrel
safety, posterior/readout safety, or red-line constraints. Texas also flips
weighted anchor agreement from a negative movement in V56A to a positive
movement in V57A.
```

Main caution:

```text
The improvement is gate-level rather than full SOTA-level. Chameleon ACC drops
relative to V56A, and Flickr remains a weak-anchor exception despite a large ACC
gain. V57A should be expanded carefully, with no tuning and no full-length run
until a separate expansion review is written.
```

Recommended next artifact:

```text
V57A_EXPANSION_REVIEW.md
```

## 12. No-Fabrication Status

All numbers in this document come from the local V57A mixed-stress run and its
diagnostics:

```text
results/archive/v51-v57/unified_aptc_9datasets_v57a_mass_floor_normalized_residual_anchor.csv
results/archive/v51-v57/unified_aptc_9datasets_v57a_mass_floor_normalized_residual_anchor_diagnostics.jsonl
```

No V57A full-run result exists.
