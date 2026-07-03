# V57A Expansion Review

This document reviews whether `v57a_mass_floor_normalized_residual_anchor` may
expand beyond the preregistered 6-dataset / 80-epoch mixed-stress run.

Decision:

```text
AUTHORIZE LIMITED 9-DATASET / 80-EPOCH SMOKE ONLY.
```

This review does not authorize a 260-epoch full run, seed sweep, target-mass
sweep, max-mass-scale sweep, reliability-cap sweep, beta-bound sweep,
soft-power sweep, hybrid-compensation sweep, schedule variant, reliability
formula variant, threshold sweep, or V50A anchor hyperparameter change.

## 1. Evidence Basis

Local artifacts:

```text
V57A_PREREGISTRATION.md
V57A_IMPLEMENTATION_REVIEW.md
V57A_CONNECTIVITY_VERDICT.md
V57A_FIRST_MIXED_STRESS_VERDICT.md
results/archive/v51-v57/unified_aptc_9datasets_v57a_mass_floor_normalized_residual_anchor.csv
results/archive/v51-v57/unified_aptc_9datasets_v57a_mass_floor_normalized_residual_anchor_diagnostics.jsonl
```

V57A passed every first-stage gate:

| Gate | Verdict |
| --- | --- |
| Red-line | PASS |
| Mass-normalization | PASS |
| Reliability non-collapse | PASS |
| Posterior/readout safety | PASS |
| Anchor usefulness | PASS |
| Heterophily stress | PASS |

This is the first route in the V54A-V57A rescue chain to pass reliability
non-collapse.

## 2. Expansion Rationale

The expansion is justified because V57A tested the intended mechanism:

```text
Detached V56A raw reliability remains the ranking signal.
Fixed mass-floor normalization lifts DBLP/Texas effective mass.
The same formula applies across datasets.
Final labels remain q_refined.
```

The mixed-stress result is promising but not final:

```text
DBLP and Texas cross the 0.08 mass floor.
Squirrel remains above 0.2800.
Flickr improves but remains a weak-anchor exception.
Chameleon drops relative to V56A.
```

Therefore the next step may test coverage on all 9 datasets at the same
80-epoch scale, but must not jump to 260 epochs.

## 3. Authorized Command

Only this command is authorized:

```powershell
conda run --no-capture-output -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v57a_mass_floor_normalized_residual_anchor --datasets "acm,dblp,texas,wisconsin,cornell,chameleon,squirrel,actor,flickr" --epochs 80 --device cuda --log-level WARNING
```

Use the same fixed seed and runner defaults as the existing script.

## 4. Required Verdict Artifact

After the limited smoke, write:

```text
V57A_9DATASET_80E_SMOKE_VERDICT.md
```

It must include:

- exact command;
- 9-dataset ACC/NMI/ARI table;
- red-line table;
- mass-normalization table;
- raw-vs-scaled reliability table;
- reliability non-collapse table;
- posterior/readout safety table;
- anchor usefulness table;
- per-dataset risk notes for Wisconsin, Cornell, and Actor;
- explicit stop/continue decision.

## 5. Expansion Gates

### 5.1 Red-Line Gate

Must pass on 9/9:

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

### 5.2 Mass-Normalization Gate

Must pass on 9/9:

```text
v57a_target_mass = 0.08
v57a_max_mass_scale = 1.50
v57a_max_reliability_cap = 0.90
1.0 <= v57a_mass_scale <= 1.50
0.0 <= v57a_reliability_mean <= 0.90
```

### 5.3 Reliability Gate

Must pass on at least 6/9:

```text
0.08 <= v57a_reliability_mean <= 0.90
v57a_effective_anchor_mass >= 0.08
```

And at least 4/9:

```text
v57a_reliable_node_ratio >= 0.05
```

Allowed weak-anchor exception:

```text
Flickr may fail mass if anchor evidence remains weak.
```

Hard fail:

```text
v57a_reliability_mean < 0.03 on ACM, DBLP, Squirrel, Chameleon, Wisconsin,
Cornell, or Actor
v57a_reliability_mean > 0.97 on any dataset
```

### 5.4 Safety Gate

Must pass:

```text
abs(embedding_posterior_gap) <= 0.04 on at least 8/9
no dataset abs(embedding_posterior_gap) > 0.08
```

### 5.5 Floor Gate

Must pass:

```text
ACM ACC >= 0.8888
DBLP ACC >= 0.6610
Squirrel ACC >= 0.2800
```

No additional dataset-specific ACC floor may be invented after seeing the run.

### 5.6 Anchor Usefulness Gate

Must pass on at least 6/9:

```text
v57a_weighted_q_anchor_agreement_epoch_80 >
v57a_weighted_q_anchor_agreement_epoch_1
```

## 6. Stop Conditions

Stop after the 9-dataset 80-epoch smoke if any occurs:

- red-line violation;
- non-finite loss or diagnostic;
- mass-normalization mismatch;
- reliability hard fail;
- safety hard fail;
- ACM drops below 0.8888;
- DBLP drops below 0.6610;
- Squirrel drops below 0.2800;
- reliability gate fails;
- anchor usefulness gate fails.

Do not repair by tuning:

```text
target_mass
max_mass_scale
max_reliability_cap
beta bounds
soft power
hybrid compensation
schedule
reliability formula
thresholds
V50A anchor construction
dataset-specific branches
```

## 7. Boundary After Smoke

If the 9-dataset 80-epoch smoke passes, the next artifact must be:

```text
V57A_FULL_RUN_PREREGISTRATION.md
```

It must specify whether a 260-epoch run is justified, what exact command is
allowed, what gates apply, and how to handle failure without tuning.

If the smoke fails, stop and write a failure analysis. Do not proceed to a
260-epoch run.

## 8. No-Fabrication Status

This review is based only on local V57A first-stage artifacts. No V57A
9-dataset smoke or 260-epoch full result exists yet.
