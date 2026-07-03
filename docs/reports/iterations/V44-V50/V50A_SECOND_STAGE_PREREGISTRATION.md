# v50a_spectral_compactness_anchor Second-Stage Preregistration

This document preregisters the second-stage smoke for
`v50a_spectral_compactness_anchor` after the first-stage gates passed. It does
not change the V50A mechanism, constants, loss, anchor construction, or final
label protocol.

## 1. Basis For Expansion

The first-stage smoke on ACM, DBLP, and Flickr passed all preregistered gates in
`V50A_FIRST_SMOKE_VERDICT.md`.

Latest complete 80-epoch first-stage records:

| Dataset | ACC | NMI | ARI | Anchor ACC | Agreement @1 | Agreement @80 | Emb Gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ACM | 0.9088 | 0.6969 | 0.7487 | 0.8942 | 0.4301 | 0.8982 | 0.0000 |
| DBLP | 0.6810 | 0.4002 | 0.3425 | 0.8901 | 0.2963 | 0.4972 | 0.0002 |
| Flickr | 0.4397 | 0.2856 | 0.1940 | 0.3626 | 0.1053 | 0.1251 | 0.0286 |

Interpretation fixed before second-stage execution:

```text
V50A is alive because the stop-gradient spectral compactness anchor couples to
the unified posterior without breaking red lines or posterior/readout safety.
```

Known risk:

```text
Flickr shows weak anchor quality and weak posterior-anchor agreement. Therefore
the second stage must test generalization and failure boundaries, not merely
seek more positive numbers.
```

## 2. Frozen Mechanism

The second-stage smoke must use the exact same implemented V50A variant:

```text
variant = v50a_spectral_compactness_anchor
v50a_anchor_weight = 0.04
v50a_filter_steps = 2
v50a_anchor_rank_multiplier = 1.0
v50a_anchor_temperature = 0.35
v50a_anchor_refresh = false
loss = KL(q_refined || stopgrad(q_spec))
```

No changes are allowed to:

- anchor weight;
- filter steps;
- rank rule;
- temperature;
- refresh policy;
- KL direction;
- final label protocol;
- random seed;
- dataset-specific logic.

## 3. Hard Prohibitions

Do not run or introduce:

- any weight, rank, temperature, filter-step, or refresh sweep;
- V50B/V50A-soft/V50A-high/V50A-low parameter variants;
- full 9-dataset smoke before this second-stage verdict is written;
- 260-epoch full run;
- dataset-specific module, branch, head, loss, assigner, threshold, or weight;
- legacy head as final output;
- S2CAG/ELSS/KMeans anchor output as final label;
- adaptive selector or post-processing selector;
- post-hoc selection among APTC, embedding KMeans, spectral anchor, or legacy
  labels.

The V43B-V49A failed mechanisms must remain disabled.

## 4. Second-Stage Smoke Scope

Run exactly the six remaining datasets that were not included in first-stage:

```text
datasets = pubmed,wiki,blogcatalog,texas,squirrel,chameleon
epochs = 80
seed = 42
device = cuda
```

Command:

```powershell
conda run --no-capture-output -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v50a_spectral_compactness_anchor --datasets "pubmed,wiki,blogcatalog,texas,squirrel,chameleon" --epochs 80 --device cuda --log-level WARNING
```

This is an expansion smoke only. It is not a full final result.

## 5. Required Diagnostics

For every second-stage dataset, record:

Red-line and safety:

```text
status
legacy_head_used
v43b_enabled
v44_enabled
v44b_enabled
v45a_enabled
v46a_enabled
v47a_enabled
v48a_enabled
v49a_enabled
v50a_enabled
embedding_posterior_gap
```

Anchor quality:

```text
v50a_anchor_acc_diagnostic
v50a_anchor_nmi_diagnostic
v50a_anchor_ari_diagnostic
v50a_anchor_entropy
v50a_anchor_confidence
v50a_anchor_cluster_usage_entropy
```

Coupling:

```text
v50a_anchor_loss
v50a_q_anchor_kl
v50a_q_anchor_agreement
v50a_embedding_anchor_agreement
v50a_q_anchor_agreement_epoch_1
v50a_q_anchor_agreement_epoch_40
v50a_q_anchor_agreement_epoch_80
v50a_q_anchor_kl_epoch_1
v50a_q_anchor_kl_epoch_40
v50a_q_anchor_kl_epoch_80
```

Performance context:

```text
final_acc
final_nmi
final_ari
embedding_kmeans_acc
embedding_kmeans_nmi
embedding_kmeans_ari
```

## 6. Second-Stage Gates

### 6.1 Red-Line Gate

Must pass on 6/6:

```text
status=ok
legacy_head_used=false
v43b/v44/v44b/v45a/v46a/v47a/v48a/v49a_enabled=false
v50a_enabled=true
```

Any red-line failure stops the route immediately.

### 6.2 Anchor Non-Degeneracy Gate

Must pass on at least 5/6:

```text
v50a_anchor_cluster_usage_entropy >= 0.60
v50a_anchor_entropy finite
v50a_q_anchor_kl finite
v50a_anchor_loss finite
```

If a single dataset fails this gate, the verdict must treat it as a failure
case and must not use it to justify parameter changes.

### 6.3 Coupling Gate

Primary coupling gate:

```text
v50a_q_anchor_agreement_epoch_80 > v50a_q_anchor_agreement_epoch_1
```

Must pass on at least 4/6.

Weak-coupling warning:

```text
v50a_q_anchor_agreement_epoch_80 < 0.15
```

If this warning appears on 3 or more datasets, V50A is considered
anchor-present-but-weak and must stop before full-run expansion.

### 6.4 Posterior/Readout Safety Gate

Must pass:

```text
abs(embedding_posterior_gap) <= 0.02 on at least 4/6
abs(embedding_posterior_gap) <= 0.04 on at least 5/6
```

Any dataset with `abs(embedding_posterior_gap) > 0.08` is a hard safety failure.

### 6.5 Heterophily Stress Gate

For Texas, Squirrel, and Chameleon, record but do not overfit:

```text
anchor confidence
anchor usage entropy
q-anchor agreement movement
final ACC/NMI/ARI
embedding-posterior gap
```

The route may continue only if at least 2/3 heterophily-style datasets pass both:

```text
v50a_q_anchor_agreement_epoch_80 > v50a_q_anchor_agreement_epoch_1
abs(embedding_posterior_gap) <= 0.04
```

If this fails, the next step should be a failure analysis of spectral-anchor
domain limits, not a hyperparameter sweep.

### 6.6 Performance Context Gate

Performance is supportive, not sufficient. Record whether V50A beats the latest
available in-repo context for the same dataset when a comparable prior variant
exists, but do not authorize full run from performance alone.

Expansion beyond second-stage requires all of:

- red-line gate pass;
- anchor non-degeneracy gate pass;
- coupling gate pass;
- posterior/readout safety gate pass;
- heterophily stress gate pass.

## 7. Stop Conditions

Stop after second-stage smoke if any occurs:

- any red-line failure;
- non-finite loss or diagnostic;
- anchor non-degeneracy fails on 2 or more datasets;
- coupling gate passes on fewer than 4/6;
- weak-coupling warning appears on 3 or more datasets;
- posterior/readout hard safety failure;
- heterophily stress gate fails.

Do not run second-stage twice to choose a better result. If a run fails because
of infrastructure, record the error and rerun only after documenting that the
failure was not a model result.

## 8. Required Verdict Artifact

After the second-stage smoke, write:

```text
V50A_SECOND_STAGE_SMOKE_VERDICT.md
```

It must include:

- the exact command;
- whether any duplicate or interrupted run occurred;
- a 6-dataset result table;
- red-line gate table;
- anchor non-degeneracy table;
- coupling movement table;
- posterior/readout safety table;
- heterophily stress interpretation;
- explicit stop/continue decision.

No full 9-dataset run or 260-epoch run is allowed until this verdict is written.

## 9. No-Fabrication Status

Only first-stage values in this document are completed results. All second-stage
numbers are `TBD` until the preregistered command finishes and the diagnostics
JSONL is parsed.
