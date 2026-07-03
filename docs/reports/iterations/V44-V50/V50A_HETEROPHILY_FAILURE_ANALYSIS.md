# V50A Heterophily Failure Analysis

This document analyzes why `v50a_spectral_compactness_anchor` passed the first
stage but failed second-stage expansion. It uses `ccf-idea-optimizer`
exploratory rescue mode: diagnose the mechanism and define the next rescue
question before changing code or running more experiments.

No new experiment is run in this document. No V50A hyperparameter change is
authorized here.

## 1. Evidence Basis

Main local artifacts:

```text
V50_RESCUE_ROUTE_DECISION.md
V50A_PREREGISTRATION.md
V50A_IMPLEMENTATION_REVIEW.md
V50A_FIRST_SMOKE_VERDICT.md
V50A_SECOND_STAGE_PREREGISTRATION.md
V50A_SECOND_STAGE_SMOKE_VERDICT.md
results/archive/v40-v50/unified_aptc_9datasets_v50a_spectral_compactness_anchor_diagnostics.jsonl
```

Reference signals read locally:

```text
Explicit Low-Rank Structured Subspace Learning for Fast Attributed Graph Clustering
Compactness and Consistency: A Joint Framework for Deep Graph Clustering
All Roads Lead to Rome: Exploring Edge Distribution Shift in Heterophilic Graph Learning
What is Missing in Homophily: Graph Homophily Disentanglement in GNNs
```

The local literature signal is consistent with the failure: fixed smoothing can
help when structure and attributes align, but heterophily-style graphs require
edge/structure-aware reliability instead of blindly trusting a single low-pass
subspace.

## 2. What V50A Proved

V50A proved a positive claim:

```text
A fixed low-rank graph-smoothed spectral anchor can rescue the failed topology
mask route on some datasets without violating the unified-pipeline red lines.
```

First-stage evidence:

| Dataset | ACC | Anchor ACC | Agreement @1 | Agreement @80 | Emb Gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| ACM | 0.9088 | 0.8942 | 0.4301 | 0.8982 | 0.0000 |
| DBLP | 0.6810 | 0.8901 | 0.2963 | 0.4972 | 0.0002 |
| Flickr | 0.4397 | 0.3626 | 0.1053 | 0.1251 | 0.0286 |

Interpretation:

- ACM validates the intended mechanism almost perfectly.
- DBLP shows the anchor is strong, but the posterior only partially extracts it.
- Flickr shows the route can help even when the anchor is weak, but the coupling
  is marginal.

This was enough to justify second-stage testing, but not enough to justify full
expansion.

## 3. What V50A Failed

Second-stage gates:

| Gate | Verdict |
| --- | --- |
| Red-line | PASS |
| Anchor non-degeneracy | PASS |
| Coupling | PASS, narrow |
| Posterior/readout safety | FAIL |
| Heterophily stress | FAIL |

Second-stage result summary:

| Dataset | ACC | Anchor ACC | Agreement Move | Emb Gap | Main Issue |
| --- | ---: | ---: | ---: | ---: | --- |
| PubMed | 0.6215 | 0.6581 | -0.0861 | 0.0000 | strong but rejected anchor |
| Wiki | 0.3655 | 0.4844 | +0.0832 | 0.0000 | weak coupling, many classes |
| BlogCatalog | 0.8460 | 0.4355 | +0.0458 | 0.0002 | model succeeds despite weak anchor |
| Texas | 0.7322 | 0.3880 | -0.0164 | 0.0109 | final ACC strong while anchor bad |
| Squirrel | 0.3019 | 0.2448 | +0.0160 | -0.0877 | hard posterior/readout safety failure |
| Chameleon | 0.3377 | 0.3193 | +0.0778 | 0.0000 | pass but low absolute performance |

The core failure is not collapse:

```text
All second-stage anchors have high usage entropy and finite KL.
```

The failure is reliability:

```text
V50A has no mechanism to decide when the spectral anchor should be trusted,
ignored, or treated only as a weak compactness diagnostic.
```

## 4. Dataset-Level Failure Types

### 4.1 Type A: Anchor Is Good But Not Absorbed

Datasets:

```text
DBLP, PubMed, Wiki
```

Signals:

| Dataset | Anchor ACC | Final ACC | Agreement @80 | KL |
| --- | ---: | ---: | ---: | ---: |
| DBLP | 0.8901 | 0.6810 | 0.4972 | 0.1804 |
| PubMed | 0.6581 | 0.6215 | 0.1579 | 0.7947 |
| Wiki | 0.4844 | 0.3655 | 0.1430 | 0.5217 |

Interpretation:

The anchor contains useful cluster information, but the posterior does not
consistently absorb it. This suggests the fixed KL is competing with existing
posterior geometry or target bootstrap rather than becoming a clean readout
teacher.

Rescue implication:

```text
The next route should not simply increase v50a_anchor_weight. It needs a
coupling-reliability mechanism that decides where anchor guidance is coherent.
```

### 4.2 Type B: Model Succeeds Despite Weak Anchor

Datasets:

```text
BlogCatalog, Texas
```

Signals:

| Dataset | Final ACC | Anchor ACC | Anchor Conf | Agreement Move |
| --- | ---: | ---: | ---: | ---: |
| BlogCatalog | 0.8460 | 0.4355 | 0.2161 | +0.0458 |
| Texas | 0.7322 | 0.3880 | 0.3534 | -0.0164 |

Interpretation:

The existing frontend/readout can succeed while the spectral anchor is weak or
even misaligned. Texas is especially important: final ACC is strong, but
q-anchor agreement decreases. This is exactly the case where a fixed teacher
should not be trusted.

Rescue implication:

```text
The next route needs an anchor distrust path. If the learned embedding and
spectral anchor disagree under low confidence, the model should regularize
compactness without imitating anchor assignments.
```

### 4.3 Type C: Posterior/Readout Safety Breaks

Dataset:

```text
Squirrel
```

Signals:

| ACC | Emb KMeans ACC | Emb Gap | Anchor ACC | Emb-Anchor Agreement |
| ---: | ---: | ---: | ---: | ---: |
| 0.3019 | 0.2142 | -0.0877 | 0.2448 | 0.0586 |

Interpretation:

Squirrel violates the hard posterior/readout safety ceiling. The negative gap
means the final posterior output is substantially better than embedding KMeans,
but both embedding-anchor and q-anchor coupling remain weak. The anchor is not a
stable semantic teacher; the readout is solving a different partition than the
embedding geometry.

Rescue implication:

```text
Do not force the embedding to match the anchor on Squirrel-like graphs. The
failure calls for anchor reliability diagnostics and readout/embedding
consistency constraints, not stronger spectral supervision.
```

### 4.4 Type D: Low Absolute Heterophily Performance

Dataset:

```text
Chameleon
```

Signals:

| Final ACC | Anchor ACC | Agreement Move | Emb Gap |
| ---: | ---: | ---: | ---: |
| 0.3377 | 0.3193 | +0.0778 | 0.0000 |

Interpretation:

Chameleon passes the heterophily safety condition but remains low in absolute
performance. This means V50A is safe here but not sufficiently expressive.

Rescue implication:

```text
Safety alone is not enough; the next route must preserve anchor benefits while
adding heterophily-aware discriminative structure.
```

## 5. Mechanism Diagnosis

V50A's anchor is:

```text
X -> row-normalized graph filtering with self-loops -> rank-K SVD -> soft KMeans
```

This encodes a hidden assumption:

```text
graph-smoothed attributes form a good global clustering basis.
```

That assumption is often true for ACM, DBLP, PubMed-like graphs, but it is not
uniformly true for heterophily-style graphs. Local references explain the same
risk from two angles:

- Heterophily work: edge distribution shifts blur homophilic/heterophilic edge
  semantics; robust models need edge-aware or signed treatment, not plain
  smoothing.
- Homophily disentanglement work: label, structural, and feature homophily are
  distinct; one graph-smoothed feature anchor cannot represent all three
  reliably.

Therefore V50A fails because it treats the spectral anchor as globally
trustworthy once it is non-collapsed. Non-collapse is too weak a criterion.

## 6. Why Simple Fixes Are Rejected

Do not do:

- increase `v50a_anchor_weight`;
- lower or raise `v50a_anchor_temperature`;
- change rank from `K` to `2K`;
- change `filter_steps`;
- refresh the anchor;
- select between spectral anchor and posterior by dataset;
- report anchor KMeans or embedding KMeans as final output.

Reason:

These would turn the failure into a hyperparameter search around a known unsafe
assumption. The second-stage failure was not caused by a missing scalar; it was
caused by missing reliability logic.

## 7. Optimized Rescue Idea

Recommended next route name:

```text
v51a_reliability_gated_spectral_anchor
```

Core insight:

```text
Spectral compactness is useful, but only under measurable reliability. The model
should learn from the anchor where the anchor agrees with local/global evidence,
and fall back to geometry-level compactness where assignment-level imitation is
unsafe.
```

Method blueprint:

```text
Input:
  existing V50A q_spec, q_refined, q_embed, edge masks, low/high views

Reliability signals:
  anchor confidence
  q-anchor agreement trend
  q_embed-anchor agreement
  local neighbor consistency under current masks
  low/high view disagreement
  posterior entropy

Operation:
  convert node-level reliability into a stop-gradient weight r_i in [0, 1]

Loss:
  high reliability: weighted KL(q_refined || stopgrad(q_spec))
  low reliability: compactness/geometry alignment only, no assignment imitation

Constraint:
  unified formula for every dataset; no dataset branch
```

The intended contribution shifts from:

```text
fixed spectral teacher improves clustering
```

to:

```text
reliable spectral compactness transfers low-rank structure without forcing
unsafe assignment imitation on heterophily-style graphs.
```

## 8. Candidate V51 Mechanisms

### Candidate A: Reliability-Weighted Anchor KL

Use:

```text
loss = mean_i r_i * KL(q_refined_i || stopgrad(q_spec_i))
```

Where `r_i` is based on anchor confidence, q_embed-anchor agreement, local
consistency, and posterior entropy.

Pros:

- minimal change from V50A;
- directly addresses unsafe teacher imitation;
- can be diagnosed with active ratio and weighted coupling.

Risk:

- if `r_i` becomes too small, the route may degenerate into V50A-off.

Required guard:

```text
report active_ratio, mean_weight, weighted_agreement, and no-collapse floors
```

### Candidate B: Assignment-To-Geometry Fallback

Use assignment KL only when reliable; otherwise align pairwise/prototype
geometry:

```text
high r_i: KL(q_refined || q_spec)
low r_i: match anchor similarity geometry without hard cluster assignment
```

Pros:

- safer for Texas/Squirrel-like cases;
- preserves spectral compactness signal even when labels from anchor are wrong.

Risk:

- more implementation complexity;
- pairwise geometry can be expensive.

### Candidate C: Signed Spectral Anchor

Construct a signed or edge-aware anchor using the existing homo/hetero/hard masks
instead of raw graph filtering:

```text
H_spec = signed_filter(A_homo, A_hetero, X)
```

Pros:

- matches heterophily literature more directly;
- may fix the wrong smoothing assumption.

Risk:

- depends on the edge masks that V43-V49 showed are semantically unstable;
- easy to re-enter the failed topology-mask route.

## 9. Unique Recommendation

Choose Candidate A first:

```text
v51a_reliability_gated_spectral_anchor
```

Reason:

It is the smallest scientific correction to V50A's actual failure. V50A already
proved that the spectral anchor can rescue ACM and help Flickr. The missing
piece is not a different anchor hyperparameter; it is a reliability gate that
prevents harmful or irrelevant assignment imitation.

Candidate B should be the planned fallback if V51A shows `active_ratio` collapse
or if Texas/Squirrel still fail safety. Candidate C is higher risk because it
leans back toward the topology-mask semantics that already failed repeatedly.

## 10. Required V51A Preregistration Before Code

Before implementation, write:

```text
V51A_PREREGISTRATION.md
```

It must fix:

- exact reliability formula;
- whether reliability is stop-gradient;
- minimum/maximum reliability floor;
- whether low-reliability nodes receive zero KL or a geometry fallback;
- exact diagnostics;
- first-stage dataset scope;
- hard stop conditions.

Suggested first-stage scope:

```text
datasets = acm,dblp,flickr,texas,squirrel,chameleon
epochs = 80
seed = 42
device = cuda
```

Reason:

This set includes:

- ACM/DBLP: must not lose the V50A rescue signal;
- Flickr: weak anchor but improvement context;
- Texas/Squirrel/Chameleon: heterophily stress.

Do not include all 9 datasets until V51A passes this mixed stress set.

## 11. Proposed V51A Gates

Red-line gate:

```text
legacy_head_used=false
v43b-v49a failed mechanisms disabled
v51a_enabled=true
no dataset-specific branch
```

Reliability non-collapse:

```text
0.10 <= v51a_reliability_mean <= 0.90 on at least 5/6
v51a_reliable_node_ratio >= 0.10 on at least 5/6
```

Safety:

```text
abs(embedding_posterior_gap) <= 0.04 on 6/6
no dataset abs(gap) > 0.08
```

Anchor-usefulness:

```text
weighted q-anchor agreement improves on at least 4/6
ACM ACC does not fall below V50A ACM by more than 0.02
Squirrel hard safety failure is removed
```

Heterophily stress:

```text
Texas/Squirrel/Chameleon: at least 2/3 pass safety and non-collapse
```

Stop conditions:

- reliability collapses to all-zero or all-one;
- Squirrel remains hard safety failure;
- ACM loses the V50A spectral rescue signal;
- any red-line violation;
- any non-finite loss.

## 12. No-Fabrication Status

All numbers in this document come from local V50A verdict files and diagnostics.
Literature signals are summarized from local markdown references. No V51 result
exists. V51A is a proposed rescue route, not an implemented or evaluated model.
