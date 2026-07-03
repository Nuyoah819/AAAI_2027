# v51a_reliability_gated_spectral_anchor Preregistration

This document preregisters the next rescue route after the V50A second-stage
failure. It uses `ccf-idea-optimizer` exploratory rescue mode: optimize the
mechanism before implementation. No V51A code has been implemented and no V51A
experiment has been run.

## 1. Motivation

V50A established a useful rescue signal:

```text
a fixed low-rank graph-smoothed spectral anchor can rescue the failed
topology-mask route on ACM/DBLP-style datasets without violating red lines.
```

But V50A failed second-stage expansion:

| Gate | V50A second-stage verdict |
| --- | --- |
| Red-line | PASS |
| Anchor non-degeneracy | PASS |
| Coupling | PASS, narrow |
| Posterior/readout safety | FAIL |
| Heterophily stress | FAIL |

The decisive failure was not anchor collapse. All second-stage anchors were
finite and non-collapsed. The decisive failure was reliability:

```text
V50A treats the spectral anchor as globally trustworthy once it exists.
```

Concrete failure signals:

| Dataset | Failure signal |
| --- | --- |
| Texas | final ACC strong, but q-anchor agreement decreases |
| Squirrel | embedding_posterior_gap = -0.0877 hard safety failure |
| Chameleon | safe but low absolute performance |
| PubMed | anchor has information, but q-anchor agreement decreases |

Therefore V51A must not tune V50A's anchor weight, temperature, rank, or filter
steps. It must add a unified reliability mechanism that decides where the
assignment-level anchor KL is safe.

## 2. Version Name

```text
v51a_reliability_gated_spectral_anchor
```

Core hypothesis:

```text
Spectral compactness is useful only under measurable reliability. A
stop-gradient reliability gate can preserve V50A's rescue signal on aligned
graphs while preventing unsafe assignment imitation on heterophily-style graphs.
```

## 3. Hard Prohibitions

V51A must not use:

- dataset-specific module, branch, head, loss, assigner, threshold, or weight;
- legacy head as final output;
- adaptive selector or post-processing selector;
- S2CAG/ELSS/KMeans anchor labels as final output;
- labels or test-set metrics in training;
- V50A weight, temperature, rank, filter-step, or refresh sweep;
- V43B-V49A failed loss family as the main mechanism;
- post-hoc selection among APTC, embedding KMeans, spectral anchor, or legacy
  labels;
- signed topology-mask anchor in the first V51A implementation.

The V50A spectral anchor construction stays frozen:

```text
X -> row-l2 normalize
A_filter = row-normalized adjacency with self-loops
H_spec = A_filter^2 X
U_spec = TruncatedSVD(H_spec, rank=K)
q_spec = softmax(-||U_spec - center||^2 / 0.35)
```

Frozen constants inherited from V50A:

```text
v50a_anchor_weight = 0.04
v50a_filter_steps = 2
v50a_anchor_rank_multiplier = 1.0
v50a_anchor_temperature = 0.35
v50a_anchor_refresh = false
```

## 4. Allowed Mechanism

V51A adds a node-level stop-gradient reliability gate to the V50A anchor KL.

### 4.1 Inputs

Allowed training-time tensors:

```text
q_refined      current trainable posterior
q_embed        embedding readout posterior
q_spec         fixed V50A spectral anchor
edge_index     existing candidate graph edges
```

No label, dataset name, target metric, or dataset-specific metadata may enter
the reliability computation.

### 4.2 Reliability Formula

All reliability terms are computed per node and detached before entering the
loss.

Let:

```text
K = number of clusters
a_i = normalize(q_spec_i)
q_i = normalize(q_refined_i)
e_i = normalize(q_embed_i)
```

Anchor confidence:

```text
conf_i = (max(a_i) - 1/K) / (1 - 1/K)
conf_i = clamp(conf_i, 0, 1)
```

Posterior-anchor soft agreement:

```text
qa_i = sum_c q_i[c] * a_i[c]
qa_i_norm = (qa_i - 1/K) / (1 - 1/K)
qa_i_norm = clamp(qa_i_norm, 0, 1)
```

Embedding-anchor soft agreement:

```text
ea_i = sum_c e_i[c] * a_i[c]
ea_i_norm = (ea_i - 1/K) / (1 - 1/K)
ea_i_norm = clamp(ea_i_norm, 0, 1)
```

Local anchor consistency on existing candidate edges:

```text
sim_ij = sum_c a_i[c] * a_j[c]
local_i = mean_{j: (i,j) in edge_index or (j,i) in edge_index} sim_ij
local_i_norm = (local_i - 1/K) / (1 - 1/K)
local_i_norm = clamp(local_i_norm, 0, 1)
```

If a node has no sampled candidate neighbor, use:

```text
local_i_norm = conf_i
```

Final reliability:

```text
r_raw_i = conf_i * sqrt(clamp(qa_i_norm * ea_i_norm, 0, 1)) * sqrt(local_i_norm)
r_i = detach(clamp(r_raw_i, 0, 1))
```

This formula is intentionally conservative:

- high anchor confidence alone is not enough;
- q-refined and q-embed must both be compatible with the anchor;
- local graph consistency must not contradict the anchor;
- reliability cannot be learned or optimized directly.

### 4.3 Loss

V51A replaces the global V50A anchor KL with a reliability-weighted anchor KL:

```text
kl_i = KL(q_refined_i || stopgrad(q_spec_i))
v51a_anchor_loss = sum_i r_i * kl_i / clamp(sum_i r_i, min=N * reliability_floor)
loss += v51a_anchor_weight * v51a_anchor_loss
```

Fixed constants:

```text
v51a_anchor_weight = 0.04
v51a_reliability_floor = 0.10
v51a_reliable_threshold = 0.20
v51a_min_effective_mass = 0.10
```

Important boundary:

```text
V51A does not add a low-reliability geometry fallback in the first
implementation.
```

Reason:

The first test must isolate whether reliability gating alone removes V50A's
Squirrel safety failure while preserving ACM/DBLP rescue. Geometry fallback is
reserved for a later preregistered route if V51A collapses or underuses the
anchor.

## 5. Expected Mechanism

V51A should behave differently across V50A's failure types:

| Failure type | Desired V51A behavior |
| --- | --- |
| ACM/DBLP-style aligned anchor | high reliability, preserve spectral rescue |
| PubMed good-but-rejected anchor | moderate reliability, avoid forcing all nodes |
| Texas weak anchor but strong model | low/moderate reliability, do not damage posterior |
| Squirrel unsafe anchor/readout split | low reliability, remove hard safety failure |
| Chameleon low absolute performance | safe reliability, record limit without overclaiming |

The central scientific question:

```text
Can reliability gating separate useful spectral compactness from unsafe
assignment imitation without dataset-specific routing?
```

## 6. Required Diagnostics

Red-line:

```text
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
v51a_enabled
embedding_posterior_gap
```

Anchor quality inherited from V50A:

```text
v51a_anchor_acc_diagnostic
v51a_anchor_nmi_diagnostic
v51a_anchor_ari_diagnostic
v51a_anchor_entropy
v51a_anchor_confidence
v51a_anchor_cluster_usage_entropy
```

Reliability:

```text
v51a_reliability_mean
v51a_reliability_std
v51a_reliability_p10
v51a_reliability_p50
v51a_reliability_p90
v51a_reliable_node_ratio
v51a_effective_anchor_mass
v51a_confidence_component_mean
v51a_q_anchor_component_mean
v51a_embed_anchor_component_mean
v51a_local_component_mean
```

Coupling:

```text
v51a_anchor_loss
v51a_weighted_q_anchor_kl
v51a_weighted_q_anchor_agreement
v51a_unweighted_q_anchor_agreement
v51a_embedding_anchor_agreement
v51a_weighted_q_anchor_agreement_epoch_1
v51a_weighted_q_anchor_agreement_epoch_40
v51a_weighted_q_anchor_agreement_epoch_80
v51a_reliability_mean_epoch_1
v51a_reliability_mean_epoch_40
v51a_reliability_mean_epoch_80
```

Safety/performance:

```text
final_acc
final_nmi
final_ari
embedding_kmeans_acc
embedding_kmeans_nmi
embedding_kmeans_ari
embedding_posterior_gap
```

## 7. Implementation Review Required

Before code implementation, write:

```text
V51A_IMPLEMENTATION_REVIEW.md
```

It must verify:

- reliability is computed with `detach`;
- V50A anchor construction is unchanged;
- V50A unweighted KL is not simultaneously active;
- no dataset-specific branch is introduced;
- diagnostics can distinguish useful gating from all-zero anchor avoidance;
- `q_refined`, `q_embed`, `q_spec`, and `edge_index` are available at the loss
  site;
- final labels remain the existing unified output, not the anchor labels.

## 8. Connectivity Run

Only after implementation review and code implementation, run:

```powershell
conda run --no-capture-output -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v51a_reliability_gated_spectral_anchor --datasets "acm" --epochs 1 --device cuda --log-level WARNING
```

Connectivity pass condition:

```text
status=ok
legacy_head_used=false
v51a_enabled=true
v50a_enabled=false or v50a_anchor_weight=0.0
v51a_anchor_loss finite
v51a_reliability_mean finite
v51a_effective_anchor_mass finite
```

The connectivity run is not a performance result.

## 9. First-Stage Mixed Stress Experiment

Only after connectivity passes:

```text
datasets = acm,dblp,flickr,texas,squirrel,chameleon
epochs = 80
seed = 42
device = cuda
```

Command:

```powershell
conda run --no-capture-output -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v51a_reliability_gated_spectral_anchor --datasets "acm,dblp,flickr,texas,squirrel,chameleon" --epochs 80 --device cuda --log-level WARNING
```

Reason:

- ACM/DBLP check whether V50A's rescue signal is preserved.
- Flickr checks weak-anchor recovery.
- Texas/Squirrel/Chameleon test the failure boundary that stopped V50A.

Do not run all 9 datasets before this mixed stress verdict.

## 10. First-Stage Gates

### 10.1 Red-Line Gate

Must pass on 6/6:

```text
status=ok
legacy_head_used=false
v43b/v44/v44b/v45a/v46a/v47a/v48a/v49a_enabled=false
v51a_enabled=true
no selector / no post-processing selector
```

### 10.2 Reliability Non-Collapse Gate

Must pass on at least 5/6:

```text
0.10 <= v51a_reliability_mean <= 0.90
v51a_reliable_node_ratio >= 0.10
v51a_effective_anchor_mass >= 0.10
```

Hard fail:

```text
v51a_reliability_mean < 0.03 on any dataset
v51a_reliability_mean > 0.97 on any dataset
```

### 10.3 Anchor Safety Gate

Must pass:

```text
abs(embedding_posterior_gap) <= 0.04 on 6/6
no dataset abs(embedding_posterior_gap) > 0.08
```

Required targeted repair:

```text
Squirrel must no longer have abs(embedding_posterior_gap) > 0.08
```

### 10.4 Anchor Usefulness Gate

Must pass on at least 4/6:

```text
v51a_weighted_q_anchor_agreement_epoch_80 >
v51a_weighted_q_anchor_agreement_epoch_1
```

And:

```text
ACM ACC >= V50A ACM ACC - 0.02 = 0.8888
DBLP ACC >= V50A DBLP ACC - 0.02 = 0.6610
```

This prevents the reliability gate from avoiding the anchor so aggressively that
it destroys the proven V50A rescue signal.

### 10.5 Heterophily Stress Gate

For Texas, Squirrel, and Chameleon, at least 2/3 must pass:

```text
abs(embedding_posterior_gap) <= 0.04
v51a_reliability_mean within [0.10, 0.90]
v51a_effective_anchor_mass >= 0.10
```

Performance is supportive but not sufficient. Do not authorize expansion solely
because ACC improves on one heterophily-style dataset.

## 11. Stop Conditions

Stop after first-stage mixed stress if any occurs:

- red-line violation;
- non-finite loss or diagnostic;
- reliability collapses to all-zero or all-one;
- Squirrel remains a hard posterior/readout safety failure;
- ACM drops below 0.8888;
- DBLP drops below 0.6610;
- heterophily stress gate fails;
- weighted coupling passes only because effective anchor mass is below 0.10.

Do not run:

- full 9-dataset smoke;
- 260-epoch full run;
- V51A reliability formula variants;
- reliability threshold sweep;
- V50A anchor hyperparameter sweep.

## 12. Required Verdict Artifact

After first-stage mixed stress, write:

```text
V51A_FIRST_MIXED_STRESS_VERDICT.md
```

It must include:

- exact command;
- whether any interrupted run occurred;
- 6-dataset ACC/NMI/ARI table;
- red-line table;
- reliability non-collapse table;
- weighted coupling table;
- posterior/readout safety table;
- heterophily stress table;
- explicit stop/continue decision.

## 13. No-Fabrication Status

All V50A numbers in this document come from local V50A verdict files and
diagnostics. No V51A result exists. V51A is a preregistered rescue mechanism,
not an implemented or evaluated model.
