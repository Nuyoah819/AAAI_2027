# v52a_curriculum_reliability_spectral_anchor Preregistration

This document preregisters the next rescue route after the V51A mixed-stress
failure. It uses `ccf-idea-optimizer` exploratory rescue mode: fix the mechanism
and gates before implementation. No V52A code has been implemented and no V52A
experiment has been run.

## 1. Motivation

V50A proved that a fixed graph-smoothed low-rank spectral anchor can be useful,
but it was unsafe on heterophily-style graphs:

```text
V50A: useful anchor transfer, unsafe fixed trust.
```

V51A added a node-level reliability gate and fixed Squirrel's hard
posterior/readout gap, but it failed by anchor avoidance:

```text
V51A: safe, but reliability collapses and suppresses useful anchor signal.
```

The key V51A failure was circular:

```text
The anchor is supposed to make q_refined and q_embed compatible with q_spec,
but V51A requires q_refined and q_embed to already be compatible with q_spec
before the anchor can train them.
```

Concrete V51A mixed-stress failure:

| Dataset | V50A ACC | V51A ACC | V51A Rel Mean | V51A Effective Mass |
| --- | ---: | ---: | ---: | ---: |
| ACM | 0.9088 | 0.6846 | 0.0054 | 0.0054 |
| DBLP | 0.6810 | 0.6520 | 0.0020 | 0.0020 |
| Flickr | 0.4397 | 0.3637 | 0.0000 | 0.0000 |
| Texas | 0.7322 | 0.7322 | 0.0024 | 0.0024 |
| Squirrel | 0.3019 | 0.3021 | 0.0029 | 0.0029 |
| Chameleon | 0.3377 | 0.3303 | 0.0176 | 0.0176 |

Therefore V52A must not tune V51A thresholds or V50A anchor hyperparameters. It
must remove the circular dependency by giving the model a fixed early
anchor-availability curriculum.

## 2. Version Name

```text
v52a_curriculum_reliability_spectral_anchor
```

Core hypothesis:

```text
Reliable spectral compactness needs a curriculum: early training should expose
the model to anchor confidence and local anchor consistency before posterior
agreement is used as a trust condition.
```

## 3. Hard Prohibitions

V52A must not use:

- dataset-specific module, branch, head, loss, assigner, threshold, schedule, or
  weight;
- legacy head as final output;
- adaptive selector or post-processing selector;
- S2CAG/ELSS/KMeans anchor labels as final output;
- labels or test-set metrics in training;
- V50A weight, temperature, rank, filter-step, or refresh sweep;
- V51A threshold, exponent, factor, or formula sweep;
- V43B-V49A failed topology loss family as the main mechanism;
- post-hoc selection among APTC, V50A, V51A, V52A, embedding KMeans, spectral
  anchor, or legacy labels;
- signed topology-mask anchor in the first V52A implementation;
- low-reliability geometry fallback in V52A first implementation.

The V50A spectral anchor construction remains frozen:

```text
X -> row-l2 normalize
A_filter = row-normalized adjacency with self-loops
H_spec = A_filter^2 X
U_spec = TruncatedSVD(H_spec, rank=K)
q_spec = softmax(-||U_spec - center||^2 / 0.35)
```

Frozen constants inherited from V50A:

```text
v50a_filter_steps = 2
v50a_anchor_rank_multiplier = 1.0
v50a_anchor_temperature = 0.35
v50a_anchor_refresh = false
```

## 4. Allowed Mechanism

V52A replaces V51A's fully multiplicative reliability gate with a fixed
curriculum reliability gate.

### 4.1 Inputs

Allowed training-time tensors:

```text
q_refined      current trainable posterior
q_embed        embedding readout posterior
q_spec         fixed V50A spectral anchor
edge_index     existing candidate graph edges
epoch          current training epoch
epochs         total training epochs
```

No label, dataset name, target metric, or dataset-specific metadata may enter
the reliability computation.

### 4.2 Components

All component values are computed per node and detached before entering the
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
conf_i = clamp((max(a_i) - 1/K) / (1 - 1/K), 0, 1)
```

Posterior-anchor agreement:

```text
qa_i = sum_c q_i[c] * a_i[c]
qa_i_norm = clamp((qa_i - 1/K) / (1 - 1/K), 0, 1)
```

Embedding-anchor agreement:

```text
ea_i = sum_c e_i[c] * a_i[c]
ea_i_norm = clamp((ea_i - 1/K) / (1 - 1/K), 0, 1)
```

Local anchor consistency:

```text
sim_ij = sum_c a_i[c] * a_j[c]
local_i = mean_{j: incident to i in edge_index treated as undirected} sim_ij
local_i_norm = clamp((local_i - 1/K) / (1 - 1/K), 0, 1)
```

If a node has no sampled candidate neighbor:

```text
local_i_norm = conf_i
```

### 4.3 Curriculum Reliability

V52A uses a fixed schedule:

```text
warmup_epochs = 20
ramp_epochs = 40
gamma_t = clamp((epoch + 1 - warmup_epochs) / ramp_epochs, 0, 1)
```

For the preregistered 80-epoch mixed stress:

```text
epochs 1-20: gamma_t = 0
epochs 21-60: gamma_t linearly increases from 0 to 1
epochs 61-80: gamma_t = 1
```

For a one-epoch connectivity run:

```text
gamma_t = 0
```

Base reliability:

```text
r_base_i = 0.5 * conf_i + 0.5 * local_i_norm
```

Agreement reliability:

```text
r_agree_i = 0.5 * qa_i_norm + 0.5 * ea_i_norm
```

Final reliability:

```text
r_raw_i = (1 - gamma_t) * r_base_i
        + gamma_t * (r_base_i * r_agree_i)
r_i = detach(clamp(r_raw_i, 0, 1))
```

This formula is intentionally not V51A with a looser threshold:

- early anchor availability depends only on fixed anchor evidence;
- posterior/readout agreement cannot block the anchor before the posterior has
  had time to learn from it;
- late training still uses agreement to reduce unsafe assignment imitation;
- all terms are stop-gradient reliability diagnostics, not learnable gates.

### 4.4 Loss

V52A keeps the same KL direction as V51A:

```text
kl_i = KL(q_refined_i || stopgrad(q_spec_i))
```

Reliability-weighted anchor KL:

```text
v52a_anchor_loss = sum_i r_i * kl_i / clamp(sum_i r_i, min=N * reliability_floor)
loss += v52a_anchor_weight * v52a_anchor_loss
```

Fixed constants:

```text
v52a_anchor_weight = 0.04
v52a_reliability_floor = 0.10
v52a_reliable_threshold = 0.20
v52a_min_effective_mass = 0.10
v52a_warmup_epochs = 20
v52a_ramp_epochs = 40
```

No other V52A constants may be tuned in the first implementation.

## 5. Expected Mechanism

Desired behavior:

| Dataset type | Desired V52A behavior |
| --- | --- |
| ACM/DBLP aligned anchor | nonzero early reliability, preserve V50A rescue signal |
| Flickr weak anchor | low reliability is acceptable, but not silent zero if base evidence exists |
| Texas weak anchor but strong model | low-to-moderate late reliability, avoid damaging posterior |
| Squirrel unsafe V50A behavior | early signal allowed, late agreement should prevent posterior/readout hard gap |
| Chameleon low absolute performance | safe behavior with explicit limitation |

Central scientific question:

```text
Can a fixed curriculum separate anchor availability from anchor trust, preserving
V50A's useful spectral signal while retaining V51A's safety benefit?
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
v52a_enabled
embedding_posterior_gap
```

Anchor quality:

```text
v52a_anchor_acc_diagnostic
v52a_anchor_nmi_diagnostic
v52a_anchor_ari_diagnostic
v52a_anchor_entropy
v52a_anchor_confidence
v52a_anchor_cluster_usage_entropy
```

Curriculum:

```text
v52a_gamma
v52a_gamma_epoch_1
v52a_gamma_epoch_20
v52a_gamma_epoch_40
v52a_gamma_epoch_60
v52a_gamma_epoch_80
```

Reliability:

```text
v52a_reliability_mean
v52a_reliability_std
v52a_reliability_p10
v52a_reliability_p50
v52a_reliability_p90
v52a_reliable_node_ratio
v52a_effective_anchor_mass
v52a_base_reliability_mean
v52a_agreement_reliability_mean
v52a_confidence_component_mean
v52a_q_anchor_component_mean
v52a_embed_anchor_component_mean
v52a_local_component_mean
```

Coupling:

```text
v52a_anchor_loss
v52a_weighted_q_anchor_kl
v52a_weighted_q_anchor_agreement
v52a_unweighted_q_anchor_agreement
v52a_embedding_anchor_agreement
v52a_weighted_q_anchor_agreement_epoch_1
v52a_weighted_q_anchor_agreement_epoch_40
v52a_weighted_q_anchor_agreement_epoch_80
v52a_reliability_mean_epoch_1
v52a_reliability_mean_epoch_40
v52a_reliability_mean_epoch_80
v52a_base_reliability_mean_epoch_1
v52a_agreement_reliability_mean_epoch_80
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
V52A_IMPLEMENTATION_REVIEW.md
```

It must verify:

- V52A reliability is detached;
- schedule is fixed and epoch-based, not dataset-based;
- V50A anchor construction is unchanged;
- V50A and V51A losses are disabled in the V52A variant;
- no dataset-specific branch is introduced;
- no low-reliability geometry fallback is added;
- diagnostics can distinguish base reliability from agreement reliability;
- final labels remain the existing unified `q_refined` output.

## 8. Connectivity Run

Only after implementation review and code implementation, run:

```powershell
conda run --no-capture-output -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v52a_curriculum_reliability_spectral_anchor --datasets "acm" --epochs 1 --device cuda --log-level WARNING
```

Connectivity pass condition:

```text
status=ok
legacy_head_used=false
v52a_enabled=true
v50a_enabled=false
v51a_enabled=false
v52a_anchor_loss finite
v52a_gamma = 0
v52a_reliability_mean finite
v52a_base_reliability_mean finite
v52a_effective_anchor_mass finite
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
conda run --no-capture-output -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v52a_curriculum_reliability_spectral_anchor --datasets "acm,dblp,flickr,texas,squirrel,chameleon" --epochs 80 --device cuda --log-level WARNING
```

Do not run all 9 datasets before this mixed-stress verdict.

## 10. First-Stage Gates

### 10.1 Red-Line Gate

Must pass on 6/6:

```text
status=ok
legacy_head_used=false
v43b/v44/v44b/v45a/v46a/v47a/v48a/v49a_enabled=false
v50a_enabled=false
v51a_enabled=false
v52a_enabled=true
no selector / no post-processing selector
```

### 10.2 Curriculum Gate

Must pass on 6/6:

```text
v52a_gamma_epoch_1 = 0
v52a_gamma_epoch_40 = 0.5
v52a_gamma_epoch_80 = 1
```

Allow small floating-point tolerance:

```text
abs(observed - expected) <= 1e-4
```

### 10.3 Reliability Non-Collapse Gate

Must pass on at least 5/6:

```text
0.08 <= v52a_reliability_mean <= 0.90
v52a_reliable_node_ratio >= 0.05
v52a_effective_anchor_mass >= 0.08
```

Hard fail:

```text
v52a_reliability_mean < 0.03 on ACM, DBLP, Texas, Squirrel, or Chameleon
v52a_reliability_mean > 0.97 on any dataset
```

Flickr exception:

```text
Flickr may have v52a_reliability_mean < 0.03 only if anchor confidence,
local consistency, and anchor diagnostics are also weak; this must be reported
as weak-anchor non-use, not success.
```

### 10.4 Anchor Safety Gate

Must pass:

```text
abs(embedding_posterior_gap) <= 0.04 on 6/6
no dataset abs(embedding_posterior_gap) > 0.08
```

Required targeted repair:

```text
Squirrel must no longer have abs(embedding_posterior_gap) > 0.08
```

### 10.5 Anchor Usefulness Gate

Must pass on at least 4/6:

```text
v52a_weighted_q_anchor_agreement_epoch_80 >
v52a_weighted_q_anchor_agreement_epoch_1
```

And:

```text
ACM ACC >= 0.8888
DBLP ACC >= 0.6610
```

These are inherited from V51A to ensure V52A preserves the proven V50A rescue
signal on aligned-anchor datasets.

### 10.6 Heterophily Stress Gate

For Texas, Squirrel, and Chameleon, at least 2/3 must pass:

```text
abs(embedding_posterior_gap) <= 0.04
v52a_reliability_mean within [0.08, 0.90]
v52a_effective_anchor_mass >= 0.08
```

Performance is supportive but not sufficient. Do not authorize expansion solely
because ACC improves on one heterophily-style dataset.

## 11. Stop Conditions

Stop after first-stage mixed stress if any occurs:

- red-line violation;
- non-finite loss or diagnostic;
- curriculum schedule mismatch;
- reliability collapses to near-zero on ACM or DBLP;
- reliability becomes all-one on any dataset;
- Squirrel remains a hard posterior/readout safety failure;
- ACM drops below 0.8888;
- DBLP drops below 0.6610;
- heterophily stress gate fails;
- weighted coupling passes only because effective anchor mass is below 0.08.

Do not run:

- full 9-dataset smoke;
- 260-epoch full run;
- V52A schedule variants;
- V52A reliability formula variants;
- reliability threshold sweep;
- V50A anchor hyperparameter sweep.

## 12. Required Verdict Artifact

After first-stage mixed stress, write:

```text
V52A_FIRST_MIXED_STRESS_VERDICT.md
```

It must include:

- exact command;
- whether any interrupted run occurred;
- 6-dataset ACC/NMI/ARI table;
- red-line table;
- curriculum schedule table;
- reliability non-collapse table;
- base-vs-agreement reliability table;
- weighted coupling table;
- posterior/readout safety table;
- heterophily stress table;
- explicit stop/continue decision.

## 13. No-Fabrication Status

All V50A and V51A numbers in this document come from local verdict files and
diagnostics. No V52A result exists. V52A is a preregistered rescue mechanism,
not an implemented or evaluated model.
