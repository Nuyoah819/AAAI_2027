# v53a_residual_curriculum_spectral_anchor Preregistration

This document preregisters the next rescue route after the V52A mixed-stress
failure. It uses `ccf-idea-optimizer` exploratory rescue mode: fix the mechanism
and gates before implementation. No V53A code has been implemented and no V53A
experiment has been run.

## 1. Motivation

V52A showed that a fixed curriculum is useful but incomplete:

```text
Early base reliability restores anchor availability and preserves ACM/DBLP
performance, but late gamma=1 removes the base path and recreates reliability
collapse.
```

Observed V52A pattern:

| Dataset | Rel @1 | Rel @40 | Rel @80 | ACC | Safety |
| --- | ---: | ---: | ---: | ---: | --- |
| ACM | 0.1962 | 0.1051 | 0.0197 | 0.9041 | PASS |
| DBLP | 0.1389 | 0.0725 | 0.0055 | 0.6867 | PASS |
| Texas | 0.1089 | 0.0580 | 0.0075 | 0.7377 | PASS |
| Squirrel | 0.2086 | 0.1074 | 0.0068 | 0.3003 | PASS |
| Chameleon | 0.4507 | 0.2422 | 0.0376 | 0.3329 | PASS |

Therefore V53A must not tune the V52A schedule. It must preserve a fixed
late-stage residual of base reliability while still allowing agreement to
modulate unsafe anchor imitation.

## 2. Version Name

```text
v53a_residual_curriculum_spectral_anchor
```

Core hypothesis:

```text
A nonzero late residual of anchor confidence/local consistency can prevent
reliability collapse, while agreement modulation preserves the safety gains of
V51A/V52A.
```

## 3. Hard Prohibitions

V53A must not use:

- dataset-specific module, branch, head, loss, assigner, threshold, schedule, or
  weight;
- legacy head as final output;
- adaptive selector or post-processing selector;
- S2CAG/ELSS/KMeans anchor labels as final output;
- labels or test-set metrics in training;
- V50A weight, temperature, rank, filter-step, or refresh sweep;
- V51A/V52A threshold, exponent, schedule, or formula sweep;
- beta sweep or dataset-specific beta;
- V43B-V49A failed topology loss family as the main mechanism;
- post-hoc selection among APTC, V50A, V51A, V52A, V53A, embedding KMeans,
  spectral anchor, or legacy labels;
- signed topology-mask anchor in the first V53A implementation;
- low-reliability geometry fallback in the first V53A implementation.

The V50A spectral anchor construction remains frozen:

```text
X -> row-l2 normalize
A_filter = row-normalized adjacency with self-loops
H_spec = A_filter^2 X
U_spec = TruncatedSVD(H_spec, rank=K)
q_spec = softmax(-||U_spec - center||^2 / 0.35)
```

Frozen constants:

```text
v50a_filter_steps = 2
v50a_anchor_rank_multiplier = 1.0
v50a_anchor_temperature = 0.35
v50a_anchor_refresh = false
```

## 4. Allowed Mechanism

V53A inherits V52A's components and schedule but changes the late reliability
composition.

### 4.1 Inputs

Allowed training-time tensors:

```text
q_refined
q_embed
q_spec
edge_index
epoch
epochs
```

No label, dataset name, target metric, or dataset-specific metadata may enter
the reliability computation.

### 4.2 Components

All reliability components are computed per node and detached before entering
the loss.

Anchor confidence:

```text
conf_i = clamp((max(a_i) - 1/K) / (1 - 1/K), 0, 1)
```

Local anchor consistency:

```text
local_i_norm = clamp((mean_j sum_c a_i[c] * a_j[c] - 1/K) / (1 - 1/K), 0, 1)
```

Posterior-anchor and embedding-anchor agreement:

```text
qa_i_norm = clamp((sum_c q_i[c] * a_i[c] - 1/K) / (1 - 1/K), 0, 1)
ea_i_norm = clamp((sum_c e_i[c] * a_i[c] - 1/K) / (1 - 1/K), 0, 1)
```

Base and agreement reliability:

```text
r_base_i = 0.5 * conf_i + 0.5 * local_i_norm
r_agree_i = 0.5 * qa_i_norm + 0.5 * ea_i_norm
```

### 4.3 Residual Curriculum

V53A inherits the fixed V52A schedule:

```text
warmup_epochs = 20
ramp_epochs = 40
gamma_t = clamp((epoch + 1 - warmup_epochs) / ramp_epochs, 0, 1)
```

Fixed residual:

```text
beta = 0.50
```

Final reliability:

```text
r_multiplier_i = (1 - gamma_t)
               + gamma_t * (beta + (1 - beta) * r_agree_i)
r_i = detach(clamp(r_base_i * r_multiplier_i, 0, 1))
```

Equivalent endpoints:

```text
epoch 1:  r_i = r_base_i
epoch 80: r_i = r_base_i * (0.50 + 0.50 * r_agree_i)
```

This is intentionally different from V52A:

```text
V52A epoch 80: r_i = r_base_i * r_agree_i
V53A epoch 80: r_i = r_base_i * (0.50 + 0.50 * r_agree_i)
```

The residual preserves anchor availability; agreement still reduces but cannot
erase base reliability.

### 4.4 Loss

V53A keeps the same KL direction:

```text
kl_i = KL(q_refined_i || stopgrad(q_spec_i))
v53a_anchor_loss = sum_i r_i * kl_i / clamp(sum_i r_i, min=N * reliability_floor)
loss += v53a_anchor_weight * v53a_anchor_loss
```

Fixed constants:

```text
v53a_anchor_weight = 0.04
v53a_reliability_floor = 0.10
v53a_reliable_threshold = 0.20
v53a_min_effective_mass = 0.10
v53a_warmup_epochs = 20
v53a_ramp_epochs = 40
v53a_residual_beta = 0.50
```

No other V53A constants may be tuned in the first implementation.

## 5. Required Diagnostics

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
v53a_enabled
embedding_posterior_gap
```

Anchor quality:

```text
v53a_anchor_acc_diagnostic
v53a_anchor_nmi_diagnostic
v53a_anchor_ari_diagnostic
v53a_anchor_entropy
v53a_anchor_confidence
v53a_anchor_cluster_usage_entropy
```

Curriculum and residual:

```text
v53a_gamma
v53a_residual_beta
v53a_gamma_epoch_1
v53a_gamma_epoch_40
v53a_gamma_epoch_80
v53a_residual_multiplier_mean
v53a_residual_multiplier_mean_epoch_80
```

Reliability:

```text
v53a_reliability_mean
v53a_reliability_std
v53a_reliability_p10
v53a_reliability_p50
v53a_reliability_p90
v53a_reliable_node_ratio
v53a_effective_anchor_mass
v53a_base_reliability_mean
v53a_agreement_reliability_mean
v53a_confidence_component_mean
v53a_q_anchor_component_mean
v53a_embed_anchor_component_mean
v53a_local_component_mean
```

Coupling:

```text
v53a_anchor_loss
v53a_weighted_q_anchor_kl
v53a_weighted_q_anchor_agreement
v53a_unweighted_q_anchor_agreement
v53a_embedding_anchor_agreement
v53a_weighted_q_anchor_agreement_epoch_1
v53a_weighted_q_anchor_agreement_epoch_40
v53a_weighted_q_anchor_agreement_epoch_80
v53a_reliability_mean_epoch_1
v53a_reliability_mean_epoch_40
v53a_reliability_mean_epoch_80
v53a_base_reliability_mean_epoch_1
v53a_agreement_reliability_mean_epoch_80
```

## 6. Implementation Review Required

Before code implementation, write:

```text
V53A_IMPLEMENTATION_REVIEW.md
```

It must verify:

- V53A reliability is detached;
- V52A schedule is inherited unchanged;
- `beta=0.50` is fixed and not dataset-specific;
- V50A anchor construction is unchanged;
- V50A/V51A/V52A losses are disabled in the V53A variant;
- no dataset-specific branch is introduced;
- no low-reliability geometry fallback is added;
- final labels remain the existing unified `q_refined` output.

## 7. Connectivity Run

Only after implementation review and code implementation, run:

```powershell
conda run --no-capture-output -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v53a_residual_curriculum_spectral_anchor --datasets "acm" --epochs 1 --device cuda --log-level WARNING
```

Connectivity pass condition:

```text
status=ok
legacy_head_used=false
v53a_enabled=true
v50a_enabled=false
v51a_enabled=false
v52a_enabled=false
v53a_anchor_loss finite
v53a_gamma = 0
v53a_residual_beta = 0.50
v53a_reliability_mean finite
v53a_effective_anchor_mass finite
```

Connectivity is not a performance result.

## 8. First-Stage Mixed Stress Experiment

Only after connectivity passes:

```text
datasets = acm,dblp,flickr,texas,squirrel,chameleon
epochs = 80
seed = 42
device = cuda
```

Command:

```powershell
conda run --no-capture-output -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v53a_residual_curriculum_spectral_anchor --datasets "acm,dblp,flickr,texas,squirrel,chameleon" --epochs 80 --device cuda --log-level WARNING
```

Do not run all 9 datasets before this mixed-stress verdict.

## 9. First-Stage Gates

### 9.1 Red-Line Gate

Must pass on 6/6:

```text
status=ok
legacy_head_used=false
v43b-v49a_enabled=false
v50a_enabled=false
v51a_enabled=false
v52a_enabled=false
v53a_enabled=true
no selector / no post-processing selector
```

### 9.2 Residual Schedule Gate

Must pass on 6/6:

```text
v53a_gamma_epoch_1 = 0
v53a_gamma_epoch_40 = 0.5
v53a_gamma_epoch_80 = 1
v53a_residual_beta = 0.50
```

Tolerance:

```text
abs(observed - expected) <= 1e-4
```

### 9.3 Reliability Non-Collapse Gate

Must pass on at least 5/6:

```text
0.08 <= v53a_reliability_mean <= 0.90
v53a_reliable_node_ratio >= 0.05
v53a_effective_anchor_mass >= 0.08
```

Hard fail:

```text
v53a_reliability_mean < 0.03 on ACM, DBLP, Texas, Squirrel, or Chameleon
v53a_reliability_mean > 0.97 on any dataset
```

Flickr exception:

```text
Flickr may have v53a_reliability_mean < 0.03 only if anchor confidence,
local consistency, and anchor diagnostics are weak; report as weak-anchor
non-use, not success.
```

### 9.4 Safety Gate

Must pass:

```text
abs(embedding_posterior_gap) <= 0.04 on 6/6
no dataset abs(embedding_posterior_gap) > 0.08
Squirrel must no longer have abs(embedding_posterior_gap) > 0.08
```

### 9.5 Anchor Usefulness Gate

Must pass on at least 4/6:

```text
v53a_weighted_q_anchor_agreement_epoch_80 >
v53a_weighted_q_anchor_agreement_epoch_1
```

And:

```text
ACM ACC >= 0.8888
DBLP ACC >= 0.6610
```

### 9.6 Heterophily Stress Gate

For Texas, Squirrel, and Chameleon, at least 2/3 must pass:

```text
abs(embedding_posterior_gap) <= 0.04
v53a_reliability_mean within [0.08, 0.90]
v53a_effective_anchor_mass >= 0.08
```

## 10. Stop Conditions

Stop after first-stage mixed stress if any occurs:

- red-line violation;
- non-finite loss or diagnostic;
- residual schedule mismatch;
- beta mismatch;
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
- beta sweep;
- V53A schedule variants;
- V53A reliability formula variants;
- reliability threshold sweep;
- V50A anchor hyperparameter sweep.

## 11. Required Verdict Artifact

After first-stage mixed stress, write:

```text
V53A_FIRST_MIXED_STRESS_VERDICT.md
```

It must include:

- exact command;
- 6-dataset ACC/NMI/ARI table;
- red-line table;
- residual schedule table;
- reliability non-collapse table;
- base/agreement/residual contribution table;
- weighted coupling table;
- posterior/readout safety table;
- heterophily stress table;
- explicit stop/continue decision.

## 12. No-Fabrication Status

All V50A/V51A/V52A numbers in this document come from local verdict files and
diagnostics. No V53A result exists. V53A is a preregistered rescue mechanism,
not an implemented or evaluated model.
