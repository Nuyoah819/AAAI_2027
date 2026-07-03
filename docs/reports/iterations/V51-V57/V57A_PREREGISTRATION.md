# v57a_mass_floor_normalized_residual_anchor Preregistration

This document preregisters the next rescue route after the V56A mixed-stress
STOP verdict. It uses `ccf-idea-optimizer` rescue mode: fix the mechanism and
gates before implementation.

No V57A code has been implemented and no V57A experiment has been run.

## 1. Motivation

V56A preserved the safest bounded-residual behavior so far:

```text
ACM/DBLP/Squirrel floors pass, posterior/readout safety passes, anchor
usefulness passes, and heterophily stress passes.
```

But V56A still stops:

```text
Reliability non-collapse fails because DBLP and Texas remain below the 0.08
effective-mass floor.
```

The V54A-V56A chain has now tested hard consensus, soft consensus, and a
hybrid floor-plus-compensation beta. The remaining failure is no longer simply
which consensus signal should set beta. The remaining failure is:

```text
Raw reliability mass is not operationally large enough on medium-consensus
datasets even when performance and safety are acceptable.
```

Therefore V57A must not increase beta or sweep compensation. It should keep the
V56A raw reliability ranking and test whether a detached, fixed mass-floor
normalization can make that reliability usable without changing dataset routing
or final labels.

## 2. Version Name

```text
v57a_mass_floor_normalized_residual_anchor
```

Core hypothesis:

```text
If V56A's raw reliability ranking is safe but under-massed, a bounded,
dataset-agnostic mass normalization can lift effective anchor mass on DBLP/Texas
without reintroducing Squirrel overexposure.
```

## 3. Hard Prohibitions

V57A must not use:

- dataset-specific module, branch, head, loss, selector, threshold, schedule, or
  weight;
- legacy head as final output;
- adaptive selector or post-processing selector;
- S2CAG/ELSS/KMeans anchor labels as final output;
- labels or test-set metrics in training;
- V50A weight, temperature, rank, filter-step, or refresh sweep;
- V54A/V55A/V56A beta-bound, soft-power, or hybrid-compensation sweep;
- dataset-specific mass target or mass scale;
- V43B-V49A failed topology loss family as the main mechanism;
- post-hoc selection among APTC, V50A, V51A, V52A, V53A, V54A, V55A, V56A,
  V57A, embedding KMeans, spectral anchor, or legacy labels;
- signed topology-mask anchor in the first V57A implementation;
- low-reliability geometry fallback in the first V57A implementation.

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

V57A inherits V56A's schedule, base reliability, agreement reliability, hybrid
beta, KL direction, and frozen spectral anchor. It changes only how detached
raw reliability is normalized before weighting the KL.

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

### 4.2 V56A Raw Reliability

Use the V56A raw mechanism unchanged:

```text
r_base_i = 0.5 * conf_i + 0.5 * local_i_norm
r_agree_i = 0.5 * qa_i_norm + 0.5 * ea_i_norm

h_i = 0.5 * 1[argmax(q_refined_i)=argmax(q_spec_i)]
    + 0.5 * 1[argmax(q_embed_i)=argmax(q_spec_i)]

s_i = clamp(0.5 * qa_i_norm + 0.5 * ea_i_norm, 0, 1)
c_i = sqrt(s_i)
hybrid_i = clamp(h_i + 0.50 * relu(c_i - h_i), 0, 1)
beta_i = 0.35 + (0.70 - 0.35) * hybrid_i

raw_multiplier_i = (1 - gamma_t)
                 + gamma_t * (beta_i + (1 - beta_i) * r_agree_i)
raw_r_i = detach(clamp(r_base_i * raw_multiplier_i, 0, 1))
```

### 4.3 Mass-Floor Normalization

Fixed constants:

```text
target_mass = 0.08
max_mass_scale = 1.50
max_reliability_cap = 0.90
```

Detached normalization:

```text
raw_mass = mean(raw_r_i)
mass_scale = clamp(target_mass / clamp(raw_mass, 1e-8), 1.0, max_mass_scale)
r_i = detach(clamp(raw_r_i * mass_scale, 0, max_reliability_cap))
```

Interpretation:

```text
V57A does not change which nodes are reliable according to V56A. It only tests
whether the same reliability ranking can be made operationally strong enough
for the anchor loss under a fixed mass floor.
```

This differs from V56A:

```text
V56A: raw reliability directly weights KL.
V57A: raw reliability is scaled by a fixed detached mass floor, capped for safety.
```

### 4.4 Schedule

Inherited fixed schedule:

```text
warmup_epochs = 20
ramp_epochs = 40
gamma_t = clamp((epoch + 1 - warmup_epochs) / ramp_epochs, 0, 1)
```

### 4.5 Loss

V57A keeps the same KL direction:

```text
kl_i = KL(q_refined_i || stopgrad(q_spec_i))
v57a_anchor_loss = sum_i r_i * kl_i / clamp(sum_i r_i, min=N * reliability_floor)
loss += v57a_anchor_weight * v57a_anchor_loss
```

Fixed constants:

```text
v57a_anchor_weight = 0.04
v57a_reliability_floor = 0.10
v57a_reliable_threshold = 0.20
v57a_min_effective_mass = 0.10
v57a_warmup_epochs = 20
v57a_ramp_epochs = 40
v57a_beta_min = 0.35
v57a_beta_max = 0.70
v57a_soft_power = 0.50
v57a_hybrid_compensation = 0.50
v57a_target_mass = 0.08
v57a_max_mass_scale = 1.50
v57a_max_reliability_cap = 0.90
```

No other V57A constants may be tuned in the first implementation.

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
v54a_enabled
v55a_enabled
v56a_enabled
v57a_enabled
embedding_posterior_gap
```

Hybrid and mass normalization:

```text
v57a_gamma
v57a_beta_min
v57a_beta_max
v57a_soft_power
v57a_hybrid_compensation
v57a_target_mass
v57a_max_mass_scale
v57a_max_reliability_cap
v57a_hard_consensus_mean
v57a_soft_consensus_mean
v57a_lifted_soft_consensus_mean
v57a_compensation_mean
v57a_compensation_active_ratio
v57a_hybrid_consensus_mean
v57a_beta_mean
v57a_raw_reliability_mean
v57a_mass_scale
v57a_scaled_reliability_mean
v57a_residual_multiplier_mean
```

Reliability:

```text
v57a_reliability_mean
v57a_reliability_std
v57a_reliability_p10
v57a_reliability_p50
v57a_reliability_p90
v57a_reliable_node_ratio
v57a_effective_anchor_mass
v57a_base_reliability_mean
v57a_agreement_reliability_mean
v57a_confidence_component_mean
v57a_q_anchor_component_mean
v57a_embed_anchor_component_mean
v57a_local_component_mean
```

Coupling and anchor:

```text
v57a_anchor_loss
v57a_weighted_q_anchor_kl
v57a_weighted_q_anchor_agreement
v57a_unweighted_q_anchor_agreement
v57a_embedding_anchor_agreement
v57a_anchor_entropy
v57a_anchor_confidence
v57a_anchor_cluster_usage_entropy
v57a_anchor_acc_diagnostic
v57a_anchor_nmi_diagnostic
v57a_anchor_ari_diagnostic
v57a_weighted_q_anchor_agreement_epoch_1
v57a_weighted_q_anchor_agreement_epoch_40
v57a_weighted_q_anchor_agreement_epoch_80
v57a_reliability_mean_epoch_1
v57a_reliability_mean_epoch_40
v57a_reliability_mean_epoch_80
v57a_raw_reliability_mean_epoch_80
v57a_mass_scale_epoch_80
```

## 6. Implementation Review Required

Before code implementation, write:

```text
V57A_IMPLEMENTATION_REVIEW.md
```

It must verify:

- V57A reliability is detached;
- V56A raw reliability formula is inherited unchanged;
- mass scaling is detached and dataset-agnostic;
- `target_mass=0.08`, `max_mass_scale=1.50`, and
  `max_reliability_cap=0.90` are fixed and not swept;
- V50A anchor construction is unchanged;
- V50A/V51A/V52A/V53A/V54A/V55A/V56A losses are disabled in the V57A variant;
- no dataset-specific branch is introduced;
- no low-reliability geometry fallback is added;
- final labels remain the existing unified `q_refined` output.

## 7. Connectivity Run

Only after implementation review and code implementation, run:

```powershell
conda run --no-capture-output -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v57a_mass_floor_normalized_residual_anchor --datasets "acm" --epochs 1 --device cuda --log-level WARNING
```

Connectivity pass condition:

```text
status=ok
legacy_head_used=false
v57a_enabled=true
v50a_enabled=false
v51a_enabled=false
v52a_enabled=false
v53a_enabled=false
v54a_enabled=false
v55a_enabled=false
v56a_enabled=false
v57a_anchor_loss finite
v57a_gamma = 0
v57a_target_mass = 0.08
v57a_max_mass_scale = 1.50
v57a_max_reliability_cap = 0.90
v57a_raw_reliability_mean finite
v57a_mass_scale finite and within [1.0, 1.50]
v57a_reliability_mean finite
v57a_effective_anchor_mass finite
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
conda run --no-capture-output -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v57a_mass_floor_normalized_residual_anchor --datasets "acm,dblp,flickr,texas,squirrel,chameleon" --epochs 80 --device cuda --log-level WARNING
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
v53a_enabled=false
v54a_enabled=false
v55a_enabled=false
v56a_enabled=false
v57a_enabled=true
no selector / no post-processing selector
```

### 9.2 Mass-Normalization Gate

Must pass on 6/6:

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

### 9.3 Reliability Non-Collapse Gate

Must pass on at least 4/6:

```text
0.08 <= v57a_reliability_mean <= 0.90
v57a_effective_anchor_mass >= 0.08
```

And at least 3/6:

```text
v57a_reliable_node_ratio >= 0.05
```

Hard fail:

```text
v57a_reliability_mean < 0.03 on ACM, DBLP, Squirrel, or Chameleon
v57a_reliability_mean > 0.97 on any dataset
```

Flickr weak-anchor exception remains allowed only if anchor evidence is weak.

### 9.4 Safety Gate

Must pass:

```text
abs(embedding_posterior_gap) <= 0.04 on 6/6
no dataset abs(embedding_posterior_gap) > 0.08
```

### 9.5 Anchor Usefulness Gate

Must pass on at least 4/6:

```text
v57a_weighted_q_anchor_agreement_epoch_80 >
v57a_weighted_q_anchor_agreement_epoch_1
```

And:

```text
ACM ACC >= 0.8888
DBLP ACC >= 0.6610
Squirrel ACC >= 0.2800
```

### 9.6 Heterophily Stress Gate

For Texas, Squirrel, and Chameleon, at least 2/3 must pass:

```text
abs(embedding_posterior_gap) <= 0.04
v57a_reliability_mean within [0.08, 0.90]
v57a_effective_anchor_mass >= 0.08
```

Additionally:

```text
Squirrel ACC >= 0.2800
```

## 10. Stop Conditions

Stop after first-stage mixed stress if any occurs:

- red-line violation;
- non-finite loss or diagnostic;
- mass-normalization mismatch;
- reliability collapses to near-zero on ACM, DBLP, Squirrel, or Chameleon;
- reliability becomes all-one on any dataset;
- Squirrel ACC < 0.2800;
- Squirrel has a hard posterior/readout safety failure;
- ACM drops below 0.8888;
- DBLP drops below 0.6610;
- heterophily stress gate fails;
- reliability non-collapse gate fails.

Do not run:

- full 9-dataset smoke;
- 260-epoch full run;
- seed sweep;
- target-mass sweep;
- max-mass-scale sweep;
- reliability-cap sweep;
- beta-bound sweep;
- soft-power sweep;
- hybrid-compensation sweep;
- schedule variant;
- reliability formula variant;
- threshold sweep;
- V50A anchor hyperparameter sweep.

## 11. Required Verdict Artifact

After first-stage mixed stress, write:

```text
V57A_FIRST_MIXED_STRESS_VERDICT.md
```

It must include:

- exact command;
- 6-dataset ACC/NMI/ARI table;
- red-line table;
- mass-normalization table;
- raw-vs-scaled reliability table;
- reliability non-collapse table;
- weighted coupling table;
- posterior/readout safety table;
- heterophily stress table;
- explicit stop/continue decision.

## 12. No-Fabrication Status

All V54A/V55A/V56A numbers in this document come from local verdict files and
diagnostics. No V57A result exists. V57A is a preregistered rescue mechanism,
not an implemented or evaluated model.
