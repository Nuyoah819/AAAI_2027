# v56a_hybrid_consensus_floor_residual_anchor Preregistration

This document preregisters the next rescue route after the V55A mixed-stress
STOP verdict. It uses `ccf-idea-optimizer` rescue mode: fix the mechanism and
gates before implementation.

No V56A code has been implemented and no V56A experiment has been run.

## 1. Motivation

V54A fixed Squirrel overexposure but underactivated DBLP/Texas because hard
argmax consensus was coarse. V55A replaced hard consensus with pure soft
consensus, but the soft agreement values were too small and beta dropped on
DBLP/Texas.

Therefore the next route must not replace hard consensus. It should preserve
the hard-consensus safety floor and add a bounded soft compensation only where
continuous evidence exceeds the hard floor.

## 2. Version Name

```text
v56a_hybrid_consensus_floor_residual_anchor
```

Core hypothesis:

```text
Hard argmax consensus is useful as a safety floor, while soft agreement is
useful only as a residual compensation signal for medium-evidence nodes. A
hybrid floor-plus-compensation beta can recover DBLP/Texas anchor mass without
reintroducing Squirrel overexposure.
```

## 3. Hard Prohibitions

V56A must not use:

- dataset-specific module, branch, head, loss, selector, threshold, schedule, or
  weight;
- legacy head as final output;
- adaptive selector or post-processing selector;
- S2CAG/ELSS/KMeans anchor labels as final output;
- labels or test-set metrics in training;
- V50A weight, temperature, rank, filter-step, or refresh sweep;
- V54A beta-bound sweep;
- V55A soft-power sweep;
- V56A compensation strength sweep;
- dataset-specific compensation;
- V43B-V49A failed topology loss family as the main mechanism;
- post-hoc selection among APTC, V50A, V51A, V52A, V53A, V54A, V55A, V56A,
  embedding KMeans, spectral anchor, or legacy labels;
- signed topology-mask anchor in the first V56A implementation;
- low-reliability geometry fallback in the first V56A implementation.

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

V56A inherits V54A/V55A's schedule, base reliability, agreement reliability,
KL direction, and frozen spectral anchor. It changes only node-level beta.

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

Detached components:

```text
conf_i       anchor confidence
local_i_norm local anchor consistency
qa_i_norm    posterior-anchor soft agreement
ea_i_norm    embedding-anchor soft agreement
```

Base and agreement:

```text
r_base_i = 0.5 * conf_i + 0.5 * local_i_norm
r_agree_i = 0.5 * qa_i_norm + 0.5 * ea_i_norm
```

Hard consensus:

```text
h_q_i = 1[argmax(q_refined_i) == argmax(q_spec_i)]
h_e_i = 1[argmax(q_embed_i) == argmax(q_spec_i)]
h_i = 0.5 * h_q_i + 0.5 * h_e_i
```

Soft consensus:

```text
s_i = clamp(0.5 * qa_i_norm + 0.5 * ea_i_norm, 0, 1)
c_i = sqrt(s_i)
```

### 4.3 Hybrid Consensus Floor

Fixed residual bounds:

```text
beta_min = 0.35
beta_max = 0.70
soft_power = 0.50
hybrid_compensation = 0.50
```

Hybrid consensus:

```text
hybrid_i = clamp(
    h_i + hybrid_compensation * relu(c_i - h_i),
    0,
    1
)
beta_i = beta_min + (beta_max - beta_min) * hybrid_i
```

Interpretation:

```text
If hard consensus is high, V56A keeps the V54A safety floor.
If hard consensus is low but soft evidence is meaningfully higher, V56A adds a
fixed half-strength compensation.
If both hard and soft evidence are low, beta remains low.
```

This differs from V54A and V55A:

```text
V54A: beta uses hard consensus only.
V55A: beta uses soft consensus only.
V56A: beta uses hard consensus as floor plus fixed soft compensation.
```

### 4.4 Schedule

Inherited fixed schedule:

```text
warmup_epochs = 20
ramp_epochs = 40
gamma_t = clamp((epoch + 1 - warmup_epochs) / ramp_epochs, 0, 1)
```

### 4.5 Reliability And Loss

Reliability:

```text
r_multiplier_i = (1 - gamma_t)
               + gamma_t * (beta_i + (1 - beta_i) * r_agree_i)
r_i = detach(clamp(r_base_i * r_multiplier_i, 0, 1))
```

Loss:

```text
kl_i = KL(q_refined_i || stopgrad(q_spec_i))
v56a_anchor_loss = sum_i r_i * kl_i / clamp(sum_i r_i, min=N * reliability_floor)
loss += v56a_anchor_weight * v56a_anchor_loss
```

Fixed constants:

```text
v56a_anchor_weight = 0.04
v56a_reliability_floor = 0.10
v56a_reliable_threshold = 0.20
v56a_min_effective_mass = 0.10
v56a_warmup_epochs = 20
v56a_ramp_epochs = 40
v56a_beta_min = 0.35
v56a_beta_max = 0.70
v56a_soft_power = 0.50
v56a_hybrid_compensation = 0.50
```

No other V56A constants may be tuned in the first implementation.

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
embedding_posterior_gap
```

Hybrid consensus:

```text
v56a_gamma
v56a_beta_min
v56a_beta_max
v56a_soft_power
v56a_hybrid_compensation
v56a_hard_consensus_mean
v56a_soft_consensus_mean
v56a_lifted_soft_consensus_mean
v56a_compensation_mean
v56a_compensation_active_ratio
v56a_hybrid_consensus_mean
v56a_beta_mean
v56a_beta_p10
v56a_beta_p50
v56a_beta_p90
v56a_residual_multiplier_mean
```

Reliability:

```text
v56a_reliability_mean
v56a_reliability_std
v56a_reliability_p10
v56a_reliability_p50
v56a_reliability_p90
v56a_reliable_node_ratio
v56a_effective_anchor_mass
v56a_base_reliability_mean
v56a_agreement_reliability_mean
v56a_confidence_component_mean
v56a_q_anchor_component_mean
v56a_embed_anchor_component_mean
v56a_local_component_mean
```

Coupling and anchor:

```text
v56a_anchor_loss
v56a_weighted_q_anchor_kl
v56a_weighted_q_anchor_agreement
v56a_unweighted_q_anchor_agreement
v56a_embedding_anchor_agreement
v56a_anchor_entropy
v56a_anchor_confidence
v56a_anchor_cluster_usage_entropy
v56a_anchor_acc_diagnostic
v56a_anchor_nmi_diagnostic
v56a_anchor_ari_diagnostic
v56a_weighted_q_anchor_agreement_epoch_1
v56a_weighted_q_anchor_agreement_epoch_40
v56a_weighted_q_anchor_agreement_epoch_80
v56a_reliability_mean_epoch_1
v56a_reliability_mean_epoch_40
v56a_reliability_mean_epoch_80
```

## 6. Implementation Review Required

Before code implementation, write:

```text
V56A_IMPLEMENTATION_REVIEW.md
```

It must verify:

- V56A reliability is detached;
- beta is bounded by fixed `beta_min=0.35` and `beta_max=0.70`;
- `soft_power=0.50` is fixed and not swept;
- `hybrid_compensation=0.50` is fixed and not swept;
- hard consensus is a floor, not a selector;
- soft compensation is positive-only through `relu(c_i - h_i)`;
- V52A/V53A/V54A/V55A schedule is inherited unchanged;
- V50A anchor construction is unchanged;
- V50A/V51A/V52A/V53A/V54A/V55A losses are disabled in the V56A variant;
- no dataset-specific branch is introduced;
- no low-reliability geometry fallback is added;
- final labels remain the existing unified `q_refined` output.

## 7. Connectivity Run

Only after implementation review and code implementation, run:

```powershell
conda run --no-capture-output -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v56a_hybrid_consensus_floor_residual_anchor --datasets "acm" --epochs 1 --device cuda --log-level WARNING
```

Connectivity pass condition:

```text
status=ok
legacy_head_used=false
v56a_enabled=true
v50a_enabled=false
v51a_enabled=false
v52a_enabled=false
v53a_enabled=false
v54a_enabled=false
v55a_enabled=false
v56a_anchor_loss finite
v56a_gamma = 0
v56a_beta_min = 0.35
v56a_beta_max = 0.70
v56a_soft_power = 0.50
v56a_hybrid_compensation = 0.50
v56a_hard_consensus_mean finite
v56a_soft_consensus_mean finite
v56a_compensation_mean finite
v56a_beta_mean finite and within [0.35, 0.70]
v56a_reliability_mean finite
v56a_effective_anchor_mass finite
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
conda run --no-capture-output -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v56a_hybrid_consensus_floor_residual_anchor --datasets "acm,dblp,flickr,texas,squirrel,chameleon" --epochs 80 --device cuda --log-level WARNING
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
v56a_enabled=true
no selector / no post-processing selector
```

### 9.2 Hybrid Residual Bound Gate

Must pass on 6/6:

```text
v56a_gamma_epoch_1 = 0
v56a_gamma_epoch_40 = 0.5
v56a_gamma_epoch_80 = 1
v56a_beta_min = 0.35
v56a_beta_max = 0.70
v56a_soft_power = 0.50
v56a_hybrid_compensation = 0.50
0.35 <= v56a_beta_mean <= 0.70
v56a_compensation_mean >= 0
0 <= v56a_compensation_active_ratio <= 1
```

### 9.3 Reliability Non-Collapse Gate

Must pass on at least 4/6:

```text
0.08 <= v56a_reliability_mean <= 0.90
v56a_effective_anchor_mass >= 0.08
```

And at least 3/6:

```text
v56a_reliable_node_ratio >= 0.05
```

Hard fail:

```text
v56a_reliability_mean < 0.03 on ACM, DBLP, Squirrel, or Chameleon
v56a_reliability_mean > 0.97 on any dataset
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
v56a_weighted_q_anchor_agreement_epoch_80 >
v56a_weighted_q_anchor_agreement_epoch_1
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
v56a_reliability_mean within [0.08, 0.90]
v56a_effective_anchor_mass >= 0.08
```

Additionally:

```text
Squirrel ACC >= 0.2800
```

## 10. Stop Conditions

Stop after first-stage mixed stress if any occurs:

- red-line violation;
- non-finite loss or diagnostic;
- residual bound mismatch;
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
- beta-bound sweep;
- soft-power sweep;
- hybrid-compensation sweep;
- V56A schedule variants;
- V56A reliability formula variants;
- reliability threshold sweep;
- V50A anchor hyperparameter sweep.

## 11. Required Verdict Artifact

After first-stage mixed stress, write:

```text
V56A_FIRST_MIXED_STRESS_VERDICT.md
```

It must include:

- exact command;
- 6-dataset ACC/NMI/ARI table;
- red-line table;
- hybrid-residual-bound table;
- hard/soft/compensation consensus table;
- reliability non-collapse table;
- weighted coupling table;
- posterior/readout safety table;
- heterophily stress table;
- explicit stop/continue decision.

## 12. No-Fabrication Status

All V54A/V55A numbers in this document come from local verdict files and
diagnostics. No V56A result exists. V56A is a preregistered rescue mechanism,
not an implemented or evaluated model.
