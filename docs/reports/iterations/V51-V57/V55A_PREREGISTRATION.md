# v55a_soft_consensus_bounded_residual_anchor Preregistration

This document preregisters the next rescue route after the V54A mixed-stress
STOP verdict. It uses `ccf-idea-optimizer` standard rescue mode: fix the
mechanism and gates before implementation.

No V55A code has been implemented and no V55A experiment has been run.

## 1. Motivation

V54A gave the cleanest safety signal so far:

```text
Consensus-bounded residual preserves ACM/DBLP, restores Squirrel above the
explicit safety floor, keeps posterior/readout gap clean, and passes heterophily
stress.
```

But V54A still stops:

```text
Reliability non-collapse fails because DBLP and Texas remain below the 0.08
effective-mass floor.
```

The failure is not weak-anchor overexposure. V54A fixed that. The remaining
failure is hard-consensus underactivation:

```text
Argmax consensus is too coarse for medium-consensus nodes on DBLP/Texas.
```

Therefore V55A must not raise beta globally or sweep thresholds. It must keep
V54A's bounded residual path while replacing hard argmax consensus with a
continuous, stop-gradient soft consensus score.

## 2. Version Name

```text
v55a_soft_consensus_bounded_residual_anchor
```

Core hypothesis:

```text
Residual anchor availability should be bounded by continuous agreement strength,
not only exact argmax consensus, so medium-consensus graphs can retain enough
anchor mass without reintroducing Squirrel overexposure.
```

## 3. Hard Prohibitions

V55A must not use:

- dataset-specific module, branch, head, loss, assigner, threshold, schedule, or
  weight;
- legacy head as final output;
- adaptive selector or post-processing selector;
- S2CAG/ELSS/KMeans anchor labels as final output;
- labels or test-set metrics in training;
- V50A weight, temperature, rank, filter-step, or refresh sweep;
- V51A/V52A/V53A/V54A threshold, exponent, schedule, beta, or formula sweep;
- dataset-specific beta or soft-power;
- V43B-V49A failed topology loss family as the main mechanism;
- post-hoc selection among APTC, V50A, V51A, V52A, V53A, V54A, V55A,
  embedding KMeans, spectral anchor, or legacy labels;
- signed topology-mask anchor in the first V55A implementation;
- low-reliability geometry fallback in the first V55A implementation.

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

V55A inherits V54A's schedule and base/agreement components, but replaces hard
argmax consensus with continuous soft consensus.

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

Use the same detached components as V54A:

```text
r_base_i = 0.5 * conf_i + 0.5 * local_i_norm
r_agree_i = 0.5 * qa_i_norm + 0.5 * ea_i_norm
```

where:

```text
conf_i       anchor confidence
local_i_norm local anchor consistency
qa_i_norm    posterior-anchor soft agreement
ea_i_norm    embedding-anchor soft agreement
```

### 4.3 Soft-Consensus-Bounded Residual

V55A inherits the fixed V52A/V53A/V54A schedule:

```text
warmup_epochs = 20
ramp_epochs = 40
gamma_t = clamp((epoch + 1 - warmup_epochs) / ramp_epochs, 0, 1)
```

Fixed residual bounds:

```text
beta_min = 0.35
beta_max = 0.70
soft_power = 0.50
```

Node-level soft consensus:

```text
soft_i = clamp(0.5 * qa_i_norm + 0.5 * ea_i_norm, 0, 1)
c_i = soft_i ^ soft_power
beta_i = beta_min + (beta_max - beta_min) * c_i
```

Final reliability:

```text
r_multiplier_i = (1 - gamma_t)
               + gamma_t * (beta_i + (1 - beta_i) * r_agree_i)
r_i = detach(clamp(r_base_i * r_multiplier_i, 0, 1))
```

Endpoint at epoch 80:

```text
r_i = r_base_i * (beta_i + (1 - beta_i) * r_agree_i)
```

This is intentionally different from V54A:

```text
V54A: beta_i is driven by hard argmax consensus.
V55A: beta_i is driven by continuous soft agreement with a fixed concave lift.
```

### 4.4 Loss

V55A keeps the same KL direction:

```text
kl_i = KL(q_refined_i || stopgrad(q_spec_i))
v55a_anchor_loss = sum_i r_i * kl_i / clamp(sum_i r_i, min=N * reliability_floor)
loss += v55a_anchor_weight * v55a_anchor_loss
```

Fixed constants:

```text
v55a_anchor_weight = 0.04
v55a_reliability_floor = 0.10
v55a_reliable_threshold = 0.20
v55a_min_effective_mass = 0.10
v55a_warmup_epochs = 20
v55a_ramp_epochs = 40
v55a_beta_min = 0.35
v55a_beta_max = 0.70
v55a_soft_power = 0.50
```

No other V55A constants may be tuned in the first implementation.

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
embedding_posterior_gap
```

Anchor quality:

```text
v55a_anchor_acc_diagnostic
v55a_anchor_nmi_diagnostic
v55a_anchor_ari_diagnostic
v55a_anchor_entropy
v55a_anchor_confidence
v55a_anchor_cluster_usage_entropy
```

Soft consensus:

```text
v55a_gamma
v55a_beta_min
v55a_beta_max
v55a_soft_power
v55a_soft_consensus_mean
v55a_soft_consensus_p10
v55a_soft_consensus_p50
v55a_soft_consensus_p90
v55a_beta_mean
v55a_beta_p10
v55a_beta_p50
v55a_beta_p90
v55a_residual_multiplier_mean
```

Reliability:

```text
v55a_reliability_mean
v55a_reliability_std
v55a_reliability_p10
v55a_reliability_p50
v55a_reliability_p90
v55a_reliable_node_ratio
v55a_effective_anchor_mass
v55a_base_reliability_mean
v55a_agreement_reliability_mean
v55a_confidence_component_mean
v55a_q_anchor_component_mean
v55a_embed_anchor_component_mean
v55a_local_component_mean
```

Coupling:

```text
v55a_anchor_loss
v55a_weighted_q_anchor_kl
v55a_weighted_q_anchor_agreement
v55a_unweighted_q_anchor_agreement
v55a_embedding_anchor_agreement
v55a_weighted_q_anchor_agreement_epoch_1
v55a_weighted_q_anchor_agreement_epoch_40
v55a_weighted_q_anchor_agreement_epoch_80
v55a_reliability_mean_epoch_1
v55a_reliability_mean_epoch_40
v55a_reliability_mean_epoch_80
```

## 6. Implementation Review Required

Before code implementation, write:

```text
V55A_IMPLEMENTATION_REVIEW.md
```

It must verify:

- V55A reliability is detached;
- beta is bounded by fixed `beta_min=0.35` and `beta_max=0.70`;
- `soft_power=0.50` is fixed and not swept;
- soft consensus uses only detached training-time posterior/embedding/anchor
  agreement;
- V52A/V53A/V54A schedule is inherited unchanged;
- V50A anchor construction is unchanged;
- V50A/V51A/V52A/V53A/V54A losses are disabled in the V55A variant;
- no dataset-specific branch is introduced;
- no low-reliability geometry fallback is added;
- final labels remain the existing unified `q_refined` output.

## 7. Connectivity Run

Only after implementation review and code implementation, run:

```powershell
conda run --no-capture-output -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v55a_soft_consensus_bounded_residual_anchor --datasets "acm" --epochs 1 --device cuda --log-level WARNING
```

Connectivity pass condition:

```text
status=ok
legacy_head_used=false
v55a_enabled=true
v50a_enabled=false
v51a_enabled=false
v52a_enabled=false
v53a_enabled=false
v54a_enabled=false
v55a_anchor_loss finite
v55a_gamma = 0
v55a_beta_min = 0.35
v55a_beta_max = 0.70
v55a_soft_power = 0.50
v55a_soft_consensus_mean finite
v55a_beta_mean finite
v55a_reliability_mean finite
v55a_effective_anchor_mass finite
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
conda run --no-capture-output -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v55a_soft_consensus_bounded_residual_anchor --datasets "acm,dblp,flickr,texas,squirrel,chameleon" --epochs 80 --device cuda --log-level WARNING
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
v55a_enabled=true
no selector / no post-processing selector
```

### 9.2 Soft-Residual Bound Gate

Must pass on 6/6:

```text
v55a_gamma_epoch_1 = 0
v55a_gamma_epoch_40 = 0.5
v55a_gamma_epoch_80 = 1
v55a_beta_min = 0.35
v55a_beta_max = 0.70
v55a_soft_power = 0.50
0.35 <= v55a_beta_mean <= 0.70
0.00 <= v55a_soft_consensus_mean <= 1.00
```

### 9.3 Reliability Non-Collapse Gate

Must pass on at least 4/6:

```text
0.08 <= v55a_reliability_mean <= 0.90
v55a_effective_anchor_mass >= 0.08
```

And at least 3/6:

```text
v55a_reliable_node_ratio >= 0.05
```

Hard fail:

```text
v55a_reliability_mean < 0.03 on ACM, DBLP, Squirrel, or Chameleon
v55a_reliability_mean > 0.97 on any dataset
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
v55a_weighted_q_anchor_agreement_epoch_80 >
v55a_weighted_q_anchor_agreement_epoch_1
```

And:

```text
ACM ACC >= 0.8888
DBLP ACC >= 0.6610
Squirrel ACC >= 0.2800
```

The Squirrel floor remains because V53A passed safety but overexposed a weak
anchor and dropped to 0.2119.

### 9.6 Heterophily Stress Gate

For Texas, Squirrel, and Chameleon, at least 2/3 must pass:

```text
abs(embedding_posterior_gap) <= 0.04
v55a_reliability_mean within [0.08, 0.90]
v55a_effective_anchor_mass >= 0.08
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
- V55A schedule variants;
- V55A reliability formula variants;
- reliability threshold sweep;
- V50A anchor hyperparameter sweep.

## 11. Required Verdict Artifact

After first-stage mixed stress, write:

```text
V55A_FIRST_MIXED_STRESS_VERDICT.md
```

It must include:

- exact command;
- 6-dataset ACC/NMI/ARI table;
- red-line table;
- soft-residual-bound table;
- soft-consensus table;
- reliability non-collapse table;
- weighted coupling table;
- posterior/readout safety table;
- heterophily stress table;
- explicit stop/continue decision.

## 12. No-Fabrication Status

All V50A/V51A/V52A/V53A/V54A numbers in this document come from local verdict
files and diagnostics. No V55A result exists. V55A is a preregistered rescue
mechanism, not an implemented or evaluated model.
