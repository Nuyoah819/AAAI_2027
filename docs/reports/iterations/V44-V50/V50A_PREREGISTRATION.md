# v50a_spectral_compactness_anchor Preregistration

This document preregisters a rescue mechanism after V49A failure. It is a design
document only. No V50A code has been implemented and no V50A experiment has been
run.

## 1. Motivation

V43B-V49A tried to improve the model through frontend topology masks,
frequency/conflict response, hard-band calibration, posterior-guided hard-edge
targets, and orientation/clarity reparameterization. The latest result,
`v49a_reparameterized_topology_transition`, passed safety and usage gates but
failed the primary direction gate:

| Dataset | ACC | Emb Gap | Homo Use | Hetero Use | Hard Use | Direction |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| ACM | 0.6208 | -0.0003 | 0.3515 | 0.3509 | 0.2976 | FAIL |
| DBLP | 0.6571 | 0.0000 | 0.3405 | 0.3433 | 0.3162 | PASS |
| Flickr | 0.3376 | 0.0000 | 0.3383 | 0.3417 | 0.3199 | FAIL |

ACM and Flickr both repeated the V48A wrong-direction pattern:

```text
targeted_homo_delta < 0 and targeted_hetero_delta < 0
```

Therefore V50A stops topology-mask calibration as the primary route.

Baseline code and local reference review instead indicate that strong attributed
graph clustering performance comes from stable low-rank/spectral subspaces:

- S2CAG: graph propagation + randomized SVD / modularity spectral basis + SNEM
  rounding.
- ELSS: homophily-aware filtering + PageRank anchor Nyström + explicit
  low-rank subspace.
- CoCo-style work: compact low-rank embeddings plus cross-view consistency.

## 2. Version Name

```text
v50a_spectral_compactness_anchor
```

Core hypothesis:

```text
If the current end-to-end model lacks a stable low-rank clustering basis, then a
unified spectral compactness anchor can improve posterior/embedding alignment
without adding another topology-mask loss.
```

## 3. Hard Prohibitions

V50A must not use:

- dataset-specific module, branch, head, loss, assigner, threshold, or weight;
- legacy head as final output;
- adaptive selector or post-processing selector;
- S2CAG/ELSS KMeans output as the reported final label;
- labels or test-set-driven correction;
- V49A temperature/init sweep;
- V43B-V49A failed loss family as the main mechanism;
- post-hoc selection among APTC, KMeans, spectral, or legacy heads.

The following must remain disabled:

```text
v43b_conflict_margin_weight = 0.0
v43b_band_conflict_weight = 0.0
v43b_highpass_energy_weight = 0.0
ideal_signed_embedding_weight = 0.0
ideal_band_resolution_weight = 0.0
ideal_highpass_energy_weight = 0.0
v44_topology_band_resolution_weight = 0.0
v44_conflict_highpass_corr_weight = 0.0
v44b_pre_hp_corr_weight = 0.0
v45a_edge_freq_weight = 0.0
v45a_band_guard_weight = 0.0
v46a_band_cal_weight = 0.0
v46a_balance_weight = 0.0
v46a_spread_weight = 0.0
v47a_resolution_weight = 0.0
v47a_usage_guard_weight = 0.0
v48a_enabled = false
v49a_enabled = false
```

## 4. Allowed Mechanism

### 4.1 Spectral Compactness Anchor

Construct a unified low-rank anchor from graph-smoothed input features:

```text
X_norm -> H_spec = graph_filter(A, X_norm)
H_spec -> U_spec = top-k_or_top-r low-rank basis
U_spec -> q_spec = soft cluster distribution
```

The first implementation may use a stop-gradient spectral anchor. The anchor is
not the final prediction. It is a training signal and diagnostic object.

Allowed first implementation shape:

```text
spectral_anchor_loss = KL(q_refined || stopgrad(q_spec))
```

or symmetric KL if preregistered in implementation review. The first
implementation must choose exactly one form before running.

### 4.2 Unified Constants

Use one configuration for every dataset:

```text
v50a_enabled = true
v50a_anchor_weight = TBD before implementation
v50a_filter_steps = TBD before implementation
v50a_anchor_rank_multiplier = TBD before implementation
v50a_anchor_temperature = TBD before implementation
```

Before implementation, these TBD constants must be fixed in an implementation
review. No sweep is allowed.

### 4.3 End-to-End Boundary

The anchor may be stop-gradient, but the loss must update the trainable
end-to-end model:

```text
loss -> q_refined/q_main -> embedding/frontend parameters
```

The final reported label must remain the unified model output specified by the
variant, not an external S2CAG/ELSS/KMeans selector.

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

These may use labels only for diagnostics after training, exactly like existing
diagnostic metric reporting. They must not affect training.

Training coupling:

```text
v50a_anchor_loss
v50a_q_anchor_kl
v50a_q_anchor_agreement
v50a_embedding_anchor_agreement
v50a_anchor_effective_weight
```

Safety:

```text
final_acc
nmi
ari
embedding_kmeans_acc
embedding_kmeans_nmi
embedding_kmeans_ari
embedding_posterior_gap
```

Legacy/context diagnostics:

```text
v49a_direction diagnostics may be omitted unless V49A is enabled, which it
should not be in V50A.
```

## 6. First-Stage Experiment

Only after implementation review and 1-epoch connectivity:

```text
datasets = acm,dblp,flickr
epochs = 80
seed = 42
device = cuda
```

Command template:

```powershell
conda run -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v50a_spectral_compactness_anchor --datasets "acm,dblp,flickr" --epochs 80 --device cuda --log-level WARNING
```

No second-batch smoke is allowed unless all first-stage gates pass.

## 7. First-Stage Gates

### 7.1 Red-Line Gate

Must pass:

```text
legacy_head_used=false
v43b/v44/v44b/v45a/v46a/v47a_enabled=false
v48a_enabled=false
v49a_enabled=false
v50a_enabled=true
no selector / no post-processing selector
```

### 7.2 Anchor Non-Degeneracy Gate

Must pass on 3/3:

```text
v50a_anchor_cluster_usage_entropy >= 0.60
v50a_anchor_entropy finite
v50a_q_anchor_kl finite
```

### 7.3 Coupling Gate

Must pass on at least 2/3:

```text
v50a_q_anchor_agreement improves from epoch early snapshot to final snapshot
```

If epoch snapshots are not implemented, the first implementation must instead
use a diagnostic that can distinguish "anchor present but ignored" from "anchor
used by posterior".

### 7.4 Posterior/Readout Safety Gate

Must pass:

```text
abs(embedding_posterior_gap) <= 0.02 on at least 2/3
abs(embedding_posterior_gap) <= 0.04 on 3/3
```

### 7.5 Performance Context Gate

Record but do not use alone to expand:

```text
ACM should recover above V49A ACC 0.6208
DBLP should not fall below V49A ACC 0.6571 by more than 0.01
Flickr should recover above V49A ACC 0.3376
```

Performance cannot authorize expansion unless red-line, anchor, coupling, and
safety gates pass.

## 8. Stop Conditions

Stop after first-stage smoke if any occurs:

- red-line violation;
- anchor collapse;
- anchor-present-but-ignored diagnostics;
- posterior/readout safety failure;
- ACM and Flickr both fail to recover above V49A context ACC;
- any non-finite loss or diagnostics.

Do not run:

- second-batch smoke;
- full 9-dataset smoke;
- 260-epoch full run;
- anchor weight sweep;
- filter step sweep;
- rank sweep;
- temperature sweep.

## 9. Implementation Review Required

Before code implementation, write:

```text
V50A_IMPLEMENTATION_REVIEW.md
```

It must fix:

- exact anchor construction;
- exact rank rule;
- exact filter step count;
- exact anchor temperature;
- exact loss direction;
- whether anchor is computed once before training or refreshed;
- how epoch-level coupling is diagnosed.

## 10. No-Fabrication Status

All V49A values come from `V49A_FIRST_SMOKE_VERDICT.md`. S2CAG values cited in
the route decision come from local batch result files. V50A has not been
implemented or run. All V50A results are `TBD`.
