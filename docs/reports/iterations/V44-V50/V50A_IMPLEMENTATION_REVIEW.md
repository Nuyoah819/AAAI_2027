# V50A Implementation Review

This file fixes all implementation choices before running
`v50a_spectral_compactness_anchor`. It follows the V50 preregistration and keeps
V50A as a minimal rescue test, not a sweep.

## 1. Implementation Decision

Implement Candidate A:

```text
stop-gradient spectral teacher for posterior alignment
```

The anchor is used only as a training signal and diagnostic object. It is not a
final head, selector, replacement output, or dataset-specific branch.

## 2. Fixed Anchor Construction

Use the same pipeline for every dataset:

```text
X_dense = existing preprocessed input features
A_filter = row-normalized graph adjacency with self-loops
H0 = row-l2-normalize(X_dense)
H1 = A_filter @ H0
H2 = A_filter @ H1
U_spec = TruncatedSVD(H2, rank=K)
Z_spec = row-l2-normalize(U_spec)
spec_labels, spec_centers = KMeans(Z_spec, K)
q_spec = softmax(-||Z_spec - spec_centers||^2 / temperature)
```

`K` is the dataset's number of clusters. The number of clusters is already part
of the benchmark metadata and is used by the existing clustering protocol.

## 3. Fixed Constants

These constants are fixed before implementation:

```text
v50a_enabled = true
v50a_anchor_weight = 0.04
v50a_filter_steps = 2
v50a_anchor_rank_multiplier = 1.0
v50a_anchor_temperature = 0.35
v50a_anchor_refresh = false
```

Rationale:

- `filter_steps = 2` follows the current low-pass depth and avoids a low-pass
  depth sweep.
- `rank = K` is the smallest interpretable clustering subspace; it avoids a
  2K/3K rank search.
- `temperature = 0.35` is softer than the APTC prototype temperature and avoids
  forcing brittle hard pseudo labels.
- `weight = 0.04` is deliberately below the main clustering and transport
  weights, so the anchor can guide without replacing the learned pipeline.

## 4. Fixed Loss Direction

Use exactly:

```text
v50a_anchor_loss = KL(q_refined || stopgrad(q_spec))
```

In PyTorch terms:

```text
F.kl_div(q_refined.log(), q_spec.detach(), reduction="batchmean")
```

Do not use symmetric KL in V50A. Do not align `q_flow`, `q`, or a new final head
in the first implementation.

## 5. Refresh Policy

The spectral anchor is computed once before training, immediately after feature
preprocessing and candidate-edge construction, then stored as a model buffer.

No epoch refresh is allowed in V50A.

## 6. Coupling Diagnostics

Log these every epoch through the existing diagnostics path:

```text
v50a_enabled
v50a_anchor_loss
v50a_q_anchor_kl
v50a_q_anchor_agreement
v50a_anchor_entropy
v50a_anchor_confidence
v50a_anchor_cluster_usage_entropy
v50a_anchor_effective_weight
```

The existing epoch snapshots at 1/40/80 must include:

```text
v50a_q_anchor_agreement_epoch_1
v50a_q_anchor_agreement_epoch_40
v50a_q_anchor_agreement_epoch_80
v50a_q_anchor_kl_epoch_1
v50a_q_anchor_kl_epoch_40
v50a_q_anchor_kl_epoch_80
```

This is the preregistered way to distinguish "anchor present but ignored" from
"posterior moves toward the anchor" without running a sweep.

## 7. Final Output Boundary

Final labels remain whatever the existing unified variant output protocol
specifies. V50A must not report `q_spec`, spectral KMeans labels, S2CAG labels,
ELSS labels, or a selector among them as final results.

For V50A the variant keeps:

```text
final_label_mode = aptc
postproc_subspace_margin = 1.0
```

This preserves the existing unified run behavior while ensuring the new spectral
object is not a new reported head.

## 8. Red-Line Controls

The V50A variant must explicitly set the failed loss families to inactive:

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

## 9. First Connectivity Run

Before any 80-epoch smoke, run:

```powershell
conda run -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v50a_spectral_compactness_anchor --datasets "acm" --epochs 1 --device cuda --log-level WARNING
```

Pass condition:

```text
status=ok
v50a_enabled=true
v50a_anchor_loss finite
v50a_q_anchor_kl finite
legacy_head_used=false
v48a_enabled=false
v49a_enabled=false
```

Only then allow the preregistered first-stage smoke on ACM/DBLP/Flickr.

## 10. No-Result Status

No V50A result exists at the time this review is written. Any later ACC/NMI/ARI
must come from a logged run and must not be used to revise the fixed constants
above.
