# V49A Implementation Review

Target variant:

```text
v49a_reparameterized_topology_transition
```

This review checks whether V49A can be implemented from the preregistered route
without reviving failed loss families or introducing dataset-specific logic. No
experiment result is reported in this file.

## 1. Current Boundary

V48A established:

```text
topology masks move, but movement is not reliably semantically aligned.
```

Therefore V49A must change the topology transition geometry. It must not add
another hard-edge teacher, CE target, margin loss, scalar band penalty, or
frequency-response pressure.

## 2. Code-Level Entry Points

Current frontend path:

```text
edge_features/evidences
  -> AdaptiveEdgeConfidence
  -> score, alpha, edge_logit
  -> DifferentiableTopologyContraction(score)
  -> homo, hetero, hard, low_threshold, high_threshold
```

Downstream code consumes:

```text
score
homo / hetero / hard
low_threshold / high_threshold
q_refined
```

This means V49A can be introduced at a single point in `_frontend_pass` after
`AdaptiveEdgeConfidence` and before support weights, diffusion, high-pass, APTC,
and losses.

## 3. Minimal Mechanism Decision

Use one unified, dataset-agnostic parameterization:

```text
orient = sigmoid(edge_logit / tau_orient)
clear  = sigmoid(abs(edge_logit) / tau_clear)

homo   = clear * orient
hetero = clear * (1 - orient)
hard   = 1 - clear
```

Rationale:

- `edge_logit` is already the shared edge evidence coordinate.
- `orient` uses the sign/direction of the same evidence.
- `clear` uses evidence magnitude, separating resolved-vs-hard from
  homo-vs-hetero orientation.
- No labels, dataset identifiers, selectors, or external target losses are
  introduced.
- No extra tunable center/margin is added in the first implementation.

This is a conservative first implementation of the preregistered
orientation/clarity simplex. It is smaller than adding a new edge MLP, but still
changes the geometry away from the ordered low/high threshold band.

## 4. Important Confound To Disable

The old threshold geometry still appears in two baseline losses:

```text
threshold_reg_weight
edge_quantile_anchor_weight
```

If these stay active, they can push `score`, low threshold, and high threshold
even though V49A masks no longer come from the ordered-threshold contraction.
That would make the first V49A result hard to interpret.

Therefore the V49A runner variant should explicitly set:

```text
threshold_reg_weight = 0.0
edge_quantile_anchor_weight = 0.0
```

The old thresholds may remain in diagnostics for compatibility, but they must
not contribute to the V49A objective.

## 5. Loss Boundary

V49A should add no new loss term.

The following must remain zero in the runner variant:

```text
v43b_conflict_margin_weight
v43b_band_conflict_weight
v43b_highpass_energy_weight
ideal_signed_embedding_weight
ideal_band_resolution_weight
ideal_highpass_energy_weight
v44_topology_band_resolution_weight
v44_conflict_highpass_corr_weight
v44b_pre_hp_corr_weight
v45a_edge_freq_weight
v45a_band_guard_weight
v46a_band_cal_weight
v46a_balance_weight
v46a_spread_weight
v47a_resolution_weight
v47a_usage_guard_weight
partition_spread_weight
freq_separation_weight
freq_ortho_weight
```

V47A/V48A posterior-guided target groups may be computed for diagnostics only.

## 6. Diagnostics Needed

Add V49A diagnostics:

```text
v49a_enabled
v49a_homo_usage
v49a_hetero_usage
v49a_hard_usage
v49a_band_mass
v49a_usage_entropy
v49a_clear_mean
v49a_clear_std
v49a_orient_mean
v49a_orient_std
v49a_has_prev_snapshot
v49a_sample_size
v49a_mean_abs_delta_homo
v49a_mean_abs_delta_hetero
v49a_mean_abs_delta_hard
v49a_mean_abs_delta_score
v49a_hard_mass_delta
v49a_hard_rank_corr_prev
v49a_homo_target_mass
v49a_hetero_target_mass
v49a_defer_target_mass
v49a_raw_homo_target_mass
v49a_raw_hetero_target_mass
v49a_raw_defer_target_mass
v49a_targeted_homo_delta
v49a_targeted_hetero_delta
v49a_targeted_hard_delta
```

The movement helper should use deterministic edge-prefix sampling, matching
V48A's audit discipline.

## 7. Implementation Verdict

V49A is implementable with a small, auditable patch:

- Add config fields.
- Add V49A snapshot buffers.
- Add a V49A mask mapping helper.
- Add a V49A movement/target diagnostic helper.
- Branch `_frontend_pass` to use V49A masks when enabled.
- Add diagnostics fields.
- Register the runner variant with failed losses disabled.

After implementation, only run:

```text
py_compile
1-epoch ACM CPU connectivity
```

Do not run the 80-epoch first-stage smoke until implementation notes confirm
that the connectivity diagnostics are present and red-line flags are correct.
