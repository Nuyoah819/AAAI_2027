# V54A Implementation Review

This document reviews how to implement
`v54a_consensus_bounded_residual_anchor` after
`V54A_PREREGISTRATION.md`. It follows `ccf-idea-optimizer` standard rescue
mode: optimize the mechanism and evidence boundary before expanding
experiments.

No V54A code is implemented in this document and no V54A experiment is run.

## 1. Reviewed Artifacts

Local evidence and design files:

```text
V52A_FIRST_MIXED_STRESS_VERDICT.md
V52A_LATE_RELIABILITY_COLLAPSE_ANALYSIS.md
V53A_PREREGISTRATION.md
V53A_IMPLEMENTATION_REVIEW.md
V53A_CONNECTIVITY_VERDICT.md
V53A_FIRST_MIXED_STRESS_VERDICT.md
V53A_RESIDUAL_OVEREXPOSURE_ANALYSIS.md
V54A_PREREGISTRATION.md
CRITICAL_RED_LINES.md
```

Current code surfaces:

```text
core/e2e/sect_coco_e2e.py
scripts/run_unified_aptc_9datasets.py
```

Conclusion:

```text
V54A is implementable as a narrow extension of the existing V53A residual
helper. It should add node-level consensus-bounded residual strength without a
new head, selector, dataset branch, post-processing path, geometry fallback, or
new spectral-anchor construction.
```

## 2. Idea Optimization Summary

Target venue and family:

```text
AAAI 2027 / AI-ML family, assumed from workspace.
```

Normalized idea card:

| Field | V54A position |
| --- | --- |
| Task | End-to-end graph clustering with a unified posterior output |
| Gap | V52A loses late anchor reliability; V53A repairs reliability but overexposes weak-anchor heterophily nodes |
| Root challenge | Anchor availability must persist without forcing weak anchor assignments everywhere |
| Core insight | Residual strength should be node-level and bounded by agreement among posterior, embedding readout, and fixed spectral anchor |
| Proposed mechanism | Keep V53A residual curriculum but replace global beta with stop-gradient `beta_i` from hard consensus |
| Contribution type | Method mechanism plus diagnostic evidence |
| Expected evidence | Connectivity, 6-dataset mixed stress, reliability non-collapse, Squirrel safety, anchor-usefulness gates |
| Main risk | Hard consensus may underactivate DBLP/Texas or still overexpose Squirrel if consensus is spurious |

Novelty status:

```text
needs-search. This review only optimizes the local rescue route and does not
claim literature novelty.
```

## 3. Current Interface Facts

The required tensors and code paths already exist:

| Tensor / object | Current availability | V54A use |
| --- | --- | --- |
| `out["q_refined"]` | loss site | trainable KL source and detached consensus input |
| `out["q_embed"]` | loss site | detached consensus and agreement input |
| `self.v50a_anchor_q` | model buffer | fixed spectral anchor |
| `self.edge_index` | module member | local anchor consistency |
| `current_epoch` | loss method | inherited V52A/V53A curriculum |

Anchor construction should still call:

```text
build_spectral_compactness_anchor(x_np, graph_adj, self.n_clusters, cfg)
```

and should be triggered when any of these are enabled:

```text
v50a_enabled or v51a_enabled or v52a_enabled or v53a_enabled or v54a_enabled
```

## 4. Required Code Changes

### 4.1 Config Fields

Add fields near the V53A fields:

```text
v54a_enabled: bool = False
v54a_anchor_weight: float = 0.0
v54a_reliability_floor: float = 0.10
v54a_reliable_threshold: float = 0.20
v54a_min_effective_mass: float = 0.10
v54a_warmup_epochs: int = 20
v54a_ramp_epochs: int = 40
v54a_beta_min: float = 0.35
v54a_beta_max: float = 0.70
```

Do not add beta sweeps, schedule sweeps, dataset-specific overrides, or fallback
switches.

### 4.2 Loss Helper

Add a new helper instead of modifying V53A:

```text
consensus_bounded_residual_spectral_anchor_loss(
    q,
    q_anchor,
    *,
    q_embed,
    edge_index,
    enabled,
    current_epoch,
    effective_weight,
    reliability_floor,
    reliable_threshold,
    min_effective_mass,
    warmup_epochs,
    ramp_epochs,
    beta_min,
    beta_max,
)
```

The helper may reuse V53A component logic, but must emit separate `v54a_*`
diagnostics and preserve V53A semantics.

### 4.3 Reliability Formula

The helper must implement:

```text
gamma_t = clamp((current_epoch + 1 - warmup_epochs) / ramp_epochs, 0, 1)
r_base_i = 0.5 * conf_i + 0.5 * local_i_norm
r_agree_i = 0.5 * qa_i_norm + 0.5 * ea_i_norm
h_q_i = 1[argmax(q_refined_i) == argmax(q_spec_i)]
h_e_i = 1[argmax(q_embed_i) == argmax(q_spec_i)]
h_i = 0.5 * h_q_i + 0.5 * h_e_i
beta_i = beta_min + (beta_max - beta_min) * h_i
r_multiplier_i = (1 - gamma_t)
               + gamma_t * (beta_i + (1 - beta_i) * r_agree_i)
r_i = detach(clamp(r_base_i * r_multiplier_i, 0, 1))
```

Required detach boundary:

```text
q_spec, q_refined for reliability, q_embed for reliability, local consistency,
hard consensus, beta_i, multiplier, and r_i
```

The KL term remains trainable through `q_refined`:

```text
kl_i = KL(q_refined_i || stopgrad(q_spec_i))
```

### 4.4 Loss Wiring

Add a separate total loss term:

```text
cfg.v54a_anchor_weight * v54a_anchor_loss
```

The V54A runner variant must set:

```text
v50a_enabled = False
v50a_anchor_weight = 0.0
v51a_enabled = False
v51a_anchor_weight = 0.0
v52a_enabled = False
v52a_anchor_weight = 0.0
v53a_enabled = False
v53a_anchor_weight = 0.0
v54a_enabled = True
v54a_anchor_weight = 0.04
v54a_beta_min = 0.35
v54a_beta_max = 0.70
```

Running V50A, V51A, V52A, V53A, and V54A anchor losses together is a
preregistration violation.

### 4.5 Diagnostics

Add the required V54A diagnostics under distinct keys:

```text
v54a_enabled
v54a_gamma
v54a_beta_min
v54a_beta_max
v54a_beta_mean
v54a_beta_p10
v54a_beta_p50
v54a_beta_p90
v54a_hard_q_anchor_match_ratio
v54a_hard_embed_anchor_match_ratio
v54a_hard_both_anchor_match_ratio
v54a_residual_multiplier_mean
v54a_anchor_loss
v54a_weighted_q_anchor_kl
v54a_weighted_q_anchor_agreement
v54a_unweighted_q_anchor_agreement
v54a_embedding_anchor_agreement
v54a_anchor_entropy
v54a_anchor_confidence
v54a_anchor_cluster_usage_entropy
v54a_anchor_effective_weight
v54a_reliability_mean
v54a_reliability_std
v54a_reliability_p10
v54a_reliability_p50
v54a_reliability_p90
v54a_reliable_node_ratio
v54a_effective_anchor_mass
v54a_base_reliability_mean
v54a_agreement_reliability_mean
v54a_confidence_component_mean
v54a_q_anchor_component_mean
v54a_embed_anchor_component_mean
v54a_local_component_mean
```

Epoch snapshots should include:

```text
v54a_gamma
v54a_beta_mean
v54a_residual_multiplier_mean
v54a_weighted_q_anchor_agreement
v54a_weighted_q_anchor_kl
v54a_reliability_mean
v54a_base_reliability_mean
v54a_agreement_reliability_mean
```

The existing snapshot epochs `{1, 20, 40, 60, 80}` are sufficient.

### 4.6 Final Anchor Diagnostics

Add:

```text
v54a_anchor_acc_diagnostic
v54a_anchor_nmi_diagnostic
v54a_anchor_ari_diagnostic
```

The labels are diagnostics only and must not affect training or final output.

## 5. Red-Line Review

| Red line | V54A implementation requirement | Status |
| --- | --- | --- |
| No dataset-specific module | same formula and constants for every dataset | satisfied if no dataset name enters code |
| Unified pipeline | final labels remain `q_refined.argmax` | satisfied |
| End-to-end trainability | KL trains existing posterior path only | satisfied |
| Preserve frontend innovations | no head/selector replaces frontend | satisfied |
| No post-hoc selection | anchor labels remain diagnostics only | satisfied |
| No sweep | beta bounds and schedule fixed | satisfied if implemented exactly |
| No geometry fallback | low-consensus nodes only receive bounded residual, no new fallback | satisfied |

Hard implementation failures:

- adding `if dataset == ...`;
- enabling V50A/V51A/V52A/V53A losses together with V54A;
- using anchor labels, embedding KMeans, or legacy head as final labels;
- using labels or metrics inside reliability;
- allowing gradients through reliability, hard consensus, or beta;
- changing V50A anchor construction constants;
- changing `beta_min=0.35`, `beta_max=0.70`, warmup, or ramp;
- adding signed topology-mask anchor or low-reliability geometry fallback.

## 6. Scientific Interpretation

V54A tests a narrow claim:

```text
Residual anchor availability can remain late in training while weak-anchor
exposure is bounded by node-level anchor/posterior/readout consensus.
```

It should not claim SOTA, full heterophily resolution, or general robustness
until the preregistered mixed-stress gates pass.

## 7. Implementation Order

Allowed next implementation sequence:

1. Add V54A config fields.
2. Add `consensus_bounded_residual_spectral_anchor_loss`.
3. Build the spectral anchor when V54A is enabled.
4. Wire V54A loss and diagnostics.
5. Add V54A epoch snapshots.
6. Add V54A final anchor diagnostics.
7. Add V54A runner defaults and variant.
8. Run static compilation.
9. Run the one-dataset, one-epoch connectivity command.

Do not run mixed stress until connectivity passes and
`V54A_CONNECTIVITY_VERDICT.md` is written.

## 8. Static Checks Before Connectivity

Before any V54A run:

```powershell
conda run -n aaai-e2e-subspace python -m py_compile core\e2e\sect_coco_e2e.py scripts\run_unified_aptc_9datasets.py
```

Manual checks:

```text
rg -n "v54a|consensus_bounded_residual" core/e2e/sect_coco_e2e.py scripts/run_unified_aptc_9datasets.py
rg -n "v50a_enabled|v51a_enabled|v52a_enabled|v53a_enabled|v54a_enabled" scripts/run_unified_aptc_9datasets.py
```

The second check must confirm the V54A variant disables V50A, V51A, V52A, and
V53A anchor losses.

## 9. Connectivity Authorization

After implementation and static checks, only this run is authorized:

```powershell
conda run --no-capture-output -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v54a_consensus_bounded_residual_anchor --datasets "acm" --epochs 1 --device cuda --log-level WARNING
```

Connectivity pass condition:

```text
status=ok
legacy_head_used=false
v54a_enabled=true
v50a_enabled=false
v51a_enabled=false
v52a_enabled=false
v53a_enabled=false
v54a_anchor_loss finite
v54a_gamma = 0
v54a_beta_min = 0.35
v54a_beta_max = 0.70
v54a_beta_mean finite
v54a_reliability_mean finite
v54a_effective_anchor_mass finite
```

Connectivity output is not a performance result and must not be used for model
selection.

## 10. Decision

V54A is approved for minimal implementation under this review, but not for
mixed-stress or full-run evaluation yet.

Next allowed artifact after code implementation and connectivity:

```text
V54A_CONNECTIVITY_VERDICT.md
```

Only if that verdict passes should the preregistered six-dataset mixed stress
experiment be run.

## 11. No-Fabrication Status

No V54A result exists. This document only reviews implementation feasibility,
idea mechanism, and evidence gates based on current local code and V52A/V53A
evidence.
