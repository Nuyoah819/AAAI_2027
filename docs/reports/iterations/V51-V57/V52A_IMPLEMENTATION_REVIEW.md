# V52A Implementation Review

This document reviews how to implement
`v52a_curriculum_reliability_spectral_anchor` after
`V52A_PREREGISTRATION.md`. It follows `ccf-idea-optimizer` exploratory rescue
mode: freeze the implementation boundary before changing code.

No V52A code is implemented in this document and no V52A experiment is run.

## 1. Reviewed Artifacts

Local evidence and design files:

```text
V50A_SECOND_STAGE_SMOKE_VERDICT.md
V50A_HETEROPHILY_FAILURE_ANALYSIS.md
V51A_PREREGISTRATION.md
V51A_IMPLEMENTATION_REVIEW.md
V51A_CONNECTIVITY_VERDICT.md
V51A_FIRST_MIXED_STRESS_VERDICT.md
V51A_RELIABILITY_COLLAPSE_ANALYSIS.md
V52A_PREREGISTRATION.md
CRITICAL_RED_LINES.md
```

Current code surfaces:

```text
core/e2e/sect_coco_e2e.py
scripts/run_unified_aptc_9datasets.py
```

Conclusion:

```text
V52A is implementable as a conservative curriculum extension of V51A. It does
not need a new head, selector, post-processing path, dataset-specific branch, or
new spectral anchor construction.
```

## 2. Current Interface Facts

The current implementation already exposes the required tensors:

| Tensor / object | Current availability | V52A use |
| --- | --- | --- |
| `out["q_refined"]` | loss site | KL source |
| `out["q_embed"]` | loss site | late agreement component |
| `self.v50a_anchor_q` | model buffer | fixed spectral anchor |
| `self.edge_index` | module member | local anchor consistency |
| `current_epoch` | loss method | curriculum schedule |
| `cfg.epochs` | config | optional schedule context, not dataset-specific |

V52A should reuse V50A anchor construction:

```text
build_spectral_compactness_anchor(x_np, graph_adj, self.n_clusters, cfg)
```

Anchor construction must happen when any of these are enabled:

```text
v50a_enabled or v51a_enabled or v52a_enabled
```

## 3. Required Code Changes

### 3.1 Config Fields

Add fields near the V51A fields:

```text
v52a_enabled: bool = False
v52a_anchor_weight: float = 0.0
v52a_reliability_floor: float = 0.10
v52a_reliable_threshold: float = 0.20
v52a_min_effective_mass: float = 0.10
v52a_warmup_epochs: int = 20
v52a_ramp_epochs: int = 40
```

Do not add alternate schedules, exponents, thresholds, fallback modes, or
dataset-specific controls in the first implementation.

### 3.2 Loss Helper

Add a new helper instead of modifying V51A:

```text
curriculum_reliability_spectral_anchor_loss(
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
)
```

It may share logic with V51A conceptually, but should emit separate `v52a_*`
diagnostics and preserve V51A semantics.

### 3.3 Reliability Formula

The helper must implement exactly:

```text
gamma_t = clamp((current_epoch + 1 - warmup_epochs) / ramp_epochs, 0, 1)
r_base_i = 0.5 * conf_i + 0.5 * local_i_norm
r_agree_i = 0.5 * qa_i_norm + 0.5 * ea_i_norm
r_i = detach(clamp((1 - gamma_t) * r_base_i + gamma_t * (r_base_i * r_agree_i), 0, 1))
```

All component tensors used to compute reliability must be detached:

```text
q_spec, q_refined for reliability, q_embed for reliability, local consistency
```

The KL term remains trainable through `q_refined`:

```text
kl_i = KL(q_refined_i || stopgrad(q_spec_i))
```

### 3.4 Local Anchor Consistency

Reuse the V51A incident-edge implementation:

```text
sim_ij = sum_c a_i[c] * a_j[c]
local_sum.index_add_(src/dst)
local_cnt.index_add_(src/dst)
isolated fallback = conf_i
```

The candidate graph is treated as undirected for reliability diagnostics only.
This does not change graph construction or final assignment.

### 3.5 Loss Wiring

Add a separate total loss term:

```text
cfg.v52a_anchor_weight * v52a_anchor_loss
```

The V52A runner variant must set:

```text
v50a_enabled = False
v50a_anchor_weight = 0.0
v51a_enabled = False
v51a_anchor_weight = 0.0
v52a_enabled = True
v52a_anchor_weight = 0.04
```

Running V50A, V51A, and V52A anchor losses together is a preregistration
violation.

### 3.6 Diagnostics

Add the required V52A diagnostics under distinct keys:

```text
v52a_enabled
v52a_gamma
v52a_anchor_loss
v52a_weighted_q_anchor_kl
v52a_weighted_q_anchor_agreement
v52a_unweighted_q_anchor_agreement
v52a_embedding_anchor_agreement
v52a_anchor_entropy
v52a_anchor_confidence
v52a_anchor_cluster_usage_entropy
v52a_anchor_effective_weight
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

Epoch snapshots should extend the existing snapshot list:

```text
v52a_gamma
v52a_weighted_q_anchor_agreement
v52a_weighted_q_anchor_kl
v52a_reliability_mean
v52a_base_reliability_mean
v52a_agreement_reliability_mean
```

Note:

The current runner snapshots epochs 1, 40, and 80. The preregistration also
mentions epoch 20 and 60 gamma diagnostics. The minimal implementation should
either:

1. extend the snapshot set to `{1, 20, 40, 60, 80}`, or
2. explicitly add only V52A gamma snapshots at 20 and 60.

Recommendation:

```text
extend the snapshot set to {1, 20, 40, 60, 80}
```

This is a diagnostic expansion only and does not change training.

### 3.7 Final Anchor Diagnostics

The final evaluation diagnostic currently reports V50A and V51A anchor metrics
from the same anchor buffer. V52A should add:

```text
v52a_anchor_acc_diagnostic
v52a_anchor_nmi_diagnostic
v52a_anchor_ari_diagnostic
```

The labels are used only for post-training diagnostics, like V50A/V51A. They
must not affect training.

### 3.8 Runner Variant

Add:

```text
EXPERIMENT_VARIANTS["v52a_curriculum_reliability_spectral_anchor"]
```

It should inherit from the V51A/V50A rescue base but explicitly set:

```text
v50a_enabled = False
v50a_anchor_weight = 0.0
v51a_enabled = False
v51a_anchor_weight = 0.0
v52a_enabled = True
v52a_anchor_weight = 0.04
v52a_reliability_floor = 0.10
v52a_reliable_threshold = 0.20
v52a_min_effective_mass = 0.10
v52a_warmup_epochs = 20
v52a_ramp_epochs = 40
```

All V43B-V49A failed mechanisms remain disabled.

## 4. Red-Line Review

| Red line | V52A implementation requirement | Status |
| --- | --- | --- |
| No dataset-specific module | same schedule and formula for every dataset | satisfied if no dataset name enters code |
| Unified pipeline | same forward path and `q_refined` final labels | satisfied |
| End-to-end trainability | KL trains existing posterior/frontend path | satisfied |
| Preserve frontend innovations | no head/selector replaces frontend | satisfied |
| No post-hoc selection | anchor labels diagnostics only | satisfied |
| No sweep | schedule and constants fixed | satisfied if implemented exactly |

Hard implementation failures:

- adding `if dataset == ...`;
- enabling V50A/V51A losses together with V52A;
- using anchor labels as final labels;
- using embedding KMeans as final labels;
- using labels or metrics inside reliability;
- allowing gradients through reliability;
- adding geometry fallback;
- changing V50A anchor construction constants.

## 5. Scientific Interpretation

V51A failed because it required current posterior agreement before allowing the
anchor to shape the posterior. V52A's mechanism separates:

```text
early anchor availability = anchor confidence + local consistency
late anchor trust = base reliability modulated by posterior/readout agreement
```

This is the smallest correction to V51A's actual failure. It is not a threshold
sweep and not a return to V50A's global trust.

The first V52A claim must remain narrow:

```text
A fixed curriculum can prevent reliability collapse while preserving safety.
```

Do not claim that V52A solves heterophily or reaches SOTA until mixed-stress
and later evidence support it.

## 6. Implementation Order

Allowed next implementation sequence:

1. Add V52A config fields.
2. Add `curriculum_reliability_spectral_anchor_loss`.
3. Build the spectral anchor when `v50a_enabled or v51a_enabled or v52a_enabled`.
4. Wire V52A loss and diagnostics.
5. Add V52A epoch snapshots, including gamma at 20 and 60.
6. Add V52A final anchor diagnostics.
7. Add V52A runner defaults and variant.
8. Run static compilation.
9. Run the one-dataset, one-epoch connectivity command.

Do not run mixed stress until connectivity passes and
`V52A_CONNECTIVITY_VERDICT.md` is written.

## 7. Static Checks Before Connectivity

Before any V52A run:

```powershell
conda run -n aaai-e2e-subspace python -m py_compile core\e2e\sect_coco_e2e.py scripts\run_unified_aptc_9datasets.py
```

Manual checks:

```text
rg -n "v52a|curriculum_reliability" core/e2e/sect_coco_e2e.py scripts/run_unified_aptc_9datasets.py
rg -n "v50a_enabled|v51a_enabled|v52a_enabled" scripts/run_unified_aptc_9datasets.py
```

The second check must confirm the V52A variant disables V50A and V51A anchor
losses.

## 8. Connectivity Authorization

After implementation and static checks, only this run is authorized:

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

Connectivity output is not a performance result and must not be used for model
selection.

## 9. Decision

V52A is approved for minimal implementation under this review, but not for
mixed-stress or full-run evaluation yet.

Next allowed artifact after code implementation and connectivity:

```text
V52A_CONNECTIVITY_VERDICT.md
```

Only if that verdict passes should the preregistered six-dataset mixed stress
experiment be run.

## 10. No-Fabrication Status

No V52A result exists. This document only reviews implementation feasibility
and boundaries based on current local code and V50A/V51A evidence.
