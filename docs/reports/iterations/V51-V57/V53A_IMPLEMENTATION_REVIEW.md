# V53A Implementation Review

This document reviews how to implement
`v53a_residual_curriculum_spectral_anchor` after
`V53A_PREREGISTRATION.md`. It follows `ccf-idea-optimizer` exploratory rescue
mode: freeze the implementation boundary before code changes.

No V53A code is implemented in this document and no V53A experiment is run.

## 1. Reviewed Artifacts

Local evidence and design files:

```text
V51A_FIRST_MIXED_STRESS_VERDICT.md
V51A_RELIABILITY_COLLAPSE_ANALYSIS.md
V52A_PREREGISTRATION.md
V52A_IMPLEMENTATION_REVIEW.md
V52A_CONNECTIVITY_VERDICT.md
V52A_FIRST_MIXED_STRESS_VERDICT.md
V52A_LATE_RELIABILITY_COLLAPSE_ANALYSIS.md
V53A_PREREGISTRATION.md
CRITICAL_RED_LINES.md
```

Current code surfaces:

```text
core/e2e/sect_coco_e2e.py
scripts/run_unified_aptc_9datasets.py
```

Conclusion:

```text
V53A is implementable as a conservative residual extension of the existing V52A
helper. It does not need a new head, selector, post-processing path,
dataset-specific branch, or new spectral anchor construction.
```

## 2. Current Interface Facts

The current implementation already exposes the required tensors and schedule
inputs:

| Tensor / object | Current availability | V53A use |
| --- | --- | --- |
| `out["q_refined"]` | loss site | KL source |
| `out["q_embed"]` | loss site | agreement component |
| `self.v50a_anchor_q` | model buffer | fixed spectral anchor |
| `self.edge_index` | module member | local anchor consistency |
| `current_epoch` | loss method | inherited V52A schedule |

V53A should reuse V50A anchor construction:

```text
build_spectral_compactness_anchor(x_np, graph_adj, self.n_clusters, cfg)
```

Anchor construction must happen when any of these are enabled:

```text
v50a_enabled or v51a_enabled or v52a_enabled or v53a_enabled
```

## 3. Required Code Changes

### 3.1 Config Fields

Add fields near the V52A fields:

```text
v53a_enabled: bool = False
v53a_anchor_weight: float = 0.0
v53a_reliability_floor: float = 0.10
v53a_reliable_threshold: float = 0.20
v53a_min_effective_mass: float = 0.10
v53a_warmup_epochs: int = 20
v53a_ramp_epochs: int = 40
v53a_residual_beta: float = 0.50
```

Do not add alternate schedules, beta values, fallback modes, or dataset-specific
controls.

### 3.2 Loss Helper

Add a new helper instead of modifying V52A:

```text
residual_curriculum_spectral_anchor_loss(
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
    residual_beta,
)
```

The helper may reuse the V52A component logic, but must emit separate `v53a_*`
diagnostics and preserve V52A semantics.

### 3.3 Reliability Formula

The helper must implement exactly:

```text
gamma_t = clamp((current_epoch + 1 - warmup_epochs) / ramp_epochs, 0, 1)
r_base_i = 0.5 * conf_i + 0.5 * local_i_norm
r_agree_i = 0.5 * qa_i_norm + 0.5 * ea_i_norm
beta = 0.50
r_multiplier_i = (1 - gamma_t) + gamma_t * (beta + (1 - beta) * r_agree_i)
r_i = detach(clamp(r_base_i * r_multiplier_i, 0, 1))
```

All component tensors used to compute reliability must be detached:

```text
q_spec, q_refined for reliability, q_embed for reliability, local consistency
```

The KL term remains trainable through `q_refined`:

```text
kl_i = KL(q_refined_i || stopgrad(q_spec_i))
```

### 3.4 Loss Wiring

Add a separate total loss term:

```text
cfg.v53a_anchor_weight * v53a_anchor_loss
```

The V53A runner variant must set:

```text
v50a_enabled = False
v50a_anchor_weight = 0.0
v51a_enabled = False
v51a_anchor_weight = 0.0
v52a_enabled = False
v52a_anchor_weight = 0.0
v53a_enabled = True
v53a_anchor_weight = 0.04
v53a_residual_beta = 0.50
```

Running V50A, V51A, V52A, and V53A anchor losses together is a preregistration
violation.

### 3.5 Diagnostics

Add the required V53A diagnostics under distinct keys:

```text
v53a_enabled
v53a_gamma
v53a_residual_beta
v53a_residual_multiplier_mean
v53a_anchor_loss
v53a_weighted_q_anchor_kl
v53a_weighted_q_anchor_agreement
v53a_unweighted_q_anchor_agreement
v53a_embedding_anchor_agreement
v53a_anchor_entropy
v53a_anchor_confidence
v53a_anchor_cluster_usage_entropy
v53a_anchor_effective_weight
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

Epoch snapshots should extend the existing snapshot list:

```text
v53a_gamma
v53a_residual_multiplier_mean
v53a_weighted_q_anchor_agreement
v53a_weighted_q_anchor_kl
v53a_reliability_mean
v53a_base_reliability_mean
v53a_agreement_reliability_mean
```

The existing snapshot epochs `{1, 20, 40, 60, 80}` are sufficient.

### 3.6 Final Anchor Diagnostics

Add:

```text
v53a_anchor_acc_diagnostic
v53a_anchor_nmi_diagnostic
v53a_anchor_ari_diagnostic
```

The labels are diagnostics only and must not affect training or final output.

### 3.7 Runner Variant

Add:

```text
EXPERIMENT_VARIANTS["v53a_residual_curriculum_spectral_anchor"]
```

It should inherit from the V52A rescue base but explicitly disable V50A, V51A,
and V52A losses.

## 4. Red-Line Review

| Red line | V53A implementation requirement | Status |
| --- | --- | --- |
| No dataset-specific module | same beta, schedule, and formula for every dataset | satisfied if no dataset name enters code |
| Unified pipeline | same forward path and `q_refined` final labels | satisfied |
| End-to-end trainability | KL trains existing posterior/frontend path | satisfied |
| Preserve frontend innovations | no head/selector replaces frontend | satisfied |
| No post-hoc selection | anchor labels diagnostics only | satisfied |
| No sweep | beta and schedule fixed | satisfied if implemented exactly |

Hard implementation failures:

- adding `if dataset == ...`;
- enabling V50A/V51A/V52A losses together with V53A;
- using anchor labels as final labels;
- using embedding KMeans as final labels;
- using labels or metrics inside reliability;
- allowing gradients through reliability;
- adding geometry fallback;
- changing V50A anchor construction constants;
- changing `beta=0.50`.

## 5. Scientific Interpretation

V52A failed because it made base reliability temporary. V53A's mechanism keeps:

```text
early anchor availability = r_base
late anchor availability = r_base * (0.50 + 0.50 * r_agree)
```

This is the smallest correction to V52A's actual failure. It is not a schedule
sweep, beta sweep, or return to V50A's global trust.

The first V53A claim must remain narrow:

```text
A fixed residual can prevent late reliability collapse while preserving safety.
```

Do not claim that V53A solves heterophily or reaches SOTA until mixed-stress and
later evidence support it.

## 6. Implementation Order

Allowed next implementation sequence:

1. Add V53A config fields.
2. Add `residual_curriculum_spectral_anchor_loss`.
3. Build the spectral anchor when `v50a_enabled or v51a_enabled or v52a_enabled or v53a_enabled`.
4. Wire V53A loss and diagnostics.
5. Add V53A epoch snapshots.
6. Add V53A final anchor diagnostics.
7. Add V53A runner defaults and variant.
8. Run static compilation.
9. Run the one-dataset, one-epoch connectivity command.

Do not run mixed stress until connectivity passes and
`V53A_CONNECTIVITY_VERDICT.md` is written.

## 7. Static Checks Before Connectivity

Before any V53A run:

```powershell
conda run -n aaai-e2e-subspace python -m py_compile core\e2e\sect_coco_e2e.py scripts\run_unified_aptc_9datasets.py
```

Manual checks:

```text
rg -n "v53a|residual_curriculum" core/e2e/sect_coco_e2e.py scripts/run_unified_aptc_9datasets.py
rg -n "v50a_enabled|v51a_enabled|v52a_enabled|v53a_enabled" scripts/run_unified_aptc_9datasets.py
```

The second check must confirm the V53A variant disables V50A, V51A, and V52A
anchor losses.

## 8. Connectivity Authorization

After implementation and static checks, only this run is authorized:

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

Connectivity output is not a performance result and must not be used for model
selection.

## 9. Decision

V53A is approved for minimal implementation under this review, but not for
mixed-stress or full-run evaluation yet.

Next allowed artifact after code implementation and connectivity:

```text
V53A_CONNECTIVITY_VERDICT.md
```

Only if that verdict passes should the preregistered six-dataset mixed stress
experiment be run.

## 10. No-Fabrication Status

No V53A result exists. This document only reviews implementation feasibility
and boundaries based on current local code and V50A/V51A/V52A evidence.
