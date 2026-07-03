# V51A Implementation Review

This document reviews how to implement
`v51a_reliability_gated_spectral_anchor` after `V51A_PREREGISTRATION.md`.
It follows `ccf-idea-optimizer` exploratory rescue mode: freeze the mechanism
and implementation boundary before code changes. No V51A code is implemented in
this document and no V51A experiment is run.

## 1. Reviewed Artifacts

Local evidence and design files:

```text
CRITICAL_RED_LINES.md
V50_RESCUE_ROUTE_DECISION.md
V50A_PREREGISTRATION.md
V50A_IMPLEMENTATION_REVIEW.md
V50A_FIRST_SMOKE_VERDICT.md
V50A_SECOND_STAGE_SMOKE_VERDICT.md
V50A_HETEROPHILY_FAILURE_ANALYSIS.md
V51A_PREREGISTRATION.md
```

Current code surfaces:

```text
core/e2e/sect_coco_e2e.py
scripts/run_unified_aptc_9datasets.py
```

Conclusion:

```text
V51A is implementable as a conservative V50A helper extension. It does not need
a new head, a new final selector, a dataset-specific branch, or a post-processing
assignment path.
```

## 2. Current Interface Facts

The current model already exposes the tensors required by V51A at the loss site:

| Tensor / object | Current availability | Use in V51A |
| --- | --- | --- |
| `out["q_refined"]` | available in `compute_loss` | trainable posterior for KL source |
| `out["q_embed"]` | available in `compute_loss` | embedding readout agreement component |
| `self.v50a_anchor_q` | existing spectral anchor buffer | fixed `q_spec` teacher |
| `self.edge_index` | module member | local anchor consistency component |
| `cfg.v50a_*` | existing frozen anchor construction constants | reused unchanged for V51A anchor construction |

The current V50A loss is centralized:

```text
spectral_anchor_alignment_loss(q_refined, v50a_anchor_q, q_embed, enabled, effective_weight)
```

The current final labels remain:

```text
out["q_refined"].argmax(dim=1)
```

This is compatible with the V51A red line: the anchor labels are diagnostics and
training guidance only, not final output.

## 3. Required Code Changes

### 3.1 Config

Add V51A fields to `E2ESECTCoCoConfig` near the V50A fields:

```text
v51a_enabled: bool = False
v51a_anchor_weight: float = 0.0
v51a_reliability_floor: float = 0.10
v51a_reliable_threshold: float = 0.20
v51a_min_effective_mass: float = 0.10
```

Do not add dataset-specific constants. Do not expose alternative reliability
formulas in the first implementation.

### 3.2 Anchor Construction

The V50A spectral anchor builder should be reused unchanged:

```text
build_spectral_compactness_anchor(x_np, graph_adj, self.n_clusters, cfg)
```

Allowed code change:

```text
if cfg.v50a_enabled or cfg.v51a_enabled:
    build the same anchor and store it in the existing anchor buffer
```

Frozen V50A anchor constants for the V51A variant:

```text
v50a_filter_steps = 2
v50a_anchor_rank_multiplier = 1.0
v50a_anchor_temperature = 0.35
v50a_anchor_refresh = false
```

Rejected changes:

- new rank rule;
- new temperature;
- new filter step count;
- refresh policy;
- signed or topology-mask anchor;
- using labels or validation/test metrics to construct the anchor.

### 3.3 Loss Helper

Add a new helper rather than modifying V50A semantics in place:

```text
reliability_gated_spectral_anchor_loss(
    q,
    q_anchor,
    *,
    q_embed,
    edge_index,
    enabled,
    effective_weight,
    reliability_floor,
    reliable_threshold,
    min_effective_mass,
)
```

The helper must:

1. Normalize `q`, `q_embed`, and `q_anchor`.
2. Detach `q_anchor`.
3. Compute all reliability components from detached tensors.
4. Compute `r_i = detach(clamp(r_raw_i, 0, 1))`.
5. Compute per-node `KL(q_refined_i || stopgrad(q_spec_i))`.
6. Aggregate by:

```text
sum_i r_i * kl_i / clamp(sum_i r_i, min=N * reliability_floor)
```

The reliability computation must not let gradients flow through:

```text
conf_i
qa_i_norm
ea_i_norm
local_i_norm
r_i
```

Rationale:

```text
Reliability is a safety gate, not a learnable escape route. If it is not
detached, the model can reduce the loss by changing the gate rather than by
learning a better posterior.
```

### 3.4 Local Anchor Consistency

Implement local consistency with existing `edge_index`.

Required behavior:

```text
sim_ij = sum_c a_i[c] * a_j[c]
local_i = mean similarity over incident candidate edges
isolated node fallback: local_i_norm = conf_i
```

The implementation may use `index_add_` to avoid adding a new dependency:

```text
src, dst = edge_index
sim = (anchor[src] * anchor[dst]).sum(dim=1)
local_sum.index_add_(0, src, sim)
local_cnt.index_add_(0, src, 1)
local_sum.index_add_(0, dst, sim)
local_cnt.index_add_(0, dst, 1)
```

Then:

```text
local = local_sum / local_cnt.clamp_min(1)
local = where(local_cnt > 0, local, conf)
```

This treats the candidate graph as undirected for reliability diagnostics, which
matches the preregistered "edge_index or reverse" intent without changing graph
construction.

### 3.5 Total Loss Wiring

Current V50A total loss term:

```text
cfg.v50a_anchor_weight * v50a_anchor_loss
```

V51A should add a separate term:

```text
cfg.v51a_anchor_weight * v51a_anchor_loss
```

The V51A experiment variant must set:

```text
v50a_enabled = False
v50a_anchor_weight = 0.0
v51a_enabled = True
v51a_anchor_weight = 0.04
```

This prevents simultaneous unweighted V50A KL and reliability-weighted V51A KL.
Using both at the same time is a preregistration violation.

### 3.6 Diagnostics

Add diagnostics under distinct V51A keys. Required minimum:

```text
v51a_enabled
v51a_anchor_loss
v51a_weighted_q_anchor_kl
v51a_weighted_q_anchor_agreement
v51a_unweighted_q_anchor_agreement
v51a_embedding_anchor_agreement
v51a_anchor_entropy
v51a_anchor_confidence
v51a_anchor_cluster_usage_entropy
v51a_anchor_effective_weight
v51a_reliability_mean
v51a_reliability_std
v51a_reliability_p10
v51a_reliability_p50
v51a_reliability_p90
v51a_reliable_node_ratio
v51a_effective_anchor_mass
v51a_confidence_component_mean
v51a_q_anchor_component_mean
v51a_embed_anchor_component_mean
v51a_local_component_mean
```

Epoch snapshots should mirror the existing V50A agreement snapshot pattern:

```text
v51a_weighted_q_anchor_agreement_epoch_1
v51a_weighted_q_anchor_agreement_epoch_40
v51a_weighted_q_anchor_agreement_epoch_80
v51a_reliability_mean_epoch_1
v51a_reliability_mean_epoch_40
v51a_reliability_mean_epoch_80
```

Implementation note:

The runner currently already records selected epoch diagnostics for V50A. V51A
must extend that list rather than changing V50A keys.

### 3.7 Runner Variant

Add one variant in `scripts/run_unified_aptc_9datasets.py`:

```text
EXPERIMENT_VARIANTS["v51a_reliability_gated_spectral_anchor"]
```

It should inherit the V50A rescue base and only replace the anchor-loss switch:

```text
output_stem = "unified_aptc_9datasets_v51a_reliability_gated_spectral_anchor"
v50a_enabled = False
v50a_anchor_weight = 0.0
v51a_enabled = True
v51a_anchor_weight = 0.04
v51a_reliability_floor = 0.10
v51a_reliable_threshold = 0.20
v51a_min_effective_mass = 0.10
```

Keep all V43B-V49A failed mechanisms disabled exactly as in V50A.

## 4. Red-Line Review

| Red line | V51A implementation requirement | Status |
| --- | --- | --- |
| No dataset-specific modules | One reliability formula for every dataset | Satisfied if no dataset name enters config or code |
| Unified pipeline | Same forward path, same final assignment path | Satisfied if final labels remain `q_refined.argmax` |
| End-to-end trainability | Anchor KL trains the existing posterior path | Satisfied; reliability is detached by design |
| Preserve frontend innovations | Existing frontend and contraction path remain active | Satisfied if no head/selector replaces them |
| No post-hoc selection | Anchor labels and embedding KMeans are diagnostics only | Satisfied if final output is not switched |
| No V50A hyperparameter sweep | Anchor construction constants frozen | Satisfied if variant constants match preregistration |

Hard fail conditions during implementation review:

- adding `if dataset == ...`;
- enabling V50A and V51A anchor losses together;
- using anchor labels as final labels;
- using embedding KMeans as final labels;
- using labels or metrics inside reliability;
- allowing gradients through reliability;
- adding low-reliability geometry fallback in V51A first implementation.

## 5. Scientific Interpretation Of The Mechanism

Problem:

```text
V50A's fixed spectral teacher is non-collapsed but not uniformly trustworthy,
especially under heterophily-style graph/attribute mismatch.
```

Root challenge:

```text
A useful low-rank signal and an unsafe assignment teacher can be the same object
on different nodes.
```

V51A insight:

```text
Trust should be node-local and evidence-gated. A spectral anchor should only
impose assignment-level KL where anchor confidence, model posterior, embedding
readout, and local anchor consistency agree.
```

Innovation type:

```text
Method/objective innovation with diagnostic empirical evidence.
```

The first V51A claim must remain narrow:

```text
Reliability gating can make the V50A spectral compactness anchor safer without
dataset-specific routing.
```

Do not claim:

```text
V51A solves heterophily, beats all baselines, or provides a final SOTA route.
```

Those claims require future evidence.

## 6. Implementation Order

Allowed next implementation sequence:

1. Add config fields.
2. Add `reliability_gated_spectral_anchor_loss`.
3. Build the spectral anchor when `v50a_enabled or v51a_enabled`.
4. Wire V51A loss and diagnostics.
5. Add V51A runner variant.
6. Run static compilation only.
7. Run the preregistered one-dataset, one-epoch connectivity command.

Do not run mixed stress until connectivity passes and diagnostics are confirmed.

## 7. Static Checks Before Connectivity

Before any V51A run, execute:

```powershell
conda run -n aaai-e2e-subspace python -m py_compile core\e2e\sect_coco_e2e.py scripts\run_unified_aptc_9datasets.py
```

Manual static checks:

```text
rg -n "v51a|reliability_gated" core/e2e/sect_coco_e2e.py scripts/run_unified_aptc_9datasets.py
rg -n "dataset|legacy_head|argmax|KMeans|v50a_anchor_weight" core/e2e/sect_coco_e2e.py scripts/run_unified_aptc_9datasets.py
```

The second check is not a prohibition on existing code; it is a review step to
confirm V51A did not introduce selector behavior.

## 8. Connectivity Authorization

After implementation and static checks, only this run is authorized:

```powershell
conda run --no-capture-output -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v51a_reliability_gated_spectral_anchor --datasets "acm" --epochs 1 --device cuda --log-level WARNING
```

Connectivity pass condition:

```text
status=ok
legacy_head_used=false
v51a_enabled=true
v50a_enabled=false
v51a_anchor_loss finite
v51a_reliability_mean finite
v51a_effective_anchor_mass finite
v51a_reliable_node_ratio finite
```

Connectivity output is not a performance result and must not be used for model
selection.

## 9. Decision

V51A is approved for minimal implementation under this review, but not for
mixed-stress or full-run evaluation yet.

Next allowed artifact after code implementation and connectivity:

```text
V51A_CONNECTIVITY_VERDICT.md
```

Only if that verdict passes should the preregistered six-dataset mixed stress
experiment be run.

## 10. No-Fabrication Status

No V51A result exists. This document only reviews implementation feasibility
and boundaries based on current local code and V50A evidence.
