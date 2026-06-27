# V45A Implementation-Readiness Review

Target mechanism: `v45a_edge_local_band_guarded_frequency`

Scope of this review:

- Read and audit `V45A_PREREGISTRATION.md`.
- Re-check the v44b first-smoke verdict and diagnostics logic.
- Check against `CRITICAL_RED_LINES.md` and the compressed algorithm evolution log.
- Resolve the implementation-blocking question: how should `band_reference` be defined without becoming an implicit dataset-specific mechanism?

No training code is changed here. No experiment is run here. No new result is reported here.

---

## 1. Mode

`design / implementation-readiness review`

This is not a result-presentation document. It is a pre-implementation design audit intended to prevent a red-line violation before any `v45a` code is added.

---

## 2. Venue and assumptions

Venue-family expectation: AAAI / AI-ML evidence package.

Relevant evidence requirements:

- Mechanism-level ablation and diagnostics must explain why the mechanism works.
- The protocol must avoid post-hoc selection on test-set outcomes.
- All variants must be preregistered before execution.
- The method must remain a unified end-to-end model rather than a dataset-routed system.

Project red lines that directly affect v45a:

1. No dataset-specific modules, losses, heads, assigners, or hidden branches.
2. No implicit dataset-type selector where homophilic graphs always get one behavior and heterophilic graphs another behavior.
3. Keep edge confidence, topology contraction, and frequency filtering in the unified front-end.
4. Do not continue failed variants through weight sweeps or post-hoc rescue.

---

## 3. Evidence recap from v44b

The latest v44b first-smoke verdict supports stopping v44b and redesigning the coupling form:

| Dataset | ACC | Emb-Post Gap | Pre-HP Std | Corr | Response Gap | Band Mass | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ACM | 0.7098 | 0.0000 | 0.0252 | -0.0737 | -0.0016 | 0.5009 | FAIL |
| DBLP | 0.6623 | -0.0002 | 0.0109 | 0.3538 | 0.0045 | 0.6863 | partial pass |
| Flickr | 0.3683 | -0.0042 | 0.0477 | 0.3017 | 0.0271 | 0.5033 | FAIL |

Mechanistic conclusion:

- Pre-normalization high-pass response is non-degenerate on all 3 datasets.
- The global node-level correlation loss is falsified because ACM becomes anti-aligned.
- Posterior/readout is not the failure source because `embedding_posterior_gap` stays small.
- ACM band safety fails slightly, so frequency pressure must not be allowed to increase or sustain an unsafe ambiguous band.

Therefore v45a is justified as a coupling redesign, not a v44b weight sweep.

---

## 4. Main implementation-readiness verdict

V45A is conceptually admissible only if the `band_reference` is implemented as a unified graph-adaptive safety reference, not as a dataset lookup table.

The current preregistration is mostly sound, but section 5.3 is underspecified:

```text
band_guard = ReLU(band_mass - stopgrad(band_reference))^2
```

The phrase "current v28b/v41f/v42a baseline band diagnostic if available from the same run family" is unsafe if implemented as:

```text
if dataset == "acm": band_reference = 0.4991
if dataset == "dblp": band_reference = 0.6877
if dataset == "flickr": band_reference = 0.5051
```

or as any equivalent dataset-name / dataset-family table.

That would create a dataset-specific training loss even if the code does not visibly switch heads. It would violate the red-line spirit because each dataset would receive a different safety target by identity.

---

## 5. Recommended unified definition of `band_reference`

### 5.1 Adopt this definition for first implementation

Use a frozen warmup reference measured from the same model, same graph, and same unified forward path before any v45a frequency pressure is active:

```text
warmup_epochs = W
v45a losses are inactive during epochs 1..W
band_reference = stopgrad(mean_or_ema_{epochs 1..W}(band_mass))
```

Then for epochs after warmup:

```text
band_guard = ReLU(band_mass - band_reference)^2
safe_band_gate = sigmoid(k * (band_reference - band_mass))
effective_edge_freq_loss = safe_band_gate * L_edge_freq
```

Recommended preregistered constants for first implementation:

```text
W = 5
k = 20.0
band_reference_delta = 0.0
```

If a stricter version is desired, use a universal improvement margin rather than a dataset-specific ceiling:

```text
band_reference = stopgrad(warmup_band_reference - delta)
delta = 0.0025
```

However, the first implementation should use `delta = 0.0` unless the preregistration is explicitly amended, because v45a already has multiple moving parts.

### 5.2 Why this is not dataset-specific

This definition is admissible because:

- It does not read `dataset`.
- It does not use labels.
- It does not route examples to a different head, assigner, or post-processing path.
- It uses the same formula for all graphs.
- It only asks: "Has the ambiguous band become worse than this run's own pre-v45a topology state?"

This is graph-adaptive, not dataset-routed.

### 5.3 Why fixed per-dataset ceilings must not enter the loss

The following values may remain in the first-stage gate table as diagnostic stop criteria:

```text
ACM band_mass <= 0.4991
DBLP band_mass <= 0.6877
Flickr band_mass <= 0.5051
```

But they must not be used inside the training objective or `safe_band_gate`.

Acceptable use:

```text
After the run, mark the first-stage smoke as pass/fail using preregistered dataset-specific diagnostic ceilings.
```

Unacceptable use:

```text
During training, set band_reference from a dataset-specific ceiling table.
```

This distinction is important: evaluation gates can be dataset-role-specific because ACM/DBLP/Flickr were preregistered as stress tests; the model mechanism cannot be dataset-specific.

---

## 6. Edge-local frequency objective review

The v45a edge-local idea is admissible with one correction: the masks used to form `boundary` and `safe_homo` should be detached for the frequency-response loss unless the explicit aim is to optimize topology assignment through this loss.

Recommended first implementation:

```text
edge_response_ij = 0.5 * (pre_hp_response_i + pre_hp_response_j)
boundary_weight_ij = stopgrad((hetero_ij + hard_ij).clamp(0, 1))
safe_homo_weight_ij = stopgrad(homo_ij.clamp(0, 1))
mean_boundary_response = weighted_mean(edge_response, boundary_weight)
mean_safe_response = weighted_mean(edge_response, safe_homo_weight)
edge_response_gap = mean_boundary_response - mean_safe_response
L_edge_freq = ReLU(target_edge_gap - edge_response_gap)^2
```

Recommended first implementation constants:

```text
target_edge_gap = 0.0
```

Rationale:

- v44b already showed that positive response gaps can appear; first v45a should only require boundary response not to be lower than safe-homophily response.
- A positive margin can be added only after a successful first-stage mechanism result, not before.
- Detaching masks prevents the loss from gaming the topology masks directly to create an artificial gap.

Diagnostics should still report a non-detached or detached copy consistently, but the training path should use detached masks for attribution clarity.

---

## 7. Required implementation invariants

A v45a implementation should be rejected if any of the following are found:

### 7.1 Dataset identity leakage

Forbidden:

```text
dataset name -> band_reference
dataset name -> v45a weight
dataset name -> target_edge_gap
dataset name -> k
dataset name -> warmup length
dataset homophily class -> different loss path
```

### 7.2 Selector-like behavior

Forbidden:

```text
if safe_band_gate < threshold: use old posterior else use new posterior
if graph is homophilic: suppress frequency module
if graph is heterophilic: enable frequency module
```

Allowed:

```text
effective_edge_freq_loss = continuous_safe_band_gate * L_edge_freq
```

The latter is a continuous safety weighting on one unified loss, not a head/assigner selector.

### 7.3 Hidden continuation of failed losses

Must stay disabled:

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
```

V44B diagnostics may remain, but the v44b global node-level correlation loss must be inactive.

---

## 8. Minimal implementation checklist

If code is implemented later, the minimal diff should be constrained to:

1. Add config fields:

```text
v45a_edge_freq_weight
v45a_band_guard_weight
v45a_warmup_epochs
v45a_band_gate_k
v45a_target_edge_gap
v45a_band_reference_delta
```

2. Add diagnostics:

```text
v45a_enabled
v45a_band_mass
v45a_band_reference
v45a_band_guard_loss
v45a_safe_band_gate
v45a_edge_freq_loss
v45a_boundary_response_mean
v45a_safe_homo_response_mean
v45a_edge_response_gap
v45a_edge_response_corr
v45a_boundary_mass
v45a_safe_homo_mass
```

3. Preserve existing v44b diagnostics as diagnostics only:

```text
v44b_pre_hp_response_mean/std/p10/p90
v44b_conflict_response_corr
v44b_response_gap
v44b_postnorm_hp_energy_mean/std
```

4. Add a single variant:

```text
v45a_edge_local_band_guarded_frequency
```

5. Do not add any legacy head, selector, post-processing branch, or dataset-specific config mapping.

---

## 9. Claim-evidence matrix

| Claim | Reviewer question | Evidence needed | Status |
| --- | --- | --- | --- |
| v44b measured a real pre-HP signal | Was v44a's target degenerate? | v44b std/p10/p90 non-degenerate while postnorm energy remains constant | already supported |
| v44b coupling is insufficient | Why not continue global corr? | ACM corr and response gap negative; ACM band safety fails | already supported |
| v45a is a valid redesign | Does it change the mechanism rather than tune weight? | edge-local response + band-guarded coupling, no global node corr | design-ready with fixes |
| band guard is red-line safe | Does `band_reference` create hidden dataset routing? | frozen warmup reference, no dataset lookup | must enforce in implementation |
| frequency pressure is attributable | Can the loss game topology masks? | detached topology masks for first `L_edge_freq` | recommended |

---

## 10. Execution priority

Do not run experiments yet. Recommended order:

1. Amend or annotate `V45A_PREREGISTRATION.md` with the frozen warmup `band_reference` definition.
2. Only then implement the minimal v45a code path.
3. Run a code-level sanity check that diagnostics exist and failed losses are disabled.
4. Only then run the preregistered ACM/DBLP/Flickr 80-epoch smoke.

---

## 11. No-fabrication status

This review reports no new experimental result. All numeric v44b values are copied from the existing v44b first-smoke verdict. All v45a outputs remain `TBD`.

---

## 12. Final decision

V45A is implementation-ready only after the `band_reference` definition is tightened as follows:

```text
Use a unified frozen warmup reference from the same run, not a dataset-specific ceiling table.
```

The per-dataset band ceilings may remain as preregistered evaluation gates, but they must not influence training. This resolves the main red-line risk while preserving the intended topology safety role of v45a.
