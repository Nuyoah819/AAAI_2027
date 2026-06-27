# v45a_edge_local_band_guarded_frequency Preregistration

This document defines the next mechanism after the preregistered failure of
`v44b_pre_normalization_frequency_response`. It is a mechanism and experiment
design document only. It does not report new experimental results.

## 1. Decision From v44b

`v44b_pre_normalization_frequency_response` must stop after the first-stage
80-epoch smoke on ACM, DBLP, and Flickr.

Evidence from `V44B_FIRST_SMOKE_VERDICT.md`:

| Dataset | ACC | Emb-Post Gap | Pre-HP Std | Corr | Response Gap | Band Mass | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ACM | 0.7098 | 0.0000 | 0.0252 | -0.0737 | -0.0016 | 0.5009 | FAIL |
| DBLP | 0.6623 | -0.0002 | 0.0109 | 0.3538 | 0.0045 | 0.6863 | partial pass |
| Flickr | 0.3683 | -0.0042 | 0.0477 | 0.3017 | 0.0271 | 0.5033 | FAIL |

Gate verdict:

- Red-line gate: PASS.
- Posterior/readout safety: PASS.
- Pre-HP non-degeneracy: PASS on 3/3.
- Conflict-response coupling: PASS only on DBLP/Flickr, FAIL on ACM.
- Performance gate: FAIL on ACM and Flickr.
- Topology safety: FAIL on ACM.

The next step must not be a `v44b_pre_hp_corr_weight` sweep, a stronger
version of the same global correlation loss, or a second-batch/full run.

## 2. Scientific Interpretation

v43b, v44a, and v44b jointly establish the following:

1. Direct frontend embedding separation pressure is not reliable.
   In v43b, the conflict gate activated too broadly, uncertainty saturated,
   margin violation ratio was full, `overlap_gap` became negative, and
   frontend embedding was damaged while `embedding_posterior_gap` stayed near
   zero.

2. Post-normalized high-pass energy is an invalid target.
   In v44a, `v44_highpass_energy_mean=0.5`, high-pass std was effectively zero,
   `v44_energy_gap=0`, and `v44_conflict_energy_corr=0` on ACM/DBLP/Flickr.

3. Pre-normalization high-pass response is a valid diagnostic signal.
   In v44b, `v44b_pre_hp_response_std > 1e-4` and `p90 > p10` on all three
   datasets, while post-normalized energy remained degenerate.

4. A node-level global correlation objective is not a sufficient coupling
   mechanism.
   It aligned on DBLP/Flickr but anti-aligned on ACM, where
   `v44b_conflict_response_corr=-0.0737`, `v44b_response_gap=-0.0016`, and
   `v44_band_mass=0.5009` exceeded the preregistered safety ceiling.

Therefore, v44b partially validates the measurement redesign but falsifies the
current coupling form.

## 3. Version Name

```text
v45a_edge_local_band_guarded_frequency
```

Core hypothesis:

```text
If pre-normalization frequency response is evaluated and coupled at the
edge-local topology boundary, and topology band safety is enforced before
frequency-response pressure is allowed to affect representation, then the model
can preserve homophilic safety on ACM-like graphs while retaining useful
conflict-frequency signal on DBLP/Flickr-like graphs.
```

Chinese summary:

V45A keeps V44B's pre-normalization response as a diagnostic signal, but stops
using global node-level scalar correlation as the optimization target. The next
mechanism should operate on edge-local signed frequency response and must guard
the topology ambiguous band before any frequency response pressure is coupled
back into the representation.

## 4. Prohibited Items

All project red lines remain active:

- no dataset-specific module, branch, head, loss, or assigner
- no legacy head
- no adaptive selector or post-processing selector
- no embedding cosine margin loss
- no edge-level overlap margin loss
- no v43b-style selective conflict gate revision
- no v44b weight sweep or target-corr sweep
- no full run before first-stage gate
- no test-set-based post-hoc selection

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
```

V45A may keep V44B diagnostics, but the V44B global node-level correlation loss
must not be active.

## 5. Allowed Minimal Mechanism

### 5.1 Preserve Pre-HP Response As Diagnostic

Keep the pre-normalization high-pass response diagnostic introduced by v44b:

```text
pre_hp_response_i = log1p(raw_high_response_i)
```

Required diagnostics:

```text
v44b_pre_hp_response_mean
v44b_pre_hp_response_std
v44b_pre_hp_response_p10
v44b_pre_hp_response_p90
v44b_conflict_response_corr
v44b_response_gap
v44b_postnorm_hp_energy_mean
v44b_postnorm_hp_energy_std
```

These remain diagnostics only in v45a.

### 5.2 Edge-Local Signed Frequency Response

Define an edge-local response using pre-normalization node responses and the
existing unified topology masks:

```text
edge_response_ij = 0.5 * (pre_hp_response_i + pre_hp_response_j)
edge_boundary_ij = hard_ij + hetero_ij
edge_safe_homo_ij = homo_ij
```

The allowed first objective is not a global node correlation. It should measure
whether high local boundary edges have stronger pre-normalization response than
clear safe-homophily edges:

```text
L_edge_freq = ReLU(target_edge_gap - (mean_boundary_response - mean_safe_response))^2
```

Constraints:

- `edge_boundary_ij` and `edge_safe_homo_ij` come from the same unified
  topology contraction used for all datasets.
- No label information.
- No dataset-specific threshold.
- No pairwise embedding cosine separation.
- No optimization of `overlap_gap`.

### 5.3 Band-Guard Before Frequency Pressure

V44B failed topology safety on ACM:

```text
ACM v44_band_mass = 0.5009 > 0.4991
```

Therefore V45A must include a unified band guard that prevents the ambiguous
band from increasing before frequency pressure is allowed to matter.

Allowed form:

```text
band_guard = ReLU(band_mass - stopgrad(band_reference))^2
```

The reference must be global and preregistered, not dataset-specific. The first
allowed reference is the current v28b/v41f/v42a baseline band diagnostic if
available from the same run family, or a fixed universal ceiling derived before
implementation from existing diagnostics.

Important: this is not a selector. It does not route datasets to different
heads. It is a unified safety regularizer applied identically to every dataset.

### 5.4 Coupling Rule

Frequency pressure may only be active through a band-guarded coupling:

```text
effective_edge_freq_loss = safe_band_gate * L_edge_freq
```

where `safe_band_gate` is a differentiable, unified function of band safety,
not a dataset-specific switch.

The first implementation should prefer a conservative continuous gate:

```text
safe_band_gate = sigmoid(k * (band_reference - band_mass))
```

with fixed preregistered `k`, not tuned per dataset.

## 6. Required Diagnostics

V45A must add or preserve the following diagnostics.

Red-line and safety:

```text
legacy_head_used
v43b_enabled
v44_enabled
v44b_enabled
v45a_enabled
embedding_kmeans_acc
final_acc
embedding_posterior_gap
```

Topology band safety:

```text
v45a_band_mass
v45a_band_reference
v45a_band_guard_loss
v45a_safe_band_gate
v44_band_mass
v44_hard_ratio
v44_ambiguous_ratio
v44_clear_mass
```

Edge-local frequency:

```text
v45a_edge_freq_loss
v45a_boundary_response_mean
v45a_safe_homo_response_mean
v45a_edge_response_gap
v45a_edge_response_corr
v45a_boundary_mass
v45a_safe_homo_mass
```

Pre-HP retained diagnostics:

```text
v44b_pre_hp_response_std
v44b_pre_hp_response_p10
v44b_pre_hp_response_p90
v44b_conflict_response_corr
v44b_response_gap
v44b_postnorm_hp_energy_mean
v44b_postnorm_hp_energy_std
```

## 7. First-Stage Experiment Design

Only the following first-stage smoke is allowed after implementation sanity
checks:

```text
datasets = acm,dblp,flickr
epochs = 80
seed = 42
device = cuda
```

Command template:

```powershell
conda run -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v45a_edge_local_band_guarded_frequency --datasets "acm,dblp,flickr" --epochs 80 --device cuda --log-level WARNING
```

Dataset roles:

| Dataset | Role |
| --- | --- |
| ACM | Homophily safety and band guard stress test |
| DBLP | Preserve the dataset where v44a/v44b already passed ACC |
| Flickr | Test whether edge-local frequency keeps useful heterophilic signal |

No second-batch smoke is allowed unless all gates below pass.

## 8. First-Stage Gates

### 8.1 Red-Line Gate

Must all pass:

```text
legacy_head_used=false
v43b_enabled=false
v44_enabled=false
v44b_enabled=false
v45a_enabled=true
embedding cosine margin loss disabled
post-normalized high-pass loss disabled
global node-level v44b correlation loss disabled
no selector / no post-processing selector
```

### 8.2 Pre-HP Diagnostic Gate

Must pass on 3/3:

```text
v44b_pre_hp_response_std > 1e-4
v44b_pre_hp_response_p90 > v44b_pre_hp_response_p10
```

This ensures v45a still uses a non-degenerate frequency signal.

### 8.3 Edge-Local Frequency Gate

Must pass on at least 2/3:

```text
v45a_edge_response_gap > 0
v45a_edge_response_corr >= 0.05
```

ACM-specific anti-alignment from v44b must be fixed as a safety requirement:

```text
ACM v45a_edge_response_gap >= 0
ACM v45a_edge_response_corr >= 0
```

This is not dataset-specific training logic; it is a preregistered diagnostic
gate because ACM is the known homophily safety stress test.

### 8.4 Band Safety Gate

Must pass on 3/3:

```text
ACM band_mass <= 0.4991
DBLP band_mass <= 0.6877
Flickr band_mass <= 0.5051
```

The ACM ceiling is strict because v44b failed exactly there.

### 8.5 Posterior/Readout Safety Gate

Must pass on 3/3:

```text
abs(embedding_posterior_gap) <= 0.02
```

### 8.6 Performance Gate

Must pass on 3/3:

```text
ACM ACC >= 0.80
DBLP ACC >= 0.645
Flickr ACC >= 0.45
```

If mechanism diagnostics pass but ACM/Flickr performance does not, record
mechanistic partial pass only and stop. Do not expand to second-batch smoke.

## 9. Stop Conditions

Stop immediately after first-stage smoke if any condition occurs:

- any red-line violation
- `abs(embedding_posterior_gap) > 0.02` on any first-stage dataset
- pre-HP response degenerates on any first-stage dataset
- ACM edge-local response remains negative or anti-correlated
- ACM band mass exceeds `0.4991`
- ACM ACC remains below `0.80`
- DBLP ACC falls below `0.645`
- Flickr ACC remains below `0.45`
- improvements appear only in ACC without edge-local frequency and band-safety
  diagnostics closing the loop

Do not run:

- second-batch smoke
- full 9-dataset smoke
- 260-epoch full run
- weight sweep
- target sweep
- safety correction based on observed test scores

## 10. Expansion Condition

Only if ACM/DBLP/Flickr all pass red-line, pre-HP, edge-local frequency, band
safety, posterior safety, and performance gates may the next second-batch smoke
be considered:

```text
Wiki, BlogCatalog, Texas
```

The second batch must be preregistered separately after first-stage success.

## 11. Result Templates

### 11.1 First-Stage Gate Table

| Dataset | ACC | Emb Gap | Pre-HP Std | Edge Gap | Edge Corr | Band Mass | Safe Gate | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ACM | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| DBLP | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Flickr | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### 11.2 Mechanism Comparison

| Dataset | v44b Corr | v44b Gap | v45a Edge Corr | v45a Edge Gap | v44b Band | v45a Band | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ACM | -0.0737 | -0.0016 | TBD | TBD | 0.5009 | TBD | TBD |
| DBLP | 0.3538 | 0.0045 | TBD | TBD | 0.6863 | TBD | TBD |
| Flickr | 0.3017 | 0.0271 | TBD | TBD | 0.5033 | TBD | TBD |

## 12. Claim-Evidence Matrix

| Claim | Reviewer question | Evidence needed | Dataset | Status |
| --- | --- | --- | --- | --- |
| v44b measured a real pre-HP signal | Was v44a's target truly degenerate? | pre-HP std/p10/p90 non-degenerate; postnorm energy still 0.5 | ACM/DBLP/Flickr | done |
| global node correlation is insufficient | Why not continue v44b? | ACM corr/gap negative and band safety failure | ACM | done |
| edge-local response fixes alignment | Does local boundary response match topology conflict better? | edge corr/gap positive without ACM anti-alignment | ACM/DBLP/Flickr | planned |
| band guard preserves homophily safety | Does the mechanism stop ACM band growth? | band mass under ceilings | ACM/DBLP/Flickr | planned |
| readout remains intact | Is failure caused by posterior/readout detachment? | embedding-posterior gap within 0.02 | ACM/DBLP/Flickr | planned |

## 13. No-Fabrication Status

All v43b/v44a/v44b values cited here are copied from existing verdicts and
diagnostics. All v45a values are `TBD` and must come only from future
preregistered runs.

## 14. Next Owner

The next step should be `ccf-experiment-designer` review of this preregistration
followed by a minimal implementation plan. Direct implementation is allowed
only after confirming the band reference and edge-local diagnostics do not
create dataset-specific behavior.
