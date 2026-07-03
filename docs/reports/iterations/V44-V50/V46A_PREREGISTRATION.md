# v46a_topology_band_calibration Preregistration

This document preregisters the next mechanism after the failure of
`v45a_edge_local_band_guarded_frequency`. It is a design document only. No V46A
code has been implemented and no V46A experiment has been run.

## 1. Motivation

The sequence V43B to V45A shows that direct frontend pressure on embeddings or
frequency response is not currently reliable:

- V43B: conflict margin pressure activated too broadly, uncertainty saturated,
  margin violations were full, and `overlap_gap` was negative.
- V44A: post-normalized high-pass energy was structurally degenerate.
- V44B: pre-normalization response became measurable, but global node-level
  conflict-response coupling failed on ACM.
- V45A: edge-local band-guarded response coupling failed on all three
  first-stage datasets.

The repeated stable fact is:

```text
embedding_posterior_gap ~= 0
```

Therefore the next mechanism should not add another response-pressure loss. It
should directly calibrate the topology contraction band while keeping frequency
response as diagnostics only.

## 2. Version Name

```text
v46a_topology_band_calibration
```

Core hypothesis:

```text
If persistent ambiguous topology mass is reduced through a unified
non-selector calibration loss, while decision balance prevents collapse into a
single topology mask, then the frontend can become safer on ACM-like graphs
without reviving failed embedding or frequency-response pressure losses.
```

## 3. Hard Prohibitions

V46A must not use:

- dataset-specific module, branch, head, loss, assigner, or threshold
- legacy head
- adaptive selector or post-processing selector
- embedding cosine margin loss
- edge-level overlap margin loss
- v43b-style selective conflict gate
- post-normalized high-pass energy loss
- global node-level pre-HP correlation loss
- edge-local pre-HP frequency pressure loss
- weight sweep, target sweep, or warmup sweep
- test-set-driven safety correction

The following weights must remain zero:

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
```

V44B and V45A diagnostics may remain, but they must not contribute to loss.
The first V46A implementation must also avoid confounding the new mechanism
with older generic pressure terms:

```text
partition_spread_weight = 0.0
freq_separation_weight = 0.0
freq_ortho_weight = 0.0
```

This keeps the first smoke attributable to V46A's explicitly registered
topology-band calibration rather than to inherited band/frequency losses.

## 4. Allowed Mechanism

### 4.1 Diagnostic Inputs

V46A may observe the existing topology contraction outputs:

```text
homo_ij
hetero_ij
hard_ij
score_ij
low_threshold
high_threshold
```

It may compute diagnostics from:

```text
pre_hp_response
edge_response_gap
edge_response_corr
embedding_posterior_gap
```

but these diagnostics must not be optimized directly in V46A.

### 4.2 Ambiguous Band Calibration

Use the existing topology-contraction semantics. In the current implementation,
`homo`, `hetero`, and `hard` are the normalized soft masks returned by
`DifferentiableTopologyContraction`, and `hard` is the ambiguous-band mask used
by V44/V45 diagnostics. Therefore V46A must keep the same band-mass definition
for comparability:

```text
band_ij = hard_ij
band_mass = mean(hard_ij)
```

The first allowed calibration objective is:

```text
L_band_cal = mean(band_ij^2)
```

Rationale:

- It directly targets persistent ambiguous topology mass.
- It does not use embedding cosine or frequency response.
- It is applied uniformly to every graph.
- It stays comparable with `v44_band_mass`, `v45a_band_mass`, and the
  preregistered ACM/DBLP/Flickr band gates.

Do not use:

```text
band_ij = 1 - max(homo_ij, hetero_ij, hard_ij)
```

because `hard_ij` is already the ambiguous-band mask in this codebase.

### 4.3 Decision Balance Guard

To prevent collapse into a single mask, V46A must include a balance diagnostic
and a small guard:

```text
mask_usage = mean([mean(homo), mean(hetero), mean(hard)])
target_usage = stopgrad(mask_usage from current forward normalized to sum 1)
```

First allowed guard:

```text
usage_entropy = entropy(normalize([mean(homo), mean(hetero), mean(hard)]))
L_balance = ReLU(entropy_floor - usage_entropy)^2
```

Preregistered first constant:

```text
entropy_floor = 0.60
```

This is a collapse guard, not a selector. It does not prescribe which mask a
dataset should prefer.

### 4.4 Threshold Spread Guard

V46A may include a weak threshold-spread guard to avoid collapsing the low/high
thresholds:

```text
threshold_gap = high_threshold - low_threshold
L_spread = ReLU(min_threshold_gap - threshold_gap)^2
```

Preregistered first constant:

```text
min_threshold_gap = 0.05
```

This guard is global and does not use dataset identity.

## 5. First Implementation Constants

If implemented, use exactly one first configuration:

```text
v46a_band_cal_weight = 0.01
v46a_balance_weight = 0.005
v46a_spread_weight = 0.005
v46a_entropy_floor = 0.60
v46a_min_threshold_gap = 0.05
v46a_corr_eps = 1e-8
```

These values are fixed preregistered constants for the first smoke, not a
sweep. If this first configuration fails, stop and write a verdict.

## 6. Required Diagnostics

Red-line and safety:

```text
legacy_head_used
v43b_enabled
v44_enabled
v44b_enabled
v45a_enabled
v46a_enabled
embedding_kmeans_acc
final_acc
embedding_posterior_gap
```

Topology calibration:

```text
v46a_band_cal_loss
v46a_balance_loss
v46a_spread_loss
v46a_band_mass
v46a_homo_usage
v46a_hetero_usage
v46a_hard_usage
v46a_usage_entropy
v46a_threshold_gap
v46a_low_threshold
v46a_high_threshold
```

Retained failure-family diagnostics, diagnostic only:

```text
v44b_pre_hp_response_std
v44b_pre_hp_response_p10
v44b_pre_hp_response_p90
v45a_edge_response_gap
v45a_edge_response_corr
```

## 7. First-Stage Experiment

Only after implementation sanity checks, the first-stage smoke may run:

```text
datasets = acm,dblp,flickr
epochs = 80
seed = 42
device = cuda
```

Command template:

```powershell
conda run -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v46a_topology_band_calibration --datasets "acm,dblp,flickr" --epochs 80 --device cuda --log-level WARNING
```

No second-batch smoke is allowed unless all gates pass.

## 8. First-Stage Gates

### 8.1 Red-Line Gate

Must pass:

```text
legacy_head_used=false
v43b_enabled=false
v44_enabled=false
v44b_enabled=false
v45a_enabled=false
v46a_enabled=true
no selector / no post-processing selector
no frequency-response pressure loss
```

### 8.2 Topology Band Gate

Must pass on 3/3:

```text
ACM band_mass <= 0.4991
DBLP band_mass <= 0.6877
Flickr band_mass <= 0.5051
```

Additionally, at least 2/3 must improve against V45A observed band mass:

```text
ACM reference = 0.4998
DBLP reference = 0.6859
Flickr reference = 0.5037
```

These references are evaluation gates only. They must not be used inside
training.

### 8.3 Collapse Safety Gate

Must pass on 3/3:

```text
v46a_usage_entropy >= 0.60
v46a_threshold_gap >= 0.05
```

### 8.4 Posterior/Readout Safety Gate

Must pass on 3/3:

```text
abs(embedding_posterior_gap) <= 0.02
```

### 8.5 Performance Gate

Must pass on 3/3:

```text
ACM ACC >= 0.80
DBLP ACC >= 0.645
Flickr ACC >= 0.45
```

If topology calibration passes but performance fails, record a mechanistic
partial pass and stop.

## 9. Stop Conditions

Stop immediately after first-stage smoke if any occurs:

- any red-line violation
- any revived V43B/V44/V44B/V45A failed loss
- band gate fails
- usage entropy collapses
- threshold gap collapses
- `abs(embedding_posterior_gap) > 0.02`
- ACM ACC remains below `0.80`
- DBLP ACC falls below `0.645`
- Flickr ACC remains below `0.45`

Do not run:

- second-batch smoke
- full 9-dataset smoke
- 260-epoch full run
- weight sweep
- entropy floor sweep
- threshold-gap sweep

## 10. Result Templates

### 10.1 First-Stage Gate Table

| Dataset | ACC | Emb Gap | Band Mass | Usage Entropy | Threshold Gap | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| ACM | TBD | TBD | TBD | TBD | TBD | TBD |
| DBLP | TBD | TBD | TBD | TBD | TBD | TBD |
| Flickr | TBD | TBD | TBD | TBD | TBD | TBD |

### 10.2 Mechanism Comparison

| Dataset | V45A Band | V46A Band | V45A ACC | V46A ACC | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| ACM | 0.4998 | TBD | 0.7084 | TBD | TBD |
| DBLP | 0.6859 | TBD | 0.6596 | TBD | TBD |
| Flickr | 0.5037 | TBD | 0.3694 | TBD | TBD |

## 11. No-Fabrication Status

All V43B/V44/V45 values cited here come from existing verdicts and diagnostics.
V46A has not been implemented or run. All V46A results are `TBD`.
