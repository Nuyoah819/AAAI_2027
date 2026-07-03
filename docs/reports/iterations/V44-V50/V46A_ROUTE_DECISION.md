# V46A Route Decision After V45A Failure

This document records the research decision after the preregistered failure of
`v45a_edge_local_band_guarded_frequency`. It is a route-design document only.
It does not change code, run experiments, or report new unrecorded results.

## 1. Inputs

Primary evidence:

- `V43B` first-stage diagnostics: broad conflict activation, saturated
  uncertainty, full margin violation, negative overlap gap, high band mass, and
  no posterior/readout break.
- `V44A_FIRST_SMOKE_VERDICT.md`: post-normalized high-pass energy is degenerate.
- `V44B_FIRST_SMOKE_VERDICT.md`: pre-normalization response is measurable, but
  global node-level response correlation fails on ACM.
- `V45A_FIRST_SMOKE_VERDICT.md`: edge-local band-guarded response coupling also
  fails.

No new run is introduced here.

## 2. V45A Gate Recap

| Dataset | ACC | Emb Gap | Pre-HP Std | Edge Gap | Edge Corr | Band Mass | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ACM | 0.7084 | 0.0000 | 0.0257 | -0.0057 | -0.2068 | 0.4998 | FAIL |
| DBLP | 0.6596 | 0.0000 | 0.0109 | -0.0028 | -0.1853 | 0.6859 | partial ACC only |
| Flickr | 0.3694 | 0.0001 | 0.0472 | -0.0050 | -0.0314 | 0.5037 | FAIL |

Passes:

- red-line and implementation sanity
- same-run frozen warmup reference
- pre-HP response non-degeneracy
- posterior/readout safety

Failures:

- edge-local frequency gate fails on 3/3
- ACM band safety still fails slightly
- performance gate fails on ACM and Flickr

## 3. Scientific Conclusion

V45A disproves the current family of frequency-response pressure losses.

The failure should not be explained as:

- weight too small
- warmup too short
- target edge gap too strict
- gate too conservative

Reason: `v45a_safe_band_gate` was active enough on ACM and Flickr, yet
`v45a_edge_response_gap` and `v45a_edge_response_corr` were still negative on
all three datasets. The issue is the direction of the target, not just its
strength.

Combined interpretation:

1. Pre-normalization response remains valuable as a diagnostic.
2. It should not currently be used as a direct optimization pressure.
3. Both node-level and edge-local response coupling failed to produce aligned
   mechanism behavior.
4. The remaining bottleneck is likely topology calibration and ambiguous-band
   decision quality, not lack of high-pass response magnitude.
5. Posterior/readout remains safe; the failure is still in the frontend
   topology/frequency interface.

## 4. Route Rejection

Do not continue:

- V45A second-batch smoke
- V45A full run
- V45A weight sweep
- `v45a_band_gate_k` sweep
- `v45a_warmup_epochs` sweep
- `v45a_target_edge_gap` sweep
- any V45A soft/strict/high/low variant

Do not create another variant whose main change is still:

```text
make pre_hp_response larger on conflict/boundary edges
```

That family has now failed in two forms:

- V44B: global node-level coupling
- V45A: edge-local boundary-vs-safe coupling

## 5. Candidate Next Mechanisms

### Candidate A: Topology Band Calibration Without Frequency Pressure

Name:

```text
v46a_topology_band_calibration
```

Core idea:

Use pre-HP response and view disagreement only as diagnostics, while the loss
directly calibrates topology band allocation. The goal is not to push frequency
response; it is to reduce unsafe ambiguous mass and improve boundary decisions.

Mechanism sketch:

```text
L_band_calibration = penalty for persistent ambiguous band mass
L_decision_balance = prevent collapse into all-homo or all-hetero masks
pre_hp_response = diagnostic only
```

Pros:

- Directly targets the recurring unresolved variable: band mass.
- Avoids the repeatedly failed frequency-pressure objective.
- Remains in the core frontend topology-contraction innovation.

Risk:

- If too strong, it may recreate v43b-style broad pressure. The first version
  must avoid margin/cosine pressure and must not optimize `overlap_gap`.

### Candidate B: Posterior-Topology Consistency Calibration

Name:

```text
v46a_posterior_topology_consistency
```

Core idea:

Do not pressure embedding or frequency response directly. Instead, align the
topology masks with posterior-view agreement consistency. For example, if an
edge is confidently same-posterior under the refined head, topology should not
remain hard/ambiguous; if an edge is uncertain, topology should avoid overhard
commitment.

Mechanism sketch:

```text
posterior_agreement_ij = stopgrad(sim(q_refined_i, q_refined_j))
topology_confidence_ij = homo_ij + hetero_ij
L_ptc = calibration(topology_confidence, posterior_agreement, uncertainty)
```

Pros:

- Uses the fact that `embedding_posterior_gap` is near zero, so posterior is not
  detached from embedding.
- Shifts the target from raw high-pass magnitude to semantic agreement.
- May give `q_low/q_high/q_refined` disagreement real meaning.

Risk:

- Must avoid becoming a post-processing selector or a posterior-only shortcut.
- Posterior signals must be stop-gradient or carefully gated to avoid circular
  self-confirmation.

### Candidate C: Diagnostic-Only Freeze And Return To Strongest Safe Baseline

Name:

```text
v46a_diagnostic_only_baseline_recenter
```

Core idea:

Stop adding frontend pressure losses. Keep V44B/V45A diagnostics for analysis
only and recenter on the strongest safe baseline variant, then design a smaller
calibration around existing successful mechanisms.

Pros:

- Reduces accumulated loss complexity.
- Avoids continuing a failed family.

Risk:

- Does not immediately propose a new mechanism; may be too conservative for
  the paper's novelty path.

## 6. Recommended Route

Recommended next route:

```text
v46a_topology_band_calibration
```

Reason:

The strongest repeated failure signal is not absence of frequency response; it
is unreliable topology band allocation:

- v43b: conflict gate activated too broadly and violation ratio saturated.
- v44a: band mass did not clearly improve.
- v44b: ACM band safety failed.
- v45a: ACM band safety still failed, while edge-local response went negative.

At the same time, posterior/readout safety has repeatedly passed:

```text
embedding_posterior_gap ~= 0
```

Therefore the next mechanism should stop pushing high-pass response and instead
ask whether the topology contraction itself can be calibrated without
dataset-specific routing, legacy heads, selectors, or embedding separation.

## 7. V46A Preregistration Target

Proposed version name:

```text
v46a_topology_band_calibration
```

Primary hypothesis:

```text
If the ambiguous topology band is calibrated directly with a unified
non-selector loss that discourages persistent ambiguous mass while preserving
decision balance, then frontend representations can improve without relying on
failed frequency-response pressure or embedding separation losses.
```

Must remain diagnostic only:

```text
v44b_pre_hp_response_*
v45a_edge_response_gap
v45a_edge_response_corr
```

Must be disabled:

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

## 8. Minimum First-Stage Gate

First-stage datasets remain:

```text
ACM, DBLP, Flickr
```

Required gates:

- red-line pass
- no legacy head / selector / post-processing
- no revived v43b/v44/v44b/v45a failed losses
- `abs(embedding_posterior_gap) <= 0.02`
- band safety:

```text
ACM band_mass <= 0.4991
DBLP band_mass <= 0.6877
Flickr band_mass <= 0.5051
```

- performance:

```text
ACM ACC >= 0.80
DBLP ACC >= 0.645
Flickr ACC >= 0.45
```

No second-batch smoke unless all gates pass.

## 9. Next Owner

Next step should be:

```text
ccf-experiment-designer
```

to write a full `V46A_PREREGISTRATION.md` before any code implementation.

Do not directly implement V46A yet. The exact topology-band calibration loss
must be preregistered first to avoid recreating v43b broad pressure under a new
name.

## 10. No-Fabrication Status

All numbers cited here come from existing verdicts and diagnostics. V46A has
not been implemented or run. All V46A results are `TBD`.
