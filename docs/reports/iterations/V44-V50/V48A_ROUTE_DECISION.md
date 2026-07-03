# V48A Route Decision After V47A Failure

This document records the route decision after the preregistered failure of
`v47a_posterior_guided_band_resolution`. It is a design document only. It does
not change code, run experiments, or report new results.

## 1. Evidence Chain

| Version | Mechanism | Main finding |
| --- | --- | --- |
| V43B | direct conflict/embedding pressure | gate too broad, margin pressure damages frontend |
| V44B | node-level pre-HP response coupling | response exists, coupling fails on ACM |
| V45A | edge-local response coupling | edge response gap/corr fail on 3/3 |
| V46A | direct hard-band penalty | no collapse, but hard band not reduced on ACM/Flickr |
| V47A | posterior-guided hard-band CE | targets exist, but band and ACC worsen vs V46A |

V47A first-stage result:

| Dataset | ACC | Emb Gap | Band | Homo Tgt | Hetero Tgt | Defer Tgt | Effective Tgt |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ACM | 0.6651 | 0.0000 | 0.5215 | 0.2822 | 0.1957 | 0.3089 | 0.7869 |
| DBLP | 0.6485 | 0.0000 | 0.6853 | 0.2756 | 0.1753 | 0.3223 | 0.7731 |
| Flickr | 0.3537 | 0.0000 | 0.5100 | 0.1368 | 0.2843 | 0.2453 | 0.6664 |

Critical conclusion:

```text
V47A did not fail because targets were absent.
```

Targets were non-degenerate and hard-weighted effective target mass was high.
The resolution loss was also nonzero. Yet band mass worsened on 3/3 against
V46A and performance worsened on 3/3 against V46A.

## 2. Scientific Interpretation

The failure boundary has moved.

Earlier we did not know whether the model lacked:

1. a high-pass signal,
2. a topology target,
3. a semantic target for hard edges, or
4. a movable topology parameterization.

V44B established that a high-pass diagnostic exists. V47A established that a
semantic posterior-guided target can be made non-degenerate. Therefore the next
suspect is the topology contraction parameterization and training dynamics:

```text
The masks may be resistant to external calibration losses.
```

In other words, the unresolved issue may not be "what target should hard edges
follow", but "can the current contraction/threshold parameterization actually
move hard edges in response to such targets without harming representation".

## 3. Rejected Continuations

Do not continue:

- V47A second-batch smoke
- V47A full run
- V47A weight sweep
- posterior quantile sweep
- defer threshold sweep
- stronger posterior-guided CE
- another external teacher loss that pushes `homo/hetero/hard` directly

The following families are now rejected as primary loss targets:

- embedding cosine/margin pressure
- pre-HP response pressure
- hard-mass scalar penalty
- posterior-guided hard-band CE pressure

## 4. Candidate Routes

### Candidate A: Topology Contraction Dynamics Audit

Name:

```text
v48a_topology_dynamics_audit
```

Core idea:

Do not add another target loss first. Instrument and test whether the current
topology contraction can respond to gradients at all:

- mask gradient norms
- score-to-mask sensitivity
- threshold movement
- hard-to-homo/hetero transition rate
- whether V47A resolution loss changes masks during training

Pros:

- Directly addresses the new bottleneck.
- Avoids adding another loss family blindly.
- Can explain why several valid-looking targets failed.

Risk:

- It is more diagnostic than performance-driven, so it may not immediately
  produce a publishable mechanism.

### Candidate B: Reparameterized Topology Contraction

Name:

```text
v48a_reparameterized_topology_contraction
```

Core idea:

Replace or augment the current hard-band mask parameterization with a smoother
or more direct simplex parameterization so topology masks can move under
calibration targets.

Pros:

- Changes the suspected failure point.
- Keeps the paper's core frontend contribution.

Risk:

- Larger implementation blast radius.
- Must preserve unified end-to-end pipeline and avoid dataset-specific logic.

### Candidate C: Diagnostic Recenter

Name:

```text
v48a_diagnostic_recenter
```

Core idea:

Stop mechanism additions and recenter on the strongest safe baseline, using the
failed V43B-V47A sequence as ablation evidence.

Pros:

- Low risk.
- Prevents further loss accumulation.

Risk:

- May leave the paper without a new successful mechanism.

## 5. Recommended Route

Recommended next step:

```text
v48a_topology_dynamics_audit
```

Reason:

V47A had all ingredients expected for a meaningful hard-band resolution target:

- red-line pass
- posterior gap pass
- non-degenerate hard-weighted targets
- nonzero resolution loss

Yet masks did not move in the desired direction. Therefore implementation of a
new objective before auditing topology dynamics would be weakly justified.

V48A should answer a narrower question:

```text
Is the current topology contraction module actually responsive to calibration
gradients in the way the research story assumes?
```

## 6. V48A Preregistration Target

Proposed document:

```text
V48A_PREREGISTRATION.md
```

Proposed version:

```text
v48a_topology_dynamics_audit
```

Primary hypothesis:

```text
If the current topology contraction cannot show measurable hard-to-homo/hetero
transition or score/threshold sensitivity under existing calibration signals,
then further external topology losses should stop until the contraction
parameterization is redesigned.
```

This should be a diagnostic/audit variant, not a performance-claim variant.

## 7. Required Audit Diagnostics

V48A should add diagnostics such as:

```text
mask_grad_norm_homo
mask_grad_norm_hetero
mask_grad_norm_hard
score_grad_norm
threshold_grad_norm
hard_to_homo_transition_rate
hard_to_hetero_transition_rate
hard_persistence_rate
score_delta_mean
threshold_delta_mean
```

If gradient hooks are too invasive, first implementation may use epoch-to-epoch
transition diagnostics:

```text
mean_abs_delta_homo
mean_abs_delta_hetero
mean_abs_delta_hard
hard_rank_correlation_prev
hard_mass_delta
```

## 8. First-Stage Gate Shape

V48A should not be judged primarily by ACC. It should be judged by whether it
proves or falsifies topology movability:

- red-line gate must pass
- no failed loss family revived
- posterior gap remains safe
- audit diagnostics present and finite
- at least one topology-movement diagnostic is non-degenerate on 3/3

Performance can be recorded but should not be used to expand runs.

## 9. Next Step

Next owner:

```text
ccf-experiment-designer
```

Write `V48A_PREREGISTRATION.md` before implementation. Do not directly implement
from this route note.

## 10. No-Fabrication Status

All cited V47A values come from `V47A_FIRST_SMOKE_VERDICT.md` and diagnostics.
V48A has not been implemented or run. All V48A outputs are `TBD`.
