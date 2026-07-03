# V49A Route Decision After V48A Audit

This document records the route decision after the diagnostic outcome of
`v48a_topology_dynamics_audit`. It is a mechanism-design document only. It does
not change code, run experiments, or report new results.

## 1. Evidence Chain

| Version | Mechanism | Main finding |
| --- | --- | --- |
| V43B | direct conflict / embedding pressure | gate too broad, frontend damaged |
| V44B | node-level pre-HP response coupling | response exists, coupling fails on ACM |
| V45A | edge-local response coupling | edge response gap/corr fail on 3/3 |
| V46A | direct hard-band penalty | no collapse, but hard band not reduced on ACM/Flickr |
| V47A | posterior-guided hard-band CE | targets exist, but band and ACC worsen vs V46A |
| V48A | topology dynamics audit | masks move, but ACM/Flickr move against target direction |

Latest V48A audit:

| Dataset | ACC | dHomo | dHetero | dHard | dScore | hard corr | Direction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ACM | 0.6450 | 0.0328 | 0.0695 | 0.0792 | 0.0439 | 0.8594 | FAIL |
| DBLP | 0.6522 | 0.0668 | 0.0489 | 0.1004 | 0.0633 | 0.8426 | PASS |
| Flickr | 0.3401 | 0.0338 | 0.0168 | 0.0428 | 0.0226 | 0.9728 | FAIL |

Critical conclusion:

```text
The topology masks are movable, but not reliably semantically steerable.
```

This rules out the simplest "frozen topology" explanation. The next bottleneck
is the transition geometry: the current single-score ordered-threshold
contraction moves mask values but does not reliably move hard edges toward the
intended homo/hetero directions.

## 2. Scientific Interpretation

The current topology contraction uses a single scalar score and ordered
thresholds to derive three masks:

```text
homo   = sigmoid((score - high) / tau)
hetero = sigmoid((low - score) / tau)
hard   = sigmoid((score - low) / tau) * sigmoid((high - score) / tau)
```

This couples two different decisions:

```text
orientation: homo vs hetero
clarity: resolved vs hard
```

V48A suggests that this coupling is not merely stiff. It can move, but the
movement can be directionally wrong on ACM/Flickr. Therefore the next mechanism
should change the transition geometry itself, not add another external
teacher, target, or penalty.

## 3. Rejected Continuations

Do not continue:

- V48A second-batch audit
- V48A full audit
- V48A sample-size sweep
- V47A stronger CE
- V47A quantile/defer sweep
- another external hard-edge teacher loss
- another scalar hard-band penalty
- another pre-HP/frequency response pressure loss

The following families remain rejected as primary loss targets:

- embedding cosine/margin pressure
- pre-HP response pressure
- direct hard-mass scalar penalty
- posterior-guided hard-band CE pressure

## 4. Candidate Routes

### Candidate A: Decoupled Orientation-Clarity Topology Transition

Name:

```text
v49a_reparameterized_topology_transition
```

Core idea:

Replace the single-score ordered-threshold mask geometry with a differentiable
simplex that separates:

```text
orientation_ij = homo-vs-hetero direction
clarity_ij     = resolved-vs-hard confidence
```

First mechanism sketch:

```text
clear_ij = sigmoid(clarity_logit_ij / tau_clear)
orient_ij = sigmoid(orientation_logit_ij / tau_orient)

homo_ij   = clear_ij * orient_ij
hetero_ij = clear_ij * (1 - orient_ij)
hard_ij   = 1 - clear_ij
```

Why it fits V48A:

- Hard-to-homo and hard-to-hetero movement becomes locally controllable through
  orientation without forcing all hard movement through the same scalar band.
- Hard persistence becomes a clarity decision, not the byproduct of lying
  between two ordered thresholds.
- The mechanism changes the parameterization rather than adding another target
  loss.

Main risk:

- Larger implementation blast radius than prior loss-only variants.
- The new two-coordinate topology head could become underconstrained unless
  diagnostics verify mask usage, directionality, and posterior/readout safety.

### Candidate B: Learnable Temperature / Threshold Geometry Only

Name:

```text
v49a_learned_transition_temperature
```

Core idea:

Keep the single scalar score, but learn or adapt temperature and threshold
geometry so the hard band can move more flexibly.

Why it fits:

- Smaller implementation change.

Risk:

- V48A already showed movement exists; merely changing temperature may preserve
  the same wrong-direction coupling.

### Candidate C: Diagnostic Recenter

Name:

```text
v49a_diagnostic_recenter
```

Core idea:

Stop topology mechanism additions and recenter the paper around the strongest
safe baseline plus the V43B-V48A negative evidence sequence.

Risk:

- Conservative and may not provide a successful new mechanism.

## 5. Recommended Route

Recommended:

```text
v49a_reparameterized_topology_transition
```

Reason:

V48A moved the failure boundary from "targets missing" or "topology frozen" to:

```text
topology transition is not directionally reliable.
```

The most direct response is to decouple the transition coordinates themselves.
V49A should therefore test whether a unified orientation/clarity topology
simplex can produce more reliable hard-to-homo and hard-to-hetero transitions
without reviving failed external pressure losses.

## 6. Required Design Principles For V49A

V49A must:

- use one unified parameterization for all datasets
- remain end-to-end differentiable
- avoid dataset-specific thresholds, weights, branches, or modules
- avoid legacy head / selector / post-processing selector
- keep V47A/V48A target diagnostics diagnostic-only
- disable V43B/V44/V44B/V45A/V46A/V47A failed losses
- not optimize labels
- not use test-set-driven correction
- not run a sweep before first verdict

V49A must not be presented as a performance success unless it passes both
transition-direction and safety gates.

## 7. First-Stage Gate Shape

Datasets remain:

```text
ACM, DBLP, Flickr
```

Required:

- red-line pass
- no revived failed losses
- posterior/readout safety
- non-degenerate usage of homo/hetero/hard
- V48A-style movement diagnostics present
- directional consistency improves over V48A:

```text
targeted_homo_delta > 0
targeted_hetero_delta > 0
targeted_hard_delta >= 0
```

At minimum, directional consistency should pass on 2/3 datasets and must not
repeat ACM/Flickr both moving negative on homo and hetero targets.

Performance and band mass should be recorded, but V49A should stop after the
first-stage smoke if directional consistency fails.

## 8. Next Step

Next owner:

```text
ccf-experiment-designer
```

Write a full `V49A_PREREGISTRATION.md` before implementation. Do not implement
V49A from this route note alone.

## 9. No-Fabrication Status

All V48A numbers cited here come from `V48A_FIRST_AUDIT_VERDICT.md` and
diagnostics. V49A has not been implemented or run. All V49A outputs are `TBD`.
