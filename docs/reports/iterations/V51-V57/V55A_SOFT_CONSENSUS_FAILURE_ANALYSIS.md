# V55A Soft Consensus Failure Analysis

This document analyzes why `v55a_soft_consensus_bounded_residual_anchor` stops
after the first mixed-stress gate. It follows `ccf-idea-optimizer` rescue mode:
diagnose the mechanism before any new implementation.

No new experiment is run in this document. No V55A soft-power, beta-bound,
schedule, threshold, formula, or V50A anchor hyperparameter sweep is authorized.

## 1. Evidence Basis

Local artifacts:

```text
V54A_FIRST_MIXED_STRESS_VERDICT.md
V54A_CONSENSUS_UNDERACTIVATION_ANALYSIS.md
V55A_PREREGISTRATION.md
V55A_IMPLEMENTATION_REVIEW.md
V55A_CONNECTIVITY_VERDICT.md
V55A_FIRST_MIXED_STRESS_VERDICT.md
results/archive/v51-v57/unified_aptc_9datasets_v55a_soft_consensus_bounded_residual_anchor_diagnostics.jsonl
```

V55A already stops by preregistered gates. This document extracts the next
rescue question.

## 2. What V55A Preserved

V55A remained clean on red-line and safety:

| Gate | V55A Verdict |
| --- | --- |
| Red-line | PASS |
| Soft-residual bound | PASS |
| Posterior/readout safety | PASS |
| Anchor usefulness | PASS |
| Heterophily stress | PASS |

It also preserved the key floors:

| Dataset | V55A ACC | Required Floor | Gate |
| --- | ---: | ---: | --- |
| ACM | 0.8995 | 0.8888 | PASS |
| DBLP | 0.6847 | 0.6610 | PASS |
| Squirrel | 0.3021 | 0.2800 | PASS |

Interpretation:

```text
The bounded residual anchor family is still safe enough to study. V55A did not
reintroduce the V53A Squirrel overexposure failure.
```

## 3. What V55A Failed

The reliability non-collapse gate fails:

| Dataset | Rel Mean | Effective Mass | Reliable Ratio | Gate |
| --- | ---: | ---: | ---: | --- |
| ACM | 0.1004 | 0.1004 | 0.0241 | PASS mass, FAIL ratio |
| DBLP | 0.0589 | 0.0589 | 0.0000 | FAIL mass |
| Flickr | 0.0051 | 0.0051 | 0.0000 | weak-anchor non-use |
| Texas | 0.0490 | 0.0490 | 0.0000 | FAIL mass |
| Squirrel | 0.0820 | 0.0820 | 0.0458 | PASS mass, FAIL ratio |
| Chameleon | 0.2067 | 0.2067 | 0.5081 | PASS |

The mass gate passes on only 3/6. The reliable-node ratio gate passes on only
1/6.

## 4. Mechanistic Cause

V55A replaced V54A hard consensus:

```text
hard_i = 0.5 * 1[argmax(q_refined_i)=argmax(q_spec_i)]
       + 0.5 * 1[argmax(q_embed_i)=argmax(q_spec_i)]
```

with soft consensus:

```text
soft_i = 0.5 * qa_i_norm + 0.5 * ea_i_norm
c_i = sqrt(soft_i)
beta_i = 0.35 + 0.35 * c_i
```

This was meant to lift medium-consensus nodes. Instead, the soft values were
too small:

| Dataset | Soft Mean | Beta Mean | V54A Beta Mean | V55A Rel Mean |
| --- | ---: | ---: | ---: | ---: |
| ACM | 0.0873 | 0.4481 | 0.6636 | 0.1004 |
| DBLP | 0.0351 | 0.3970 | 0.5135 | 0.0589 |
| Flickr | 0.0010 | 0.3575 | 0.3866 | 0.0051 |
| Texas | 0.0603 | 0.4131 | 0.4896 | 0.0490 |
| Squirrel | 0.0172 | 0.3697 | 0.3845 | 0.0820 |
| Chameleon | 0.0798 | 0.4194 | 0.4405 | 0.2067 |

Key observation:

```text
V55A reduced beta on the very datasets it was supposed to lift. DBLP and Texas
do not have enough raw dot-product agreement for a pure soft score to work.
```

## 5. Why V54A Hard Consensus Still Matters

V54A's hard argmax consensus was coarse, but it carried useful discrete
alignment evidence:

```text
If q_refined or q_embed already agrees with the spectral anchor at argmax
level, the node can tolerate more residual anchor availability even when the
soft dot-product score is modest.
```

V55A threw this evidence away. The result:

```text
Safety remains, but useful medium-confidence mass disappears.
```

This suggests the next route should not be "soft instead of hard." It should be:

```text
hard safety floor + soft compensation.
```

## 6. Rejected Fixes

Do not do:

- increase `soft_power` or sweep it;
- raise `beta_min` after seeing DBLP/Texas;
- choose V54A for some datasets and V55A for others;
- relax the reliability mass gate;
- add dataset-specific compensation for DBLP/Texas;
- add a low-reliability geometry fallback in the next step;
- change V50A anchor rank, temperature, filter steps, or refresh;
- use anchor labels, KMeans, S2CAG/ELSS, or legacy head as final output.

Reason:

```text
These actions would tune around the observed test behavior rather than test a
new mechanism.
```

## 7. Rescue Question For V56A

The next question should be:

```text
Can the model preserve V54A's hard-consensus safety while adding a fixed,
detached soft compensation term for medium-evidence nodes, so DBLP/Texas regain
anchor mass without exposing Squirrel to a weak anchor?
```

Recommended route name:

```text
v56a_hybrid_consensus_floor_residual_anchor
```

Core design direction:

```text
Use V54A hard consensus as the base beta signal.
Add only positive soft compensation when soft consensus exceeds the hard
consensus floor.
Keep beta in [0.35, 0.70].
Keep all reliability/beta signals detached.
Keep final labels as q_refined.
```

## 8. Expected Failure Mode

V56A may still stop if:

- the hybrid compensation still cannot lift DBLP/Texas above 0.08 mass;
- the compensation lifts Squirrel too much and ACC drops below 0.2800;
- Texas weighted anchor agreement keeps decreasing;
- reliability ratio remains too sparse even when mean mass improves.

These risks must be gate-tested in one mixed-stress run only after
implementation review and connectivity pass.

## 9. No-Fabrication Status

All numbers in this document come from local V54A/V55A verdict files and
diagnostics. No V56A code or result exists.
