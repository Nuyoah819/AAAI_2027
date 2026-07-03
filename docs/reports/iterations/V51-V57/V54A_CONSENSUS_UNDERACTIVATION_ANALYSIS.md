# V54A Consensus Underactivation Analysis

This document analyzes why `v54a_consensus_bounded_residual_anchor` still stops
after the first mixed-stress gate. It follows `ccf-idea-optimizer` standard
rescue mode: diagnose the mechanism and preregister the next idea before any
new implementation.

No new experiment is run in this document. No V54A beta-bound, schedule,
threshold, formula, or V50A anchor hyperparameter sweep is authorized here.

## 1. Evidence Basis

Local artifacts:

```text
V53A_FIRST_MIXED_STRESS_VERDICT.md
V53A_RESIDUAL_OVEREXPOSURE_ANALYSIS.md
V54A_PREREGISTRATION.md
V54A_IMPLEMENTATION_REVIEW.md
V54A_CONNECTIVITY_VERDICT.md
V54A_FIRST_MIXED_STRESS_VERDICT.md
results/archive/v51-v57/unified_aptc_9datasets_v54a_consensus_bounded_residual_anchor_diagnostics.jsonl
```

V54A already stops by preregistered gates. This document extracts the next
rescue question.

## 2. What V54A Fixed

V54A directly fixed the main V53A failure:

| Dataset | V53A ACC | V54A ACC | Delta |
| --- | ---: | ---: | ---: |
| ACM | 0.9005 | 0.9051 | +0.0046 |
| DBLP | 0.6806 | 0.6798 | -0.0008 |
| Flickr | 0.3675 | 0.3737 | +0.0062 |
| Texas | 0.7322 | 0.7322 | +0.0000 |
| Squirrel | 0.2119 | 0.3005 | +0.0886 |
| Chameleon | 0.3421 | 0.3364 | -0.0057 |

The key repair is Squirrel:

```text
V53A overexposed Squirrel to a weak anchor and dropped to 0.2119.
V54A lowers beta on weak-consensus Squirrel nodes and restores ACC to 0.3005.
```

V54A also passes:

| Gate | V54A verdict |
| --- | --- |
| Red-line | PASS |
| Residual bound | PASS |
| Posterior/readout safety | PASS |
| Anchor usefulness | PASS |
| Heterophily stress | PASS |

Interpretation:

```text
Consensus-bounded residual is the right safety direction. It prevents the
global residual from forcing weak-anchor assignments on Squirrel.
```

## 3. What V54A Still Fails

The remaining failure is reliability non-collapse:

| Dataset | Rel Mean | Reliable Ratio | Effective Mass | Gate |
| --- | ---: | ---: | ---: | --- |
| ACM | 0.1390 | 0.1481 | 0.1390 | PASS |
| DBLP | 0.0749 | 0.0000 | 0.0749 | FAIL |
| Flickr | 0.0056 | 0.0000 | 0.0056 | weak-anchor non-use |
| Texas | 0.0562 | 0.0055 | 0.0562 | FAIL |
| Squirrel | 0.0852 | 0.0525 | 0.0852 | PASS |
| Chameleon | 0.2123 | 0.4668 | 0.2123 | PASS |

The failure is narrow:

```text
DBLP is just below the 0.08 mass floor.
Texas remains below the mass floor.
No safety gate fails.
ACM/DBLP/Squirrel performance floors all pass.
```

This means the next route should not discard V54A. It should preserve the
consensus-bounded safety idea while fixing underactivation on medium-consensus
nodes.

## 4. Mechanistic Cause

V54A uses hard argmax consensus:

```text
h_q_i = 1[argmax(q_refined_i) == argmax(q_spec_i)]
h_e_i = 1[argmax(q_embed_i) == argmax(q_spec_i)]
h_i = 0.5 * h_q_i + 0.5 * h_e_i
beta_i = beta_min + (beta_max - beta_min) * h_i
```

This creates three beta levels:

```text
0.35, 0.525, 0.70
```

That discreteness is useful for Squirrel, where hard consensus is weak:

| Dataset | Hard Q | Hard Embed | Hard Both | Beta Mean | ACC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Squirrel | 0.1375 | 0.0598 | 0.0386 | 0.3845 | 0.3005 |

But it underactivates medium-consensus datasets:

| Dataset | Hard Q | Hard Embed | Hard Both | Beta Mean | Rel Mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| DBLP | 0.4797 | 0.4548 | 0.4318 | 0.5135 | 0.0749 |
| Texas | 0.3989 | 0.3989 | 0.3934 | 0.4896 | 0.0562 |

DBLP and Texas have nontrivial soft agreement:

| Dataset | Q-Anchor Component | Embed-Anchor Component | Agreement Mean |
| --- | ---: | ---: | ---: |
| DBLP | 0.0413 | 0.0273 | 0.0343 |
| Texas | 0.0673 | 0.0547 | 0.0610 |

The hard consensus ignores this continuous evidence except at exact argmax
matches. As a result, nodes with partial alignment receive the same low or
middle residual as nodes with nearly no useful agreement.

## 5. Why Simple Fixes Are Rejected

Do not do:

- raise `beta_min` after seeing DBLP/Texas;
- sweep `beta_min` or `beta_max`;
- relax the reliability mass threshold;
- add a Texas/DBLP-specific branch;
- select V53A for DBLP/Texas and V54A for Squirrel;
- use labels, target metrics, or validation performance in reliability;
- add geometry fallback or signed topology-mask anchor in the next step;
- change final labels away from `q_refined`.

Reason:

```text
The failure is not that V54A is too conservative everywhere. It is correctly
conservative on Squirrel and Flickr. The failure is that hard consensus is too
coarse for medium-consensus graphs.
```

## 6. Rescue Question For V55A

The next research question should be:

```text
Can consensus-bounded residual preserve V54A's weak-anchor safety while using
continuous soft agreement to avoid underactivating medium-consensus nodes?
```

This shifts the mechanism from:

```text
hard argmax consensus beta
```

to:

```text
soft consensus beta
```

## 7. Candidate V55A Mechanism

Recommended route name:

```text
v55a_soft_consensus_bounded_residual_anchor
```

Key idea:

```text
Keep V54A's bounded residual path, but replace hard consensus h_i with a
continuous stop-gradient consensus score based on already computed soft
posterior-anchor and embedding-anchor agreement.
```

Candidate node-level soft consensus:

```text
soft_i = clamp(0.5 * qa_i_norm + 0.5 * ea_i_norm, 0, 1)
```

Because raw soft agreement is small on all datasets, do not map it directly.
Use a fixed concave lift:

```text
c_i = sqrt(soft_i)
beta_i = beta_min + (beta_max - beta_min) * c_i
```

Fixed constants for preregistration:

```text
beta_min = 0.35
beta_max = 0.70
soft_power = 0.50
```

Why this is a new mechanism rather than a sweep:

- beta remains node-level and bounded;
- the signal is continuous, not an argmax threshold;
- constants are fixed before implementation;
- the score uses already available detached agreement components;
- final labels remain `q_refined`.

## 8. Required Safeguards

Any V55A preregistration must include:

- fixed `soft_power = 0.50`;
- fixed `beta_min = 0.35`, `beta_max = 0.70`;
- no power sweep or beta-bound sweep;
- inherited V52A/V53A/V54A schedule;
- inherited V50A anchor construction;
- no dataset-specific branch;
- no final selector;
- explicit Squirrel ACC floor `>= 0.2800`;
- reliability mass and safety gates;
- anchor usefulness gates;
- soft-consensus diagnostics.

## 9. Expected Failure Mode

V55A may still stop if:

- soft consensus over-lifts Squirrel and repeats V53A overexposure;
- soft consensus remains too small and DBLP/Texas stay under 0.08 reliability
  mass;
- Texas loses anchor usefulness because weighted agreement keeps decreasing;
- Chameleon reliability or ACC drops due to excessive residual exposure.

These risks should be gate-tested in one mixed-stress run only after
implementation review and connectivity pass.

## 10. No-Fabrication Status

All numbers in this document come from local V53A/V54A verdict files and
diagnostics. No V55A code or result exists.
