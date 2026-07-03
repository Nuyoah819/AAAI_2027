# V53A Residual Overexposure Analysis

This document analyzes why `v53a_residual_curriculum_spectral_anchor` still
stops after the first mixed-stress gate. It follows `ccf-idea-optimizer`
exploratory rescue mode: diagnose the mechanism before proposing new code.

No new experiment is run in this document. No V53A beta, schedule, threshold,
formula, or V50A anchor hyperparameter sweep is authorized here.

## 1. Evidence Basis

Local artifacts:

```text
V52A_FIRST_MIXED_STRESS_VERDICT.md
V52A_LATE_RELIABILITY_COLLAPSE_ANALYSIS.md
V53A_PREREGISTRATION.md
V53A_IMPLEMENTATION_REVIEW.md
V53A_CONNECTIVITY_VERDICT.md
V53A_FIRST_MIXED_STRESS_VERDICT.md
results/archive/v51-v57/unified_aptc_9datasets_v53a_residual_curriculum_spectral_anchor_diagnostics.jsonl
```

V53A already stops by preregistered gates. This document extracts the next
rescue question.

## 2. What V53A Fixed

V53A fixed the major V52A late-reliability collapse:

| Dataset | V52A Rel @80 | V53A Rel @80 |
| --- | ---: | ---: |
| ACM | 0.0197 | 0.1080 |
| DBLP | 0.0055 | 0.0722 |
| Texas | 0.0075 | 0.0582 |
| Squirrel | 0.0068 | 0.1074 |
| Chameleon | 0.0376 | 0.2441 |

It also passed several gates that V51A/V52A could not pass together:

| Gate | V53A verdict |
| --- | --- |
| Red-line | PASS |
| Residual schedule | PASS |
| Posterior/readout safety | PASS |
| Anchor usefulness | PASS |
| Heterophily stress | PASS |

Interpretation:

```text
A residual base-reliability path is necessary. V53A is the first route that
keeps the anchor alive late enough to preserve ACM/DBLP and pass heterophily
stress safety.
```

## 3. What V53A Broke

V53A failed reliability non-collapse and damaged Squirrel:

| Dataset | V52A ACC | V53A ACC | Delta | V53A Anchor ACC | V53A Rel |
| --- | ---: | ---: | ---: | ---: | ---: |
| ACM | 0.9041 | 0.9005 | -0.0036 | 0.8942 | 0.1080 |
| DBLP | 0.6867 | 0.6806 | -0.0062 | 0.8901 | 0.0722 |
| Flickr | 0.3733 | 0.3675 | -0.0058 | 0.3626 | 0.0071 |
| Texas | 0.7377 | 0.7322 | -0.0055 | 0.3880 | 0.0582 |
| Squirrel | 0.3003 | 0.2119 | -0.0884 | 0.2448 | 0.1074 |
| Chameleon | 0.3329 | 0.3421 | +0.0092 | 0.3193 | 0.2441 |

The decisive failure is Squirrel:

```text
V53A raises reliability on Squirrel from 0.0068 to 0.1074, but the anchor
diagnostic ACC is only 0.2448 and final ACC falls to 0.2119.
```

This is not a posterior/readout safety failure:

```text
embedding_posterior_gap = 0.0000 on Squirrel
```

It is an assignment-quality failure:

```text
The model safely follows a weak residual anchor too much.
```

## 4. Mechanistic Cause

V53A late reliability is:

```text
r_i = r_base_i * (0.50 + 0.50 * r_agree_i)
```

The fixed `0.50` residual protects anchor availability, but it is global. It
does not ask whether a specific node has enough anchor-consensus evidence for
assignment-level residual imitation.

This creates a new failure mode:

```text
When the anchor is globally weak or semantically mismatched, a fixed residual can
preserve too much assignment KL even though posterior/readout safety remains
clean.
```

Squirrel exposes this failure because:

- anchor ACC is weak;
- agreement reliability remains very low (`0.0170`);
- residual reliability is still lifted above the mass floor;
- final performance collapses.

## 5. Why Simple Fixes Are Rejected

Do not do:

- lower `beta` from 0.50 after seeing Squirrel;
- sweep beta values;
- lower the reliability mass or ratio gates;
- select V52A for Squirrel and V53A for ACM/DBLP;
- tune V50A anchor rank, temperature, filter steps, or refresh;
- change the final output to anchor labels, embedding KMeans, or a selector.

Reason:

```text
The failure is not that residuals are bad. The failure is that residual strength
is not bounded by node-level anchor-consensus evidence.
```

## 6. Rescue Question For V54A

The next research question should be:

```text
Can residual anchor availability be preserved while bounding residual strength
by unsupervised anchor-consensus evidence, so that weak-anchor heterophily nodes
are not overexposed?
```

This shifts the mechanism from:

```text
global residual beta
```

to:

```text
consensus-bounded residual beta
```

## 7. Candidate V54A Mechanism

Recommended route name:

```text
v54a_consensus_bounded_residual_anchor
```

Key idea:

```text
Keep a residual path, but make the residual stronger only where q_refined and
q_embed agree with the spectral anchor at the assignment level.
```

Candidate node-level residual:

```text
h_q_i = 1[argmax(q_refined_i) == argmax(q_spec_i)]
h_e_i = 1[argmax(q_embed_i) == argmax(q_spec_i)]
h_i = 0.5 * h_q_i + 0.5 * h_e_i
beta_i = beta_min + (beta_max - beta_min) * h_i
```

Fixed constants for preregistration:

```text
beta_min = 0.35
beta_max = 0.70
```

Late reliability:

```text
r_i = r_base_i * (beta_i + (1 - beta_i) * r_agree_i)
```

Why this is a new mechanism rather than a beta sweep:

- beta is no longer a single global scalar;
- the node-level residual is bounded by consensus evidence already available in
  the unified training path;
- all gates are stop-gradient diagnostics, not a learned selector;
- final labels remain `q_refined`.

## 8. Required Safeguards

Any V54A preregistration must include:

- fixed `beta_min` and `beta_max`;
- no beta sweep;
- inherited V52A/V53A schedule;
- inherited V50A anchor construction;
- no dataset-specific branch;
- no final selector;
- explicit Squirrel performance floor to guard against residual overexposure;
- reliability mass and safety gates;
- anchor usefulness gates.

## 9. No-Fabrication Status

All numbers in this document come from local V52A/V53A verdict files and
diagnostics. No V54A code or result exists.
