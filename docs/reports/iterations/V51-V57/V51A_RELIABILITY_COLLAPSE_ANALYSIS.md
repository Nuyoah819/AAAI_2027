# V51A Reliability Collapse Analysis

This document analyzes why `v51a_reliability_gated_spectral_anchor` failed the
first mixed-stress gate. It follows `ccf-idea-optimizer` exploratory rescue
mode: diagnose the mechanism before proposing any new implementation.

No new experiment is run in this document. No V51A sweep or formula variant is
authorized here.

## 1. Evidence Basis

Local artifacts:

```text
V50A_SECOND_STAGE_SMOKE_VERDICT.md
V50A_HETEROPHILY_FAILURE_ANALYSIS.md
V51A_PREREGISTRATION.md
V51A_IMPLEMENTATION_REVIEW.md
V51A_CONNECTIVITY_VERDICT.md
V51A_FIRST_MIXED_STRESS_VERDICT.md
results/archive/v40-v50/unified_aptc_9datasets_v50a_spectral_compactness_anchor_diagnostics.jsonl
results/archive/v51-v57/unified_aptc_9datasets_v51a_reliability_gated_spectral_anchor_diagnostics.jsonl
```

The mixed-stress verdict already stops V51A. This document explains what can be
learned and what the next rescue route must avoid.

## 2. What V51A Fixed

V51A fixed the original hard safety symptom:

| Dataset | V50A Emb Gap | V51A Emb Gap |
| --- | ---: | ---: |
| Squirrel | -0.0877 | 0.0000 |
| Chameleon | 0.0000 | -0.0312 |
| Texas | 0.0109 | 0.0055 |

Interpretation:

```text
Reliability gating can prevent unsafe assignment imitation from creating a
posterior/readout split.
```

This is a real safety signal. However, it is not sufficient because the gate
achieves safety by suppressing almost all anchor supervision.

## 3. What V51A Broke

V51A failed the central rescue requirement:

| Dataset | V50A ACC | V51A ACC | Delta | V51A Rel Mean | V51A Effective Mass |
| --- | ---: | ---: | ---: | ---: | ---: |
| ACM | 0.9088 | 0.6846 | -0.2241 | 0.0054 | 0.0054 |
| DBLP | 0.6810 | 0.6520 | -0.0291 | 0.0020 | 0.0020 |
| Flickr | 0.4397 | 0.3637 | -0.0760 | 0.0000 | 0.0000 |
| Texas | 0.7322 | 0.7322 | 0.0000 | 0.0024 | 0.0024 |
| Squirrel | 0.3019 | 0.3021 | +0.0002 | 0.0029 | 0.0029 |
| Chameleon | 0.3377 | 0.3303 | -0.0075 | 0.0176 | 0.0176 |

The decisive counterexample is ACM:

```text
V50A anchor ACC = 0.8942
V50A final ACC = 0.9088
V51A reliability mean = 0.0054
V51A final ACC = 0.6846
```

ACM proves the gate is not merely filtering bad heterophily anchor signal. It is
also rejecting a strong aligned anchor.

## 4. Mechanistic Cause

V51A reliability is:

```text
r_i = conf_i * sqrt(qa_i_norm * ea_i_norm) * sqrt(local_i_norm)
```

This creates an early-training bootstrapping problem:

```text
The anchor is supposed to make q_refined and q_embed anchor-compatible, but the
gate requires q_refined and q_embed to already be anchor-compatible before the
anchor is allowed to train them.
```

In other words:

```text
V51A asks for the effect before allowing the cause.
```

Component evidence:

| Dataset | Conf | q-anchor | embed-anchor | local | Rel Mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| ACM | 0.3011 | 0.0753 | 0.0315 | 0.0913 | 0.0054 |
| DBLP | 0.2268 | 0.0392 | 0.0263 | 0.0510 | 0.0020 |
| Flickr | 0.0264 | 0.0020 | 0.0000 | 0.0021 | 0.0000 |
| Texas | 0.1917 | 0.0667 | 0.0544 | 0.0261 | 0.0024 |
| Squirrel | 0.3316 | 0.0155 | 0.0194 | 0.0857 | 0.0029 |
| Chameleon | 0.6298 | 0.0479 | 0.0745 | 0.2715 | 0.0176 |

Even when anchor confidence and local consistency are non-trivial, the product
with `q-anchor` and `embed-anchor` agreement drives the gate near zero.

## 5. Why Simple Fixes Are Rejected

Do not do:

- lower `v51a_reliable_threshold`;
- lower `v51a_reliability_floor`;
- change the multiplicative exponents;
- drop only one factor post hoc;
- increase `v51a_anchor_weight`;
- return to V50A unweighted KL;
- tune V50A rank, temperature, filter steps, or refresh;
- select V50A on ACM/DBLP and V51A on Squirrel/heterophily datasets.

Reason:

```text
The failure is not a missing scalar. The failure is a circular dependency
between posterior agreement and anchor training.
```

Threshold changes would only hide the collapse diagnostic. Increasing anchor
weight would still multiply by near-zero reliability. Returning to V50A would
reintroduce Squirrel's hard safety failure.

## 6. Rescue Question For V52A

The next research question should be:

```text
How can the model receive enough early spectral compactness signal to become
anchor-compatible, while still preventing unsafe assignment imitation on nodes
where the anchor is unreliable?
```

This shifts the mechanism from:

```text
trust only after q and embedding already agree with anchor
```

to:

```text
provide a nonzero safe anchor curriculum, then use posterior agreement as a
later reliability signal.
```

## 7. Candidate V52A Mechanism

Recommended route name:

```text
v52a_curriculum_reliability_spectral_anchor
```

Core idea:

```text
Separate anchor availability from assignment imitation strength.
```

V52A should not simply make V51A less strict. It should use a curriculum:

Stage 1, early training:

```text
Use stop-gradient anchor confidence and local anchor consistency only.
Do not require q-anchor or embed-anchor agreement yet.
Keep assignment KL weak and mass-controlled.
```

Stage 2, after the posterior has had a chance to align:

```text
Introduce q-anchor and embed-anchor agreement as reliability factors.
Keep the same unified formula for all datasets.
```

Potential reliability structure:

```text
r_base_i = sqrt(conf_i * local_i_norm)
r_agree_i = sqrt(q_anchor_i * embed_anchor_i)
gamma_t = scheduled agreement gate in [0, 1]
r_i = detach(clamp(r_base_i * ((1 - gamma_t) + gamma_t * r_agree_i), 0, 1))
```

This is not yet authorized code. It is a candidate mechanism for a new
preregistration.

## 8. Required Safeguards For Any V52A

Any V52A preregistration must include:

- a fixed curriculum schedule, not a tuned one;
- no dataset-specific schedule;
- no V50A anchor hyperparameter changes;
- no V51A threshold sweep;
- no geometry fallback unless explicitly preregistered as the main mechanism;
- a hard effective-mass lower and upper gate;
- a safety gate at least as strict as V51A;
- ACM/DBLP preservation floors inherited from V51A;
- heterophily stress gate inherited from V51A.

Minimum first test should remain:

```text
datasets = acm,dblp,flickr,texas,squirrel,chameleon
epochs = 80
seed = 42
device = cuda
```

But before any code, write:

```text
V52A_PREREGISTRATION.md
```

## 9. Reviewer-Facing Risk Register

| Risk | Type | Why it matters | Required response |
| --- | --- | --- | --- |
| Anchor curriculum looks like tuning | evidence-fixable | Reviewers may see V52A as repairing metrics post hoc | Fix schedule before running and report V51A failure honestly |
| Curriculum reintroduces Squirrel gap | requires-new-result | V50A failed exactly here | Keep hard posterior/readout safety gate |
| Reliability mass becomes artificial | design-fixable | A floor can fake usage without useful learning | Diagnose actual mean, ratio, weighted agreement, and performance floors |
| Novelty too incremental | writing/design-fixable | V50-V52 are iterative rescue variants | Frame contribution as reliable low-rank compactness transfer under graph/attribute mismatch |
| Full-run temptation | integrity risk | V51A failed preregistered gates | Stop until V52A is preregistered and passes mixed stress |

## 10. Decision

V51A is stopped.

The next allowed move is:

```text
Write V52A_PREREGISTRATION.md for a curriculum reliability spectral anchor.
```

Not allowed:

```text
V51A sweep, V51A formula variant, full run, or dataset-specific selector.
```

## 11. No-Fabrication Status

All numbers in this document come from local V50A and V51A verdict files and
diagnostics. No V52A code or result exists.
