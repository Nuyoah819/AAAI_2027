# V52A Late Reliability Collapse Analysis

This document analyzes why `v52a_curriculum_reliability_spectral_anchor` failed
the first mixed-stress gate. It follows `ccf-idea-optimizer` exploratory rescue
mode: diagnose the mechanism before proposing any new implementation.

No new experiment is run in this document. No V52A schedule, threshold, formula,
or V50A anchor hyperparameter sweep is authorized here.

## 1. Evidence Basis

Local artifacts:

```text
V51A_FIRST_MIXED_STRESS_VERDICT.md
V51A_RELIABILITY_COLLAPSE_ANALYSIS.md
V52A_PREREGISTRATION.md
V52A_IMPLEMENTATION_REVIEW.md
V52A_CONNECTIVITY_VERDICT.md
V52A_FIRST_MIXED_STRESS_VERDICT.md
results/archive/v51-v57/unified_aptc_9datasets_v52a_curriculum_reliability_spectral_anchor_diagnostics.jsonl
```

V52A already stops by preregistered gates. This document only extracts the
mechanistic lesson and defines the next rescue question.

## 2. What V52A Fixed

V52A fixed V51A's immediate all-off reliability problem.

ACM one-epoch connectivity:

| Signal | V51A | V52A |
| --- | ---: | ---: |
| Reliability mean | 0.0014 | 0.1962 |
| Reliable node ratio | 0.0000 | 0.5041 |
| Effective anchor mass | 0.0014 | 0.1962 |

Mixed-stress positives:

| Dataset | V52A ACC | Preservation / Safety Signal |
| --- | ---: | --- |
| ACM | 0.9041 | passes 0.8888 floor |
| DBLP | 0.6867 | passes 0.6610 floor |
| Texas | 0.7377 | strong and safe |
| Squirrel | 0.3003 | hard posterior/readout failure remains fixed |

Interpretation:

```text
Early base reliability is useful. It restores anchor availability and preserves
the V50A aligned-anchor rescue signal better than V51A.
```

## 3. What V52A Failed

V52A failed reliability non-collapse and heterophily stress:

| Dataset | Rel @1 | Rel @40 | Rel @80 | Effective Mass @80 | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| ACM | 0.1962 | 0.1051 | 0.0197 | 0.0197 | HARD FAIL |
| DBLP | 0.1389 | 0.0725 | 0.0055 | 0.0055 | HARD FAIL |
| Flickr | 0.0142 | 0.0071 | 0.0000 | 0.0000 | weak-anchor non-use |
| Texas | 0.1089 | 0.0580 | 0.0075 | 0.0075 | HARD FAIL |
| Squirrel | 0.2086 | 0.1074 | 0.0068 | 0.0068 | HARD FAIL |
| Chameleon | 0.4507 | 0.2422 | 0.0376 | 0.0376 | FAIL |

The failure is not early access. The failure is late replacement:

```text
V52A makes base reliability temporary. At gamma=1, the gate becomes
r_base * r_agree, so late training inherits V51A's agreement bottleneck.
```

## 4. Mechanistic Cause

V52A reliability is:

```text
r_i = (1 - gamma_t) * r_base_i + gamma_t * (r_base_i * r_agree_i)
```

At epoch 80:

```text
gamma_t = 1
r_i = r_base_i * r_agree_i
```

This removes the very mechanism that rescued early training. The diagnostic
pattern is clear:

| Dataset | Base Reliability | Agreement Reliability | Final Reliability |
| --- | ---: | ---: | ---: |
| ACM | 0.1962 | 0.0861 | 0.0197 |
| DBLP | 0.1389 | 0.0345 | 0.0055 |
| Texas | 0.1089 | 0.0606 | 0.0075 |
| Squirrel | 0.2086 | 0.0189 | 0.0068 |
| Chameleon | 0.4507 | 0.0809 | 0.0376 |

The agreement term is useful as a safety modulator, but too weak to be a final
gate on its own.

## 5. Why Simple Fixes Are Rejected

Do not do:

- slow down the V52A ramp;
- stop gamma at 0.5 post hoc;
- change warmup or ramp epochs;
- lower reliability thresholds;
- change the V52A formula without preregistration;
- increase anchor weight;
- tune V50A anchor rank, temperature, filter steps, or refresh;
- choose V52A for ACM/DBLP and another variant for heterophily datasets.

Reason:

```text
The failure is structural: base reliability is treated as a temporary scaffold
instead of a persistent safety-preserving anchor availability signal.
```

Changing the schedule would be a disguised sweep. The next route must be a new
mechanism with a fixed late residual, not a V52A parameter tweak.

## 6. Rescue Question For V53A

The next research question should be:

```text
Can a fixed nonzero base-reliability residual preserve anchor availability at
late training while still using agreement to reduce unsafe imitation?
```

This shifts the mechanism from:

```text
early base reliability -> late agreement bottleneck
```

to:

```text
early base reliability -> late residual base plus agreement modulation
```

## 7. Candidate V53A Mechanism

Recommended route name:

```text
v53a_residual_curriculum_spectral_anchor
```

Candidate reliability:

```text
r_base_i = 0.5 * conf_i + 0.5 * local_i_norm
r_agree_i = 0.5 * qa_i_norm + 0.5 * ea_i_norm
gamma_t = fixed V52A schedule
beta = 0.50
r_multiplier_i = (1 - gamma_t) + gamma_t * (beta + (1 - beta) * r_agree_i)
r_i = detach(clamp(r_base_i * r_multiplier_i, 0, 1))
```

At epoch 1:

```text
gamma_t = 0
r_i = r_base_i
```

At epoch 80:

```text
gamma_t = 1
r_i = r_base_i * (0.50 + 0.50 * r_agree_i)
```

This is not a threshold change. It preserves a fixed amount of base reliability
at late training while still letting agreement reduce trust where model/readout
and anchor disagree.

## 8. Expected Diagnostic Difference

The expected V53A behavior is not "higher reliability everywhere" without
control. The required pattern is:

```text
V53A final reliability should stay nonzero on ACM/DBLP/Texas/Squirrel/Chameleon,
while posterior/readout safety remains within V52A's safety gate.
```

V53A should be considered a failure if:

- final reliability passes only because it ignores agreement completely;
- Squirrel's hard posterior/readout failure returns;
- ACM/DBLP performance floors fail;
- Texas/Squirrel/Chameleon fail safety despite more anchor mass.

## 9. Required Next Artifact

Before any code implementation, write:

```text
V53A_PREREGISTRATION.md
```

It must fix:

- exact residual formula;
- fixed `beta`;
- whether V52A schedule is inherited unchanged;
- diagnostics for residual/base/agreement contributions;
- first-stage mixed stress gates;
- hard stop conditions;
- explicit prohibition of V52A schedule or beta sweep.

## 10. No-Fabrication Status

All numbers in this document come from local V52A verdict files and diagnostics.
No V53A code or result exists.
