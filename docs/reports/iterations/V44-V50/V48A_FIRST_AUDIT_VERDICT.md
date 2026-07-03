# V48A First-Stage Audit Verdict

Variant:

```text
v48a_topology_dynamics_audit
```

Command:

```powershell
conda run -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v48a_topology_dynamics_audit --datasets "acm,dblp,flickr" --epochs 80 --device cuda --log-level WARNING
```

This was the preregistered first-stage topology dynamics audit. It is not a
performance-claim run. No second-batch smoke, full run, sweep, or follow-up
experiment was run.

## 1. Result Summary

| Dataset | ACC | NMI | ARI | Emb Gap |
| --- | ---: | ---: | ---: | ---: |
| ACM | 0.6450 | 0.2783 | 0.2929 | 0.0000 |
| DBLP | 0.6522 | 0.3604 | 0.2997 | -0.0007 |
| Flickr | 0.3401 | 0.2052 | 0.1298 | 0.0263 |

Performance is recorded only for context. It is not an expansion gate for V48A.

## 2. Red-Line Gate

| Dataset | legacy | v43b | v44 | v44b | v45a | v46a | v47a | v48a | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ACM | false | false | false | false | false | false | false | true | PASS |
| DBLP | false | false | false | false | false | false | false | true | PASS |
| Flickr | false | false | false | false | false | false | false | true | PASS |

Red-line gate passes on 3/3 datasets.

## 3. Diagnostic Completeness Gate

| Dataset | has prev snapshot | sample size | finite movement diagnostics | Verdict |
| --- | --- | ---: | --- | --- |
| ACM | true | 20000 | yes | PASS |
| DBLP | true | 20000 | yes | PASS |
| Flickr | true | 20000 | yes | PASS |

Diagnostic completeness passes on 3/3 datasets.

## 4. Movement Non-Degeneracy Gate

Preregistered criterion: at least one of `dHomo`, `dHetero`, `dHard`, or
`dScore` must be greater than `1e-6` on each dataset.

| Dataset | dHomo | dHetero | dHard | dScore | hard delta | threshold delta | hard rank corr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ACM | 0.032827 | 0.069549 | 0.079167 | 0.043893 | 0.002069 | 0.000221 | 0.8594 |
| DBLP | 0.066800 | 0.048889 | 0.100356 | 0.063345 | 0.000638 | 0.000191 | 0.8426 |
| Flickr | 0.033845 | 0.016754 | 0.042781 | 0.022618 | 0.005683 | 0.000227 | 0.9728 |

Movement non-degeneracy gate passes on 3/3 datasets.

Interpretation:

```text
The topology masks are not frozen. The current contraction path can move homo,
hetero, hard, and score values between epochs.
```

However, hard rank correlation remains high, especially on Flickr. This means
the hard ordering is stable even while mask values move. V48A therefore does
not support the explanation that previous failures were caused by a completely
immobile topology module.

## 5. Directional Consistency Diagnostics

These diagnostics were preregistered as interpretive, not pass/fail expansion
gates.

Expected signs:

```text
targeted_homo_delta > 0
targeted_hetero_delta > 0
targeted_hard_delta >= 0
```

| Dataset | Homo target mass | Hetero target mass | Defer target mass | targeted homo delta | targeted hetero delta | targeted hard delta | Direction Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ACM | 0.3037 | 0.1994 | 0.3035 | -0.001801 | -0.001988 | 0.003623 | FAIL |
| DBLP | 0.2852 | 0.1752 | 0.3135 | 0.002328 | 0.004466 | 0.002717 | PASS |
| Flickr | 0.1201 | 0.2975 | 0.2541 | -0.000474 | -0.002498 | 0.003840 | FAIL |

Directional consistency holds only on DBLP. It fails on ACM and Flickr because
posterior-targeted homo and hetero groups move in the wrong direction.

## 6. Core Verdict

V48A audit passes the instrumentation gates but exposes a mechanism failure:

```text
Topology is movable, but not reliably target-responsive.
```

This is more specific than the V47A failure. V47A showed that semantic
posterior-guided targets were present but did not improve band or ACC. V48A now
shows that the masks do move between epochs, so the issue is not simply frozen
topology. The failure is directional:

- ACM and Flickr move against the intended homo/hetero target directions.
- DBLP is directionally consistent, matching the earlier pattern where DBLP was
  the easiest dataset for these mechanisms to help.
- Hard mass changes are small while hard rank correlation stays high, so the
  model mostly perturbs mask values without reliably reordering hard edges into
  the desired semantic regions.

## 7. Stop Decision

Stop V48A here.

Do not run:

- second-batch smoke
- 9-dataset audit
- 260-epoch full run
- V48A sample-size sweep
- V48A threshold or target sweep
- another stronger posterior-guided CE loss

## 8. Recommended Next Route

Do not continue designing external hard-edge target losses as the next primary
route. The evidence chain now says:

```text
targets exist, masks move, but movement is not semantically aligned.
```

The next mechanism should focus on topology parameterization and transition
geometry, not on adding another teacher signal. A reasonable next route is:

```text
v49a_reparameterized_topology_transition
```

The V49A preregistration should first specify how the topology contraction will
make hard-to-homo and hard-to-hetero transitions locally controllable under a
single unified, non-dataset-specific parameterization. It should not proceed
directly to code without a preregistered design, because V48A is diagnostic
evidence that the existing parameterization can move but does not reliably move
in the intended direction.

## 9. No-Fabrication Status

All numbers in this verdict come from:

```text
results/archive/v40-v50/unified_aptc_9datasets_v48a_topology_dynamics_audit.csv
results/archive/v40-v50/unified_aptc_9datasets_v48a_topology_dynamics_audit_diagnostics.jsonl
```

The prior 1-epoch ACM connectivity record in the diagnostics file was not used
for the 80-epoch audit verdict. For ACM, the latest 80-epoch record was used.
