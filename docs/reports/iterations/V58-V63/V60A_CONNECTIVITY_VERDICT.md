# V60A Connectivity Verdict

This file records the authorized ACM 1-epoch connectivity run for
`v60a_ema_self_distillation_drift_guard`.

It follows `V60A_IMPLEMENTATION_REVIEW.md`.

Connectivity is a wiring and diagnostic check only. It is not a performance
result.

## 1. Static Check

Command:

```powershell
conda run -n aaai-e2e-subspace python -m py_compile core\e2e\sect_coco_e2e.py scripts\run_unified_aptc_9datasets.py
```

Verdict:

```text
PASS
```

No residual training process was present before launch.

## 2. Connectivity Run

Authorized command:

```powershell
conda run --no-capture-output -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v60a_ema_self_distillation_drift_guard --datasets "acm" --epochs 1 --device cuda --log-level WARNING
```

Run status:

```text
status=ok
```

Output files:

```text
results/archive/v58-v63/unified_aptc_9datasets_v60a_ema_self_distillation_drift_guard.csv
results/archive/v58-v63/unified_aptc_9datasets_v60a_ema_self_distillation_drift_guard_diagnostics.jsonl
```

## 3. Red-Line Check

| Flag | Value | Gate |
| --- | --- | --- |
| legacy_head_used | false | PASS |
| v50a_enabled | false | PASS |
| v51a_enabled | false | PASS |
| v52a_enabled | false | PASS |
| v53a_enabled | false | PASS |
| v54a_enabled | false | PASS |
| v55a_enabled | false | PASS |
| v56a_enabled | false | PASS |
| v57a_enabled | false | PASS |
| v58a_enabled | false | PASS |
| v59a_enabled | false | PASS |
| v60a_enabled | true | PASS |

Verdict:

```text
PASS
```

## 4. Anchor And Guard Check

| Diagnostic | Value | Expected | Gate |
| --- | ---: | ---: | --- |
| v60a_release_gamma | 1.0000 | 1.0000 | PASS |
| v60a_anchor_loss | 0.1644 | finite | PASS |
| v60a_teacher_ready | false | false | PASS |
| v60a_teacher_epoch | -1.0000 | -1.0000 | PASS |
| v60a_guard_gamma | 0.0000 | 0.0000 | PASS |
| v60a_guard_loss | 0.0000 | 0.0000 | PASS |
| v60a_guard_kl | 0.0000 | 0.0000 | PASS |
| v60a_teacher_active_ratio | 0.0000 | 0.0000 before teacher | PASS |

Interpretation:

```text
The V60A guard is correctly inactive before the epoch-80 teacher snapshot.
The anchor side remains active through the inherited V59A/V57A mechanism.
```

## 5. Fixed Constants

| Diagnostic | Value | Gate |
| --- | ---: | --- |
| v60a_target_mass | 0.0800 | PASS |
| v60a_max_mass_scale | 1.5000 | PASS |
| v60a_max_reliability_cap | 0.9000 | PASS |

## 6. Safety Check

| Diagnostic | Value | Gate |
| --- | ---: | --- |
| embedding_posterior_gap | 0.0026 | PASS |

## 7. Decision

Connectivity verdict:

```text
PASS V60A CONNECTIVITY.
```

This authorizes only the preregistered first mixed-stress run:

```powershell
conda run --no-capture-output -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v60a_ema_self_distillation_drift_guard --datasets "acm,dblp,flickr,texas,squirrel,chameleon" --epochs 100 --device cuda --log-level WARNING
```

No V60A 260-epoch run, seed sweep, confidence-threshold sweep, guard-weight
sweep, teacher-epoch sweep, EMA update, or final-label selector is authorized.

## 8. No-Fabrication Status

All values in this verdict come from the local V60A connectivity run and its
diagnostics.
