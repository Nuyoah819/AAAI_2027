# V62A Expansion Review

This review follows `V62A_FIRST_MIXED_STRESS_VERDICT.md`.

Variant:

```text
v62a_drift_responsive_self_distillation_guard
```

Decision:

```text
AUTHORIZE ONE SUPPORTED 9-DATASET / 260-EPOCH FULL-RUN TEST.
```

No sweep, rerun-and-keep-best, dataset-specific branch, drift-floor change,
drift-scale change, drift-boost change, guard-weight change, coverage change,
teacher-epoch change, EMA change, or final-label selector is authorized.

## 1. Rationale

V62A passed the preregistered 6-dataset / 100-epoch mixed-stress gate:

```text
6/6 status=ok
teacher_ready by epoch 80 on 6/6
guard_gamma_epoch_80 = 0.0 on 6/6
guard_gamma_epoch_100 = 1.0 on 6/6
drift_gamma_epoch_100 = 0.0 on 6/6
effective_guard_multiplier_epoch_100 = 1.0 on 6/6
teacher_active_ratio_epoch_80 >= 0.10 on 6/6
teacher_topk_active_ratio_epoch_80 >= 0.10 on 6/6
abs(embedding_posterior_gap) <= 0.04 on 6/6
ACM ACC = 0.8992 >= 0.8888
DBLP ACC = 0.6788 >= 0.6610
Squirrel ACC = 0.3013 >= 0.2800
```

The expansion question is whether the bounded drift multiplier becomes active
after epoch 100 where needed and repairs the V61A 260-epoch long-run drift
failure without breaking coverage, anchor/release, or final-label rules.

## 2. Dataset Universe

Use exactly:

```text
acm,dblp,pubmed,wiki,flickr,blogcatalog,squirrel,texas,chameleon
```

Do not use unsupported datasets:

```text
wisconsin,cornell,actor
```

## 3. Frozen Configuration

V62A constants remain:

```text
v62a_anchor_weight=0.04
v62a_guard_weight=0.02
v62a_absolute_floor=0.45
v62a_min_teacher_coverage=0.10
v62a_start_epoch=80
v62a_guard_ramp_epochs=20
v62a_max_gamma=1.0
v62a_drift_start_epoch=100
v62a_drift_floor=0.02
v62a_drift_scale=0.06
v62a_drift_boost=1.0
v62a_max_effective_guard_multiplier=2.0
v62a_reliability_floor=0.10
v62a_reliable_threshold=0.20
v62a_min_effective_mass=0.10
v62a_warmup_epochs=20
v62a_ramp_epochs=40
v62a_beta_min=0.35
v62a_beta_max=0.70
v62a_soft_power=0.50
v62a_hybrid_compensation=0.50
v62a_target_mass=0.08
v62a_max_mass_scale=1.50
v62a_max_reliability_cap=0.90
v62a_release_start_epoch=80
v62a_release_decay_epochs=60
v62a_release_floor=0.25
```

All previous active rescue losses must remain disabled:

```text
v50a_enabled=false
v51a_enabled=false
v52a_enabled=false
v53a_enabled=false
v54a_enabled=false
v55a_enabled=false
v56a_enabled=false
v57a_enabled=false
v58a_enabled=false
v59a_enabled=false
v60a_enabled=false
v61a_enabled=false
v62a_enabled=true
```

Final labels must remain:

```text
q_refined
```

## 4. Required Pre-Run Checks

Run:

```powershell
conda run -n aaai-e2e-subspace python -m py_compile core\e2e\sect_coco_e2e.py scripts\run_unified_aptc_9datasets.py
```

Then confirm there is no residual training process:

```powershell
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'run_unified_aptc_9datasets|aaai-e2e-subspace' } | Select-Object ProcessId,Name,CommandLine,CreationDate
```

## 5. Authorized Command

Only this command is authorized:

```powershell
conda run --no-capture-output -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v62a_drift_responsive_self_distillation_guard --datasets "acm,dblp,pubmed,wiki,flickr,blogcatalog,squirrel,texas,chameleon" --epochs 260 --device cuda --log-level WARNING
```

## 6. Full-Run Gates

Execution:

```text
9/9 supported datasets complete with status=ok
```

Red-line:

```text
legacy_head_used=false on 9/9
v50a-v61a_enabled=false on 9/9
v62a_enabled=true on 9/9
teacher used only as loss target, not final labels
```

Teacher and base guard:

```text
v62a_teacher_ready_epoch_80 = true on 9/9
v62a_guard_gamma_epoch_80 = 0.0 on 9/9
v62a_guard_gamma_epoch_100 = 1.0 on 9/9
v62a_guard_gamma at final epoch = 1.0 on 9/9
v62a_teacher_active_ratio_epoch_80 >= 0.10 on 9/9
v62a_teacher_topk_active_ratio_epoch_80 >= 0.10 on 9/9
v62a_guard_loss is finite on 9/9
```

Drift response:

```text
v62a_drift_gamma_epoch_100 = 0.0 on 9/9
v62a_effective_guard_multiplier_epoch_100 = 1.0 on 9/9
v62a_effective_guard_multiplier at final epoch <= 2.0 on 9/9
v62a_drift_score at final epoch is finite on 9/9
```

At least one historically drifting dataset among `flickr,squirrel,texas` should
show a nonzero final drift response if its final drift score exceeds the fixed
floor. If no drift response activates anywhere, the verdict must explicitly
explain that V62A did not test the intended mechanism.

Release schedule:

```text
v62a_release_gamma_epoch_1 = 1.0
v62a_release_gamma_epoch_80 = 1.0
v62a_release_gamma at final epoch = 0.25
```

Mass and anchor:

```text
effective anchor mass >= 0.08 on at least 6/9 datasets
v62a_reliability_mean >= 0.03 on ACM, DBLP, PubMed, Wiki, Squirrel, Texas, and Chameleon
v62a_reliability_mean <= 0.97 on 9/9
```

Flickr and BlogCatalog may remain weak-anchor exceptions only if posterior
safety passes and they do not violate the drift-repair floors.

Posterior/readout safety:

```text
abs(embedding_posterior_gap) <= 0.04 on at least 8/9 datasets
no dataset has abs(embedding_posterior_gap) > 0.08
```

Preservation floors:

```text
ACM ACC >= 0.8888
DBLP ACC >= 0.6610
Squirrel ACC >= 0.2800
```

Long-run drift-repair floors:

```text
PubMed ACC >= 0.5200
Flickr ACC >= 0.3500
Squirrel ACC >= 0.2800
Texas ACC >= 0.7000
```

## 7. Required Verdict Artifact

After the run, write:

```text
V62A_FULL_RUN_VERDICT.md
```

It must include:

```text
exact command
pre-run compile result
process-cleanliness statement
9-dataset ACC/NMI/ARI table
red-line table
teacher/base-guard coverage table
drift-response table
release schedule table
mass/reliability table
posterior/readout safety table
preservation-floor table
drift-repair table versus V61A 260e and V62A 100e where relevant
pass/stop decision
no-fabrication status
```

## 8. Stop Conditions

Stop and write failure analysis if any of the following occurs:

```text
any dataset status != ok
teacher snapshot missing or not ready by epoch 80
teacher active ratio @80 < 0.10 on any dataset
guard loss nonzero before teacher readiness
legacy head used
v50a-v61a active loss enabled
teacher used as final labels
v62a_effective_guard_multiplier > 2.0
Squirrel ACC < 0.2800
ACM ACC < 0.8888
embedding_posterior_gap > 0.08 on any dataset
```

## 9. No-Fabrication Status

This review contains no V62A 260-epoch result. It only authorizes one fixed
supported 9-dataset / 260-epoch full-run evaluation.
