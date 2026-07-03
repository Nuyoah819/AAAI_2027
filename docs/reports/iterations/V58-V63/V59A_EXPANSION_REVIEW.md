# V59A Expansion Review

This review follows `V59A_FIRST_MIXED_STRESS_VERDICT.md`.

Variant:

```text
v59a_post80_anchor_release_residual_compactness
```

## 1. Decision

Decision:

```text
AUTHORIZE ONE SUPPORTED 9-DATASET / 260-EPOCH FULL-RUN TEST.
```

Rationale:

```text
V59A passed the preregistered 6-dataset / 80-epoch mixed-stress gate and
restored the V57A early absorption window that V58A disrupted. Because V59A is
defined to be V57A-equivalent through epoch 80, another 80-epoch 9-dataset smoke
would mainly repeat the already established early-window check. The central
V59A hypothesis is post-80 release, so it must be tested at 260 epochs.
```

This review does not authorize tuning, schedule changes, seed sweeps, or any
dataset-specific branch.

## 2. Scientific Question

The V57A full run failed because:

```text
Flickr:   0.4133 at 80e -> 0.2779 at 260e
PubMed:   0.5822 at 80e -> 0.4788 at 260e
Texas:    0.7377 at 80e -> 0.6175 at 260e
Squirrel: 0.3005 at 80e -> 0.2102 at 260e
```

V59A tests:

```text
Can preserving V57A through epoch 80 and releasing anchor pressure only after
epoch 80 reduce this full-length drift while keeping a nonzero residual anchor?
```

## 3. Dataset Universe

Use exactly the supported local 9-dataset universe:

```text
acm,dblp,pubmed,wiki,flickr,blogcatalog,squirrel,texas,chameleon
```

Do not use unsupported datasets:

```text
wisconsin,cornell,actor
```

## 4. Frozen Configuration

V59A constants remain fixed:

```text
v59a_anchor_weight=0.04
v59a_reliability_floor=0.10
v59a_reliable_threshold=0.20
v59a_min_effective_mass=0.10
v59a_warmup_epochs=20
v59a_ramp_epochs=40
v59a_beta_min=0.35
v59a_beta_max=0.70
v59a_soft_power=0.50
v59a_hybrid_compensation=0.50
v59a_target_mass=0.08
v59a_max_mass_scale=1.50
v59a_max_reliability_cap=0.90
v59a_release_start_epoch=80
v59a_release_decay_epochs=60
v59a_release_floor=0.25
```

All earlier active rescue losses must remain disabled:

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
v59a_enabled=true
```

Final labels must remain unified `q_refined`.

## 5. Required Pre-Run Checks

Before launching the full run, execute:

```powershell
conda run -n aaai-e2e-subspace python -m py_compile core\e2e\sect_coco_e2e.py scripts\run_unified_aptc_9datasets.py
```

Also check there is no residual training process:

```powershell
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'run_unified_aptc_9datasets|aaai-e2e-subspace' } | Select-Object ProcessId,Name,CommandLine,CreationDate
```

## 6. Authorized Command

Only this command is authorized:

```powershell
conda run --no-capture-output -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v59a_post80_anchor_release_residual_compactness --datasets "acm,dblp,pubmed,wiki,flickr,blogcatalog,squirrel,texas,chameleon" --epochs 260 --device cuda --log-level WARNING
```

## 7. Full-Run Gates

### 7.1 Execution

Pass only if:

```text
9/9 supported datasets complete with status=ok
```

### 7.2 Red-Line

Pass only if all 9 datasets satisfy:

```text
legacy_head_used=false
v50a-v58a_enabled=false
v59a_enabled=true
```

### 7.3 Release Schedule

Pass only if all 9 datasets satisfy:

```text
v59a_release_gamma_epoch_1 = 1.0
v59a_release_gamma_epoch_40 = 1.0
v59a_release_gamma_epoch_80 = 1.0
v59a_release_gamma at final epoch = 0.25
```

At epoch 260, the release multiplier should be at the residual floor.

### 7.4 Mass And Reliability

Pass only if:

```text
effective anchor mass >= 0.08 on at least 6/9 datasets
reliable-node ratio >= 0.05 on at least 4/9 datasets
```

Flickr and BlogCatalog may remain weak-anchor exceptions only if posterior/readout
safety passes and their ACC does not create a new hard failure.

Hard fail:

```text
v59a_reliability_mean < 0.03 on ACM, DBLP, PubMed, Wiki, Squirrel, Texas, or Chameleon
v59a_reliability_mean > 0.97 on any dataset
effective anchor mass < 0.08 on more than 3/9 datasets
```

### 7.5 Posterior/Readout Safety

Pass only if:

```text
abs(embedding_posterior_gap) <= 0.04 on at least 8/9 datasets
no dataset has abs(embedding_posterior_gap) > 0.08
```

### 7.6 Preservation Floors

Pass only if:

```text
ACM ACC >= 0.8888
DBLP ACC >= 0.6610
Squirrel ACC >= 0.2800
```

### 7.7 Drift-Repair Gate

V59A must improve the V57A 260e failure boundary on the long-run stress
datasets:

```text
Squirrel ACC >= 0.2800
Texas ACC >= 0.7000
Flickr ACC >= 0.3500
PubMed ACC >= 0.5200
```

These thresholds are not SOTA claims. They are minimum rescue checks against
the V57A full-run drift:

```text
V57A 260e Squirrel = 0.2102
V57A 260e Texas    = 0.6175
V57A 260e Flickr   = 0.2779
V57A 260e PubMed   = 0.4788
```

If V59A fails this drift-repair gate, stop and write a failure analysis.

## 8. Required Verdict Artifact

After the run, write:

```text
V59A_FULL_RUN_VERDICT.md
```

It must include:

```text
exact command
pre-run compile result
process-cleanliness statement
9-dataset ACC/NMI/ARI table
red-line table
release schedule table
mass/reliability table
posterior/readout safety table
preservation-floor table
drift-repair table versus V57A 260e
pass/stop decision
no-fabrication status
```

## 9. Hard Denylist

Do not perform:

```text
seed sweep
restart and keep best
schedule sweep
release-floor sweep
release-start sweep
reliability formula change
mass constant change
V57A/V58A fallback selection
dataset-specific branch or stop rule
post-hoc selector among q_refined, q_embed, q_anchor, KMeans, or legacy labels
```

## 10. No-Fabrication Status

This review contains no V59A 260-epoch result. It only authorizes one fixed
supported 9-dataset / 260-epoch V59A evaluation.
