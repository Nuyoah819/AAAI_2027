# V57A Full-Run Preregistration

This file preregisters the first full-length evaluation for
`v57a_mass_floor_normalized_residual_anchor`.

It follows:

```text
V57A_PREREGISTRATION.md
V57A_IMPLEMENTATION_REVIEW.md
V57A_CONNECTIVITY_VERDICT.md
V57A_FIRST_MIXED_STRESS_VERDICT.md
V57A_EXPANSION_REVIEW_AMENDMENT.md
V57A_SUPPORTED_9DATASET_80E_SMOKE_VERDICT.md
CRITICAL_RED_LINES.md
```

No V57A 260-epoch result exists at the time this preregistration is written.

## 1. Decision

Decision:

```text
AUTHORIZE ONE SUPPORTED 9-DATASET / 260-EPOCH FULL RUN.
```

Rationale:

```text
V57A is the first rescue route in the V50A-V57A chain to pass both the
preregistered 6-dataset mixed-stress gate and the corrected supported
9-dataset / 80-epoch smoke gate. Its mechanism is fixed: keep the V56A
reliability ranking, then apply detached mass-floor normalization so
medium-consensus datasets receive enough operational anchor mass.
```

This is not a hyperparameter search. The full run is allowed only because the
same fixed mechanism already passed the preregistered short-run gates.

## 2. Frozen Variant

Variant:

```text
v57a_mass_floor_normalized_residual_anchor
```

The full run must use the existing runner variant without code or parameter
changes:

```text
v57a_enabled=true
v57a_anchor_weight=0.04
v57a_reliability_floor=0.10
v57a_reliable_threshold=0.20
v57a_min_effective_mass=0.10
v57a_warmup_epochs=20
v57a_ramp_epochs=40
v57a_beta_min=0.35
v57a_beta_max=0.70
v57a_soft_power=0.50
v57a_hybrid_compensation=0.50
v57a_target_mass=0.08
v57a_max_mass_scale=1.50
v57a_max_reliability_cap=0.90
```

All earlier rescue losses must remain disabled:

```text
v50a_enabled=false
v51a_enabled=false
v52a_enabled=false
v53a_enabled=false
v54a_enabled=false
v55a_enabled=false
v56a_enabled=false
```

Final labels must remain the unified model output:

```text
q_refined
```

The anchor labels, KMeans labels, legacy head labels, posterior selector, or
any dataset-specific selector must not be used as final labels.

## 3. Dataset Universe

Use exactly the supported local 9-dataset universe:

```text
acm,dblp,pubmed,wiki,flickr,blogcatalog,squirrel,texas,chameleon
```

Do not add:

```text
wisconsin,cornell,actor
```

Those datasets are not supported by the current local pipeline and were already
recorded as an invalid expansion attempt.

## 4. Required Pre-Run Checks

Before launching the full run, execute:

```powershell
conda run -n aaai-e2e-subspace python -m py_compile core\e2e\sect_coco_e2e.py scripts\run_unified_aptc_9datasets.py
```

Also check for residual training processes:

```powershell
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'run_unified_aptc_9datasets|aaai-e2e-subspace' } | Select-Object ProcessId,Name,CommandLine,CreationDate
```

If a prior training process is still running, do not launch the full run.

## 5. Authorized Command

Only the following command is authorized:

```powershell
conda run --no-capture-output -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v57a_mass_floor_normalized_residual_anchor --datasets "acm,dblp,pubmed,wiki,flickr,blogcatalog,squirrel,texas,chameleon" --epochs 260 --device cuda --log-level WARNING
```

No other epoch count, dataset list, seed, variant, or configuration change is
authorized by this file.

## 6. Hard Denylist

The following are explicitly forbidden before, during, and after this full run:

```text
seed sweep
restart and keep best
target-mass sweep
max-mass-scale sweep
reliability-cap sweep
beta-bound sweep
soft-power sweep
hybrid-compensation sweep
schedule variant
reliability formula variant
threshold sweep
V50A anchor hyperparameter sweep
dataset-specific branch, head, module, loss, or selector
post-hoc selector among q_refined, q_embed, q_anchor, KMeans, or legacy head
changing the full-run verdict because one dataset is inconvenient
```

If the 260-epoch run fails a gate, stop and write a failure analysis. Do not
repair V57A by tuning constants against this full-run result.

## 7. Full-Run Gates

The full run must be judged using the latest 260-epoch record for each of the
9 supported datasets.

### 7.1 Execution Gate

Pass only if:

```text
9/9 datasets complete with status=ok
```

If any dataset crashes, record the crash and stop. Do not silently drop the
dataset.

### 7.2 Red-Line Gate

Pass only if all 9 datasets satisfy:

```text
legacy_head_used=false
v43b-v49a_enabled=false
v50a_enabled=false
v51a_enabled=false
v52a_enabled=false
v53a_enabled=false
v54a_enabled=false
v55a_enabled=false
v56a_enabled=false
v57a_enabled=true
```

Any dataset-specific final-label behavior is an immediate failure.

### 7.3 Mass-Normalization Gate

Pass only if all 9 datasets satisfy:

```text
v57a_target_mass = 0.08
v57a_max_mass_scale = 1.50
v57a_max_reliability_cap = 0.90
1.0 <= v57a_mass_scale <= 1.50
0.0 <= v57a_reliability_mean <= 0.90
```

The full-run verdict must report raw reliability, mass scale, scaled
reliability, reliable-node ratio, and effective anchor mass.

### 7.4 Reliability Non-Collapse Gate

Pass only if:

```text
effective anchor mass >= 0.08 on at least 6/9 datasets
reliable-node ratio >= 0.05 on at least 4/9 datasets
```

Allowed weak-anchor exceptions:

```text
Flickr may remain below the effective-mass floor.
BlogCatalog may remain below the effective-mass floor only if ACC remains
stable and posterior/readout safety passes.
```

Hard fail:

```text
v57a_reliability_mean < 0.03 on ACM, DBLP, PubMed, Wiki, Squirrel, Texas, or Chameleon
v57a_reliability_mean > 0.97 on any dataset
effective anchor mass < 0.08 on more than 3/9 datasets
```

### 7.5 Anchor-Usefulness Gate

Pass only if:

```text
v57a_weighted_q_anchor_agreement_epoch_260 >
v57a_weighted_q_anchor_agreement_epoch_1 on at least 6/9 datasets
```

Flickr may fail this movement gate as the known weak-anchor exception. If two
or more of ACM, DBLP, PubMed, Wiki, Texas, Squirrel, or Chameleon fail the
movement gate, stop and write a failure analysis.

### 7.6 Posterior/Readout Safety Gate

Pass only if:

```text
abs(embedding_posterior_gap) <= 0.04 on at least 8/9 datasets
no dataset has abs(embedding_posterior_gap) > 0.08
```

This gate protects the unified `q_refined` output from being silently worse
than the embedding posterior.

### 7.7 Preservation Floors

Pass only if:

```text
ACM ACC >= 0.8888
DBLP ACC >= 0.6610
Squirrel ACC >= 0.2800
```

These floors come from earlier preregistered safety boundaries and must not be
changed after seeing the 260-epoch result.

### 7.8 Full-Length Drift Watch

The verdict must explicitly discuss:

```text
Flickr: whether weak anchor non-use remains harmless or becomes collapse.
BlogCatalog: whether low reliability mass conflicts with its high ACC.
Wiki: whether 260 epochs improve or worsen the weak 80e ACC.
Chameleon: whether the 80e ACC drop recovers or deepens.
```

These notes are diagnostic. They cannot be used to add dataset-specific
patches after the run.

## 8. Required Verdict Artifact

After the run, write:

```text
V57A_FULL_RUN_VERDICT.md
```

It must include:

```text
exact command
pre-run py_compile result
process-cleanliness statement
9-dataset ACC/NMI/ARI table
red-line table
mass-normalization table
raw-vs-scaled reliability table
anchor-usefulness table using epoch 1 and epoch 260 snapshots
posterior/readout safety table
preservation-floor table
Flickr/BlogCatalog/Wiki/Chameleon drift notes
pass/stop decision
no-fabrication status
```

## 9. Interpretation Boundary

If the full run passes, it authorizes only paper-facing analysis planning and
mechanism ablations to be preregistered separately. It does not authorize
additional V57A tuning or a new V58A chosen by looking at the 260-epoch result.

If the full run fails, the next artifact must be a failure analysis. The failure
analysis may identify a future rescue hypothesis, but it must not tune V57A
constants against the failed full-run metrics.

## 10. No-Fabrication Status

This preregistration contains no 260-epoch result. It only records the fixed
protocol and gates for the first full-length V57A evaluation.
