# V57A Expansion Review Amendment

This amendment corrects the dataset universe in `V57A_EXPANSION_REVIEW.md`.

Decision:

```text
AUTHORIZE ONE CORRECTED SUPPORTED 9-DATASET / 80-EPOCH SMOKE.
```

This amendment does not authorize a 260-epoch full run, seed sweep,
target-mass sweep, max-mass-scale sweep, reliability-cap sweep, beta-bound
sweep, soft-power sweep, hybrid-compensation sweep, schedule variant,
reliability formula variant, threshold sweep, or V50A anchor hyperparameter
change.

## 1. Correction

The original expansion review used unsupported datasets:

```text
wisconsin,cornell,actor
```

The local project's supported 9-dataset universe is:

```text
acm,dblp,pubmed,wiki,flickr,blogcatalog,squirrel,texas,chameleon
```

Evidence:

```text
scripts/run_e2e_experiments.py
core/data/data_utils.py
D:/study/graduate_student/papers/AAAI2027/data
```

## 2. Corrected Authorized Command

Only this corrected command is authorized:

```powershell
conda run --no-capture-output -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v57a_mass_floor_normalized_residual_anchor --datasets "acm,dblp,pubmed,wiki,flickr,blogcatalog,squirrel,texas,chameleon" --epochs 80 --device cuda --log-level WARNING
```

The variant, seed, epochs, formulas, constants, and final-label path remain
unchanged.

## 3. Required Verdict Artifact

After the corrected smoke, write:

```text
V57A_SUPPORTED_9DATASET_80E_SMOKE_VERDICT.md
```

It must include:

- exact command;
- 9-dataset ACC/NMI/ARI table;
- red-line table;
- mass-normalization table;
- raw-vs-scaled reliability table;
- reliability non-collapse table;
- posterior/readout safety table;
- anchor usefulness table;
- notes for PubMed, Wiki, and BlogCatalog;
- explicit stop/continue decision.

## 4. Gates

Use the same gates as `V57A_EXPANSION_REVIEW.md`, but apply them to:

```text
acm,dblp,pubmed,wiki,flickr,blogcatalog,squirrel,texas,chameleon
```

Reliability gate:

```text
mass pass on at least 6/9
reliable-node ratio pass on at least 4/9
Flickr may remain the weak-anchor exception
```

Safety gate:

```text
abs(embedding_posterior_gap) <= 0.04 on at least 8/9
no dataset abs(embedding_posterior_gap) > 0.08
```

Floor gate:

```text
ACM ACC >= 0.8888
DBLP ACC >= 0.6610
Squirrel ACC >= 0.2800
```

No additional dataset-specific floor may be added after seeing the corrected
run.

## 5. Boundary

If the corrected supported 9-dataset 80-epoch smoke passes, the next artifact
must be:

```text
V57A_FULL_RUN_PREREGISTRATION.md
```

If it fails, stop and write a failure analysis. Do not proceed to a 260-epoch
run.

## 6. No-Fabrication Status

No corrected supported 9-dataset V57A smoke result exists yet.
