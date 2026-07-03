# V57A 9-Dataset 80E Smoke Invalid Verdict

This file records an invalid expansion attempt for
`v57a_mass_floor_normalized_residual_anchor`.

Decision:

```text
INVALID AS 9-DATASET SMOKE.
```

This is a protocol correction, not a V57A mechanism verdict.

## 1. Attempted Command

```powershell
conda run --no-capture-output -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v57a_mass_floor_normalized_residual_anchor --datasets "acm,dblp,texas,wisconsin,cornell,chameleon,squirrel,actor,flickr" --epochs 80 --device cuda --log-level WARNING
```

## 2. Failure Cause

The command used three unsupported dataset names:

```text
wisconsin
cornell
actor
```

The local loader supports:

```text
acm
dblp
pubmed
wiki
flickr
blogcatalog
squirrel
texas
chameleon
```

Evidence:

```text
core/data/data_utils.py
scripts/run_e2e_experiments.py
```

The unsupported datasets produced loader errors:

```text
Unsupported dataset 'wisconsin'
Unsupported dataset 'cornell'
Unsupported dataset 'actor'
```

## 3. Partial Results Are Not A Verdict

The command produced valid rows for:

```text
acm
dblp
texas
chameleon
squirrel
flickr
```

But the run did not cover the actual supported 9-dataset universe and therefore
must not be treated as:

```text
9-dataset smoke pass
9-dataset smoke fail
V57A full expansion evidence
```

No model hyperparameter, seed, formula, or code behavior is changed based on
this invalid attempt.

## 4. Corrective Action

Write:

```text
V57A_EXPANSION_REVIEW_AMENDMENT.md
```

It must replace the unsupported command with the supported local 9-dataset
universe:

```text
acm,dblp,pubmed,wiki,flickr,blogcatalog,squirrel,texas,chameleon
```

Only that corrected 80-epoch smoke may be used for the V57A 9-dataset verdict.

## 5. No-Fabrication Status

This document records a command/protocol error only. It does not reinterpret
partial metric rows as scientific evidence.
