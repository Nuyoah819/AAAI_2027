# v50a_spectral_compactness_anchor First Smoke Verdict

This file records the first-stage smoke result for
`v50a_spectral_compactness_anchor`. It is a post-run verdict, not a change to
the preregistration or implementation review.

## Run

Implementation sanity checks:

```powershell
conda run -n aaai-e2e-subspace python -m py_compile core\e2e\sect_coco_e2e.py scripts\run_unified_aptc_9datasets.py
conda run --no-capture-output -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v50a_spectral_compactness_anchor --datasets "acm" --epochs 1 --device cuda --log-level WARNING
```

The first ACM 1-epoch command without `--no-capture-output` printed only a
Conda wrapper error. The same connectivity check with `--no-capture-output`
completed successfully:

```text
ACM 1 epoch: status=ok, ACC=0.7375
```

First-stage smoke command:

```powershell
conda run --no-capture-output -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v50a_spectral_compactness_anchor --datasets "acm,dblp,flickr" --epochs 80 --device cuda --log-level WARNING
```

The 80-epoch command was run twice. The first 80-epoch run completed, but then a
missing preregistered diagnostic field
`v50a_embedding_anchor_agreement` was noticed. The code change only added that
read-only diagnostic and did not change loss, constants, anchor construction, or
final-label logic. The verdict below uses the latest 80-epoch record per
dataset, which contains the complete diagnostic set.

Output files:

```text
results/archive/v40-v50/unified_aptc_9datasets_v50a_spectral_compactness_anchor.csv
results/archive/v40-v50/unified_aptc_9datasets_v50a_spectral_compactness_anchor_diagnostics.jsonl
```

## Implementation Summary

V50A adds a fixed stop-gradient spectral compactness anchor:

```text
X_dense -> row-l2 normalize
A_filter = row-normalized adjacency with self-loops
H_spec = A_filter^2 X
U_spec = TruncatedSVD(H_spec, rank=K)
q_spec = softmax(-||U_spec - center||^2 / 0.35)
loss += 0.04 * KL(q_refined || stopgrad(q_spec))
```

Registered constants:

| Field | Value |
| --- | ---: |
| `v50a_enabled` | true |
| `v50a_anchor_weight` | 0.04 |
| `v50a_filter_steps` | 2 |
| `v50a_anchor_rank_multiplier` | 1.0 |
| `v50a_anchor_temperature` | 0.35 |
| `v50a_anchor_refresh` | false |

Boundary:

- no dataset-specific branch;
- no legacy head;
- no S2CAG/ELSS/KMeans output as final label;
- no V43B-V49A failed loss family as active mechanism;
- labels are used only for post-training diagnostics already present in the
  project.

## Result Summary

Latest 80-epoch records:

| Dataset | ACC | NMI | ARI | V49A ACC Context | ACC Recovery |
| --- | ---: | ---: | ---: | ---: | ---: |
| ACM | 0.9088 | 0.6969 | 0.7487 | 0.6208 | +0.2880 |
| DBLP | 0.6810 | 0.4002 | 0.3425 | 0.6571 | +0.0240 |
| Flickr | 0.4397 | 0.2856 | 0.1940 | 0.3376 | +0.1021 |

Anchor diagnostics:

| Dataset | Anchor ACC | Anchor NMI | Anchor ARI | Anchor Entropy | Anchor Conf | Usage Entropy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ACM | 0.8942 | 0.6630 | 0.7144 | 0.9057 | 0.5340 | 0.9990 |
| DBLP | 0.8901 | 0.6958 | 0.7523 | 0.9339 | 0.4201 | 0.9983 |
| Flickr | 0.3626 | 0.2379 | 0.1381 | 0.9943 | 0.1346 | 0.9977 |

Coupling diagnostics:

| Dataset | Agreement @1 | Agreement @40 | Agreement @80 | Final KL | Emb-Anchor Agree |
| --- | ---: | ---: | ---: | ---: | ---: |
| ACM | 0.4301 | 0.8866 | 0.8982 | 0.0320 | 0.8982 |
| DBLP | 0.2963 | 0.4710 | 0.4972 | 0.1804 | 0.4590 |
| Flickr | 0.1053 | 0.1184 | 0.1251 | 0.0972 | 0.1122 |

Posterior/readout safety:

| Dataset | Embedding Posterior Gap | Gate |
| --- | ---: | --- |
| ACM | 0.0000 | PASS |
| DBLP | 0.0002 | PASS |
| Flickr | 0.0286 | PASS under 0.04 |

## Gate Verdict

### Red-Line Gate

PASS on 3/3:

| Dataset | Legacy | v43b | v44 | v44b | v45a | v46a | v47a | v48a | v49a | v50a |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ACM | false | false | false | false | false | false | false | false | false | true |
| DBLP | false | false | false | false | false | false | false | false | false | true |
| Flickr | false | false | false | false | false | false | false | false | false | true |

### Anchor Non-Degeneracy Gate

PASS on 3/3:

```text
v50a_anchor_cluster_usage_entropy >= 0.60
v50a_anchor_entropy finite
v50a_q_anchor_kl finite
```

All three usage entropies are above 0.997.

### Coupling Gate

PASS on 3/3:

```text
v50a_q_anchor_agreement improves from epoch 1 to epoch 80
```

ACM has strong coupling. DBLP has moderate coupling. Flickr coupling is weak but
still moves in the intended direction.

### Posterior/Readout Safety Gate

PASS:

```text
abs(embedding_posterior_gap) <= 0.02 on at least 2/3
abs(embedding_posterior_gap) <= 0.04 on 3/3
```

ACM and DBLP are essentially zero-gap. Flickr is 0.0286, inside the 0.04
ceiling.

### Performance Context Gate

PASS:

```text
ACM > V49A 0.6208
DBLP >= V49A 0.6571 - 0.01
Flickr > V49A 0.3376
```

V50A is not SOTA, but it decisively rescues the failed V49A direction on ACM and
Flickr while keeping DBLP stable.

## Interpretation

V50A supports the rescue hypothesis:

```text
The missing ingredient is a stable low-rank graph-attribute clustering basis,
not another topology-mask calibration loss.
```

The most important scientific signal is that the spectral anchor itself is
strong on ACM and DBLP, and the trainable posterior can couple to it without
breaking the unified pipeline. Flickr remains harder: its anchor is weak and
high-entropy, but even this weak anchor improves over the V49A context.

The route is alive.

## Limits

- The current anchor is too soft on Flickr: confidence is only 0.1346.
- DBLP final ACC is far below the anchor diagnostic ACC, so the model is not yet
  extracting the full spectral signal.
- The CSV contains an earlier 1-epoch connectivity row and an earlier 80-epoch
  row; verdict tables use the latest complete 80-epoch row per dataset.
- No second-batch, 9-dataset, or 260-epoch run has been executed after this
  verdict.

## Next Allowed Move

Because all first-stage gates pass, a second-stage smoke is now allowed, but it
should be preregistered before running. The next document should fix:

```text
V50A_SECOND_STAGE_PREREGISTRATION.md
```

Recommended second-stage scope:

```text
datasets = pubmed,wiki,blogcatalog,texas,squirrel,chameleon
epochs = 80
seed = 42
device = cuda
```

No weight, rank, temperature, filter-step, or refresh sweep is authorized by
this verdict.
