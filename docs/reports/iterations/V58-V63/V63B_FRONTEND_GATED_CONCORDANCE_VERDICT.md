# V63B Frontend-Gated Concordance Verdict

Variant key:

```text
v63b_concordance_ood_dual_diffusion
```

Final output stem:

```text
unified_aptc_9datasets_v63b_frontend_gated_concordance_dual_diffusion
```

## 1. Implementation

V63B adds a reference-inspired, label-free frontend gate for attribute-structure
mismatch:

- edge concordance rescue on feature-KNN candidate edges;
- graph-level mismatch gate based on `(1 - attribute similarity) * (1 - attribute-structure concordance)`;
- high-pass suppression on concordant feature edges when the graph-level mismatch gate is active;
- deterministic edge OOD diagnostic helper retained, but its loss weight is disabled in the final variant;
- V62A self-distillation guard is restored unchanged in the final variant.

The final V63B choice keeps only the frontend-gated rescue active. The earlier
confusion-aware teacher guard improved some noisy cases but increased late-run
regression risk, so it is disabled for the final run.

## 2. Verification

Static check:

```bash
scripts/run_in_venv.sh -m py_compile core/e2e/sect_coco_e2e.py scripts/run_unified_aptc_9datasets.py
```

Smoke:

```bash
scripts/run_in_venv.sh scripts/run_unified_aptc_9datasets.py --variant v63b_concordance_ood_dual_diffusion --datasets texas --epochs 2 --device cpu --log-level WARNING
```

Full run:

```bash
scripts/run_in_venv.sh scripts/run_unified_aptc_9datasets.py --variant v63b_concordance_ood_dual_diffusion --datasets acm,dblp,pubmed,wiki,flickr,blogcatalog,squirrel,texas,chameleon --epochs 260 --device cuda --log-level INFO
```

Run status: 9/9 datasets completed with `status=ok`.

## 3. Full-Run Results

| Dataset | ACC | NMI | ARI |
| --- | ---: | ---: | ---: |
| ACM | 0.9081 | 0.6927 | 0.7467 |
| DBLP | 0.7266 | 0.4770 | 0.4358 |
| PubMed | 0.5163 | 0.0793 | 0.0761 |
| Wiki | 0.4216 | 0.3902 | 0.2312 |
| Flickr | 0.2866 | 0.1681 | 0.0868 |
| BlogCatalog | 0.8507 | 0.6672 | 0.6752 |
| Squirrel | 0.2415 | 0.0148 | 0.0088 |
| Texas | 0.7213 | 0.4699 | 0.5608 |
| Chameleon | 0.3404 | 0.1635 | 0.0730 |

## 4. Comparison Against V62A Full Run

| Dataset | V62A ACC | V63B ACC | Delta ACC | V62A NMI | V63B NMI | Delta NMI | V62A ARI | V63B ARI | Delta ARI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ACM | 0.9131 | 0.9081 | -0.0050 | 0.7054 | 0.6927 | -0.0127 | 0.7606 | 0.7467 | -0.0139 |
| DBLP | 0.7190 | 0.7266 | +0.0076 | 0.4778 | 0.4770 | -0.0008 | 0.4391 | 0.4358 | -0.0033 |
| PubMed | 0.5203 | 0.5163 | -0.0041 | 0.0848 | 0.0793 | -0.0055 | 0.0815 | 0.0761 | -0.0054 |
| Wiki | 0.3277 | 0.4216 | +0.0940 | 0.2886 | 0.3902 | +0.1016 | 0.1340 | 0.2312 | +0.0971 |
| Flickr | 0.2964 | 0.2866 | -0.0098 | 0.1740 | 0.1681 | -0.0060 | 0.0894 | 0.0868 | -0.0026 |
| BlogCatalog | 0.8537 | 0.8507 | -0.0031 | 0.6687 | 0.6672 | -0.0015 | 0.6830 | 0.6752 | -0.0078 |
| Squirrel | 0.2103 | 0.2415 | +0.0311 | 0.0133 | 0.0148 | +0.0015 | 0.0003 | 0.0088 | +0.0085 |
| Texas | 0.7213 | 0.7213 | +0.0000 | 0.4602 | 0.4699 | +0.0096 | 0.5741 | 0.5608 | -0.0133 |
| Chameleon | 0.3390 | 0.3404 | +0.0013 | 0.1637 | 0.1635 | -0.0002 | 0.0754 | 0.0730 | -0.0024 |

Aggregate deltas versus V62A:

| Metric | Avg Delta | Wins |
| --- | ---: | ---: |
| ACC | +0.0125 | 4/9 |
| NMI | +0.0096 | 3/9 |
| ARI | +0.0063 | 2/9 |

## 5. Verdict

V63B improves average full-run clustering quality over V62A and notably repairs
Wiki and Squirrel relative to the V62A long-run failure mode. It does not solve
the project SOTA target: the gains are not broad enough, and Flickr remains
below the V62A full-run result.

Recommended next step: keep the graph-level mismatch gate, but replace the
current feature-edge rescue with a reliability-normalized dual diffusion branch
so Flickr can benefit without weakening ACM/PubMed/BlogCatalog.
