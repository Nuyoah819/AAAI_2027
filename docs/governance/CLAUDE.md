# SECT-CoCo-E2E - Unified End-to-End Graph Clustering (AAAI 2027)

## Project directory

`D:\study\graduate_student\papers\AAAI2027\AAAI0622\`

## Workspace versioning

This visible workspace is not a Git repository. Do not run `git init`,
`git add`, `git commit`, `git status`, or `git push` here as part of the
normal workflow.

A Stop hook mirrors the edited files into a hidden clone repository and
performs the actual `commit + push` automatically after each editing session.

## One-line summary

Unified end-to-end graph clustering pipeline:
edge confidence -> differentiable topology contraction -> frequency-aware graph
filtering -> APTC clustering head.

## Key files

| Purpose | Path |
|---|---|
| Main model | `core/e2e/sect_coco_e2e.py` |
| 9-dataset experiment runner | `scripts/run_unified_aptc_9datasets.py` |
| Data loading | `core/data/data_utils.py` |
| Evaluation metrics | `core/eval/metrics.py` |
| Algorithm evolution log | `docs/evolution/README_Algorithm_Evolution.md` |
| Red lines (must read first) | `docs/governance/CRITICAL_RED_LINES.md` |
| Core problem statement | `docs/background/频率解耦.md` |
| Documentation index | `docs/README.md` |
| Legacy multi-head reference | `core/legacy/sect_coco_legacy.py` |
| Results and tables | `results/` |
| Reference papers | `reference_md/` |

## 9 datasets

| Dataset | Homophily (h) | Classifier |
|---|---:|---|
| ACM | 0.821 | Homophilic |
| DBLP | 0.670 | Homophilic |
| PubMed | 0.802 | Homophilic |
| Wiki | 0.610 | Mixed |
| Flickr | 0.239 | Heterophilic |
| BlogCatalog | 0.401 | Heterophilic |
| Texas | 0.108 | Strong heterophilic |
| Squirrel | 0.223 | Heterophilic |
| Chameleon | 0.235 | Heterophilic |

## SOTA targets (from AAAI0617 multi-head model)

| Dataset | ACC | NMI | ARI |
|---|---:|---:|---:|
| ACM | 93.62 | 75.88 | 81.89 |
| DBLP | 93.69 | 79.74 | 84.83 |
| PubMed | 76.17 | 37.71 | 42.66 |
| Wiki | 64.82 | 59.79 | 48.51 |
| Flickr | 83.89 | 71.25 | 67.52 |
| BlogCatalog | 91.72 | 78.60 | 81.63 |
| Texas | 74.32 | 51.49 | 60.86 |
| Squirrel | 30.51 | 6.28 | 5.47 |
| Chameleon | 35.84 | 16.85 | 6.63 |

## Red lines (non-negotiable)

0. No dataset-specific modules: 9 datasets share identical forward pass, loss,
   and cluster assigner.
1. Unified pipeline: one code path, no `if dataset == 'X'` branching.
2. End-to-end differentiable: gradients must flow from the loss back to all
   front-end parameters.
3. Preserve core innovations: edge confidence, topology contraction, and
   frequency filtering must stay.
4. Target SOTA: solve attribute-structure mismatch and hit the tracked targets.

## Evaluation protocol

- Seed: fixed 42
- Smoke run: 80 epochs reduced batch
- Full run: 260 epochs full batch
- Report ACC / NMI / ARI per dataset

## Development rules

- Use the project virtual environment `.venv` for all code execution. Prefer
  `scripts/run_in_venv.sh <python-script> [args...]`; for interactive debugging,
  run `source .venv/bin/activate`.
- Read `docs/governance/CRITICAL_RED_LINES.md` before any code change.
- Read `docs/evolution/README_Algorithm_Evolution.md` before proposing new
  architecture to avoid repeating failed ideas.
- If stuck on a dataset, diagnose edge scores, partition quality, and the
  embedding-posterior gap first.
- No dataset-specific routing. Robustness must come from unified math, not
  conditional branches.
