# SECT-CoCo-E2E — Unified End-to-End Graph Clustering (AAAI 2027)

## Project directory
`D:\study\graduate_student\papers\AAAI2027\AAAI0622\`

## Workspace versioning
This visible workspace is not a Git repository. Do not run `git init`,
`git add`, `git commit`, `git status`, or `git push` here as part of the normal
workflow.

A Stop hook mirrors the edited files into a hidden clone repository and
performs the actual `commit + push` automatically after each editing session.

## One-line summary
统一端到端图聚类框架，核心管线：边置信度评分 → 可微拓扑收缩 → 频率感知图滤波 → APTC 聚类头（Sinkhorn 最优传输 + 拓扑精修）。

## Key files
| Purpose | Path |
|---|---|
| Main model (2500+ lines) | `core/e2e/sect_coco_e2e.py` |
| 9-dataset experiment runner | `scripts/run_unified_aptc_9datasets.py` |
| Data loading | `core/data/data_utils.py` |
| Evaluation metrics | `core/eval/metrics.py` |
| Algorithm evolution log | `README_Algorithm_Evolution.md` |
| Red lines (MUST read first) | `CRITICAL_RED_LINES.md` |
| Core problem statement | `频率解耦.md` |
| Legacy multi-head reference | `core/legacy/sect_coco_legacy.py` |
| Results & tables | `results/` |
| Reference papers | `reference_md/` |

## 9 datasets
| Dataset | Homophily (h) | Classifier |
|---|---|---|
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
|---|---|---|---|
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
0. **No dataset-specific modules** — 9 datasets share identical forward pass, loss, cluster assigner
1. **Unified pipeline** — one code path, no `if dataset == 'X'` branching
2. **End-to-end differentiable** — gradient from loss back to all front-end params
3. **Preserve core innovations** — edge confidence, topology contraction, frequency filtering must stay
4. **Target SOTA** — solve attribute-structure mismatch, hit the numbers above

## Current algorithm state (v24g)
- **Edge confidence**: Dirichlet-smoothed (ε_α=0.08) fusion of 3 evidence sources
- **Topology contraction**: learnable ordered sigmoid thresholds (l, h), soft tri-partition edges → homo/hetero/ambiguous
- **Frequency filtering**: low-pass on homo subgraph + high-pass on hetero subgraph
- **APTC head**: Sinkhorn transport + topology refinement, with residual correction Δ=ReLU(Q_embed−Q_base), Q_mix=normalize(Q_base+α·g_amp·Δ), where g_amp has soft floor=0.15
- **Core bottleneck**: KMeans on learned embeddings ≈80% ACC on ACM, but APTC posterior only ≈41% — the transport head is losing most discriminative signal
- **Current frontier**: improvements on homophilic citation graphs (DBLP) conflict with structurally difficult large graphs (BlogCatalog)

## Evaluation protocol
- **Seed**: fixed 42
- **Smoke run**: 80 epochs reduced batch (for rapid diagnosis)
- **Full run**: 260 epochs full batch (for final results)
- Report ACC / NMI / ARI per dataset

## Development rules
- **执行任何代码前，必须先激活 Conda 环境 `aaai-e2e-subspace`**：`conda activate aaai-e2e-subspace`
- Read `CRITICAL_RED_LINES.md` before any code change
- Read `README_Algorithm_Evolution.md` before proposing new architecture to avoid repeating failed ideas (v0–v24o)
- If stuck on a dataset, diagnose edge scores / partition quality / embedding-posterior gap first
- No dataset-specific routing — robustness comes from unified math, not conditional branches
