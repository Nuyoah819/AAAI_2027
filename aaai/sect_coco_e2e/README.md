# SECT-CoCo-E2E — Unified End-to-End Graph Clustering

The latest evolution of our AAAI 2027 method. A single end-to-end differentiable
pipeline for attributed graph clustering that works across both homophilic and
heterophilic graphs with **one identical forward pass, loss, and cluster
assigner** — no dataset-specific routing.

Mirrored to https://github.com/Nuyoah819/AAAI_2027 under `aaai/sect_coco_e2e/`
automatically after each editing session (via the Stop hook in
`.claude/settings.json`).

## Pipeline

```
edge confidence  →  differentiable topology contraction  →
frequency-aware graph filtering  →  APTC clustering head
```

1. **Edge confidence** — Dirichlet-smoothed (ε_α=0.08) fusion of three evidence
   sources into a per-edge homophily score.
2. **Topology contraction** — learnable ordered sigmoid thresholds `(l, h)` soft
   tri-partition edges into homophilic / heterophilic / ambiguous.
3. **Frequency-aware filtering** — low-pass on the homophilic subgraph +
   high-pass on the heterophilic subgraph.
4. **APTC head** — Sinkhorn optimal transport + topology refinement, with
   residual correction `Δ=ReLU(Q_embed−Q_base)`,
   `Q_mix=normalize(Q_base+α·g_amp·Δ)` (`g_amp` has a soft floor of 0.15).

## Directory layout

```
core/
  e2e/sect_coco_e2e.py      # main model (2500+ lines)
  data/data_utils.py        # 9-dataset loading
  eval/metrics.py           # ACC / NMI / ARI
  legacy/sect_coco_legacy.py
scripts/
  run_unified_aptc_9datasets.py   # 9-dataset experiment runner
  run_e2e_experiments.py
  analysis/                       # homophily, table builders
  baselines/                      # ELSS + lightweight baselines
results/                  # curated key results (tables, CSVs, summaries)
CLAUDE.md                 # project instructions (read first)
CRITICAL_RED_LINES.md     # non-negotiable constraints (read before any change)
频率解耦.md                # core problem statement
README_Algorithm_Evolution.md  # algorithm evolution log v0–v24o (avoid repeating failed ideas)
README_0617_Original.md
```

## Datasets

| Dataset | Homophily (h) | Type |
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

Datasets are loaded by `core/data/data_utils.py`. The default data root is
`D:\study\graduate_student\papers\AAAI2027\data` (overridable via the `data_root`
argument of `load_dataset`). Datasets themselves are **not** vendored — place
`.mat` / raw `npz` / Geom-GCN files under your data root before running.

## Quick start

```powershell
conda activate aaai-e2e-subspace
# smoke run (80 epochs, reduced batch — rapid diagnosis)
python scripts/run_unified_aptc_9datasets.py --smoke
# full run (260 epochs, full batch — final results)
python scripts/run_unified_aptc_9datasets.py
```

Evaluation protocol: fixed seed 42; report ACC / NMI / ARI per dataset.

## SOTA targets

Tracked against the AAAI0617 multi-head model — see `CLAUDE.md` for the per-table
targets and `results/main_results_tables_9datasets.tex` for current numbers.

## Maintenance notes

- **Read `CRITICAL_RED_LINES.md` before any code change.** The red lines
  (no dataset-specific modules, one unified pipeline, end-to-end differentiable,
  preserve the four core innovations) are non-negotiable.
- **Read `README_Algorithm_Evolution.md` before proposing a new architecture** —
  it logs v0–v24o and prevents repeating failed ideas.
- `DATA_ROOT` in `core/data/data_utils.py` is a hardcoded absolute Windows path.
  Pass `data_root=` explicitly or refactor to an env var when porting to another
  machine.
- `results/` here is a **curated** subset (main tables, 9-dataset CSV, baseline
  summaries). The large per-run `*_diagnostics.jsonl` logs and `.err` / `.out`
  captures are intentionally excluded — see `.gitignore`.
