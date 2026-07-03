# SECT-CoCo-E2E - Unified End-to-End Graph Clustering

The latest evolution of our AAAI 2027 method. A single end-to-end differentiable
pipeline for attributed graph clustering that works across both homophilic and
heterophilic graphs with one identical forward pass, loss, and cluster
assigner, with no dataset-specific routing.

This project is maintained as a Git repository and is associated with
[Nuyoah819/AAAI_2027](https://github.com/Nuyoah819/AAAI_2027).

## Repository workflow

This working directory is the canonical local Git worktree for SECT-CoCo-E2E.
Use normal Git commands from the project root:

```bash
git status
git add <files>
git commit -m "your message"
```

The GitHub remote is:

```bash
https://github.com/Nuyoah819/AAAI_2027.git
```

The existing GitHub repository also contains earlier project material outside
this directory. Coordinate branch and layout choices before pushing changes to
`master`, especially if converting the historical `aaai/sect_coco_e2e/` mirror
into this standalone project layout.

Local datasets, virtual environments, Python caches, logs, and regenerable
diagnostics are intentionally excluded by `.gitignore`.

## Pipeline

```text
edge confidence -> differentiable topology contraction -> frequency-aware graph filtering -> APTC clustering head
```

1. Edge confidence: Dirichlet-smoothed fusion of three evidence sources into a
   per-edge homophily score.
2. Topology contraction: learnable ordered sigmoid thresholds `(l, h)` softly
   tri-partition edges into homophilic, heterophilic, and ambiguous groups.
3. Frequency-aware filtering: low-pass on the homophilic subgraph plus
   high-pass on the heterophilic subgraph.
4. APTC head: Sinkhorn optimal transport plus topology refinement, with
   residual correction and normalized posterior mixing.

## Directory layout

```text
core/
  e2e/sect_coco_e2e.py          # main model
  data/data_utils.py            # 9-dataset loading
  eval/metrics.py               # ACC / NMI / ARI
  legacy/sect_coco_legacy.py
scripts/
  run_unified_aptc_9datasets.py # 9-dataset experiment runner
  run_e2e_experiments.py
  analysis/                     # homophily, table builders
  baselines/                    # ELSS + lightweight baselines
results/                        # experiment outputs and historical snapshots
  auxiliary/                    # probes, literature extracts, baseline checks
  archive/                      # versioned unified_aptc historical snapshots
  logs/                         # raw stdout / stderr / run logs
docs/
  README.md                     # documentation index
  governance/                   # project instructions and red lines
  background/                   # problem statement and early notes
  evolution/                    # algorithm evolution log
  planning/                     # route selection notes
  reports/iterations/           # V44A-V63A iteration reports
  writing/                      # manuscript and presentation materials
```

## Datasets

| Dataset | Homophily (h) | Type |
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

Datasets are loaded by `core/data/data_utils.py`. The default data root is
`/mnt/data/users/liusong/data` and can be overridden through the `data_root`
argument of `load_dataset`.

## Quick start

Create or refresh the project virtual environment:

```bash
scripts/setup_env.sh
```

Run all project commands through the checked-in wrapper so they use `.venv`:

```bash
scripts/run_in_venv.sh scripts/run_unified_aptc_9datasets.py --datasets texas --epochs 1
scripts/run_in_venv.sh scripts/run_unified_aptc_9datasets.py
```

For the CSTC implementation under `CODE/`, use the same environment wrapper:

```bash
scripts/run_in_venv.sh CODE/run_cstc_experiments.py --datasets texas --smoke
```

Evaluation protocol: fixed seed 42; report ACC / NMI / ARI per dataset.

## SOTA targets

Tracked against the AAAI0617 multi-head model. See
`docs/governance/CLAUDE.md` for the per-table targets and
`results/main_results_tables_9datasets.tex` for current numbers.

## Maintenance notes

- Read `docs/governance/CRITICAL_RED_LINES.md` before any code change.
- Read `docs/evolution/README_Algorithm_Evolution.md` before proposing a new
  architecture to avoid repeating failed ideas.
- `DATA_ROOT` in `core/data/data_utils.py` is a hardcoded absolute local path:
  `/mnt/data/users/liusong/data`. Pass `data_root=` explicitly or refactor to an
  env var when porting to another machine.
- Use `scripts/run_in_venv.sh ...` for project commands so they run with the
  project virtual environment at `.venv`.
- `results/` keeps runnable outputs and historical snapshots. Put probes,
  literature notes, and baseline checks under `results/auxiliary/`, and raw
  command captures under `results/logs/`.
