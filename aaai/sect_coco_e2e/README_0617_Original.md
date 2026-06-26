# SECT-CoCo Codebase Cleanup

This directory has been reorganized around the latest end-to-end SECT-CoCo implementation.

## What Is The Mainline

The current research mainline is the end-to-end pipeline under `core/e2e/`.

- `core/e2e/sect_coco_e2e.py`: latest E2E SECT-CoCo model and training loop
- `scripts/run_e2e_experiments.py`: main experiment entry point for the latest model
- `core/data/data_utils.py`: dataset loading and preprocessing
- `core/eval/metrics.py`: clustering metrics

The E2E implementation was identified as the newest version because it contains:

- learnable ordered thresholds implemented with `nn.Parameter`
- soft homophily, heterophily, and hard masks in one differentiable forward pass
- a single optimizer and a unified `loss.backward()` update path

## Legacy Compatibility

The older alternating SECT-CoCo pipeline has been isolated under `core/legacy/` and `archive/legacy_pipeline/`.

- `core/legacy/sect_coco_legacy.py` is intentionally retained because the latest E2E code still reuses some legacy heads and bridge modes for dataset-specific final labeling
- `archive/legacy_pipeline/` stores the old experiment runners and search scripts
- `archive/probes/` stores one-off diagnostic scripts that are not part of the main workflow

## Directory Tree

```text
CODE/
|-- README.md
|-- cleanup_report.md
|-- core/
|   |-- data/
|   |   |-- data_utils.py
|   |-- e2e/
|   |   |-- sect_coco_e2e.py
|   |-- eval/
|   |   |-- metrics.py
|   |-- legacy/
|   |   |-- sect_coco_legacy.py
|-- scripts/
|   |-- run_e2e_experiments.py
|   |-- analysis/
|   |   |-- build_nine_dataset_tables.py
|   |   |-- compute_homophily.py
|   |-- baselines/
|   |   |-- run_elss0610_baseline.py
|   |   |-- run_elss_on_local_datasets.py
|   |   |-- run_lightweight_baselines.py
|   |   |-- run_missing_fast_baselines.py
|-- archive/
|   |-- legacy_pipeline/
|   |   |-- run_experiments.py
|   |   |-- grid_search_seed42.py
|   |-- probes/
|   |   |-- compare_elss_loaders.py
|   |   |-- probe_fast_elss_align.py
|   |   |-- probe_pubmed.py
|   |   |-- probe_pubmed_graph.py
|-- results/
|-- __pycache__/
```

## What Each Area Does

### `core/`

This folder contains reusable code that defines the cleaned project backbone.

- `core/e2e/`: latest end-to-end SECT-CoCo
- `core/data/`: data loading, normalization, and dataset adapters
- `core/eval/`: evaluation helpers
- `core/legacy/`: compatibility layer for old SECT-CoCo components still referenced by the E2E code

### `scripts/`

This folder contains supported runnable scripts.

- `scripts/run_e2e_experiments.py`: the main runner for the cleaned E2E pipeline
- `scripts/analysis/`: paper-oriented analysis scripts
- `scripts/baselines/`: retained baseline runners for comparison experiments

All of these scripts now resolve outputs back into `CODE/results/`.

### `archive/`

This folder contains historical or diagnostic code that should not be confused with the current open-source mainline.

- `archive/legacy_pipeline/`: the old alternating-training SECT-CoCo experiment flow
- `archive/probes/`: one-off PubMed and ELSS debugging probes

## Recommended Entry Points

Run the latest E2E experiments:

```powershell
python D:\study\graduate_student\papers\AAAI2027\AAAI0617\CODE\scripts\run_e2e_experiments.py --datasets "acm,dblp,pubmed,wiki,flickr,blogcatalog,squirrel,texas,chameleon"
```

Regenerate the nine-dataset LaTeX table:

```powershell
python D:\study\graduate_student\papers\AAAI2027\AAAI0617\CODE\scripts\analysis\build_nine_dataset_tables.py
```

Recompute dataset homophily statistics:

```powershell
python D:\study\graduate_student\papers\AAAI2027\AAAI0617\CODE\scripts\analysis\compute_homophily.py
```

## Notes

- No files were deleted during this cleanup.
- Suspicious or one-off scripts were archived instead of removed.
- `cleanup_report.md` records which scripts look unused or diagnostic-only and why.
