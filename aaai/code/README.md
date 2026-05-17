# End-to-End Heterophily-Aware Subspace Clustering

This directory contains a PyTorch prototype that keeps ELSS as the main
attributed graph clustering pipeline and injects a GREET-style frontend for
heterophily-aware edge discrimination.

- GREET frontend: differentiable edge homophily discrimination, low-pass
  homophilic channel, and high-pass heterophilic channel.
- ELSS main body: PageRank-guided anchor selection, differentiable Nyström
  low-rank basis extraction, and ELSS-style postprocessing for clustering.
- End-to-end objective: remove InfoNCE, optimize a subspace smoothness loss,
  and keep GREET's pivot-anchored ranking loss for the edge discriminator.

The implementation is intentionally independent from the original codebases, so
it can be modified without touching `ELSS/` or `GREET/`.

## Files

- `model.py`: core PyTorch modules, including the GREET-style discriminator,
  sparse dual-channel encoder, and differentiable Nyström module.
- `losses.py`: subspace smoothness and pivot-anchored ranking losses.
- `utils.py`: sparse graph loading, PageRank, structural encoding, and helper
  functions.
- `metrics.py`: clustering metrics (`ACC`, `NMI`, `ARI`).
- `train_smoke.py`: small CPU smoke test that verifies forward, backward, and
  parameter updates.
- `run_pubmed_clustering.py`: real clustering script for `PubMed`.
- `environment.yml`: conda environment specification.

## Quick Start

```powershell
conda activate aaai-e2e-subspace
python aaai/code/train_smoke.py
python aaai/code/run_pubmed_clustering.py --dataset pubmed --epochs 5 --eval-every 1 --num-anchors 96 --hidden-dim 32 --disc-hidden-dim 64 --ranking-weight 0.2
```

The smoke test uses a tiny synthetic graph and checks that gradients flow from
the subspace loss through the Nyström module back into the GNN and edge
discriminator.

The `PubMed` script reports clustering metrics directly:

- `ACC`: clustering accuracy after Hungarian matching.
- `NMI`: normalized mutual information.
- `ARI`: adjusted Rand index.

## Notes

- The current implementation is clustering-oriented, not node-classification
  oriented.
- The graph propagation path is sparse so it can handle larger ELSS-style
  attributed graphs such as `PubMed`.
- The ELSS postprocessing step uses a stabilized Gram eigendecomposition instead
  of a direct dense SVD, because the latter was numerically fragile on `PubMed`.
