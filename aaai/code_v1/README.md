# End-to-End Heterophily-Aware Subspace Clustering

This directory contains the first PyTorch prototype that directly fuses the
main ideas of GREET and ELSS:

- GREET frontend: differentiable edge homophily discrimination, low-pass
  homophilic channel, and high-pass heterophilic channel.
- ELSS backend: PageRank-guided anchor selection and differentiable Nyström
  low-rank subspace extraction.
- End-to-end objective: replace InfoNCE with a subspace smoothness objective
  and keep GREET's pivot-anchored ranking loss for the edge discriminator.

This version is the original lightweight prototype before we later rewired the
project around ELSS-style attributed graph clustering on PubMed.

## Files

- `model.py`: core PyTorch modules.
- `losses.py`: subspace and ranking losses.
- `utils.py`: graph utilities, PageRank, normalization, synthetic data.
- `train_smoke.py`: small CPU smoke test that verifies forward, backward, and
  parameter updates.
- `environment.yml`: conda environment specification.

## Quick Start

```powershell
conda activate aaai-e2e-subspace
python aaai/code_v1/train_smoke.py
```

The smoke test uses a tiny synthetic graph and checks that gradients flow from
the subspace loss through the Nyström module back into the GNN and edge
discriminator.
