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

## ACM Clustering

```powershell
conda activate aaai-e2e-subspace
python aaai/code_v1/run_acm_clustering.py --data-root /home/liusong/Project/AAAI/ELSS/data --epochs 30 --eval-every 5
```

The ACM script loads a MATLAB attributed graph file with `W`, `fea`, and `gnd`,
trains the end-to-end prototype, and reports clustering `ACC/NMI/ARI`.
For ACM, its defaults now mirror the original ELSS setup more closely:
`rank=num_clusters+1`, `num_anchors=500`, and PageRank damping `0.875`.

The unified path keeps the ELSS backend explicit while learning dynamic
homophilic/heterophilic edge weights. PubMed can be run with:

```powershell
python aaai/code_v1/run_unified_acm_clustering.py --dataset pubmed --data-root /home/liusong/Project/AAAI/aaai/data --epochs 1 --eval-every 1 --power 360 --dynamic-power 2 --num-anchors 200 --rank 4 --pagerank-damping 0.955 --dtype float64 --dynamic-strength 0.08 --edge-feedback-weight 0.001 --ranking-weight 0.0001 --lr 0.0001 --nystrom-graph mix --no-tf-idf
```
