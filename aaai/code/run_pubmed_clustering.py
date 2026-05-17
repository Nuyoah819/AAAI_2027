from __future__ import annotations

import argparse
import os
import time

import torch

from metrics import evaluate_clustering
from model import EndToEndUnifiedClustering, elss_cluster_assignments
from utils import load_mat_attributed_graph, sample_pagerank_anchors, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ELSS-oriented end-to-end clustering on PubMed.")
    parser.add_argument("--dataset", type=str, default="pubmed")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--disc-hidden-dim", type=int, default=128)
    parser.add_argument("--gnn-layers", type=int, default=2)
    parser.add_argument("--num-anchors", type=int, default=256)
    parser.add_argument("--ranking-weight", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--structure-dim", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-every", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    # Reuse ELSS-style attributed graph input and labels so the end product is
    # clustering quality rather than node classification accuracy.
    data = load_mat_attributed_graph(
        args.dataset,
        root=root,
        structure_dim=args.structure_dim,
    )

    x = data.x.float()
    discriminator_inputs = torch.cat([x, data.structural_encoding], dim=1).float()
    edge_index = data.edge_index.long()
    edge_weight = data.edge_weight.float()
    norm_adj = data.norm_adj.float()
    # Anchors are sampled once from PageRank scores, following the ELSS view
    # that structural landmarks stay fixed while the model is optimized.
    anchor_indices = sample_pagerank_anchors(data.pagerank_scores, args.num_anchors)

    model = EndToEndUnifiedClustering(
        feature_dim=x.size(1),
        discriminator_input_dim=discriminator_inputs.size(1),
        hidden_dim=args.hidden_dim,
        rank=data.num_clusters,
        discriminator_hidden_dim=args.disc_hidden_dim,
        gnn_layers=args.gnn_layers,
        dropout=0.1,
        high_pass_alpha=0.5,
        gumbel_temperature=0.7,
        hard_gumbel=False,
        nystrom_ridge=1e-4,
        ranking_weight=args.ranking_weight,
        margin_homo=0.3,
        margin_hetero=0.3,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best = {"acc": -1.0, "nmi": -1.0, "ari": -1.0, "epoch": -1}
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(x, discriminator_inputs, edge_index, edge_weight, anchor_indices, norm_adj)
        output.loss.backward()
        optimizer.step()
        print(
            f"train epoch={epoch:03d} "
            f"loss={output.loss.item():.6f} "
            f"subspace={output.subspace_loss.item():.6f} "
            f"ranking={output.ranking_loss.item():.6f}",
            flush=True,
        )

        if epoch % args.eval_every == 0 or epoch == 1 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                output = model(x, discriminator_inputs, edge_index, edge_weight, anchor_indices, norm_adj)
                # Final clustering still follows the ELSS route:
                # low-rank basis -> postprocessing -> KMeans -> ACC/NMI/ARI.
                pred = elss_cluster_assignments(output.basis, data.num_clusters, random_state=args.seed)
                nmi, ari, acc = evaluate_clustering(data.labels, pred)
                if acc > best["acc"]:
                    best.update({"acc": acc, "nmi": nmi, "ari": ari, "epoch": epoch})
                print(
                    f"epoch={epoch:03d} "
                    f"loss={output.loss.item():.6f} "
                    f"subspace={output.subspace_loss.item():.6f} "
                    f"ranking={output.ranking_loss.item():.6f} "
                    f"ACC={acc:.4f} NMI={nmi:.4f} ARI={ari:.4f}"
                )

    elapsed = time.time() - start
    print(
        f"BEST epoch={best['epoch']:03d} "
        f"ACC={best['acc']:.4f} NMI={best['nmi']:.4f} ARI={best['ari']:.4f} "
        f"time={elapsed:.2f}s"
    )


if __name__ == "__main__":
    main()
