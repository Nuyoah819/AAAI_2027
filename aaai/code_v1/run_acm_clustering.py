from __future__ import annotations

import argparse
import os
import time

import torch

from metrics import evaluate_clustering
from model import EndToEndUnifiedClustering, elss_cluster_assignments
from utils import (
    edge_index_to_dense_adj,
    load_mat_attributed_graph,
    normalized_laplacian_from_adj,
    pagerank_scores,
    sample_pagerank_anchors,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run code_v1 clustering on ACM.")
    parser.add_argument("--dataset", type=str, default="acm")
    parser.add_argument("--data-path", type=str, default=None, help="Explicit path to <dataset>.mat.")
    parser.add_argument("--data-root", type=str, default=None, help="Directory containing <dataset>.mat.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--disc-hidden-dim", type=int, default=128)
    parser.add_argument("--gnn-layers", type=int, default=2)
    parser.add_argument("--rank", type=int, default=None, help="Nyström rank. Defaults to num_clusters + 1.")
    parser.add_argument("--num-anchors", type=int, default=500)
    parser.add_argument("--pagerank-damping", type=float, default=0.875)
    parser.add_argument("--ranking-weight", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--high-pass-alpha", type=float, default=0.5)
    parser.add_argument("--gumbel-temperature", type=float, default=0.7)
    parser.add_argument("--nystrom-ridge", type=float, default=1e-4)
    parser.add_argument("--margin-homo", type=float, default=0.3)
    parser.add_argument("--margin-hetero", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-every", type=int, default=5)
    return parser.parse_args()


def run_experiment(args: argparse.Namespace) -> dict[str, float | int | str]:
    set_seed(args.seed)
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data = load_mat_attributed_graph(
        args.dataset,
        root=root,
        data_path=args.data_path,
        data_root=args.data_root,
    )

    x = data.x.float()
    edge_index = data.edge_index.long()
    num_anchors = min(args.num_anchors, x.size(0))

    pr = pagerank_scores(edge_index, x.size(0), damping=args.pagerank_damping)
    anchors = sample_pagerank_anchors(pr, num_anchors=num_anchors)
    raw_adj = edge_index_to_dense_adj(edge_index, x.size(0), make_symmetric=True, add_self_loops=False).to(dtype=x.dtype)
    laplacian = normalized_laplacian_from_adj(raw_adj)

    model = EndToEndUnifiedClustering(
        input_dim=x.size(1),
        hidden_dim=args.hidden_dim,
        rank=args.rank or data.num_clusters + 1,
        discriminator_hidden_dim=args.disc_hidden_dim,
        gnn_layers=args.gnn_layers,
        dropout=args.dropout,
        high_pass_alpha=args.high_pass_alpha,
        gumbel_temperature=args.gumbel_temperature,
        hard_gumbel=False,
        nystrom_ridge=args.nystrom_ridge,
        ranking_weight=args.ranking_weight,
        margin_homo=args.margin_homo,
        margin_hetero=args.margin_hetero,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(
        f"dataset={args.dataset} nodes={x.size(0)} features={x.size(1)} "
        f"edges={edge_index.size(1)} clusters={data.num_clusters} "
        f"rank={args.rank or data.num_clusters + 1} anchors={num_anchors}",
        flush=True,
    )

    best = {"acc": -1.0, "nmi": -1.0, "ari": -1.0, "epoch": -1}
    final = {"acc": -1.0, "nmi": -1.0, "ari": -1.0, "epoch": -1}
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(x, edge_index, anchors, laplacian=laplacian)
        if not torch.isfinite(output.loss):
            raise RuntimeError(f"Non-finite loss at epoch {epoch}: {output.loss.item()}")
        output.loss.backward()
        optimizer.step()

        print(
            f"train epoch={epoch:03d} loss={output.loss.item():.6f} "
            f"subspace={output.subspace_loss.item():.6f} "
            f"ranking={output.ranking_loss.item():.6f}",
            flush=True,
        )

        if epoch % args.eval_every == 0 or epoch == 1 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                output = model(x, edge_index, anchors, laplacian=laplacian)
                pred = elss_cluster_assignments(output.basis, data.num_clusters, random_state=args.seed)
                nmi, ari, acc = evaluate_clustering(data.labels, pred)
                final.update({"acc": acc, "nmi": nmi, "ari": ari, "epoch": epoch})
                if acc > best["acc"]:
                    best.update({"acc": acc, "nmi": nmi, "ari": ari, "epoch": epoch})
                print(
                    f"eval epoch={epoch:03d} ACC={acc:.4f} NMI={nmi:.4f} ARI={ari:.4f}",
                    flush=True,
                )

    elapsed = time.time() - start
    print(
        f"BEST epoch={best['epoch']:03d} ACC={best['acc']:.4f} "
        f"NMI={best['nmi']:.4f} ARI={best['ari']:.4f} time={elapsed:.2f}s",
        flush=True,
    )
    return {
        "dataset": args.dataset,
        "best_epoch": best["epoch"],
        "best_acc": best["acc"],
        "best_nmi": best["nmi"],
        "best_ari": best["ari"],
        "final_epoch": final["epoch"],
        "final_acc": final["acc"],
        "final_nmi": final["nmi"],
        "final_ari": final["ari"],
        "time_sec": elapsed,
    }


def main() -> None:
    run_experiment(parse_args())


if __name__ == "__main__":
    main()
