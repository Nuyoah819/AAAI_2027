from __future__ import annotations

import argparse
import os
import time

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.cluster import KMeans

from metrics import evaluate_clustering
from model import square_feat_map
from utils import (
    load_mat_attributed_graph,
    pagerank_scores,
    sample_pagerank_anchors,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce the original ELSS-style ACM clustering path.")
    parser.add_argument("--dataset", type=str, default="acm")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--power", type=int, default=2)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--num-anchors", type=int, default=500)
    parser.add_argument("--alpha2", type=float, default=5e-5)
    parser.add_argument("--pagerank-damping", type=float, default=0.875)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def original_normalized_adjacency(data) -> torch.Tensor:
    adj = data.adj.astype(np.float64)
    adj = adj + sp.eye(adj.shape[0], format=adj.format)
    adj = sp.coo_matrix(adj)
    row_sum = np.asarray(adj.sum(1)).ravel()
    inv_sqrt = np.power(row_sum, -0.5, where=row_sum > 0)
    inv_sqrt[~np.isfinite(inv_sqrt)] = 0.0
    sym_adj = sp.diags(inv_sqrt).dot(adj).dot(sp.diags(inv_sqrt)).tocoo()

    row_adj = sym_adj + sp.eye(sym_adj.shape[0], format=sym_adj.format)
    row_sum = np.asarray(row_adj.sum(1)).ravel()
    inv = np.power(row_sum, -1.0, where=row_sum > 0)
    inv[~np.isfinite(inv)] = 0.0
    row_adj = sp.diags(inv).dot(row_adj).tocoo()

    indices = np.vstack((row_adj.row, row_adj.col)).astype(np.int64)
    dense = torch.sparse_coo_tensor(
        torch.from_numpy(indices),
        torch.from_numpy(row_adj.data.astype(np.float64)),
        row_adj.shape,
        dtype=torch.float64,
    ).to_dense()
    return dense


def original_elss_nystrom(
    h: torch.Tensor,
    laplacian: torch.Tensor,
    anchors: torch.Tensor,
    *,
    alpha2: float,
    rank: int,
) -> torch.Tensor:
    h_anchor = h.index_select(0, anchors)
    kernel_anchor = h_anchor.matmul(h_anchor.t())
    l_cols = laplacian[:, anchors]
    w = kernel_anchor - alpha2 * l_cols[anchors, :]
    c = h.matmul(h_anchor.t()) - alpha2 * l_cols

    w = 0.5 * (w + w.t())
    evals, evecs = torch.linalg.eigh(w + 1e-12 * torch.eye(w.size(0), dtype=w.dtype))
    mask = evals > 1e-12
    if not torch.any(mask):
        raise RuntimeError("All Nyström anchor eigenvalues are non-positive.")
    q = c.matmul(evecs[:, mask]).matmul(torch.diag(evals[mask].rsqrt()))
    u_full, _, _ = torch.linalg.svd(q, full_matrices=False)
    return u_full[:, :rank]


def cluster_from_basis(basis: torch.Tensor, num_clusters: int, seed: int) -> np.ndarray:
    z = square_feat_map(basis)
    u_full, _, _ = torch.linalg.svd(z, full_matrices=False)
    q = u_full[:, 1 : num_clusters + 1].detach().cpu().numpy()
    return KMeans(n_clusters=num_clusters, random_state=seed, n_init=10).fit_predict(q)


def run_experiment(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data = load_mat_attributed_graph(
        args.dataset,
        root=root,
        data_path=args.data_path,
        data_root=args.data_root,
    )
    x = data.x.double()
    edge_index = data.edge_index.long()
    norm_adj = original_normalized_adjacency(data)
    laplacian = torch.eye(x.size(0), dtype=torch.float64) - norm_adj

    h = x
    for _ in range(args.power):
        h = norm_adj.matmul(h)

    pr = pagerank_scores(edge_index, x.size(0), damping=args.pagerank_damping).double()
    anchors = sample_pagerank_anchors(pr.float(), num_anchors=min(args.num_anchors, x.size(0)))

    start = time.time()
    basis = original_elss_nystrom(
        h,
        laplacian,
        anchors,
        alpha2=args.alpha2,
        rank=args.rank,
    )
    pred = cluster_from_basis(basis, data.num_clusters, args.seed)
    nmi, ari, acc = evaluate_clustering(data.labels, pred)
    elapsed = time.time() - start
    print(
        f"ELSS_BASELINE dataset={args.dataset} nodes={x.size(0)} features={x.size(1)} "
        f"clusters={data.num_clusters} rank={args.rank} anchors={anchors.numel()} "
        f"ACC={acc:.4f} NMI={nmi:.4f} ARI={ari:.4f} time={elapsed:.2f}s",
        flush=True,
    )


def main() -> None:
    run_experiment(parse_args())


if __name__ == "__main__":
    main()
