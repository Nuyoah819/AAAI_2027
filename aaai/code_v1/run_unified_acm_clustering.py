from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from losses import pivot_anchored_ranking_loss, subspace_smoothness_loss
from metrics import evaluate_clustering
from model import square_feat_map
from utils import load_mat_attributed_graph, sample_pagerank_anchors, set_seed


@dataclass
class UnifiedResult:
    loss: torch.Tensor
    subspace_loss: torch.Tensor
    ranking_loss: torch.Tensor
    edge_feedback_loss: torch.Tensor
    orth_loss: torch.Tensor
    basis: torch.Tensor
    embeddings: torch.Tensor
    weights_homo: torch.Tensor
    weights_hetero: torch.Tensor


class EdgeDiscriminator(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        temperature: float = 1.0,
        hard_gumbel: bool = False,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.hard_gumbel = hard_gumbel
        self.node_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.edge_mlp = nn.Linear(2 * hidden_dim, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.node_mlp(x)
        src, dst = edge_index
        forward = self.edge_mlp(torch.cat([h[src], h[dst]], dim=-1)).squeeze(-1)
        backward = self.edge_mlp(torch.cat([h[dst], h[src]], dim=-1)).squeeze(-1)
        logits = 0.5 * (forward + backward) / self.temperature
        if self.training:
            weights_homo = F.gumbel_softmax(
                torch.stack([torch.zeros_like(logits), logits], dim=-1),
                tau=self.temperature,
                hard=self.hard_gumbel,
                dim=-1,
            )[:, 1]
        else:
            weights_homo = torch.sigmoid(logits)
        return weights_homo, 1.0 - weights_homo


def scipy_to_torch_sparse(matrix: sp.spmatrix, *, dtype: torch.dtype) -> torch.Tensor:
    matrix = matrix.tocoo()
    indices = np.vstack((matrix.row, matrix.col)).astype(np.int64)
    values = matrix.data.astype(np.float32 if dtype == torch.float32 else np.float64)
    return torch.sparse_coo_tensor(
        torch.from_numpy(indices),
        torch.from_numpy(values).to(dtype=dtype),
        matrix.shape,
        dtype=dtype,
    ).coalesce()


def normalized_adjacency(data, *, dtype: torch.dtype, mode: str) -> torch.Tensor:
    adj = data.adj.astype(np.float64)
    if mode in {"sym", "elss"}:
        adj = adj + sp.eye(adj.shape[0], format=adj.format)
        adj = sp.coo_matrix(adj)
        row_sum = np.asarray(adj.sum(1)).ravel()
        inv_sqrt = np.power(row_sum, -0.5, where=row_sum > 0)
        inv_sqrt[~np.isfinite(inv_sqrt)] = 0.0
        adj = sp.diags(inv_sqrt).dot(adj).dot(sp.diags(inv_sqrt)).tocoo()
        if mode == "sym":
            return scipy_to_torch_sparse(adj, dtype=dtype)

    row_adj = adj + sp.eye(adj.shape[0], format=adj.format)
    row_sum = np.asarray(row_adj.sum(1)).ravel()
    inv = np.power(row_sum, -1.0, where=row_sum > 0)
    inv[~np.isfinite(inv)] = 0.0
    row_adj = sp.diags(inv).dot(row_adj).tocoo()
    return scipy_to_torch_sparse(row_adj, dtype=dtype)


def pagerank_scores_from_scipy(
    adj: sp.spmatrix,
    *,
    damping: float,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> torch.Tensor:
    adj = adj.tocsr().astype(np.float64)
    row_sum = np.asarray(adj.sum(1)).ravel()
    inv = np.power(row_sum, -1.0, where=row_sum > 0)
    inv[~np.isfinite(inv)] = 0.0
    transition = sp.diags(inv).dot(adj).tocsr()
    n = transition.shape[0]
    pr = np.full(n, 1.0 / n, dtype=np.float64)
    teleport = np.full(n, 1.0 / n, dtype=np.float64)
    for _ in range(max_iter):
        prev = pr.copy()
        pr = (1.0 - damping) * teleport + damping * transition.T.dot(prev)
        if np.abs(pr - prev).sum() < tol:
            break
    return torch.from_numpy(pr.astype(np.float32))


@torch.no_grad()
def pagerank_scores_from_torch_sparse(
    adj: torch.Tensor,
    *,
    damping: float,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> torch.Tensor:
    adj = adj.coalesce()
    n = adj.size(0)
    row = adj.indices()[0]
    value = adj.values()
    degree = torch.zeros(n, dtype=value.dtype, device=value.device)
    degree.index_add_(0, row, value)
    pr = torch.full((n,), 1.0 / n, dtype=value.dtype, device=value.device)
    teleport = torch.full_like(pr, 1.0 / n)
    for _ in range(max_iter):
        prev = pr
        scaled = pr / degree.clamp_min(1e-12)
        pr = (1.0 - damping) * teleport + damping * torch.sparse.mm(adj.t(), scaled[:, None]).squeeze(1)
        if torch.norm(pr - prev, p=1) < tol:
            break
    return pr.float().cpu()


@torch.no_grad()
def sample_numpy_pagerank_anchors(scores: torch.Tensor, num_anchors: int, seed: int) -> torch.Tensor:
    probs = scores.double().cpu().numpy()
    probs = probs / np.maximum(probs.sum(), 1e-12)
    rng = np.random.RandomState(seed)
    indices = rng.choice(scores.numel(), size=num_anchors, replace=False, p=probs)
    return torch.from_numpy(indices.astype(np.int64))


def row_normalize_sparse(indices: torch.Tensor, values: torch.Tensor, size: tuple[int, int], eps: float = 1e-12) -> torch.Tensor:
    row = indices[0]
    degree = torch.zeros(size[0], dtype=values.dtype, device=values.device)
    degree.index_add_(0, row, values)
    norm_values = values / degree[row].clamp_min(eps)
    return torch.sparse_coo_tensor(indices, norm_values, size, dtype=values.dtype, device=values.device).coalesce()


def weighted_sparse_view(
    edge_index: torch.Tensor,
    weights: torch.Tensor,
    num_nodes: int,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    row, col = edge_index.to(device=weights.device)
    value = weights.to(dtype=dtype)
    indices = torch.cat([torch.stack([row, col]), torch.stack([col, row])], dim=1)
    values = torch.cat([value, value])
    diag = torch.arange(num_nodes, device=weights.device)
    indices = torch.cat([indices, torch.stack([diag, diag])], dim=1)
    values = torch.cat([values, torch.ones(num_nodes, dtype=dtype, device=weights.device)])
    return row_normalize_sparse(indices, values, (num_nodes, num_nodes))


def propagate(x: torch.Tensor, adj: torch.Tensor, power: int) -> torch.Tensor:
    h = x
    for _ in range(power):
        h = torch.sparse.mm(adj, h) if adj.is_sparse else adj.matmul(h)
    return h


def sparse_laplacian_columns(norm_adj: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    n = norm_adj.size(0)
    m = anchors.numel()
    col = torch.arange(m, device=anchors.device)
    selector = torch.sparse_coo_tensor(
        torch.stack([anchors, col]),
        torch.ones(m, dtype=norm_adj.dtype, device=anchors.device),
        (n, m),
        dtype=norm_adj.dtype,
        device=anchors.device,
    ).coalesce()
    cols = -torch.sparse.mm(norm_adj, selector.to_dense())
    cols[anchors, col] += 1.0
    return cols


def sparse_smoothness_loss(u: torch.Tensor, norm_adj: torch.Tensor) -> torch.Tensor:
    au = torch.sparse.mm(norm_adj, u) if norm_adj.is_sparse else norm_adj.matmul(u)
    return torch.sum(u * (u - au)) / max(1, u.size(0))


@torch.no_grad()
def structural_encoding(norm_adj: torch.Tensor, dim: int, dtype: torch.dtype) -> torch.Tensor:
    n = norm_adj.size(0)
    vec = torch.eye(n, dtype=dtype)[:, : min(n, dim)]
    # For large PubMed, diagonal random-walk returns are enough and cheap.
    adj = norm_adj.coalesce()
    row, col = adj.indices()
    val = adj.values()
    enc = [torch.zeros(n, dtype=dtype)]
    diag_mask = row == col
    if diag_mask.any():
        enc[0].index_add_(0, row[diag_mask], val[diag_mask])
    power_vec = torch.ones(n, 1, dtype=dtype) / n
    for _ in range(1, dim):
        power_vec = torch.sparse.mm(adj, power_vec)
        enc.append(power_vec.squeeze(1))
    return torch.stack(enc, dim=1)


class UnifiedSubspaceClustering(nn.Module):
    def __init__(
        self,
        input_dim: int,
        disc_input_dim: int,
        num_clusters: int,
        *,
        rank: int,
        discriminator_hidden_dim: int,
        power: int,
        dynamic_power: int,
        dynamic_strength: float,
        high_pass_alpha: float,
        alpha2: float,
        ranking_weight: float,
        edge_feedback_weight: float,
        edge_feedback_tau: float,
        orth_weight: float,
        temperature: float,
        hard_gumbel: bool,
        skip_edge: bool,
        nystrom_graph: str,
    ) -> None:
        super().__init__()
        self.edge_discriminator = EdgeDiscriminator(disc_input_dim, discriminator_hidden_dim, temperature, hard_gumbel)
        self.num_clusters = num_clusters
        self.rank = rank
        self.power = power
        self.dynamic_power = dynamic_power
        self.dynamic_strength = dynamic_strength
        self.high_pass_alpha = high_pass_alpha
        self.alpha2 = alpha2
        self.ranking_weight = ranking_weight
        self.edge_feedback_weight = edge_feedback_weight
        self.edge_feedback_tau = edge_feedback_tau
        self.orth_weight = orth_weight
        self.skip_edge = skip_edge
        self.nystrom_graph = nystrom_graph

    def elss_nystrom(
        self,
        h: torch.Tensor,
        norm_adj: torch.Tensor,
        anchors: torch.Tensor,
    ) -> torch.Tensor:
        h_anchor = h.index_select(0, anchors.to(device=h.device))
        l_cols = sparse_laplacian_columns(norm_adj, anchors.to(device=h.device)).to(device=h.device, dtype=h.dtype)
        w = h_anchor.matmul(h_anchor.t()) - self.alpha2 * l_cols[anchors.to(device=h.device), :]
        c = h.matmul(h_anchor.t()) - self.alpha2 * l_cols
        w = 0.5 * (w + w.t())
        evals, evecs = torch.linalg.eigh(w)
        mask = evals > 1e-12
        if not torch.any(mask):
            eye = torch.eye(w.size(0), dtype=w.dtype, device=w.device)
            evals, evecs = torch.linalg.eigh(w + 1e-8 * eye)
            mask = evals > 1e-12
        q = c.matmul(evecs[:, mask]).matmul(torch.diag(evals[mask].rsqrt()))
        u_full, _, _ = torch.linalg.svd(q, full_matrices=False)
        return u_full[:, : self.rank]

    def forward(
        self,
        x: torch.Tensor,
        disc_x: torch.Tensor,
        edge_index: torch.Tensor,
        anchors: torch.Tensor,
        base_adj: torch.Tensor,
        norm_adj: torch.Tensor,
    ) -> UnifiedResult:
        h_base = propagate(x, base_adj, self.power)
        if disc_x is x:
            disc_features = x
        elif disc_x.numel() == 0:
            disc_features = h_base
        else:
            disc_features = disc_x
        if self.skip_edge or self.dynamic_strength == 0:
            weights_homo = torch.full((edge_index.size(1),), 0.5, dtype=x.dtype, device=x.device)
            weights_hetero = 1.0 - weights_homo
            embeddings = h_base
        else:
            weights_homo, weights_hetero = self.edge_discriminator(disc_features, edge_index)
            adj_homo = weighted_sparse_view(edge_index, weights_homo, x.size(0), dtype=x.dtype)
            adj_hetero = weighted_sparse_view(edge_index, weights_hetero, x.size(0), dtype=x.dtype)
            h_low = propagate(x, adj_homo, self.dynamic_power)
            h_high = x - self.high_pass_alpha * propagate(x, adj_hetero, self.dynamic_power)
            embeddings = h_base + self.dynamic_strength * (h_low + h_high)
        nystrom_adj = norm_adj
        if not (self.skip_edge or self.dynamic_strength == 0):
            if self.nystrom_graph == "homo":
                nystrom_adj = adj_homo
            elif self.nystrom_graph == "mix":
                nystrom_adj = weighted_sparse_view(edge_index, 0.5 + 0.5 * weights_homo, x.size(0), dtype=x.dtype)
        basis = self.elss_nystrom(embeddings, nystrom_adj, anchors)

        subspace_loss = sparse_smoothness_loss(basis, nystrom_adj)
        if self.ranking_weight == 0:
            ranking_loss = embeddings.new_tensor(0.0)
        else:
            ranking_loss = pivot_anchored_ranking_loss(
                embeddings,
                edge_index,
                weights_homo,
                weights_hetero,
                margin_homo=0.3,
                margin_hetero=0.3,
            )
        if self.edge_feedback_weight == 0 or self.skip_edge or self.dynamic_strength == 0:
            edge_feedback_loss = embeddings.new_tensor(0.0)
        else:
            src, dst = edge_index
            sq_dist = (basis[src] - basis[dst]).pow(2).sum(dim=1)
            target_homo = torch.exp(-sq_dist.detach() / self.edge_feedback_tau).clamp(1e-4, 1.0 - 1e-4)
            weights = weights_homo.clamp(1e-4, 1.0 - 1e-4)
            edge_feedback_loss = F.binary_cross_entropy(weights, target_homo)
        gram = basis.t().matmul(basis)
        eye = torch.eye(gram.size(0), dtype=gram.dtype, device=gram.device)
        orth_loss = (gram - eye).pow(2).mean()
        loss = (
            subspace_loss
            + self.ranking_weight * ranking_loss
            + self.edge_feedback_weight * edge_feedback_loss
            + self.orth_weight * orth_loss
        )
        return UnifiedResult(
            loss,
            subspace_loss,
            ranking_loss,
            edge_feedback_loss,
            orth_loss,
            basis,
            embeddings,
            weights_homo,
            weights_hetero,
        )


@torch.no_grad()
def cluster_from_basis(
    basis: torch.Tensor,
    num_clusters: int,
    seed: int,
    *,
    postprocess: str,
    extra_embedding: torch.Tensor | None = None,
    extra_dim: int = 0,
    extra_weight: float = 1.0,
) -> np.ndarray:
    z = square_feat_map(basis)
    if postprocess == "gram":
        col_norm = z.norm(dim=0, keepdim=True)
        z = torch.where(col_norm > 1e-12, z / col_norm, torch.zeros_like(z))
        gram = z.t().matmul(z)
        gram = 0.5 * (gram + gram.t())
        evals, evecs = torch.linalg.eigh(gram + 1e-3 * torch.eye(gram.size(0), dtype=gram.dtype, device=gram.device))
        order = torch.argsort(evals, descending=True)
        top = order[: min(num_clusters + 1, evecs.size(1))]
        u_full = z.matmul(evecs[:, top]) / evals[top].clamp_min(1e-8).sqrt().unsqueeze(0)
    else:
        u_full, _, _ = torch.linalg.svd(z, full_matrices=False)
    q = u_full[:, 1 : num_clusters + 1].detach().cpu().numpy()
    if extra_embedding is not None and extra_dim > 0 and extra_weight != 0:
        extra = extra_embedding.detach().cpu().numpy()
        extra = TruncatedSVD(extra_dim, random_state=seed).fit_transform(extra)
        q = np.concatenate([normalize(q), extra_weight * normalize(extra)], axis=1)
    return KMeans(n_clusters=num_clusters, random_state=seed, n_init=10).fit_predict(q)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end unified subspace clustering.")
    parser.add_argument("--dataset", type=str, default="pubmed")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--power", type=int, default=None)
    parser.add_argument("--dynamic-power", type=int, default=None)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--num-anchors", type=int, default=None)
    parser.add_argument("--alpha2", type=float, default=5e-5)
    parser.add_argument("--dynamic-strength", type=float, default=0.01)
    parser.add_argument("--high-pass-alpha", type=float, default=0.5)
    parser.add_argument("--ranking-weight", type=float, default=0.0)
    parser.add_argument("--edge-feedback-weight", type=float, default=0.0)
    parser.add_argument("--edge-feedback-tau", type=float, default=0.02)
    parser.add_argument("--orth-weight", type=float, default=0.0)
    parser.add_argument("--disc-hidden-dim", type=int, default=32)
    parser.add_argument("--structure-dim", type=int, default=0)
    parser.add_argument("--disc-input", type=str, default="x", choices=["x", "h_base", "x_struct"])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--hard-gumbel", action="store_true")
    parser.add_argument("--skip-edge", action="store_true")
    parser.add_argument("--nystrom-graph", type=str, default="base", choices=["base", "homo", "mix"])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--pagerank-damping", type=float, default=0.875)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])
    parser.add_argument("--postprocess", type=str, default="svd", choices=["svd", "gram"])
    parser.add_argument("--fusion-extra-dim", type=int, default=0)
    parser.add_argument("--fusion-extra-weight", type=float, default=1.0)
    parser.add_argument("--graph-norm", type=str, default="elss", choices=["elss", "sym", "row"])
    parser.add_argument("--no-tf-idf", action="store_true")
    return parser.parse_args()


def run_experiment(args: argparse.Namespace) -> dict[str, float | int | str]:
    set_seed(args.seed)
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data = load_mat_attributed_graph(
        args.dataset,
        root=root,
        data_path=args.data_path,
        data_root=args.data_root,
        tf_idf=not args.no_tf_idf,
    )
    if args.power is None:
        args.power = 136 if args.dataset == "pubmed" else 2
    if args.dynamic_power is None:
        args.dynamic_power = 2
    if args.num_anchors is None:
        args.num_anchors = 50 if args.dataset == "pubmed" else 500
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    x = data.x.to(dtype=dtype)
    edge_index = data.edge_index.long()
    base_adj = normalized_adjacency(data, dtype=dtype, mode=args.graph_norm)
    if args.disc_input == "h_base":
        disc_x = torch.empty(0, dtype=dtype)
        disc_input_dim = x.size(1)
    elif args.structure_dim > 0 or args.disc_input == "x_struct":
        if args.structure_dim <= 0:
            args.structure_dim = 4
        disc_x = torch.cat([x, structural_encoding(base_adj, args.structure_dim, dtype=dtype)], dim=1)
        disc_input_dim = disc_x.size(1)
    else:
        disc_x = x
        disc_input_dim = x.size(1)
    pr = pagerank_scores_from_torch_sparse(base_adj, damping=args.pagerank_damping)
    anchors = sample_numpy_pagerank_anchors(pr, min(args.num_anchors, x.size(0)), args.seed)

    model = UnifiedSubspaceClustering(
        input_dim=x.size(1),
        disc_input_dim=disc_input_dim,
        num_clusters=data.num_clusters,
        rank=args.rank,
        discriminator_hidden_dim=args.disc_hidden_dim,
        power=args.power,
        dynamic_power=args.dynamic_power,
        dynamic_strength=args.dynamic_strength,
        high_pass_alpha=args.high_pass_alpha,
        alpha2=args.alpha2,
        ranking_weight=args.ranking_weight,
        edge_feedback_weight=args.edge_feedback_weight,
        edge_feedback_tau=args.edge_feedback_tau,
        orth_weight=args.orth_weight,
        temperature=args.temperature,
        hard_gumbel=args.hard_gumbel,
        skip_edge=args.skip_edge,
        nystrom_graph=args.nystrom_graph,
    )
    model = model.to(dtype=dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(
        f"dataset={args.dataset} nodes={x.size(0)} features={x.size(1)} clusters={data.num_clusters} "
        f"rank={args.rank} anchors={anchors.numel()} power={args.power} dynamic_power={args.dynamic_power} "
        f"dynamic={args.dynamic_strength} "
        f"ranking_weight={args.ranking_weight}",
        flush=True,
    )

    best = {"acc": -1.0, "nmi": -1.0, "ari": -1.0, "epoch": -1}
    final = {"acc": -1.0, "nmi": -1.0, "ari": -1.0, "epoch": -1}
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(x, disc_x, edge_index, anchors, base_adj, base_adj)
        if not torch.isfinite(output.loss):
            raise RuntimeError(f"Non-finite loss at epoch {epoch}: {output.loss.item()}")
        if output.loss.requires_grad:
            output.loss.backward()
            optimizer.step()
        print(
            f"train epoch={epoch:03d} loss={output.loss.item():.6f} "
            f"subspace={output.subspace_loss.item():.6f} ranking={output.ranking_loss.item():.6f} "
            f"edge_fb={output.edge_feedback_loss.item():.6f} orth={output.orth_loss.item():.6f}",
            flush=True,
        )

        if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
            model.eval()
            output = model(x, disc_x, edge_index, anchors, base_adj, base_adj)
            pred = cluster_from_basis(
                output.basis,
                data.num_clusters,
                args.seed,
                postprocess=args.postprocess,
                extra_embedding=output.embeddings,
                extra_dim=args.fusion_extra_dim,
                extra_weight=args.fusion_extra_weight,
            )
            nmi, ari, acc = evaluate_clustering(data.labels, pred)
            final.update({"acc": acc, "nmi": nmi, "ari": ari, "epoch": epoch})
            if acc > best["acc"]:
                best.update({"acc": acc, "nmi": nmi, "ari": ari, "epoch": epoch})
            print(f"eval epoch={epoch:03d} ACC={acc:.4f} NMI={nmi:.4f} ARI={ari:.4f}", flush=True)

    elapsed = time.time() - start
    print(
        f"BEST epoch={best['epoch']:03d} ACC={best['acc']:.4f} "
        f"NMI={best['nmi']:.4f} ARI={best['ari']:.4f} time={elapsed:.2f}s",
        flush=True,
    )
    return {
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
