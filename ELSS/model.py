import numpy as np
import torch
from sklearn.cluster import KMeans
from utils_cym import set_seed
import torch.nn.functional as F


class RobustGraphSubClustering(torch.nn.Module):
    def __init__(self, n_clusters=5, alpha1=0.1, k_rank=10, n_anchors=100, power=2, d=0.875, alpha2=0., gamma=0.1,
                 max_irls_iter=10, irls_eps=1e-6, filter_coef=0.1, using_pgrank=True, device='cuda', random_state=None):
        super().__init__()
        self.n_clusters = n_clusters
        self.alpha1 = alpha1
        self.k_rank = k_rank
        self.n_anchors = n_anchors
        self.power = power
        self.alpha2 = alpha2
        self.gamma = gamma
        self.filter_coef = filter_coef
        self.max_irls_iter = max_irls_iter 
        self.irls_eps = irls_eps  
        self.using_pgrank = using_pgrank
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.random_state = random_state

        self.d = d
        set_seed(self.random_state)

    def _get_sparse_eye(self, n):
        indices = torch.arange(n, device=self.device).repeat(2, 1)
        values = torch.ones(n, device=self.device, dtype=torch.float64)
        return torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()

    def _compute_laplacian(self, adj):
        n = adj.shape[0]
        I = self._get_sparse_eye(n)
        return I - adj

    def convolve(self, X, adj_normalized, power):
        # X_neigh = torch.sparse.mm(adj_normalized, X)  # [N, d]
        # Xn = F.normalize(X, dim=1)
        # Xn_neigh = F.normalize(X_neigh, dim=1)
        coef = self.filter_coef
        if self.filter_coef is None:
            adj_coalesced = adj_normalized.coalesce()
            row, col = adj_coalesced.indices()
            mask = row != col 
            row = row[mask]
            col = col[mask]

            values = torch.ones_like(row, dtype=adj_normalized.dtype)
            adj_binary = torch.sparse_coo_tensor(
                torch.stack([row, col], dim=0),
                values,
                size=adj_normalized.shape
            )
            X_neigh = torch.sparse.mm(adj_binary, X)
            deg = torch.sparse.sum(adj_binary, dim=1).to_dense().clamp(min=1.0)  # [N]
            X_neigh = X_neigh / deg.unsqueeze(1)
            Xn = F.normalize(X, dim=1)
            Xn_neigh = F.normalize(X_neigh, dim=1)
            s = ((Xn * Xn_neigh).sum(dim=1) + 1)/2
            coef = (1 - s.unsqueeze(1)) ** 2

        X_0 = X.clone()
        for _ in range(power):
            X = (1 - coef) * torch.sparse.mm(adj_normalized, X) + coef * X_0
        return X

    def compute_pagerank(self, adj_sparse, d=0.85, max_iter=10):
        n = adj_sparse.size(0)
        deg = torch.sparse.sum(adj_sparse, dim=1).to_dense().view(-1, 1)
        deg_inv = 1.0 / torch.clamp(deg, min=1e-10)
        pr = torch.ones(n, 1, device=self.device, dtype=torch.float64) / n
        teleport = torch.ones(n, 1, device=self.device, dtype=torch.float64) / n
        for _ in range(max_iter):
            x = pr * deg_inv
            pr = (1 - d) * teleport + d * torch.sparse.mm(adj_sparse.t(), x)
        return pr.squeeze()

    def get_sparse_subcolumns(self, L_sparse, indices):
        n = L_sparse.size(0)
        m = len(indices)
        col_idx = torch.arange(m, device=self.device)
        p_indices = torch.stack([indices, col_idx])
        p_values = torch.ones(m, device=self.device, dtype=torch.float64)
        P = torch.sparse_coo_tensor(p_indices, p_values, (n, m)).coalesce()
        return torch.sparse.mm(L_sparse, P.to_dense())

    def _rbf_kernel_torch(self, X, Y, gamma=0.1):
        X_norm = (X ** 2).sum(1).view(-1, 1)
        Y_norm = (Y ** 2).sum(1).view(1, -1)
        dist = torch.clamp(X_norm + Y_norm - 2.0 * torch.mm(X, Y.t()), min=0.0)
        return torch.exp(-gamma * dist)

    def nystrom_approx_top_k(self, W, C, topk):
        W = (W + W.t()) * 0.5 
        S, V = torch.linalg.eigh(W)
        mask = S > 1e-12
        S_valid, V_valid = S[mask], V[:, mask]
        Q = C @ V_valid @ torch.diag(1.0 / torch.sqrt(S_valid))
        U_full, s_vals, _ = torch.linalg.svd(Q, full_matrices=False)
        return U_full[:, :topk]

    def solve_irls(self, H, L_sparse, alpha, n_anchor, rank, indices):
        N, D = H.shape
        if indices is None:
            indices = torch.randperm(N, device=self.device)[:n_anchor]
        H_anchor = H[indices, :]
        L_sub_cols = self.get_sparse_subcolumns(L_sparse, indices)
        L_sub_sub = L_sub_cols[indices, :]
        K_XXT_anchor = self._rbf_kernel_torch(H, H_anchor, gamma=self.gamma)
        K_anchor_anchor = K_XXT_anchor[indices, :]

        W = K_anchor_anchor - alpha * L_sub_sub
        C = K_XXT_anchor - alpha * L_sub_cols
        U = self.nystrom_approx_top_k(W, C, rank)
        return U

    def fit(self, X, norm_adj, original_adj=None):
        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=torch.float64, device=self.device)
        if not torch.is_tensor(norm_adj):
            norm_adj = norm_adj.tocoo()
            indices = torch.from_numpy(np.vstack((norm_adj.row, norm_adj.col)).astype(np.int64))
            values = torch.from_numpy(norm_adj.data.astype(np.float64))
            norm_adj = torch.sparse_coo_tensor(indices, values, norm_adj.shape, device=self.device).coalesce()
        X = X.to(self.device)
        norm_adj = norm_adj.to(self.device)
        if original_adj is not None:
            original_adj = original_adj.to(self.device)
        else:
            original_adj = norm_adj
        L = self._compute_laplacian(norm_adj)
        H = self.convolve(X, norm_adj, power=self.power)
        if self.using_pgrank:
            pr_values = self.compute_pagerank(original_adj)
            print(torch.sort(pr_values, descending=True))
            prob = (pr_values / pr_values.sum()).cpu().numpy()
            indices = np.random.choice(X.shape[0], size=self.n_anchors, replace=False, p=prob)
            indices = torch.tensor(indices, device=self.device, dtype=torch.long)
        else:
            indices = torch.randperm(X.shape[0], device=self.device)[:self.n_anchors]
        self.embedding_ = self.solve_irls(H, L, alpha=self.alpha2, n_anchor=self.n_anchors, rank=self.k_rank,
                                          indices=indices)

        Z = self.square_feat_map(self.embedding_)
        U_full, _, _ = torch.linalg.svd(Z, full_matrices=False)
        Q = U_full[:, 1:self.n_clusters + 1].detach().cpu().numpy()

        self.kmeans_ = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        self.labels_ = self.kmeans_.fit_predict(Q)
        return self

    def square_feat_map(self, z, c=2 ** -0.5):
        n, d = z.shape
        bias = torch.ones(n, 1, device=self.device, dtype=z.dtype)
        linear = z
        quadratic = []
        for i in range(d):
            quadratic.append(z[:, i:i + 1] ** 2)
            for j in range(i + 1, d):
                quadratic.append(z[:, i:i + 1] * z[:, j:j + 1])
        quadratic = torch.cat(quadratic, dim=1)
        x = torch.cat([bias, linear, quadratic], dim=1)

        num_feats = x.shape[1]
        coefs = torch.ones(num_feats, device=self.device, dtype=z.dtype)
        coefs[0], coefs[1:d + 1], coefs[d + 1:] = c, np.sqrt(2 * c), np.sqrt(2.0)
        return x * coefs.unsqueeze(0)

    def predict(self, X=None, adjacency=None):
        return self.labels_