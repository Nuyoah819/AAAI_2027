import os
import numpy as np
import scipy.io as io
import scipy.sparse as sp
import torch
from ogb.nodeproppred import NodePropPredDataset
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize

def datagen(dataset):
    if dataset in ['wiki', 'pubmed', 'computers', 'acm', 'dblp']:
        data = io.loadmat(os.path.join('data', f'{dataset}.mat'))
        features = data['fea'].astype(float)
        adj = data.get('W')
        if adj is not None:
            adj = adj.astype(float)
            if not sp.issparse(adj):
                adj = sp.csc_matrix(adj)

        labels = data['gnd'].reshape(-1) #- 1
        n_classes = len(np.unique(labels))
        return adj, features, labels, n_classes

    if dataset == 'arxiv':
        dataset = NodePropPredDataset(name='ogbn-arxiv', root='data')
        graph = dataset[0]
        data = graph[0]
        labels = graph[1].reshape(-1)

        features = data['node_feat']
        row_ind = data['edge_index'][0]
        col_ind = data['edge_index'][1]
        data = np.ones(len(row_ind))

        N = M = len(features)
        adj = sp.csr_matrix((data, (row_ind, col_ind)), shape=(M, N))
        adj = (adj + adj.T)

        n_classes = len(np.unique(labels))
        return adj, features, labels, n_classes



def preprocess_dataset(adj, features, row_norm=True, sym_norm=False, feat_norm='l2', tf_idf=False, sparse=False, alpha=1,
                       beta=1):
    if sym_norm:
        adj = aug_normalized_adjacency(adj, True, alpha=alpha)
    if row_norm:
        adj = row_normalize(adj, True, alpha=beta)

    if tf_idf:
        features = TfidfTransformer(norm=feat_norm).fit_transform(features)
    else:
        features = normalize(features, feat_norm)

    if not sparse and sp.issparse(features):
        features = features.toarray()
    return adj, features


def aug_normalized_adjacency(adj, add_loops=True, alpha=1):
    if add_loops:
        adj = adj + alpha * sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    n_nodes = adj.shape[0]
    mask = (adj.row < n_nodes) & (adj.col < n_nodes)
    adj = sp.coo_matrix((adj.data[mask], (adj.row[mask], adj.col[mask])), shape=adj.shape)

    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def row_normalize(mx, add_loops=True, alpha=1):
    if add_loops:
        mx = mx + alpha * sp.eye(mx.shape[0])
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def symmetric_normalize(mx, add_loops=True, alpha=1):
    if not sp.issparse(mx):
        mx = sp.csr_matrix(mx)
    n = mx.shape[0]
    if add_loops:
        mx = mx + alpha * sp.eye(n, format=mx.format)
    rowsum = np.array(mx.sum(1)).flatten()
    r_inv_sqrt = np.power(rowsum + 1e-8, -0.5)
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    D_tilde_sqrt_inv = sp.diags(r_inv_sqrt, format=mx.format)
    mx_sym = D_tilde_sqrt_inv.dot(mx).dot(D_tilde_sqrt_inv)
    return mx_sym


def convert_sparse_matrix_to_sparse_tensor(X):
    if not sp.isspmatrix_coo(X):
        X = X.tocoo()

    shape = X.shape
    n_nodes = shape[0]
    mask = (X.row < n_nodes) & (X.col < n_nodes)
    row = X.row[mask]
    col = X.col[mask]
    data = X.data[mask]
    indices = np.vstack((row, col)).astype(np.int64)
    return torch.sparse_coo_tensor(
        torch.from_numpy(indices),
        torch.from_numpy(data.astype('float64')),
        torch.Size(shape)
    ).coalesce() 


def clustering_accuracy(y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    from scipy.optimize import linear_sum_assignment

    def ordered_confusion_matrix(y_true, y_pred):
        conf_mat = confusion_matrix(y_true, y_pred)
        w = np.max(conf_mat) - conf_mat
        row_ind, col_ind = linear_sum_assignment(w)
        conf_mat = conf_mat[row_ind, :]
        conf_mat = conf_mat[:, col_ind]
        return conf_mat

    conf_mat = ordered_confusion_matrix(y_true, y_pred)
    return np.trace(conf_mat) / np.sum(conf_mat)
