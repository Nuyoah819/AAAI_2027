from data_load import *
import argparse
import warnings
from evaluate import evaluate_clustering
from model import RobustGraphSubClustering as FGSC
warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description="Setting Training Params.")
    parser.add_argument('--dataset', type=str, default='dblp',
                        help='Name of the graph dataset (acm, dblp, arxiv, pubmed or wiki)')
    parser.add_argument('--power', type=int, default=2, help='Propagation order')
    args = parser.parse_args()
    return args


def main(args):
    dataset = args.dataset
    p = args.power
    adj, X, y, n_classes = datagen(dataset)
    norm_adj, X = preprocess_dataset(adj, X,
                                     row_norm=True,
                                     sym_norm=True,
                                     tf_idf=True,
                                     sparse=True
                                     )
    if sp.issparse(X):
        X = X.toarray()
    k = n_classes
    norm_adj = convert_sparse_matrix_to_sparse_tensor(norm_adj.astype(np.float64))
    adj = convert_sparse_matrix_to_sparse_tensor(adj.astype(np.float64))
    X = torch.tensor(X.astype('float64'), dtype=torch.float64)
    acc = []
    for i in np.arange(1, 2, 1):
        model = FGSC(n_clusters=k, alpha1=0., k_rank=k+1, n_anchors=200, using_pgrank=True, device='cuda',
                     power=20, alpha2=0.00005, gamma=0.005, random_state=42, filter_coef=None)
        model.fit(X, norm_adj, original_adj=None)
        labels_nys = model.predict(X)

        nmi_nys, ari_nys, acc_nys = evaluate_clustering(y, labels_nys)
        acc.append(acc_nys)

        print(f"{dataset}  Power={i}")
        print(f"Nyström: ACC={acc_nys:.4f}，NMI={nmi_nys:.4f}, ARI={ari_nys:.4f}")

    print(acc)

if __name__ == "__main__":
    args = parse_args()
    main(args)
