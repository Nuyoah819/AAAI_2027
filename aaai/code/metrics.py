from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, confusion_matrix, normalized_mutual_info_score


def clustering_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    conf_mat = confusion_matrix(y_true, y_pred)
    cost = np.max(conf_mat) - conf_mat
    row_ind, col_ind = linear_sum_assignment(cost)
    conf_mat = conf_mat[row_ind, :]
    conf_mat = conf_mat[:, col_ind]
    return float(np.trace(conf_mat) / np.sum(conf_mat))


def evaluate_clustering(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    acc = clustering_accuracy(y_true, y_pred)
    return nmi, ari, acc
