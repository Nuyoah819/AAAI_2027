from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, confusion_matrix, f1_score, normalized_mutual_info_score


def map_cluster_labels(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    labels = np.unique(np.concatenate([np.unique(y_true), np.unique(y_pred)]))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    rows, cols = linear_sum_assignment(matrix.max() - matrix)
    mapping = {labels[cols[i]]: labels[rows[i]] for i in range(len(rows))}
    return np.asarray([mapping.get(label, label) for label in y_pred])


def evaluate_clustering(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mapped = map_cluster_labels(y_true, y_pred)
    return {
        "acc": float(np.mean(np.asarray(y_true).reshape(-1) == mapped)),
        "nmi": float(normalized_mutual_info_score(y_true, y_pred)),
        "ari": float(adjusted_rand_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, mapped, average="macro")),
    }
