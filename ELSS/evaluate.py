import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from typing import Tuple
from data_load import *

def evaluate_clustering(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    acc = clustering_accuracy(y_true, y_pred)
    return nmi, ari, acc
