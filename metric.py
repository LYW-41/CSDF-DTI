import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


def classification_metrics(labels, probabilities, threshold=0.5):
    labels = np.asarray(labels).astype(int)
    probabilities = np.asarray(probabilities)
    predictions = (probabilities >= threshold).astype(int)
    return {
        "auroc": roc_auc_score(labels, probabilities),
        "auprc": average_precision_score(labels, probabilities),
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, zero_division=0),
        "mcc": matthews_corrcoef(labels, predictions),
    }
