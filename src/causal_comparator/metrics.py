import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score


def evaluate_binary_classification(y_true, y_scores, optimize=True):
    """
    Generic evaluator for binary classification tasks.
    """
    y_true = np.array(y_true).flatten()
    y_scores = np.array(y_scores).flatten()

    # 1. Ranking Metrics
    results = {
        'auc_roc': roc_auc_score(y_true, y_scores),
        'aupr': average_precision_score(y_true, y_scores)
    }

    if optimize:
        # 2. Bootstrap logic: Search for the best threshold
        best_f1 = 0
        best_threshold = 0
        thresholds = np.linspace(0, 1, 100)
        
        for tau in thresholds:
            y_pred = (y_scores >= tau).astype(int)
            current_f1 = f1_score(y_true, y_pred, zero_division=0)
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_threshold = tau
        
        y_best_pred = (y_scores >= best_threshold).astype(int)
    else:
        # 3. Naive logic: Use the data as-is (already 0/1)
        # this happens only when the naive method is used, where y_scores are already binary predictions
        best_f1 = f1_score(y_true, y_scores, zero_division=0)
        best_threshold = "N/A"
        y_best_pred = y_scores

    # 4. Finalize
    results.update({
        'best_f1': best_f1,
        'best_threshold': best_threshold,
        'precision': precision_score(y_true, y_best_pred, zero_division=0),
        'recall': recall_score(y_true, y_best_pred, zero_division=0)
    })

    return results