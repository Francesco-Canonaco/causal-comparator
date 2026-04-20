import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
from numpy.typing import ArrayLike
from typing import Dict


"""def evaluate_binary_classification_old(y_true:ArrayLike, y_scores:ArrayLike, optimize=True)->Dict:
    # Metrics require 1D arrays for comparison.
    y_true = np.array(y_true).flatten() 
    y_scores = np.array(y_scores).flatten()

    # Ranking threshold-independent metrics.
    results = {
        'auc_roc': roc_auc_score(y_true, y_scores),
        'aupr': average_precision_score(y_true, y_scores)
    }

    if optimize:
        best_f1 = 0
        best_threshold = 0
        # Search 100 points to find the global F1 maximum.
        thresholds = np.linspace(0, 1, 100)
        
        for tau in thresholds:
            y_pred = (y_scores >= tau).astype(int)
            current_f1 = f1_score(y_true, y_pred, zero_division=0)
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_threshold = tau
        
        y_best_pred = (y_scores >= best_threshold).astype(int)
    else:
        # Optimization is skipped if inputs are already binary predictions. (Naive approach).
        best_f1 = f1_score(y_true, y_scores, zero_division=0)
        best_threshold = "N/A"
        y_best_pred = y_scores

    results.update({
        'best_f1': best_f1,
        'best_threshold': best_threshold,
        'precision': precision_score(y_true, y_best_pred, zero_division=0),
        'recall': recall_score(y_true, y_best_pred, zero_division=0)
    })

    return results
"""

def evaluate_binary_classification(y_true:ArrayLike, y_scores:ArrayLike, optimize:bool = True)->Dict:
    """Evaluate the predicted values

    Args:
        y_true (ArrayLike): True labels
        y_scores (ArrayLike): Predicted scores
        optimize (bool, optional): Optimization of the threshold when y_score isn't binary. Defaults to True.

    Returns:
        Dict: Evaluation of the predictions containing auc_roc, aupr, F1, threshold, precision and recall. 
    """

    # Metrics require 1D arrays for comparison.
    y_true = np.array(y_true).flatten()
    y_scores = np.array(y_scores).flatten()

    # Ranking threshold-independent metrics
    results = {
        'auc_roc': roc_auc_score(y_true, y_scores),
        'aupr': average_precision_score(y_true, y_scores)
    }

    if optimize:
        best_f1 = 0.0
        best_threshold = 0.0
        
        # Search optimization: use unique values (Scikit-Learn style)
        # Optimization is skipped if inputs are already binary predictions
        thresholds = np.unique(y_scores)
        
        for tau in thresholds:
            y_pred = (y_scores >= tau).astype(int)
            current_f1 = f1_score(y_true, y_pred, zero_division=0)
            
            if current_f1 >= best_f1:
                best_f1 = current_f1
                best_threshold = tau
        
        y_best_pred = (y_scores >= best_threshold).astype(int)
    else:
        # Optimization skipped: assume y_scores are already binary (0 or 1)
        best_f1 = f1_score(y_true, (y_scores >= 0.5).astype(int), zero_division=0)
        best_threshold = "N/A"
        y_best_pred = (y_scores >= 0.5).astype(int)

    results.update({
        'best_f1': best_f1,
        'best_threshold': best_threshold,
        'precision': precision_score(y_true, y_best_pred, zero_division=0),
        'recall': recall_score(y_true, y_best_pred, zero_division=0)
    })

    return results


def calculate_node_scores(delta:ArrayLike, mode:str = 'incoming'):
    """
    Transforms an edge-level delta matrix into the node-level score vector y_scores
    
    Args:
        delta (np.ndarray): The (p x p) difference matrix (B1 - B2).
        mode (str): 'incoming' (mechanism shift), 'outgoing' (influence shift), 
                    or 'total' (volatility).
    
    Returns:
        np.ndarray: A vector of length p containing the scores (y_scores).
    """
    abs_delta = np.abs(delta)
    
    if mode == 'incoming':
        # Changes in the parents of each node
        return np.sum(abs_delta, axis=1)
    elif mode == 'outgoing':
        # Changes in the children of each node
        return np.sum(abs_delta, axis=0)
    elif mode == 'total':
        # Changes in both children and parents of each node
        return np.sum(abs_delta, axis=1) + np.sum(abs_delta, axis=0)
    else:
        raise ValueError("Mode must be 'incoming', 'outgoing', or 'total'")
    
