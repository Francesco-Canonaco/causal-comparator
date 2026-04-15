import numpy as np

def SHD_vectorized(
    B_true: np.ndarray,
    B_est: np.ndarray,
    threshold: float = 1e-8,
    double_for_anticausal: bool = True,
) -> int:
    """
    Vectorized Structural Hamming Distance for directed adjacency matrices.

    Assumes:
    - same node ordering
    - same edge convention in B_true and B_est
    """

    if B_true.shape != B_est.shape:
        raise ValueError("B_true and B_est must have the same shape.")

    A_true = (np.abs(B_true) > threshold).astype(int)
    A_est = (np.abs(B_est) > threshold).astype(int)

    np.fill_diagonal(A_true, 0)
    np.fill_diagonal(A_est, 0)

    total_diffs = np.sum(A_true != A_est)

    if double_for_anticausal:
        return int(total_diffs)

    reversals = (
        (A_true == 1) & (A_true.T == 0) &
        (A_est == 0) & (A_est.T == 1)
    )

    return int(total_diffs - np.sum(reversals))