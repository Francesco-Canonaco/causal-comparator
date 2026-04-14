import numpy as np
import pandas as pd
import networkx as nx

def sample_lingam_weight():
    """
    Samples weights based on Shimizu 2011: |b_ij| in [0.5, 1.5].
    Ensures the signal is strong enough to be identifiable.
    """
    sign = np.random.choice([-1, 1])
    mag = np.random.uniform(0.5, 1.5)
    return sign * mag

def create_experiment_setup(n_nodes, edge_prob, n_perturbations):
    """
    Creates G1, G2, and True Delta using the LiNGAM convention:
    B[i, j] is the effect of variable j on variable i.
    Uses a strictly lower triangular structure to guarantee a DAG.
    """
    # 1. Generate G1 (Strictly Lower Triangular: i > j)
    # This means node 0 can cause node 1, node 1 can cause node 2, etc.
    B1 = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i):  # j < i ensures strictly lower triangular
            if np.random.rand() < edge_prob:
                B1[i, j] = sample_lingam_weight()

    # 2. Generate G2 by perturbing G1
    B2 = B1.copy()
    # Get all possible indices in the strictly lower triangle
    indices = list(zip(*np.tril_indices(n_nodes, k=-1)))
    perturb_idx_choices = np.random.choice(len(indices), n_perturbations, replace=False)
    
    for idx in perturb_idx_choices:
        row, col = indices[idx]
        if B2[row, col] == 0:
            B2[row, col] = sample_lingam_weight()  # Addition
        else:
            B2[row, col] = 0  # Deletion
            
    # 3. Ground Truth Binary Delta (Strictly edge-wise)
    true_delta = (B1 != B2).astype(int)
    
    return B1, B2, true_delta

def simulate_from_graph(B, n_samples, noise_type='uniform', seed=None):
    """
    Generates data: X = BX + e  => X = (I - B)^-1 * e
    B is (n_nodes x n_nodes) where B[i, j] is j -> i
    """
    if seed is not None:
        np.random.seed(seed)
        
    n_nodes = B.shape[0]
    I = np.eye(n_nodes)
    
    # Transformation matrix: (I - B)^-1
    # Because B is strictly lower triangular, (I - B) is always invertible.
    transformation = np.linalg.inv(I - B)
    
    # Generate Noise (e)
    if noise_type == 'uniform':
        e = np.random.uniform(-1, 1, (n_nodes, n_samples))
    elif noise_type == 'laplace':
        e = np.random.laplace(0, 1, (n_nodes, n_samples))
    else:
        raise ValueError("Use 'uniform' or 'laplace'.")
        
    # Generate Data: X = (p x n_samples)
    data = transformation @ e
    
    # Transpose to (n_samples x p) for standard DataFrame format
    return pd.DataFrame(data.T, columns=[f'v{k}' for k in range(n_nodes)])