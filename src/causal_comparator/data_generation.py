import numpy as np
import pandas as pd
import networkx as nx
from typing import Tuple, Optional

def sample_lingam_weight(rng: np.random.Generator) -> float:
    """
    Samples weights based on [1] using a deterministic generator.
    Ensures the signal is strong enough to be identifiable.

    References:
    --------
    [1] Shimizu, Shohei, et al. "DirectLiNGAM: A direct method for learning 
    a linear non-Gaussian structural equation model." Journal of Machine 
    Learning Research-JMLR 12.Apr (2011): 1225-1248.

    Args:
    ----
        rng: A NumPy random generator for reproducibility.

    Returns:
    --------
        A float representing the sampled weight in range [-1.5, -0.5] U [0.5, 1.5].
    """
    sign = rng.choice([-1, 1])
    mag = rng.uniform(0.5, 1.5)
    return float(sign * mag)


def create_experiment_setup(
    n_nodes: int, 
    edge_prob: float, 
    n_positives: int, 
    rng: Optional[np.random.Generator] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Creates G1 and G2 adjacency matrices and the directional ground truth delta.

    The total number of differences (Nd) is 2 * n_positives (balanced add/delete).
    References:
    --------
    [1] Ma, Sisi, and Roshan Tourani. "Comparing Causal Bayesian Networks Estimated from Data." Entropy 26.3 (2024): 228.
    [2] Shimizu, Shohei, et al. "DirectLiNGAM: A direct method for learning 
    a linear non-Gaussian structural equation model." Journal of Machine 
    Learning Research-JMLR 12.Apr (2011): 1225-1248.

    Args:
    --------
        n_nodes: Number of variables in the system.
        edge_prob: Probability of an edge existing between any two nodes in G1.
        n_positives: Number of edges to remove from G1 (True Positives for E1 - E2).
        seed: Random seed for reproducibility.

    Returns:
    --------
        B1: Adjacency matrix for System 1 (j -> i).
        B2: Adjacency matrix for System 2 (j -> i).
        true_delta: Binary matrix where 1 indicates an edge in E1 but not in E2.
    """
    
    if rng is None:
        # Fallback to a default generator if none is provided
        rng = np.random.default_rng()
        
    # Generate Baseline G1 (Strictly Lower Triangular)
    B1 = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i):
            if rng.random() < edge_prob:
                # Ensure your weight sampler also accepts the generator
                B1[i, j] = sample_lingam_weight(rng)

    # Identify candidates for perturbation
    # We use a boolean mask to stay within the lower triangle (DAG)
    is_lower_tri = np.tril(np.ones((n_nodes, n_nodes), dtype=bool), k=-1)
    
    existing_edges = np.argwhere((B1 != 0) & is_lower_tri)
    potential_new_edges = np.argwhere((B1 == 0) & is_lower_tri)

    # Create G2 via Balanced Perturbation (Deletions + Additions)
    B2 = B1.copy()
    
    # Check if we have enough edges to satisfy the request
    if len(existing_edges) < n_positives:
        raise ValueError(f"G1 has only {len(existing_edges)} edges; cannot delete {n_positives}.")
    if len(potential_new_edges) < n_positives:
        raise ValueError(f"Not enough empty slots to add {n_positives} edges.")

    # Randomly DELETE edges from G1 (Directional Positives for E1 - E2)
    del_indices = rng.choice(len(existing_edges), n_positives, replace=False)
    for idx in del_indices:
        r, c = existing_edges[idx]
        B2[r, c] = 0

    # Randomly ADD edges to G2 (Directional Negatives for E1 - E2)
    add_indices = rng.choice(len(potential_new_edges), n_positives, replace=False)
    for idx in add_indices:
        r, c = potential_new_edges[idx]
        B2[r, c] = sample_lingam_weight(rng)

    # Ground Truth Delta (E1 - E2)
    # Logical check: Edge exists in B1 AND is zero in B2
    true_delta = np.logical_and(B1 != 0, B2 == 0).astype(int)
    
    return B1, B2, true_delta


def simulate_from_graph(
    B: np.ndarray, 
    n_samples: int, 
    noise_type: str = 'uniform', 
    rng: Optional[np.random.Generator] = None
) -> pd.DataFrame:
    """
    Generates data from a causal graph: X = BX + e => X = (I - B)^-1 * e.
    
    Args:
        B: Adjacency matrix (n_nodes x n_nodes) where B[i, j] is j -> i.
        n_samples: Number of observations to generate.
        noise_type: The non-Gaussian noise distribution ('uniform' or 'laplace').
        rng: A NumPy random generator for reproducibility.

    Returns:
        pd.DataFrame: Simulated data in (n_samples x n_nodes) format.
    """
    if rng is None:
        # Fallback to a default generator if none is provided
        rng = np.random.default_rng()
        
    n_nodes = B.shape[0]
    I = np.eye(n_nodes)
    
    # Transformation matrix: (I - B)^-1
    # Because B is strictly lower triangular (DAG), (I - B) is always invertible.
    transformation = np.linalg.inv(I - B)
    
    # Generate Noise (e) using the Generator API
    if noise_type == 'uniform':
        # Standard non-Gaussian noise used in LiNGAM papers
        e = rng.uniform(-1, 1, (n_nodes, n_samples))
    elif noise_type == 'laplace':
        e = rng.laplace(0, 1, (n_nodes, n_samples))
    else:
        raise ValueError("noise_type must be 'uniform' or 'laplace'.")
        
    # Generate Data: X = (n_nodes x n_samples)
    data = transformation @ e
    
    # TODO: Add colmns permutation logic as in shimizu's code to ensure that the order of columns isn't the same as the causal order. This will make the task more realistic and challenging.
    # TODO: Add the mean shift applied in shimizu's code.
    
    # Transpose to (n_samples x n_nodes) for standard DataFrame format
    return pd.DataFrame(data.T, columns=[f'v{k}' for k in range(n_nodes)])