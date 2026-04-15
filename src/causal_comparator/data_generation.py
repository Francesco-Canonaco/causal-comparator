import numpy as np
import pandas as pd
import networkx as nx
from abc import ABC, abstractmethod
from typing import Tuple, Optional

class BaseSimulator(ABC):
    def __init__(self, n_nodes: int, noise_type: str = 'uniform', rng: np.random.Generator = None):
        self.n_nodes = n_nodes
        self.noise_type = noise_type
        self.rng = rng if rng else np.random.default_rng()
    
    def sample_lingam_weight(self) -> float:
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
        sign = self.rng.choice([-1, 1])
        mag = self.rng.uniform(0.5, 1.5)
        return float(sign * mag)

    @abstractmethod
    def create_graphs(self, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Abstract method to be implemented by subclasses.
        Must return (B1_true, B2_true, delta_true).
        """
        pass

    def simulate_data(self, B: np.ndarray, n_samples: int) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Simulates data according to Shimizu et al. (2011) protocol [1].
        
        References:
        -----------
        [1] Shimizu, Shohei, et al. "DirectLiNGAM: A direct method for learning 
        a linear non-Gaussian structural equation model." Journal of Machine 
        Learning Research-JMLR 12.Apr (2011): 1225-1248.

        Args:
        -----
            B: Strictly lower triangular adjacency matrix.
            n_samples: Number of observations.
            noise_type: 'uniform' or 'laplace'.
            rng: Random generator.

        Returns:
        --------
            df: The shuffled DataFrame.
            permutation: The array of indices used to shuffle the nodes.
        """
        # 1. Sample Noise Variances sigma^2 from [1, 3]
        variances = self.rng.uniform(1, 3, self.n_nodes)
        stds = np.sqrt(variances)
        
        # 2. Generate and Standardize noise to unit variance
        if self.noise_type == 'uniform':
            e = self.rng.uniform(-1, 1, (self.n_nodes, n_samples))
        elif self.noise_type == 'laplace':
            e = self.rng.laplace(0, 1, (self.n_nodes, n_samples))
        else:
            raise ValueError("Use 'uniform' or 'laplace'.")
        
        # Standardize to Var=1, then scale to the sampled stds
        e = e / np.std(e, axis=1, keepdims=True)
        e = e * stds.reshape(-1, 1)
        
        # 3. Structural Equation: X = (I - B)^-1 * e
        I = np.eye(self.n_nodes)
        transformation = np.linalg.inv(I - B)
        data = transformation @ e
        
        # 4. Create the Permutation 
        p = self.rng.permutation(self.n_nodes)
        
        # Shuffle the data (rows represent nodes before the transpose)
        data_shuffled = data[p, :]
        
        # 5. Create DataFrame with 'blind' labels
        df = pd.DataFrame(
            data_shuffled.T, 
            columns=[f'x{i}' for i in range(self.n_nodes)]
        )
        
        return df, p
            
class EdgePerturbationSimulator(BaseSimulator):
    
    def create_graphs(self, edge_prob: float, n_positives: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Creates B1 and B2 adjacency matrices and the directional ground truth delta.

        The total number of differences (Nd) is 2 * n_positives (balanced add/delete).

        References:
        -----------
        [1] Ma, Sisi, and Roshan Tourani. "Comparing Causal Bayesian Networks Estimated from Data." Entropy 26.3 (2024): 228.

        [2] Shimizu, Shohei, et al. "DirectLiNGAM: A direct method for learning 
        a linear non-Gaussian structural equation model." Journal of Machine 
        Learning Research-JMLR 12.Apr (2011): 1225-1248.

        Args:
        --------
            edge_prob: Probability of an edge existing between any two nodes in G1.
            n_positives: Number of edges to remove from G1, number of edges to add to G1.

        Returns:
        --------
            B1: Adjacency matrix for System 1 (j -> i).
            B2: Adjacency matrix for System 2 (j -> i).
            true_delta: Binary matrix where 1 indicates an edge in E1 but not in E2.
        """

        # Generate Baseline G1 (Strictly Lower Triangular)
        B1 = np.zeros((self.n_nodes, self.n_nodes))
        for i in range(self.n_nodes):
            for j in range(i):
                if self.rng.random() < edge_prob:
                    B1[i, j] = self.sample_lingam_weight()

        # Identify candidates for perturbation
        # We use a boolean mask to stay within the lower triangle (DAG)
        is_lower_tri = np.tril(np.ones((self.n_nodes, self.n_nodes), dtype=bool), k=-1)
        
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
        del_indices = self.rng.choice(len(existing_edges), n_positives, replace=False)
        for idx in del_indices:
            r, c = existing_edges[idx]
            B2[r, c] = 0

        # Randomly ADD edges to G2 (Directional Negatives for E1 - E2)
        add_indices = self.rng.choice(len(potential_new_edges), n_positives, replace=False)
        for idx in add_indices:
            r, c = potential_new_edges[idx]
            B2[r, c] = self.sample_lingam_weight()

        # Ground Truth Delta (E1 - E2)
        # Logical check: Edge exists in B1 AND is zero in B2
        true_delta = np.logical_and(B1 != 0, B2 == 0).astype(int)
        
        return B1, B2, true_delta
