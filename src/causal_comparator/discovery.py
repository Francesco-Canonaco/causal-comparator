import lingam
import numpy as np
from joblib import Parallel, delayed
class CausalComparator:
    def __init__(self, data_i, data_j,  model_class = lingam.DirectLiNGAM, **model_kwargs ):
        """
        Initializes the CausalComparator to compare two causal systems.

        Args:
            data_i (pd.DataFrame): Reference dataset (System I).
            data_j (pd.DataFrame): Treatment/Comparison dataset (System J).
            model_class: The LiNGAM model class to use (default: DirectLiNGAM).
            **model_kwargs: Arguments passed to the LiNGAM model initialization.
        """
        self.data_i = data_i
        self.data_j = data_j
        # ensure columns alignment
        self._validate_data()

        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.nodes = data_i.columns.to_list()

        self.freq_i = None
        self.freq_j = None
        self.delta = None
    
    def _validate_data(self):
        """
        Ensures both datasets have the same columns and aligns their order.
        
        Raises:
            ValueError: If the datasets do not contain the same set of variables.
        """
        if not self.data_i.columns.equals(self.data_j.columns):
            # Sort them if they have the same columns but different order
            if set(self.data_i.columns) == set(self.data_j.columns):
                self.data_j = self.data_j[self.data_i.columns]
            else:
                raise ValueError("Datasets must have the same set of variables/columns.")

    def _fit_base(self, data):
        """
        Fits a single LiNGAM model and returns a binary adjacency matrix.
        
        Args:
            data (pd.DataFrame): The dataset to fit.
            
        Returns:
            np.ndarray: Binary matrix where 1 indicates an edge and 0 indicates no edge.
        """
        model = self.model_class(**self.model_kwargs)
        model.fit(data)
        return (np.abs(model.adjacency_matrix_) > 0).astype(int)
    
    def _fit_bootstrap(self, data, n_sampling):
        base_model = self.model_class(**self.model_kwargs)
        model = base_model.bootstrap(data, n_sampling=n_sampling)
        return model.get_probabilities()
    
    def estimate_naive(self):
        """
        Estimates structural differences using a single LiNGAM run on each dataset.
        
        The resulting delta matrix contains:
         1 : Edge Removal (Present in System I, Absent in System J)
         0 : No structural change
        -1 : Edge Addition (Absent in System I, Present in System J)
        
        Returns:
            np.ndarray: The raw difference matrix in {-1, 0, 1}.
        """
        results = Parallel(n_jobs=2)(
            delayed(self._fit_base)(data)
            for data in [self.data_i, self.data_j]
        )
        self.freq_i, self.freq_j = results
        self.delta = self.freq_i - self.freq_j
        return self.delta
    
    def estimate_bootstrap(self, n_sampling=100):
        """
        Estimates structural differences using standard bootstrapping on each dataset.
        
        Args:
            n_sampling (int): Number of bootstrap samples to draw.
            
        Returns:
            np.ndarray: Delta matrix representing change in selection probabilities.
        """
        results = Parallel(n_jobs=2)(
            delayed(self._fit_bootstrap)(data, n_sampling)
            for data in [self.data_i, self.data_j]
        )
        self.freq_i, self.freq_j = results
        self.delta = self.freq_i - self.freq_j
        return self.delta
    
    def estimate_rsbs(self, n_sampling=100, seed = None):
        """
        Estimates structural differences using Relative Sample-size Bootstrap Stability (RSBS).
        
        This method downsamples the larger dataset to match the size of the smaller one, 
        ensuring equalized discovery power and more robust change detection.
        
        Args:
            n_sampling (int): Number of bootstrap samples to draw.
            seed (int, optional): Random seed for reproducibility.
            
        Returns:
            np.ndarray: Delta matrix of selection probability differences.
        """
        ni, nj = len(self.data_i), len(self.data_j)
        n_min = min(ni, nj)
        
        # Downsample larger dataset to level the playing field
        d_i_rs = self.data_i.sample(n_min, random_state = seed) if ni > n_min else self.data_i
        d_j_rs = self.data_j.sample(n_min, random_state = seed) if nj > n_min else self.data_j
        
        # Execute both RSBS bootstrap fits in parallel using 2 cores[cite: 4]
        results = Parallel(n_jobs=2)(
            delayed(self._fit_bootstrap)(data, n_sampling)
            for data in [d_i_rs, d_j_rs]
        )
        self.freq_i, self.freq_j = results
        self.delta = self.freq_i - self.freq_j
        return self.delta


        