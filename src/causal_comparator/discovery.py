import lingam
import numpy as np

class CausalComparator:
    def __init__(self, data_i, data_j,  model_class = lingam.DirectLiNGAM, **model_kwargs ):
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
        if not self.data_i.columns.equals(self.data_j.columns):
            # Sort them if they have the same columns but different order
            if set(self.data_i.columns) == set(self.data_j.columns):
                self.data_j = self.data_j[self.data_i.columns]
            else:
                raise ValueError("Datasets must have the same set of variables/columns.")

    def _fit_base(self, data):
        model = self.model_class(**self.model_kwargs)
        model.fit(data)
        return (np.abs(model.adjacency_matrix_) > 0).astype(int)
    
    def _fit_bootstrap(self, data, n_sampling):
        base_model = self.model_class(**self.model_kwargs)
        model = base_model.bootstrap(data, n_sampling=n_sampling)
        return model.get_probabilities()
    
    def estimate_naive(self):
        self.freq_i = self._fit_base(self.data_i)
        self.freq_j = self._fit_base(self.data_j)
        self.delta = self.freq_i - self.freq_j
        return self.delta
    
    def estimate_bootstrap(self, n_sampling=100):
        self.freq_i = self._fit_bootstrap(self.data_i, n_sampling)
        self.freq_j = self._fit_bootstrap(self.data_j, n_sampling)
        self.delta = self.freq_i - self.freq_j
        return self.delta
    
    def estimate_rsbs(self, n_sampling=100, seed = None):
        ni, nj = len(self.data_i), len(self.data_j)
        n_min = min(ni, nj)
        
        # Downsample larger dataset to level the playing field
        d_i_rs = self.data_i.sample(n_min, random_state = seed) if ni > n_min else self.data_i
        d_j_rs = self.data_j.sample(n_min, random_state = seed) if nj > n_min else self.data_j
        
        self.freq_i = self._fit_bootstrap(d_i_rs, n_sampling)
        self.freq_j = self._fit_bootstrap(d_j_rs, n_sampling)
        self.delta = self.freq_i - self.freq_j
        return self.delta


        