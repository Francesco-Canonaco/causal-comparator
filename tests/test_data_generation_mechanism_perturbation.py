import numpy as np
import lingam
import pytest
import pandas as pd
from causal_comparator.data_generation import MechanismPerturbationSimulator
from causal_comparator.utils import SHD_vectorized
from sklearn.metrics import mean_absolute_error

def test_mechanism_perturbation_simulator_logic():
    """Verifies internal logic for target selection and swap counts."""
    n_nodes = 10
    n_perturbed = 2
    n_positives = 1
    sim = MechanismPerturbationSimulator(n_nodes=n_nodes)
    
    B1, B2, y_true = sim.create_graphs(
        edge_prob=0.5, 
        n_perturbed_nodes=n_perturbed, 
        n_positives=n_positives
    )
    
    # Check ground truth vector
    assert len(y_true) == n_nodes, "y_true must match the number of nodes."
    assert np.sum(y_true) == n_perturbed, f"Expected {n_perturbed} marked nodes."
    
    # Check row-wise changes
    # For every node i, if y_true[i] == 0, row B1[i,:] must equal B2[i,:]
    for i in range(n_nodes):
        if y_true[i] == 0:
            np.testing.assert_array_equal(
                B1[i, :], B2[i, :], 
                err_msg=f"Stable node {i} has modified incoming edges."
            )