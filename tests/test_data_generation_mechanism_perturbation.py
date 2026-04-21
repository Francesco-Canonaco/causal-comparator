import numpy as np
import lingam
import pytest
import pandas as pd
from causal_comparator.data_generation import MechanismPerturbationSimulator
from causal_comparator.utils import SHD_vectorized
from sklearn.metrics import mean_absolute_error

def test_mechanism_perturbation_simulator_logic():
    """Verifies internal logic for target selection and perturbation."""
    n_nodes = 10
    n_perturbed = 2
    n_positives = 1
    sim = MechanismPerturbationSimulator(n_nodes=n_nodes)
    
    B1, B2, y_true = sim.create_graphs(
        edge_prob=0.5, 
        n_perturbed_nodes=n_perturbed, 
        n_positives=n_positives
    )
    
    # The ground truth should have same length of n_nodes
    assert len(y_true) == n_nodes, "y_true must match the number of nodes."
    # The sum of 1's in y_true is equal to the perturbed nodes
    assert np.sum(y_true) == n_perturbed, f"Expected {n_perturbed} marked nodes."
    
    # For every node i, if y_true[i] == 0, row B1[i,:] must equal B2[i,:]
    # In fact when y_true[i] == 0 no addition or deletion where performed
    # On that node. On the contrary when y_true[i] == 1 row row B1[i,:] must be different B2[i,:].
    for i in range(n_nodes):
        # Calculate in-degree (number of parents) for both systems
        in_degree_b1 = np.count_nonzero(B1[i, :])
        in_degree_b2 = np.count_nonzero(B2[i, :])

        if y_true[i] == 0:
            # Causal structure of the node i hasn't been perturbed
            np.testing.assert_array_equal(
                B1[i, :], B2[i, :], 
                err_msg=f"Stable node {i} has modified incoming edges."
            )
        else:
            # Causal structure of the node i has been perturbed
            assert not np.array_equal(B1[i, :], B2[i, :]), (
                f"Perturbed node {i} is marked as changed, but B1 and B2 "
                "rows are identical (the simulator failed to apply the change)."
            )
            # Since we delete n_positives and add n_positives, 
            # the total count of non-zero entries must remain the same.
            assert in_degree_b1 == in_degree_b2, (
                f"Node {i} density changed. B1 in-degree: {in_degree_b1}, "
                f"B2 in-degree: {in_degree_b2}. Expected a balanced perturbation."
            )

def test_simulator_integrity():
    """Verifies the structural and numerical integrity of simulated outputs.
    
    This ensures that the generated causal matrices respect DAG constraints 
    and that the resulting datasets are numerically valid for discovery tasks.
    """
    n_nodes = 10
    n_samples = 100
    sim = MechanismPerturbationSimulator(n_nodes=n_nodes)
    
    # Generate graphs and data
    B1, B2, _ = sim.create_graphs(edge_prob=0.6, n_perturbed_nodes=1, n_positives=1)
    df, perm = sim.simulate_data(B1, n_samples)
    
    # Verify the Directed Acyclic Graph (DAG) property
    # Strict lower triangularity ensures no cycles and a valid causal ordering.
    np.testing.assert_array_equal(np.triu(B1), np.zeros((n_nodes, n_nodes)))
    np.testing.assert_array_equal(np.triu(B2), np.zeros((n_nodes, n_nodes)))
    
    # Validate the structure and integrity of the output DataFrame
    assert df.shape == (n_samples, n_nodes)
    assert not df.isnull().values.any()
    
    # Ensure column names follow the expected naming convention
    expected_columns = [f'x{i}' for i in range(n_nodes)]
    assert list(df.columns) == expected_columns
    
    # Check that the data is not constant
    # A successful simulation must produce variables with non-zero variance.
    assert np.all(df.std() > 0), "Simulated data contains constant features."
    
    # Verify the permutation vector is valid
    # It must contain every node index exactly once.
    assert len(perm) == n_nodes
    assert set(perm) == set(range(n_nodes))