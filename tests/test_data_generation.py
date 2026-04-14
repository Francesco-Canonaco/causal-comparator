import numpy as np
import lingam
import pytest
from causal_comparator.data_generation import (
    create_experiment_setup, 
    simulate_from_graph, 
    sample_lingam_weight
)
def test_pure_causal_recovery():
    """
    Asymptotic Recovery Test: 
    With 10k samples, DirectLiNGAM must recover the EXACT structure.
    No arbitrary thresholds allowed.
    """
    n_nodes = 5
    B_true = np.zeros((n_nodes, n_nodes))
    B_true[1, 0] = 1.2
    B_true[2, 0] = -0.5
    B_true[4, 2] = 0.8
    
    # Generate high-fidelity data
    df = simulate_from_graph(B_true, n_samples=10000, seed=1)
    
    model = lingam.DirectLiNGAM()
    model.fit(df)
    B_est = model.adjacency_matrix_
    
    # 1. Structural Check: 
    # Any non-zero in truth must be non-zero in estimate.
    # Any zero in truth must be EXACTLY zero (or within float precision) in estimate.
    true_mask = (B_true != 0)
    est_mask = (np.abs(B_est) > 1e-10) # Using machine epsilon tolerance only
    print("True Adjacency Matrix:\n", B_true)
    print("Estimated Adjacency Matrix:\n", B_est)
    assert np.array_equal(true_mask, est_mask), (
        f"Structural mismatch! \nTrue edges: {np.where(true_mask)} "
        f"\nEst edges: {np.where(est_mask)}"
    )

    # 2. Value Check:
    # The weights should be almost identical to the truth.
    assert np.allclose(B_true, B_est, atol=1e-1), "Weight estimation is outside acceptable asymptotic error."