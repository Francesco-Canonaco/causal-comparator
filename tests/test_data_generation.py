import numpy as np
import lingam
import pytest
import pandas as pd
from causal_comparator.data_generation import (
    create_experiment_setup, 
    simulate_from_graph, 
    sample_lingam_weight
)


def test_pure_causal_recovery():
    """
    Asymptotic Recovery Test: 
    With 10k samples, DirectLiNGAM must recover the EXACT structure.
    """
    # Initialize generator for this specific test
    rng = np.random.default_rng(seed=1)
    
    n_nodes = 5
    B_true = np.zeros((n_nodes, n_nodes))
    B_true[1, 0] = 1.2
    B_true[2, 0] = -0.5
    B_true[4, 2] = 0.8
    
    # Generate high-fidelity data using the rng
    df = simulate_from_graph(B_true, n_samples=10000, rng=rng)
    
    model = lingam.DirectLiNGAM()
    model.fit(df)
    B_est = model.adjacency_matrix_
    
    # Structural Recovery Check
    true_mask = (B_true != 0)
    est_mask = (np.abs(B_est) > 1e-10) 
    
    assert np.array_equal(true_mask, est_mask), (
        f"Structural mismatch! \nTrue edges: {np.where(true_mask)} "
        f"\nEst edges: {np.where(est_mask)}"
    )
    
    # Weight Recovery Check (atol=0.1 allows for small finite sample variance)
    assert np.allclose(B_true, B_est, atol=1e-1), "Weight estimation is outside acceptable asymptotic error."

def test_delta_structure_recovery():
    """
    Integration Test: Verifies that the directional difference (E1 - E2) 
    matches the ground truth delta from the paper-aligned generator.
    """
    # Initialize a master generator for the entire pipeline
    rng = np.random.default_rng(seed=42)
    
    n_nodes = 10
    n_samples = 10000
    n_positives = 4 # Renamed to match the refactored create_experiment_setup
    
    # 1. Generate Truth (Uses rng for structure and weights)
    # Note: Using the refactored parameter name 'n_positives'
    B1_true, B2_true, delta_true = create_experiment_setup(
        n_nodes=n_nodes, 
        edge_prob=0.3, 
        n_positives=n_positives,
        rng=rng
    )
    
    # 2. Simulate High-Fidelity Data (Uses the SAME rng to continue the sequence)
    df1 = simulate_from_graph(B1_true, n_samples, rng=rng)
    df2 = simulate_from_graph(B2_true, n_samples, rng=rng)
    
    # 3. Estimation
    model1 = lingam.DirectLiNGAM()
    model2 = lingam.DirectLiNGAM()
    
    model1.fit(df1)
    model2.fit(df2)
    
    # 4. Binary Extraction
    eps = 1e-10
    e1_bin = (np.abs(model1.adjacency_matrix_) > eps).astype(int)
    e2_bin = (np.abs(model2.adjacency_matrix_) > eps).astype(int)
    
    # 5. Calculate Estimated DIRECTIONAL Delta (E1 - E2)
    delta_est = np.logical_and(e1_bin == 1, e2_bin == 0).astype(int)
    
    # 6. Final Assertion
    assert np.array_equal(delta_true, delta_est), (
        f"Delta mismatch! Found {np.sum(delta_est)} positives in E1-E2, "
        f"expected {n_positives}."
    )

# TODO: run a test to check that B1 and B2 are correctly estimated and that the delta is correct.
