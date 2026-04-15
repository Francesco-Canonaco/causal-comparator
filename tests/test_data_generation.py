import numpy as np
import lingam
import pytest
import pandas as pd
from causal_comparator.data_generation import EdgePerturbationSimulator, BaseSimulator

def test_simulator_logic():
    n_nodes = 12
    n_pos = 3
    seed = 42
    
    # Initialize the generator
    rng = np.random.default_rng(seed)
    
    # Instantiate with the generator injected
    sim = EdgePerturbationSimulator(n_nodes=n_nodes, rng=rng)
    
    # Generate graphs
    B1, B2, delta_true = sim.create_graphs(edge_prob=0.2, n_positives=n_pos)
    
    # 1. DAG Property
    assert np.all(np.triu(B1) == 0), "G1 is not strictly lower triangular."
    assert np.all(np.triu(B2) == 0), "G2 is not strictly lower triangular."
    
    # 2. Deletions (E1 \ E2)
    actual_deletions = np.logical_and(B1 != 0, B2 == 0)
    assert np.sum(actual_deletions) == n_pos, f"Expected {n_pos} deletions, found {np.sum(actual_deletions)}"
    
    # 3. Additions (E2 \ E1)
    actual_additions = np.logical_and(B2 != 0, B1 == 0)
    assert np.sum(actual_additions) == n_pos, f"Expected {n_pos} additions, found {np.sum(actual_additions)}"
    
    # 4. Weight Persistence
    mask_changed = actual_deletions | actual_additions
    assert np.all(B1[~mask_changed] == B2[~mask_changed]), "Untouched edges were modified."

def test_delta_structure_recovery():
    """
    Verifies that the new class architecture can correctly recover 
    the ground truth delta from high-fidelity (N=10k) shuffled data.
    """
    
    # 1. Setup Parameters
    seed = 42
    n_nodes = 10
    n_samples = 10000
    n_positives = 4 
    
    # 2. Instantiate the new Simulator (Injecting the RNG here)
    rng = np.random.default_rng(seed=seed)
    simulator = EdgePerturbationSimulator(n_nodes=n_nodes, rng=rng)
    
    # 3. Generate Truth (Canonical Order) using class methods
    # B1_true, B2_true, and delta_true are strictly lower triangular
    B1_true, B2_true, delta_true = simulator.create_graphs(
        edge_prob=0.3, 
        n_positives=n_positives
    )
    
    # 4. Simulate Shuffled Data
    # Each call uses the internal simulator.rng to create unique permutations p1 and p2
    df1, p1 = simulator.simulate_data(B1_true, n_samples)
    df2, p2 = simulator.simulate_data(B2_true, n_samples)
    
    # 5. Estimation
    # DirectLiNGAM estimates structure from the shuffled DataFrames
    model1 = lingam.DirectLiNGAM()
    model2 = lingam.DirectLiNGAM()
    
    model1.fit(df1)
    model2.fit(df2)
    
    # 6. Un-shuffle logic (Mapping back to original 0..n node space)
    # argsort(p) gives the mapping: "Which original index is at this column position?"
    idx1 = np.argsort(p1)
    idx2 = np.argsort(p2)
    
    B1_recovered = model1.adjacency_matrix_[idx1, :][:, idx1]
    B2_recovered = model2.adjacency_matrix_[idx2, :][:, idx2]
    
    # 7. Binary Extraction (Thresholding noise)
    eps = 1e-10
    e1_bin = (np.abs(B1_recovered) > eps).astype(int)
    e2_bin = (np.abs(B2_recovered) > eps).astype(int)
    
    # 8. Calculate Estimated Delta (E1 \ E2)
    # Alignment is guaranteed because of the un-shuffling in Step 6
    delta_est = np.logical_and(e1_bin == 1, e2_bin == 0).astype(int)
    
    # 9. Final Assertion
    assert np.array_equal(delta_true, delta_est), (
        f"Delta mismatch! Found {np.sum(delta_est)} positives in E1-E2, "
        f"expected {n_positives}."
    )


