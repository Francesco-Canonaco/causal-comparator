import numpy as np
import lingam
import pytest
import pandas as pd
from causal_comparator.data_generation import EdgePerturbationSimulator, BaseSimulator

# TODO: add tests for the adjacency matrices generation. We want B2 to have n_positives edges additions and n_positives edges deletions compared to B1, and the delta to be correct. We also want to check that the weights are sampled correctly (in the right range and with the right distribution).
def test_simulator_logic():
    n_nodes = 12
    n_pos = 3
    seed = 42
    
    # Initialize the generator
    rng = np.random.default_rng(seed)
    
    # Instantiate with the generator injected
    sim = EdgePerturbationSimulator(n_nodes=n_nodes, rng=rng)
    
    # Generate graphs
    B1, B2, delta_pos = sim.create_graphs(edge_prob=0.2, n_positives=n_pos)
    
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

    print(f"✅ Structural Integrity Test: PASSED (Seed: {seed})")
    print(f"   -> Verified {n_pos} causal losses (G1 \\ G2)")
    print(f"   -> Verified {n_pos} causal gains (G2 \\ G1)")

"""
def test_delta_structure_recovery():
    
    # Initialize a master generator for the entire pipeline
    rng = np.random.default_rng(seed=42)
    
    n_nodes = 10
    n_samples = 10000
    n_positives = 4 
    
    # 1. Generate Truth (Canonical Order)
    B1_true, B2_true, delta_true = create_experiment_setup(
        n_nodes=n_nodes, 
        edge_prob=0.3, 
        n_positives=n_positives,
        rng=rng
    )
    
    # 2. Simulate High-Fidelity Data (Returns shuffled DataFrame and permutation key)
    # Note: df1 and df2 will likely have DIFFERENT permutations (p1 and p2)
    df1, p1 = simulate_from_graph(B1_true, n_samples, rng=rng)
    df2, p2 = simulate_from_graph(B2_true, n_samples, rng=rng)
    
    # 3. Estimation (LiNGAM operates on the shuffled column order)
    model1 = lingam.DirectLiNGAM()
    model2 = lingam.DirectLiNGAM()
    
    model1.fit(df1)
    model2.fit(df2)
    
    # 4. Un-shuffle the results back to canonical order using the inverse of p
    # idx contains the mapping: "Which original index is now at this position?"
    idx1 = np.argsort(p1)
    idx2 = np.argsort(p2)
    
    B1_recovered = model1.adjacency_matrix_[idx1, :][:, idx1]
    B2_recovered = model2.adjacency_matrix_[idx2, :][:, idx2]
    
    # 5. Binary Extraction (using machine epsilon)
    eps = 1e-10
    e1_bin = (np.abs(B1_recovered) > eps).astype(int)
    e2_bin = (np.abs(B2_recovered) > eps).astype(int)
    
    # 6. Calculate Estimated DIRECTIONAL Delta (E1 - E2)
    # This must be done on the recovered/aligned matrices
    delta_est = np.logical_and(e1_bin == 1, e2_bin == 0).astype(int)
    
    # 7. Final Assertion
    # Now both delta_true and delta_est are in the same 0..n node space
    assert np.array_equal(delta_true, delta_est), (
        f"Delta mismatch! Found {np.sum(delta_est)} positives in E1-E2, "
        f"expected {n_positives}."
    )
"""
