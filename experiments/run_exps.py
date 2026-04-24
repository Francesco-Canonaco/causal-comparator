import pandas as pd
import numpy as np
import time
import os
import itertools
from joblib import Parallel, delayed
from datetime import datetime

# --- IMPORT YOUR CUSTOM MODULES ---
import lingam
from causal_comparator.data_generation import MechanismPerturbationSimulator
from causal_comparator.discovery import CausalComparator
from causal_comparator.utils import align_matrix
from causal_comparator.metrics import calculate_node_scores, evaluate_binary_classification

# =================================================================
# PARAMETER GRID - Add values to these lists to expand the study
# =================================================================
GRID_PARAMS = {
    "n_nodes": [10],              # e.g., [25, 30, 50]
    "n_sampling": [60],           # Bootstrap samples
    "edge_prob": [0.4],           # Graph density
    "n_perturbed_nodes": [1],     # Number of nodes to perturb
    "n_positives": [1],           # Strength of perturbation
    "n1": [1000],                 # Pre-perturbation samples
    "n2": [100],                  # Post-perturbation samples
    "n_iterations": range(20),    # This defines the 20 seeds
}

OUTPUT_DIR = "./results"
N_JOBS = -1  # Use all 100 cores on the server

def run_task(p):
    """
    p is a dictionary containing one specific combination of parameters.
    """
    # Initialize RNG with the specific seed from the grid
    rng = np.random.default_rng(p['n_iterations'])
    
    # 1. Simulator
    simulator = MechanismPerturbationSimulator(n_nodes=p['n_nodes'], rng=rng)
    
    # 2. Graphs
    B1_true, B2_true, y_true_nodes = simulator.create_graphs(
        edge_prob=p['edge_prob'], 
        n_perturbed_nodes=p['n_perturbed_nodes'], 
        n_positives=p['n_positives']
    )
    
    # 3. Data
    df1, p1 = simulator.simulate_data(B1_true, p['n1'])
    df2, p2 = simulator.simulate_data(B2_true, p['n2'])
    
    # 4. Discovery
    comparator = CausalComparator(df1, df2, model_class=lingam.DirectLiNGAM)
    methods = ['Naive', 'Bootstrap', 'RSBS']
    task_results = []

    for name in methods:
        if name == 'Naive':
            comparator.estimate_naive()
        elif name == 'Bootstrap':
            comparator.estimate_bootstrap(n_sampling=p['n_sampling'])
        elif name == 'RSBS':
            comparator.estimate_rsbs(n_sampling=p['n_sampling'])

        # 5. Metrics
        B1_aligned = align_matrix(comparator.freq_i, p1)
        B2_aligned = align_matrix(comparator.freq_j, p2)
        delta = B2_aligned - B1_aligned
        scores = calculate_node_scores(delta, mode='incoming')

        metrics = evaluate_binary_classification(y_true_nodes, scores, optimize=True)
        
        # Merge metrics with the parameters used for this specific run
        full_result = {**metrics, **p, 'method': name}
        task_results.append(full_result)
        
    return task_results

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # --- Generate the Combinatorial Grid ---
    keys, values = zip(*GRID_PARAMS.items())
    # This creates a list of dicts for every combination
    task_list = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    total_tasks = len(task_list)
    print(f"--- STARTING GRID EXPERIMENT ---")
    print(f"Total Parameter Combinations (including seeds): {total_tasks}")
    print(f"Utilizing {N_JOBS if N_JOBS != -1 else 'all'} cores...")

    # --- Start Global Timer ---
    start_time = time.time()
    
    # Run Parallel Grid
    results_nested = Parallel(n_jobs=N_JOBS)(
        delayed(run_task)(params) for params in task_list
    )
    
    # --- End Global Timer ---
    end_time = time.time()
    total_duration = (end_time - start_time) / 60
    
    # Flatten and Save
    flattened_results = [res for sublist in results_nested for res in sublist]
    df_results = pd.DataFrame(flattened_results)
    
    # Add performance metadata
    df_results['total_execution_time_min'] = total_duration
    df_results['timestamp'] = datetime.now().isoformat()

    # Dynamic filename based on first node count and total count
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    fname = f"grid_results_{total_tasks}tasks_{ts}.csv"
    save_path = os.path.join(OUTPUT_DIR, fname)
    
    df_results.to_csv(save_path, index=False)
    
    print(f"--- FINISHED ---")
    print(f"Total wall-clock time: {total_duration:.2f} minutes")
    print(f"Results saved to {save_path}")