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
# PARAMETER GRID - Now includes 'method' for total parallelism
# =================================================================
GRID_PARAMS = {
    "n_nodes": [10],              
    "method": ["Naive", "Bootstrap", "RSBS"], # Treated as independent tasks
    "seed": range(10),            # Renamed for plotting.py compatibility
    "n_sampling": [100],           
    "edge_prob": [0.5],           
    "n_perturbed_nodes": [3],     
    "n_positives": [1],           
    "n1": [700],                 
    "n2": [100],                  
}

OUTPUT_DIR = "./results"


N_JOBS = -1  

def run_task(p):
    """
    Processes a single (Method + Seed) combination.
    """
    # 1. Initialize RNG and Simulator
    rng = np.random.default_rng(p['seed'])
    simulator = MechanismPerturbationSimulator(n_nodes=p['n_nodes'], rng=rng)
    
    # 2. Generate Ground Truth and Data[cite: 5]
    B1_true, B2_true, y_true_nodes = simulator.create_graphs(
        edge_prob=p['edge_prob'], 
        n_perturbed_nodes=p['n_perturbed_nodes'], 
        n_positives=p['n_positives']
    )
    df1, p1 = simulator.simulate_data(B1_true, p['n1'])
    df2, p2 = simulator.simulate_data(B2_true, p['n2'])
    
    # 3. Discovery Engine
    # Uses the parallelized methods in discovery_3.py
    comparator = CausalComparator(df1, df2, model_class=lingam.DirectLiNGAM)
    
    method_name = p['method']
    if method_name == 'Naive':
        comparator.estimate_naive()
    elif method_name == 'Bootstrap':
        comparator.estimate_bootstrap(n_sampling=p['n_sampling'])
    elif method_name == 'RSBS':
        comparator.estimate_rsbs(n_sampling=p['n_sampling'])

    # 4. Alignment and Scoring[cite: 5]
    B1_aligned = align_matrix(comparator.freq_i, p1)
    B2_aligned = align_matrix(comparator.freq_j, p2)
    delta = B2_aligned - B1_aligned
    scores = calculate_node_scores(delta, mode='incoming')

    # 5. Evaluation[cite: 5]
    metrics = evaluate_binary_classification(y_true_nodes, scores, optimize=True)
    
    # Return a single result dictionary for this task[cite: 5]
    return {**metrics, **p}

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Generate the combinatorial grid[cite: 5]
    keys, values = zip(*GRID_PARAMS.items())
    task_list = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    total_tasks = len(task_list)
    print(f"--- STARTING OPTIMIZED GRID EXPERIMENT ---")
    print(f"Total Tasks: {total_tasks} | Parallel Workers: {N_JOBS}")

    start_time = time.time()
    
    # Launch Parallel Grid
    # Each task in task_list is one method for one seed[cite: 5]
    results = Parallel(n_jobs=N_JOBS)(
        delayed(run_task)(params) for params in task_list
    )
    
    end_time = time.time()
    total_duration = (end_time - start_time) / 60
    
    # Create DataFrame and add metadata[cite: 5]
    df_results = pd.DataFrame(results)
    df_results['total_execution_time_min'] = total_duration
    df_results['timestamp'] = datetime.now().isoformat()

    # Save Results[cite: 5]
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    node_val = GRID_PARAMS['n_nodes'][0]
    fname = f"grid_results_{node_val}nodes_{total_tasks}tasks_{ts}.csv"
    save_path = os.path.join(OUTPUT_DIR, fname)
    
    df_results.to_csv(save_path, index=False)
    
    print(f"--- FINISHED ---")
    print(f"Wall-clock time: {total_duration:.2f} minutes")
    print(f"Results saved to {save_path}")