# causal-comparator

A Python package for detecting **structural and mechanism-level changes between two causal networks**, learned independently with [DirectLiNGAM](https://jmlr.org/papers/volume12/shimizu11a/shimizu11a.pdf). It extends the edge-level network-comparison framework of [Ma & Tourani (2024)](https://doi.org/10.3390/e26030228) to the weighted linear-SEM setting, and to **node-level mechanism-shift detection**, letting you ask not just "did this edge change?" but "did this variable's entire causal mechanism change?"

`causal-comparator` was developed to benchmark the estimators described in *Node-Level Mechanism-Shift Detection in Linear Non-Gaussian Causal Networks* (Canonaco, Acerbi & Stella), and to give other researchers a reusable framework for constructing synthetic and mechanism-perturbation benchmarks with known ground truth.

## Why compare two causal graphs?

Given two datasets — a reference system and a comparison system (e.g. healthy vs. disease, pre- vs. post-intervention, two timepoints) — a naive approach fits a causal discovery algorithm on each dataset separately and diffs the results. This is fast but has no notion of statistical confidence, and is especially unreliable when the two datasets differ substantially in sample size. `causal-comparator` implements three estimators that address this:

| Estimator | What it does | When to use it |
|---|---|---|
| **Naive** | A single discovery run per dataset, differenced directly | Baseline only — no resampling, sensitive to sampling noise |
| **Bootstrap** | Standard bootstrap resampling (with replacement) on each dataset, differencing the resulting edge-selection frequencies | Best when the two datasets are roughly the same size |
| **RSBS** (Relative Sample-size Bootstrap Stability) | Down-samples the larger dataset once to match the smaller one, then bootstraps both at equal size | Best when sample sizes are imbalanced — the regime where plain Bootstrap loses its advantage over Naive |

On top of edge-level differences, the package aggregates results into **node-level mechanism scores** — summing the absolute change over a node's incoming edges (its full parent set) rather than looking at single edges in isolation — which is the quantity evaluated against a mechanism-perturbation ground truth.

## Installation

```bash
git clone https://github.com/Francesco-Canonaco/causal-comparator.git
cd causal-comparator
pip install -e .
```

Requires Python ≥ 3.9. Core dependencies (installed automatically): `numpy`, `pandas`, `networkx`, `scikit-learn`, `scipy`, `statsmodels`, `lingam`, `causal-learn`, `matplotlib`, `seaborn`. Install the `dev` extra (`pip install -e ".[dev]"`) for `pytest`, `black`, `isort`, and `flake8`.

## Quickstart

### 1. Compare two systems with a known edge-level ground truth

```python
import numpy as np
import lingam
from causal_comparator.data_generation import EdgePerturbationSimulator
from causal_comparator.discovery import CausalComparator
from causal_comparator.metrics import evaluate_binary_classification
from causal_comparator.utils import align_matrix

# 1. Simulate a baseline graph and a perturbed version of it
simulator = EdgePerturbationSimulator(n_nodes=10, noise_type="uniform", rng=np.random.default_rng(42))
B1_true, B2_true, delta_true = simulator.create_graphs(edge_prob=0.2, n_positives=2)

# 2. Generate data from each system (node labels are shuffled to blind the causal order)
df1, p1 = simulator.simulate_data(B1_true, n_samples=1000)
df2, p2 = simulator.simulate_data(B2_true, n_samples=100)

# 3. Run discovery + comparison
comparator = CausalComparator(data_i=df1, data_j=df2, model_class=lingam.DirectLiNGAM)

comparator.estimate_naive()          # fast, no resampling
comparator.estimate_bootstrap(n_sampling=100)   # standard bootstrap
comparator.estimate_rsbs(n_sampling=100, seed=42)  # equal-sample-size resampling

# 4. Re-align to the ground-truth node order and evaluate
B1 = align_matrix(comparator.freq_i, p1)
B2 = align_matrix(comparator.freq_j, p2)
scores = evaluate_binary_classification(y_true=delta_true, y_scores=B1 - B2, optimize=True)
print(scores)  # {'auc_roc': ..., 'aupr': ..., 'best_f1': ..., 'precision': ..., 'recall': ...}
```

### 2. Detect node-level mechanism shifts

```python
from causal_comparator.data_generation import MechanismPerturbationSimulator
from causal_comparator.metrics import calculate_node_scores, evaluate_binary_classification

simulator = MechanismPerturbationSimulator(n_nodes=15, rng=np.random.default_rng(0))

# n_perturbed_nodes nodes each get n_positives parents removed AND n_positives added
B1_true, B2_true, y_true_nodes = simulator.create_graphs(
    edge_prob=0.4, n_perturbed_nodes=3, n_positives=1
)

df1, p1 = simulator.simulate_data(B1_true, n_samples=1000)
df2, p2 = simulator.simulate_data(B2_true, n_samples=200)

comparator = CausalComparator(df1, df2, model_class=lingam.DirectLiNGAM)
comparator.estimate_rsbs(n_sampling=100, seed=0)

B1 = align_matrix(comparator.freq_i, p1)
B2 = align_matrix(comparator.freq_j, p2)

# Reduce the p x p edge-difference matrix to a p-dimensional node score,
# summing absolute changes over each node's incoming edges (its mechanism)
node_scores = calculate_node_scores(B2 - B1, mode="incoming")
metrics = evaluate_binary_classification(y_true_nodes, node_scores, optimize=True)
print(metrics)
```

See `examples/causal_comparator_quickstart.ipynb` for the full walkthrough, including how to run and plot a multi-seed benchmark.

## Package structure

```
src/causal_comparator/
├── data_generation.py   # BaseSimulator, EdgePerturbationSimulator, MechanismPerturbationSimulator
├── discovery.py          # CausalComparator: estimate_naive / estimate_bootstrap / estimate_rsbs
├── metrics.py             # evaluate_binary_classification, calculate_node_scores
├── plotting.py            # plot_ranking_metrics, plot_classification_metrics, plot_summary_performance
└── utils.py                # align_matrix, SHD_vectorized

examples/     # Quickstart notebook and the notebook used to produce the paper's figures
experiments/  # Scripts and cached results used to generate the paper's synthetic and real-data benchmarks
tests/        # pytest suite (data generation, metrics, replication, SHD)
```

### Core API

- **`data_generation.BaseSimulator`** — abstract base handling weight sampling (Shimizu et al., 2011 protocol) and data simulation from a structural equation model, `X = (I - B)^{-1} e`, with node labels randomly permuted to blind the causal order.
  - **`EdgePerturbationSimulator.create_graphs(edge_prob, n_positives)`** — generates a baseline graph and a version with `n_positives` edges removed and `n_positives` added, anywhere in the graph.
  - **`MechanismPerturbationSimulator.create_graphs(edge_prob, n_perturbed_nodes, n_positives)`** — generates a baseline graph and a version where `n_perturbed_nodes` target nodes each have `n_positives` parents removed and `n_positives` new parents added, returning a node-level binary ground truth.
- **`discovery.CausalComparator(data_i, data_j, model_class=lingam.DirectLiNGAM, **model_kwargs)`** — runs any compatible LiNGAM variant (`model_class` is swappable, e.g. `ICALiNGAM`, `VARLiNGAM`) as a discovery algorithm and exposes:
  - `estimate_naive()` — single fit per dataset, binarized difference.
  - `estimate_bootstrap(n_sampling=100)` — bootstrap edge-selection frequencies, differenced.
  - `estimate_rsbs(n_sampling=100, seed=None)` — down-samples the larger dataset once, then bootstraps both at equal size.
- **`metrics.evaluate_binary_classification(y_true, y_scores, optimize=True)`** — computes AUCROC, AUPR, and threshold-optimized F1/precision/recall (grid search over unique scores when `optimize=True`, or a fixed 0.5 threshold for already-binary Naive scores).
- **`metrics.calculate_node_scores(delta, mode="incoming")`** — reduces a `p x p` edge-difference matrix to a `p`-dimensional node score, summing absolute row-wise (`incoming`, i.e. mechanism/parent-set changes), column-wise (`outgoing`), or both (`total`).
- **`utils.align_matrix(B_est, p)`** / **`utils.SHD_vectorized(B_true, B_est)`** — undo the node-label permutation used to blind discovery, and compute Structural Hamming Distance between two adjacency matrices.

## Reproducing the paper's experiments

The `experiments/` directory contains the scripts and cached baselines used to produce the synthetic benchmark (`run_exps_final.py`, `results_bootstrap_vs_naive*/`) and the semi-synthetic real-microbiome validation (`real_data_exp/`) reported in the paper, along with the table-generation script (`generate_tables.py`).

## Citation

If you use this software, please cite:

```bibtex
@article{canonaco_mechanismshift,
  title   = {Node-Level Mechanism-Shift Detection in Linear Non-Gaussian Causal Networks: A Bootstrap-Based Framework with Application to the Human Gut Microbiome},
  author  = {Canonaco, Francesco and Acerbi, Enzo and Stella, Fabio},
  journal = {},
  year    = {2026}
}
```

and the original resampling framework this work extends:

> Ma, Sisi, and Roshan Tourani. "Comparing Causal Bayesian Networks Estimated from Data." *Entropy* 26.3 (2024): 228.

> Shimizu, Shohei, et al. "DirectLiNGAM: A Direct Method for Learning a Linear Non-Gaussian Structural Equation Model." *Journal of Machine Learning Research* 12 (2011): 1225–1248.

## License

MIT
