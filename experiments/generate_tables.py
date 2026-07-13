#!/usr/bin/env python3
"""
generate_tables.py

Regenerates Table 4 ("Mean AUCROC / AUPR by sample-size ratio r = N2/N1") and
Table 5 ("Paired comparison at a chosen sample-size pair"), directly from the raw grid_results_*.csv files, and writes each as its own CSV.

Designed to live inside the experiments/ folder (next to
results_bootstrap_vs_naive_rsbs/) and be run from there, but every path and
parameter is a command-line flag, so it works from anywhere and against any
source folder or (N1, N2) pair.

--------------------------------------------------------------------------
Table 4 -- ratio summary
--------------------------------------------------------------------------
For every method and every sample-size ratio r = N2/N1 actually present in
the data, reports the mean and SD of AUCROC and AUPR, pooled over network
size, edge density, perturbation settings, and seed. The set of ratios is
detected from the data, not hardcoded -- if you point this at a different
source file with different (N1, N2) combinations, the ratios in the output
CSV update automatically.

--------------------------------------------------------------------------
Table 5 -- paired comparison at a chosen (N1, N2)
--------------------------------------------------------------------------
For a single, specific (N1, N2) pair (set via --n1 / --n2), runs a paired
comparison between every pair of methods present (Naive, Bootstrap, RSBS):
for each of AUCROC, AUPR, and F1, computes the mean difference, the paired
t-statistic, and a two-sided p-value across all matched (seed, parameter
combination) runs. "Paired" means: for a fixed seed and fixed
n_nodes/edge_prob/n_perturbed_nodes/n_positives, every method was run on the
exact same underlying graphs and simulated data (see data_generation.py /
run_exps_final.py), so the comparison isolates the effect of the estimator.

If scipy is installed, p-values use the exact t-distribution
(scipy.stats.ttest_rel); otherwise the script falls back to a normal
approximation (accurate for the sample sizes used here, typically n > 500
pairs) so it still runs with no extra dependencies beyond pandas/numpy.

--------------------------------------------------------------------------
Usage
--------------------------------------------------------------------------
    # from inside experiments/experiments/, using the defaults:
    python generate_tables.py

    # pointing at a different source folder / different sample sizes:
    python generate_tables.py \
        --data-dir path/to/results_bootstrap_vs_naive_rsbs \
        --n1 400 --n2 200 \
        --out-table4 table4_ratio_summary.csv \
        --out-table5 table5_paired_comparison.csv

Requires: pandas, numpy. scipy is optional (used automatically if present).
"""

import argparse
import glob
import math
import os

import numpy as np
import pandas as pd

try:
    from scipy import stats as _scipy_stats

    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False

# Columns that are model *outputs* (metrics), not part of a run's identity.
_OUTPUT_COLS = {"auc_roc", "aupr", "best_f1", "precision", "recall", "best_threshold"}
# Columns that are pure run metadata, not experimental parameters.
_METADATA_COLS = {"total_execution_time_min", "timestamp", "density_folder", "source_file"}


def load_grid_results(data_dir: str) -> pd.DataFrame:
    """Load MASTER_grid_results_*nodes_ALL.csv if present in data_dir;
    otherwise fall back to a recursive glob over raw grid_results_*.csv
    files (deduplicated), so this works whether or not the merge step has
    been run yet."""
    master_files = sorted(glob.glob(os.path.join(data_dir, "MASTER_grid_results_*nodes_ALL.csv")))

    if master_files:
        print(f"[load] Found {len(master_files)} MASTER file(s):")
        for f in master_files:
            print("   -", f)
        df = pd.concat([pd.read_csv(f) for f in master_files], ignore_index=True)
    else:
        raw_files = sorted(glob.glob(os.path.join(data_dir, "**", "grid_results_*.csv"), recursive=True))
        if not raw_files:
            raise FileNotFoundError(
                f"No MASTER_grid_results_*.csv or grid_results_*.csv found under '{data_dir}'. "
                f"Check --data-dir."
            )
        print(f"[load] No MASTER files found; loading {len(raw_files)} raw grid_results file(s):")
        for f in raw_files:
            print("   -", f)
        df = pd.concat([pd.read_csv(f) for f in raw_files], ignore_index=True)
        df = df.drop_duplicates()

    required_cols = {"method", "auc_roc", "aupr", "best_f1", "n1", "n2", "seed"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input data is missing expected column(s): {sorted(missing)}")

    return df


def paired_ttest(x, y):
    """Two-sided paired t-test. Returns (mean_diff, std_diff, t_stat, dof, p_value, n)."""
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    n = len(d)
    mean_d = float(d.mean())
    sd_d = float(d.std(ddof=1)) if n > 1 else float("nan")
    se = sd_d / math.sqrt(n) if n > 1 and sd_d > 0 else float("nan")
    t = mean_d / se if se and se > 0 else float("inf")
    dof = n - 1

    if _HAVE_SCIPY:
        p = float(2 * _scipy_stats.t.sf(abs(t), df=dof))
    else:
        p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))

    return mean_d, sd_d, t, dof, p, n


def build_table4(df: pd.DataFrame, out_path: str) -> pd.DataFrame:
    """Mean AUCROC / AUPR by (method, ratio = N2/N1), pooled over every other
    column (network size, density, perturbation settings, seed)."""
    d = df.copy()
    d["ratio_n2_over_n1"] = d["n2"] / d["n1"]

    rows = []
    for (method, ratio), g in d.groupby(["method", "ratio_n2_over_n1"]):
        rows.append(
            {
                "method": method,
                "ratio_n2_over_n1": ratio,
                "n_runs": len(g),
                "auc_roc_mean": g["auc_roc"].mean(),
                "auc_roc_std": g["auc_roc"].std(ddof=1),
                "aupr_mean": g["aupr"].mean(),
                "aupr_std": g["aupr"].std(ddof=1),
            }
        )

    out = pd.DataFrame(rows).sort_values(["ratio_n2_over_n1", "method"]).reset_index(drop=True)
    out.to_csv(out_path, index=False)
    print(f"[table4] Wrote {len(out)} rows ({out['ratio_n2_over_n1'].nunique()} ratio(s) "
          f"x {out['method'].nunique()} method(s)) to {out_path}")
    return out


def build_table5(
    df: pd.DataFrame, n1: int, n2: int, metrics: list, method_order: list, out_path: str
) -> pd.DataFrame:
    """Paired comparison of every pair of methods present, at a single
    (n1, n2) sample-size pair, for each metric in `metrics`.

    `method_order` fixes both which comparisons are generated and their sign
    convention (row = earlier method in the list, column = later method, so
    the reported difference is "earlier minus later"). The default
    ["RSBS", "Bootstrap", "Naive"] reproduces the row order and sign
    convention of Table 5 in the paper (RSBS-Naive, RSBS-Bootstrap,
    Bootstrap-Naive); any method not listed is appended alphabetically."""
    d = df[(df["n1"] == n1) & (df["n2"] == n2)].copy()
    if d.empty:
        available = sorted(set(zip(df["n1"], df["n2"])))
        raise ValueError(
            f"No rows found for n1={n1}, n2={n2}. Available (n1, n2) pairs in this data: {available}"
        )

    missing_metrics = [m for m in metrics if m not in d.columns]
    if missing_metrics:
        raise ValueError(f"Requested metric(s) not present in the data: {missing_metrics}")

    key_cols = [c for c in d.columns if c not in _OUTPUT_COLS | _METADATA_COLS | {"method"}]
    pivot = d.pivot_table(index=key_cols, columns="method", values=metrics)
    n_pairs = len(pivot)
    print(f"[table5] {n_pairs} paired (seed, parameter-combo) row(s) at n1={n1}, n2={n2}")

    methods_present = set(d["method"].unique())
    ordered_methods = [m for m in method_order if m in methods_present]
    ordered_methods += sorted(methods_present - set(ordered_methods))  # append any unlisted methods

    ordered_comparisons = [
        (ordered_methods[i], ordered_methods[j])
        for i in range(len(ordered_methods))
        for j in range(i + 1, len(ordered_methods))
    ]

    rows = []
    for a, b in ordered_comparisons:
        for metric in metrics:
            if (metric, a) not in pivot.columns or (metric, b) not in pivot.columns:
                continue
            x, y = pivot[(metric, a)], pivot[(metric, b)]
            valid = x.notna() & y.notna()
            mean_d, sd_d, t, dof, p, n = paired_ttest(x[valid], y[valid])
            rows.append(
                {
                    "comparison": f"{a}-{b}",
                    "metric": metric,
                    "n_pairs": n,
                    "mean_diff": mean_d,
                    "std_diff": sd_d,
                    "t_stat": t,
                    "dof": dof,
                    "p_value": p,
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    print(f"[table5] Wrote {len(out)} rows to {out_path}")
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate Table 4 (AUCROC/AUPR by sample-size ratio) and "
        "Table 5 (paired estimator comparison at a chosen N1/N2) from the raw "
        "grid_results CSVs, writing each as its own CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        default="results_bootstrap_vs_naive_rsbs",
        help="Folder containing MASTER_grid_results_*nodes_ALL.csv, or (as a fallback) "
        "raw grid_results_*.csv files searched recursively. Change this to point at a "
        "different experiment / source directory.",
    )
    parser.add_argument("--n1", type=int, default=800, help="Larger sample size N1 for Table 5.")
    parser.add_argument("--n2", type=int, default=100, help="Smaller sample size N2 for Table 5.")
    parser.add_argument(
        "--metrics",
        default="auc_roc,aupr,best_f1",
        help="Comma-separated list of metric columns to compare in Table 5.",
    )
    parser.add_argument(
        "--method-order",
        default="RSBS,Bootstrap,Naive",
        help="Comma-separated priority order for Table 5's comparisons and sign convention "
        "(row method minus column method). Default matches the paper's Table 5 "
        "(RSBS-Naive, RSBS-Bootstrap, Bootstrap-Naive). Any method present in the data but "
        "not listed here is appended alphabetically.",
    )
    parser.add_argument("--out-table4", default="table4_ratio_summary.csv", help="Output CSV path for Table 4.")
    parser.add_argument("--out-table5", default="table5_paired_comparison.csv", help="Output CSV path for Table 5.")
    args = parser.parse_args()

    print(
        f"[info] scipy available: {_HAVE_SCIPY} "
        f"({'exact t-distribution p-values' if _HAVE_SCIPY else 'normal-approximation p-values (install scipy for exact values)'})"
    )

    df = load_grid_results(args.data_dir)
    print(
        f"[info] Loaded {len(df)} total rows. Methods: {sorted(df['method'].unique())}. "
        f"n_nodes present: {sorted(df['n_nodes'].unique()) if 'n_nodes' in df.columns else 'n/a'}."
    )

    build_table4(df, args.out_table4)
    build_table5(
        df,
        args.n1,
        args.n2,
        [m.strip() for m in args.metrics.split(",")],
        [m.strip() for m in args.method_order.split(",")],
        args.out_table5,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
