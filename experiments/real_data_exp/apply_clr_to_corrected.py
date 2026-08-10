"""
Applies CLR to the batch-corrected healthy cohort (healthy_family_corrected.csv,
the output of run_mmuphin_healthy.R), using the same convention as the rest of
this project (preprocess_corrected.py / clean_healthy_cohort.py): pseudocount =
half the smallest nonzero value in the family matrix, log-transform, then
subtract each sample's own row-mean of logs.

Standalone -- doesn't import from clean_healthy_cohort.py, so it can be run or
copied on its own.
"""
import pandas as pd
import numpy as np

IN_PATH = "healthy_family_corrected.csv"          # output of run_mmuphin_healthy.R
OUT_PATH = "healthy_family_batchcorrected_clr.csv"


def apply_clr(df, family_cols, verbose=True):
    """
    Applies CLR to `family_cols` in df. Returns a new dataframe with all
    other columns unchanged and the family columns replaced by their CLR
    values.
    """
    out = df.copy()
    vals = out[family_cols].values.astype(float)

    nonzero = vals[vals > 0]
    pseudocount = nonzero.min() / 2 if len(nonzero) else 1e-6

    log_vals = np.log(vals + pseudocount)
    clr_vals = log_vals - log_vals.mean(axis=1, keepdims=True)
    out[family_cols] = clr_vals

    if verbose:
        print(f"CLR applied to {len(family_cols)} families, pseudocount={pseudocount:.6g}")

    return out


if __name__ == "__main__":
    # index_col=0: this file was written by R with row.names = TRUE
    # (subject_id as the row identifier) -- see run_mmuphin_healthy.R
    df = pd.read_csv(IN_PATH, index_col=0, low_memory=False)

    family_cols = [c for c in df.columns if c.startswith("f__")]
    print(f"Loaded {IN_PATH}: {df.shape[0]} samples, {len(family_cols)} family columns")

    clr_df = apply_clr(df, family_cols)

    clr_df.to_csv(OUT_PATH)
    print(f"Saved {OUT_PATH}: {clr_df.shape[0]} samples, {clr_df.shape[1]} columns")
