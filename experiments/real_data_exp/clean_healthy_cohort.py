import pandas as pd
import numpy as np


def clean_healthy_cohort(
    df,
    exclude_non_westernized=False,   # judgment call -- off by default, see note below
    min_reads=1_000_000,             # literature-grounded default -- see note below; None disables
    verbose=True,
):
    """
    Filters a metadata+family-abundance dataframe (one row per sample) down
    to a clean healthy-adult-stool cohort, one sample per subject.

    Steps, in order:
      1. disease == 'healthy'            (the file wasn't actually filtered to this yet)
      2. antibiotics_current_use != 'yes' (drop confirmed antibiotic users; NaN/'no' kept)
      3. pregnant != 'yes'
      4. lactating != 'yes'
      5. [optional] non_westernized != 'yes'
      6. number_reads >= min_reads (default 1,000,000 -- see note in
         filter_prevalent_families()/docs below for the literature basis;
         pass None to disable)
      7. collapse to one row per subject_id: prefer the earliest visit
         (lowest visit_number, NaN treated as earliest/only visit), tie-break
         by earliest days_from_first_collection.

    Returns the cleaned dataframe. Prints a before/after diagnostic summary
    (sample counts, per-study counts, small-study warning) unless verbose=False.
    """
    n0 = len(df)
    out = df.copy()

    # 1. disease filter -- the actual fix for the mislabeled input file
    out = out[out["disease"] == "healthy"]
    n1 = len(out)

    # 2. antibiotics: drop confirmed current users; keep 'no' and NaN (unrecorded)
    out = out[out["antibiotics_current_use"] != "yes"]
    n2 = len(out)

    # 3. pregnancy
    out = out[out["pregnant"] != "yes"]
    n3 = len(out)

    # 4. lactation
    out = out[out["lactating"] != "yes"]
    n4 = len(out)

    # 5. optional: westernization
    if exclude_non_westernized:
        out = out[out["non_westernized"] != "yes"]
    n5 = len(out)

    # 6. optional: sequencing depth floor
    if min_reads is not None:
        out = out[out["number_reads"] >= min_reads]
    n6 = len(out)

    # 7. one sample per subject: earliest visit, then earliest days_from_first_collection
    out = out.sort_values(
        by=["subject_id", "visit_number", "days_from_first_collection"],
        na_position="first",
    )
    out = out.drop_duplicates(subset="subject_id", keep="first")
    n7 = len(out)

    if verbose:
        print("Filtering funnel:")
        print(f"  start                                   : {n0}")
        print(f"  disease == 'healthy'                    : {n1}  (-{n0-n1})")
        print(f"  drop antibiotics_current_use == 'yes'   : {n2}  (-{n1-n2})")
        print(f"  drop pregnant == 'yes'                  : {n3}  (-{n2-n3})")
        print(f"  drop lactating == 'yes'                 : {n4}  (-{n3-n4})")
        if exclude_non_westernized:
            print(f"  drop non_westernized == 'yes'           : {n5}  (-{n4-n5})")
        if min_reads is not None:
            print(f"  number_reads >= {min_reads:<24,}: {n6}  (-{n5-n6})")
        print(f"  one sample per subject_id (earliest)    : {n7}  (-{n6-n7})")
        print()
        print(f"Final: {n7} samples, {out['subject_id'].nunique()} unique subjects "
              f"(should match), {out['study_name'].nunique()} studies")
        print()
        study_counts = out["study_name"].value_counts()
        small = study_counts[study_counts < 10]
        if len(small):
            print(f"Studies with <10 samples after cleaning ({len(small)}):")
            print(small)
        else:
            print("No studies with <10 samples after cleaning.")

    return out


def filter_prevalent_families(
    df,
    min_prevalence=0.50,
    min_mean_rel_abundance=1e-4,     # 0.01% -- matches preprocess_microbiome.py's ACTUAL applied
                                      # threshold (its min_mean=1e-2 was on a 0-100 scale, i.e. 1e-4
                                      # on a 0-1 scale, NOT 1%)
    verbose=True,
):
    """
    Drops family (f__...) columns that are too rare/low-abundance to be
    reliably modeled, using the same prevalence+abundance convention as the
    rest of this project (preprocess_microbiome.py): a family must be
    present (>0 reads) in at least `min_prevalence` of samples AND average
    at least `min_mean_rel_abundance` RELATIVE abundance.

    The f__ columns in this file are raw read counts (see module docstring),
    so relative abundance is computed internally (count / that sample's
    total family-level read count) purely to make this decision -- the
    columns returned are the original raw counts for whichever families
    survive, not the normalized values.

    Returns (filtered_df, keep_families) where filtered_df has all
    non-family columns plus only the surviving f__ columns.
    """
    fam_cols = [c for c in df.columns if c.startswith("f__")]
    counts = df[fam_cols].fillna(0.0)
    rel = counts.div(counts.sum(axis=1), axis=0)   # per-sample relative abundance, for filtering only

    prevalence = (counts > 0).mean(axis=0)
    mean_rel_abundance = rel.mean(axis=0)

    keep_families = sorted(
        f for f in fam_cols
        if prevalence[f] >= min_prevalence and mean_rel_abundance[f] >= min_mean_rel_abundance
    )

    non_fam_cols = [c for c in df.columns if c not in fam_cols]
    filtered = df[non_fam_cols + keep_families].copy()

    if verbose:
        print(f"Family filter: prevalence >= {min_prevalence:.0%}, "
              f"mean relative abundance >= {min_mean_rel_abundance:.1%}")
        print(f"  {len(fam_cols)} families in -> {len(keep_families)} families out")
        print()
        print("Surviving families:")
        print(keep_families)

    return filtered, keep_families


def apply_clr(df, keep_families, verbose=True):
    """
    Applies CLR to the raw-count family columns, same convention used
    throughout this project (preprocess_corrected.py): pseudocount = half
    the smallest nonzero value in the family matrix, log-transform, then
    subtract each sample's own row-mean of logs.

    Returns a new dataframe: all non-family metadata columns unchanged,
    plus the family columns replaced by their CLR values.
    """
    out = df.copy()
    vals = out[keep_families].values.astype(float)
    nonzero = vals[vals > 0]
    pseudocount = nonzero.min() / 2 if len(nonzero) else 1e-6
    log_vals = np.log(vals + pseudocount)
    clr_vals = log_vals - log_vals.mean(axis=1, keepdims=True)
    out[keep_families] = clr_vals

    if verbose:
        print(f"CLR applied to {len(keep_families)} families, pseudocount={pseudocount:.6g}")

    return out


def study_variance_share(clr_df, keep_families, verbose=True):
    """
    Quick PERMANOVA-style diagnostic (Euclidean, one-way on study_name):
    what fraction of total variance in the CLR-transformed family profile
    is "explained by" which study a sample came from? This is exactly the
    kind of variance MMUPHin's batch correction would remove -- computing
    it tells you how much is actually at stake in skipping that step,
    rather than guessing.

    R2 = SS_between(study) / SS_total, both computed as summed squared
    Euclidean distances in the 24-dimensional CLR space.
    """
    X = clr_df[keep_families].values.astype(float)
    grand_mean = X.mean(axis=0)
    ss_total = ((X - grand_mean) ** 2).sum()

    ss_between = 0.0
    for study, idx in clr_df.groupby("study_name").groups.items():
        Xi = X[clr_df.index.get_indexer(idx)]
        study_mean = Xi.mean(axis=0)
        ss_between += len(Xi) * ((study_mean - grand_mean) ** 2).sum()

    r2 = ss_between / ss_total
    if verbose:
        print(f"Variance in CLR family profile explained by study_name (R^2): {r2:.1%}")
        print(f"(SS_between={ss_between:.1f}, SS_total={ss_total:.1f})")
    return r2


if __name__ == "__main__":
    df = pd.read_csv(
        "df_healthy_crc_ibd.csv",
        low_memory=False,
    )
    clean = clean_healthy_cohort(df)
    print()
    clean_trimmed, keep_families = filter_prevalent_families(clean)
    print()

    # Version 1: no CLR (raw counts, cleaned + family-filtered)
    no_clr_path = "df_healthy_final_no_clr.csv"
    clean_trimmed.to_csv(no_clr_path, index=False)
    print(f"Saved (no CLR) dataset to {no_clr_path}, shape {clean_trimmed.shape}")
    print()

    # Version 2: CLR applied, still no batch correction
    clr_df = apply_clr(clean_trimmed, keep_families)
    clr_path = "df_healthy_final_clr.csv"
    clr_df.to_csv(clr_path, index=False)
    print(f"Saved (CLR) dataset to {clr_path}, shape {clr_df.shape}")
    print()

    study_variance_share(clr_df, keep_families)
