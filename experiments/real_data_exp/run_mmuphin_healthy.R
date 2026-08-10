# 1. Load Libraries
if (!require("BiocManager", quietly = TRUE)) install.packages("BiocManager")
if (!require("MMUPHin", quietly = TRUE)) BiocManager::install("MMUPHin")
library(MMUPHin)
library(magrittr)

# 2. Load the Data
# CHANGED: now points at the cleaned, healthy-only, family-level, NOT-YET-CLR'd
# dataset (df_healthy_final_no_clr.csv) instead of the old species-level,
# disease-mixed healthy_crc_ibd.csv.
# CHANGED: row.names = 1 replaced with row.names = "subject_id". The old
# input file had a row-index as its literal first column, matching
# row.names = 1. This file doesn't (it was saved with index=False from
# pandas) -- its first column is the real "study_name" data column, so
# row.names = 1 would silently swallow that instead. "subject_id" is
# explicitly named here rather than relying on column position, and is
# guaranteed unique/complete since the cleaning step deduplicated to one
# sample per subject.
full_df <- read.csv("./notebooks/00_exploratory/df_healthy_final_no_clr.csv", row.names = "subject_id", check.names = FALSE)

# 3. Separate Metadata and Abundance
# MMUPHin expects taxa as ROWS and samples as COLUMNS
# CHANGED: this file is FAMILY-level, not species-level, so taxa columns
# are prefixed 'f__' (e.g. f__Bacteroidaceae), not 's__'. The old regex
# would have matched zero columns here.
taxon_cols <- grep("^f__", colnames(full_df), value = TRUE)
metadata_cols <- setdiff(colnames(full_df), taxon_cols)

# Transpose abundance: Features x Samples
abd_tab <- t(full_df[, taxon_cols])
meta_tab <- full_df[, metadata_cols]

# --- THE FIX: HANDLE MISSING VALUES ---
# 1. Replace any NA/NaN with 0 (just in case they slipped through)
abd_tab[is.na(abd_tab)] <- 0

# 2. Drop any samples that ended up with a zero sum in R
# (MMUPHin cannot process columns with a sum of 0)
keep_samples <- colSums(abd_tab) > 0
abd_tab <- abd_tab[, keep_samples]
meta_tab <- meta_tab[keep_samples, ]

# 3. Re-normalize to exact proportions for MMUPHin validation
# (abd_tab holds raw read counts at this point -- see module note in
# clean_healthy_cohort.py -- so this step is still required and correct)
abd_tab <- sweep(abd_tab, 2, colSums(abd_tab), "/")
# --------------------------------------
# 4. Perform Batch Adjustment
# batch: The 'Study Name' identifying the clusters seen in your PCoA
# CHANGED: covariates = "disease" removed. This cohort is healthy-only now,
# so `disease` is constant (every row is "healthy") -- passing a covariate
# with a single level either errors on a rank-deficient design matrix or
# does nothing. Running batch-only (covariates = NULL, the package default)
# is correct here. If you want to explicitly protect a real biological
# covariate now that disease isn't doing that job, `non_westernized` is
# the candidate worth considering (~7% of samples) -- that's optional and
# not something this script assumes for you.
fit_adjust <- adjust_batch(feature_abd = abd_tab,
                           batch = "study_name",
                           covariates = NULL,
                           data = meta_tab,
                           control = list(verbose = TRUE))

# 5. Extract the Adjusted Abundances
# These are the "Clean" values for your LiNGAM model
corrected_abd <- fit_adjust$feature_abd_adj

# 6. Reformat and Save
# Transpose back: Samples x Taxa for your Python/LiNGAM workflow
# CHANGED: output filename -- avoids colliding with/overwriting the old
# species-level, disease-mixed master_dataset_corrected.csv.
# row.names = TRUE (the default, kept explicit here) writes subject_id
# back out as the CSV's first (unnamed-header) column -- same format as
# every other R-corrected file in this project, so read it back into
# Python the same way: pd.read_csv(..., index_col=0).
final_df <- cbind(meta_tab, t(corrected_abd))
write.csv(final_df, "healthy_family_corrected.csv", row.names = TRUE)

cat("\n✅ Success! Batch-corrected data saved to 'healthy_family_corrected.csv'\n")
