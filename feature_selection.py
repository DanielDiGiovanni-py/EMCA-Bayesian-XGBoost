#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 15:14:32 2025

@author: daniel

Provides functions for filtering features based on reproducibility (ICC checks)
and removing highly correlated features using Spearman correlation.
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr

def combat_location_scale_radiomics(
    df: pd.DataFrame,
    batch: pd.Series,
    radiomic_cols: list,
    eps: float = 1e-8
) -> pd.DataFrame:
    """
    A simplified ComBat-like location-scale correction applied
    ONLY to specified radiomic feature columns.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing both radiomic features and other columns
        (e.g., patient IDs, outcomes, etc.).
    batch : pd.Series or array-like of shape (N,)
        Batch labels for each sample (e.g., site/scanner).
        Must align with df's rows.
    radiomic_cols : list of str
        Columns in df corresponding to radiomic features to be harmonized.
    eps : float
        Small constant to avoid division by zero in std calculations.

    Returns
    -------
    df_corrected : pd.DataFrame
        A copy of the original df but with the specified radiomic_cols
        corrected via location-scale harmonization.
        All other columns remain unchanged.
    """

    # Ensure the row alignment is consistent between df and batch
    if len(df) != len(batch):
        raise ValueError("df and batch must have the same number of rows.")

    # We'll work on a copy so we don't modify the original df in place
    df_corrected = df.copy()

    # Extract only the radiomic subset to be harmonized
    X = df_corrected[radiomic_cols].values  # shape (N, F)
    batch = np.asarray(batch)
    unique_batches = np.unique(batch)

    # 1) Compute global (pooled) mean and std for each radiomic feature
    global_means = np.nanmean(X, axis=0)        # shape (F,)
    global_stds  = np.nanstd(X, axis=0) + eps   # shape (F,)

    # Prepare an output array for the corrected subset
    X_corrected = np.zeros_like(X)

    # 2) For each batch, compute batch-specific mean/std, then correct
    for b in unique_batches:
        idx_b = np.where(batch == b)[0]  # sample indices for batch b
        X_b = X[idx_b, :]               # shape (nb, F)

        batch_means = np.nanmean(X_b, axis=0)
        batch_stds  = np.nanstd(X_b, axis=0) + eps

        # Standardize within this batch
        X_b_std = (X_b - batch_means) / batch_stds

        # Rescale to global distribution
        X_b_corrected = X_b_std * global_stds + global_means

        # Store in the correct slice
        X_corrected[idx_b, :] = X_b_corrected

    # 3) Put the corrected radiomic data back into df_corrected
    df_corrected.loc[:, radiomic_cols] = X_corrected

    return df_corrected


def filter_features(
    df: pd.DataFrame,
    target_col: str = 'y',
    reproducibility_thresh: float = 0.8,
    correlation_thresh: float = 0.95,
    exclude_columns: list = None
) -> pd.DataFrame:
    """
    Filters features in the dataset based on:
      1) Reproducibility: 
         - Use rows where ICC == 1 to calculate Spearman correlation
           between R1 and R2 features for each imaging sequence.
         - Keep only features with correlation >= reproducibility_thresh.
      2) High correlation among reproducible features:
         - Compute Spearman correlation among these features (over all data),
           and remove any feature that has correlation > correlation_thresh
           with a feature chosen earlier.

    Parameters
    ----------
    df : pd.DataFrame
        The full dataset containing features, 'ICC', 'Hospital', etc.
    target_col : str
        The name of the target column in the dataset (e.g. 'y').
    reproducibility_thresh : float
        Minimum Spearman correlation between R1 and R2 for a feature
        to be considered reproducible.
    correlation_thresh : float
        Maximum allowable correlation between any two features to keep both.
    exclude_columns : list
        Columns that are never considered features (e.g. 'Hospital', 'ICC', 
        'LVSI_extent', 'pathology', or the target column, if desired).
        They will be preserved in the final DataFrame but excluded 
        from correlation checks.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame, containing only:
          - The reproducible, non-correlated features
          - The target column
          - Any columns in exclude_columns
          - Any columns that were not numeric to begin with
            (if you need them).
    """
    # ----------------------------------------------------------------------
    # 1) Set up which columns to exclude from numeric correlation checks
    # ----------------------------------------------------------------------
    if exclude_columns is None:
        # By default, exclude some known columns + target_col
        exclude_columns = ['Hospital', 'ICC', 'LVSI_extent', 'pathology', target_col]
    else:
        # Ensure the target_col is also in exclude_columns if not already
        if target_col not in exclude_columns:
            exclude_columns.append(target_col)

    # All numeric columns (potential features) excluding any we definitely do not want
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_candidates = [c for c in numeric_cols if c not in exclude_columns]

    # ----------------------------------------------------------------------
    # 2) Reproducibility check (using subset where ICC == 1)
    #    - Evaluate correlation between R1_ and R2_ versions of each feature
    #    - Keep only those with correlation >= reproducibility_thresh
    # ----------------------------------------------------------------------
    reproducibility_data = df[df['ICC'] == 1]  # subset where ICC == 1
    reproducible_features = []

    # Imaging sequences to check
    sequences = ['ADC', 'CE60', 'CE120', 'FOCUS', 'pre', 'T2']

    for seq in sequences:
        # R1 columns for this sequence
        seq_r1 = reproducibility_data.filter(regex=f"^{seq}_R1_")
        # R2 columns for this sequence
        seq_r2 = reproducibility_data.filter(regex=f"^{seq}_R2_")

        # Identify the "base names" after removing R1_ or R2_
        r1_feature_names = [col.replace(f"{seq}_R1_", '') for col in seq_r1.columns]
        r2_feature_names = [col.replace(f"{seq}_R2_", '') for col in seq_r2.columns]
        common_prefixes = set(r1_feature_names).intersection(r2_feature_names)

        # Build aligned R1, R2 arrays
        aligned_r1_features = []
        aligned_r2_features = []
        prefix_list = []  # keep track of which prefixes line up

        for prefix in sorted(common_prefixes):
            col_r1 = f"{seq}_R1_{prefix}"
            col_r2 = f"{seq}_R2_{prefix}"
            # Make sure these columns exist
            if col_r1 in seq_r1.columns and col_r2 in seq_r2.columns:
                aligned_r1_features.append(seq_r1[col_r1].values)
                aligned_r2_features.append(seq_r2[col_r2].values)
                prefix_list.append(prefix)

        if len(prefix_list) == 0:
            continue  # no matching features for this sequence

        aligned_r1_features = np.column_stack(aligned_r1_features)  # shape: (n_samples, n_common_features)
        aligned_r2_features = np.column_stack(aligned_r2_features)

        # Calculate reproducibility feature-by-feature
        for i, prefix in enumerate(prefix_list):
            r1_vals = aligned_r1_features[:, i]
            r2_vals = aligned_r2_features[:, i]
            if len(r1_vals) > 0 and len(r2_vals) > 0:
                corr, _ = spearmanr(r1_vals, r2_vals)
                if abs(corr) >= reproducibility_thresh:
                    # We'll keep the R1 name in the final set. 
                    # (You could also keep R2 or both, but typically you'd pick one.)
                    feature_name_r1 = f"{seq}_R1_{prefix}"
                    # Only add it if it's among the numeric feature candidates
                    if feature_name_r1 in feature_candidates:
                        reproducible_features.append(feature_name_r1)

    reproducible_features = list(set(reproducible_features))  # unique

    # ----------------------------------------------------------------------
    # 3) Among reproducible features, remove those that are highly correlated 
    #    with one another (Spearman correlation across ALL data).
    # ----------------------------------------------------------------------
    if len(reproducible_features) == 0:
        # If nothing is reproducible, return the original DataFrame 
        # with just the exclude_columns + target_col
        return df[exclude_columns].copy()

    # Subset DataFrame to reproducible features for correlation check
    reproducible_df = df[reproducible_features].copy()

    # Compute Spearman correlation across these features (all rows)
    corr_matrix = reproducible_df.corr(method='spearman').abs()
    # Zero out the diagonal
    np.fill_diagonal(corr_matrix.values, 0)

    final_features = []
    remaining_cols = set(corr_matrix.columns)

    while len(remaining_cols) > 0:
        # Pick one feature from the remaining set
        current_feature = list(remaining_cols)[0]
        final_features.append(current_feature)

        # Identify any features that are highly correlated with `current_feature`
        correlated = corr_matrix.loc[current_feature][corr_matrix.loc[current_feature] > correlation_thresh].index
        
        # Remove them from the pool
        to_drop = set(list(correlated) + [current_feature])
        remaining_cols = remaining_cols.difference(to_drop)

    # ----------------------------------------------------------------------
    # Build the final set of columns to keep
    cols_to_keep = list(final_features) + list(set(exclude_columns)) + list(['age', 'CA125'])


    # Keep them in the DataFrame if they actually exist
    cols_to_keep = [c for c in cols_to_keep if c in df.columns]

    return df[cols_to_keep].copy()
