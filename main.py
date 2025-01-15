#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 15:17:08 2025

@author: daniel

Orchestrates:
1) Data loading
2) Two ways of splitting the data (random 80/20 and by Hospital)
3) Three modeling approaches:
   A) Bayesian logistic regression with spike-and-slab (PyMC)
   B) XGBoost
   C) XGBoost with features selected by the Bayesian model
4) K-fold majority down-sampling in the training stage
5) Produces six accuracy results (3 models × 2 splitting strategies)
   using an ensemble across folds for *all* models.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

from feature_selection import combat_location_scale_radiomics, filter_features
from balanced_kfold import BalancedKFold
from balanced_kfold_no_test import BalancedKFoldNoTest
from bayes import (
    train_bayes_single_pass,
    predict_bayes_logistic_spike_slab,
    bayesian_logistic_spike_slab,
    combine_bayes_traces,
    get_significant_features_from_bayes
)
from xgboost_model import (
    train_xgboost,
    train_xgboost_single_pass,
    predict_xgboost,
    predict_proba_xgboost
)


# ---------------------------------------------------------------------
# Function to Scale DataFrame
# ---------------------------------------------------------------------
def scale_dataframe(df, exclude_columns=None):
    """
    Scales all numeric columns in a DataFrame except those explicitly excluded.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing data to scale.
    exclude_columns : list of str, optional
        List of column names to exclude from scaling. Default is None.

    Returns
    -------
    df_scaled : pd.DataFrame
        A new DataFrame with scaled values for the included columns.
    scaler : StandardScaler
        The scaler used to transform the data (can be reused for test sets).
    """
    if exclude_columns is None:
        exclude_columns = []

    # Identify columns to scale
    columns_to_scale = [col for col in df.columns if col not in exclude_columns]

    # Initialize scaler
    scaler = StandardScaler()

    # Scale the selected columns
    scaled_values = scaler.fit_transform(df[columns_to_scale])

    # Create a copy of the DataFrame to retain structure
    df_scaled = df.copy()

    # Reassign the scaled values to the appropriate columns
    df_scaled[columns_to_scale] = scaled_values

    return df_scaled, scaler


# -------------------------------------------------------------------------
# Metrics Helper: returns accuracy, AUC, sensitivity, specificity, balanced accuracy
# -------------------------------------------------------------------------
def compute_metrics(y_true, y_pred, y_proba=None):
    """
    Compute accuracy, AUC, sensitivity, specificity, and balanced accuracy.
    y_pred: predicted labels (0/1)
    y_proba: predicted probabilities for the positive class (optional for AUC).
    """
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # AUC (requires probability). If y_proba is None, fallback to NaN.
    if y_proba is not None:
        try:
            auc = roc_auc_score(y_true, y_proba)
        except:
            auc = np.nan
    else:
        auc = np.nan

    # Sensitivity/Specificity from confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    # Balanced Accuracy = (Sensitivity + Specificity) / 2
    if not np.isnan(sensitivity) and not np.isnan(specificity):
        balanced_acc = (sensitivity + specificity) / 2
    else:
        balanced_acc = np.nan

    return {
        "accuracy": accuracy,
        "auc": auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": balanced_acc
    }

# -------------------------------------------------------------------------
# Helper: K-fold training with majority down-sampling
# Each fold uses ALL minority examples but a different fold of the majority
# until all majority samples have been used.
# -------------------------------------------------------------------------

def train_kfold_balanced_bayes(X, y, n_splits=5, shuffle=True, random_state=42):
    """
    Train Bayesian logistic regression on K folds (with BalancedKFold)
    and return a list of posterior traces, one per fold.
    """
    bkf = BalancedKFoldNoTest(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    fold_traces = []
    
    for fold_idx, (train_idx, _) in enumerate(bkf.split(X, y)):
        X_train_fold = X.loc[train_idx]
        y_train_fold = y.loc[train_idx]

        # Optionally, we can define a validation set. 
        # In BalancedKFold, 'test_idx' is what we might treat as a "fold hold-out." 
        # For purely training, we just train on X_train_fold, y_train_fold:
        _, trace = bayesian_logistic_spike_slab(X_train_fold, y_train_fold)
        fold_traces.append(trace)
    
    return fold_traces


def train_kfold_balanced_xgboost(X, y, n_splits=5, shuffle=True, random_state=42, **kwargs):
    """
    Train XGBoost on K folds (BalancedKFold).
    Return a list of trained XGBoost models, one per fold.
    """
    bkf = BalancedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    models = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(bkf.split(X, y)):
        # Balanced training set
        X_train_fold = X.loc[train_idx]
        y_train_fold = y.loc[train_idx]
        
        # If you want a small hold-out for early stopping, you can do so 
        # from the training fold itself (like a second-level split), 
        # or you can treat the test_idx as your validation set. 
        # For a direct approach, let's treat test_idx as validation:
        X_val_fold = X.loc[test_idx]
        y_val_fold = y.loc[test_idx]

        model = train_xgboost(X_train_fold, y_train_fold, X_val_fold, y_val_fold, **kwargs)
        models.append(model)
    
    return models

# -------------------------------------------------------------------------
# Ensemble predictions
# -------------------------------------------------------------------------


def predict_bayes_ensemble(traces, X_test, feature_cols):
    """
    Ensemble majority vote from multiple Bayesian traces.
    We'll approximate predictions by using average coefficient 
    from each posterior trace and returning 0/1. 
    """
    preds_list = []
    proba_list = []  # We'll keep probabilities for AUC if desired

    X_ = X_test[feature_cols].values

    import numpy as np

    for trace in traces:
        # Posterior samples: inclusion, beta, intercept
        inclusion_samps = trace.posterior["inclusion"].stack(draws=("chain", "draw")).values
        beta_samps = trace.posterior["beta"].stack(draws=("chain", "draw")).values
        intercept_samps = trace.posterior["intercept"].stack(draws=("chain", "draw")).values

        coeff_mean = (inclusion_samps * beta_samps).mean(axis=1)
        intercept_mean = intercept_samps.mean()

        logits = intercept_mean + X_.dot(coeff_mean)
        p = 1 / (1 + np.exp(-logits))  # probability for y=1
        preds = (p > 0.5).astype(int)

        preds_list.append(preds)
        proba_list.append(p)

    # Majority vote across folds
    preds_array = np.array(preds_list)  # shape = (n_folds, n_samples)
    preds_array = preds_array.T         # shape = (n_samples, n_folds)
    final_preds = [np.bincount(row).argmax() for row in preds_array]

    # Average probability across folds (for AUC)
    proba_array = np.array(proba_list).T  # shape = (n_samples, n_folds)
    final_proba = proba_array.mean(axis=1)  # average across folds

    return np.array(final_preds), final_proba


def predict_xgboost_ensemble(models, X_test):
    """
    Ensemble majority vote across multiple XGBoost models.
    Also average predicted probabilities for AUC.
    """
    import numpy as np

    preds_list = []
    proba_list = []

    for m in models:
        preds_list.append(m.predict(X_test))
        proba_list.append(m.predict_proba(X_test)[:, 1])  # prob of class=1

    # shape => (n_folds, n_samples)
    preds_array = np.array(preds_list).T
    final_preds = [np.bincount(row).argmax() for row in preds_array]

    proba_array = np.array(proba_list).T  # shape = (n_samples, n_folds)
    final_proba = proba_array.mean(axis=1)

    return np.array(final_preds), final_proba

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

def plot_all_confusion_matrices(all_predictions, output_dir=None):
    for experiment_name, data in all_predictions.items():
        y_true = data["y_true"]
        y_pred = data["y_pred"]
        
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])

        fig, ax = plt.subplots(figsize=(5,4))
        disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False)
        ax.set_title(f"Confusion Matrix - {experiment_name}")
        
        if output_dir:
            plt.savefig(f"{output_dir}/confmat_{experiment_name}.png", dpi=150, bbox_inches='tight')
        else:
            plt.show()
        plt.close(fig)

def plot_feature_correlation(df, feature_list, title, output_dir=None):
    if len(feature_list) < 2:
        print(f"Not enough features to plot correlation for {title}")
        return

    df_sub = df[feature_list].copy()
    corr_matrix = df_sub.corr().abs()
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(corr_matrix, ax=ax, annot=False, cmap='coolwarm', vmin=0, vmax=1, square=True)
    ax.set_title(title)

    if output_dir:
        plt.savefig(f"{output_dir}/{title.replace(' ','_')}_corr.png", dpi=150, bbox_inches='tight')
    else:
        plt.show()
    plt.close(fig)


def main():
    # ---------------------------------------------------------------------
    # 1) Load your data
    # ---------------------------------------------------------------------
    np.random.seed(42)
    df = pd.read_excel('/media/daniel/b6eaf548-4cbb-4781-8be0-fea0091c0087/EMCA_Study/Scripts/all_radiomic_data_combined.xlsx')
    
    exclude_columns = ['PatientID', 'Hospital', 'ICC', 'LVSI_extent', 'pathology']
    df_scaled, scaler = scale_dataframe(df, exclude_columns)
    
    # ---------------------------------------------------------------------
    # 2) Filter features 
    # ---------------------------------------------------------------------
    # Apply ComBat
    # Identify the columns that are definitely NOT radiomic:
    metadata_cols = ["PatientID", "Hospital", "age", "CA125", "ICC", "pathology", "LVSI_extent"]
    radiomic_cols = [col for col in df.columns if col not in metadata_cols]

    df_corrected = combat_location_scale_radiomics(
        df=df_scaled,
        batch=df['Hospital'],
        radiomic_cols=radiomic_cols
    )
    
    # Apply Feature Filtering
    exclude_cols = ['Hospital', 'ICC', 'LVSI_extent', 'pathology']
    df_filtered = filter_features(df_corrected,
        'LVSI_extent',
        0.8,
        0.95,
        exclude_cols)
    feature_cols = [c for c in df_filtered.columns if c not in exclude_cols]

    # ---------------------------------------------------------------------
    # 3) Prepare for the experiments
    # ---------------------------------------------------------------------
    results = {}
    # We'll keep a dictionary for predictions, as well as
    # a dictionary for "used_features" if you want correlation plots.
    all_predictions = {}
    used_features = {}

    # ========== SPLIT A) Random 80/20 ========== 
    train_df_a, test_df_a = train_test_split(
        df_filtered, test_size=0.2, random_state=42, stratify=df_filtered['pathology']
    )

    X_train_a = train_df_a[feature_cols]
    y_train_a = train_df_a['pathology']
    X_test_a = test_df_a[feature_cols]
    y_test_a = test_df_a['pathology']

    # --------------------------------------------------------------------
    # MODEL A1) Bayesian logistic regression (Ensemble across folds)
    # --------------------------------------------------------------------
    bayes_traces_a1 = train_bayes_single_pass(X_train_a, y_train_a)
    preds_a1, proba_a1 = predict_bayes_logistic_spike_slab(bayes_traces_a1, X_test_a, feature_cols)
    metrics_a1 = compute_metrics(y_test_a, preds_a1, y_proba=proba_a1)
    results["RandomSplit_Bayes"] = metrics_a1
    all_predictions["RandomSplit_Bayes"] = {
        "y_true": y_test_a,
        "y_pred": preds_a1
    }
    used_features["RandomSplit_Bayes"] = feature_cols  # or the subset if you do spike-and-slab


    # --------------------------------------------------------------------
    # MODEL A2) XGBoost (all features, ensemble)
    # --------------------------------------------------------------------
    xgb_model_a2 = train_xgboost_single_pass(X_train_a, y_train_a)
    preds_a2 = predict_xgboost(xgb_model_a2, X_test_a)
    proba_a2 = predict_proba_xgboost(xgb_model_a2, X_test_a)
    metrics_a2 = compute_metrics(y_test_a, preds_a2, y_proba=proba_a2)
    results["RandomSplit_XGBoost"] = metrics_a2
    
    all_predictions["RandomSplit_XGBoost"] = {
        "y_true": y_test_a,
        "y_pred": preds_a2
    }
    used_features["RandomSplit_XGBoost"] = feature_cols 

    # --------------------------------------------------------------------
    # MODEL A3) XGBoost w/ Bayes-selected features
    # --------------------------------------------------------------------
    # We'll use the last fold’s trace (or any fold) to pick features
    # from the Bayesian model.  Typically you'd pick from a combined perspective
    # or a single fold. We pick from the last fold for simplicity.
    
    selected_bayes_features_a = get_significant_features_from_bayes(
        bayes_traces_a1, feature_cols, 0.5
        )
    if len(selected_bayes_features_a) == 0:
        selected_bayes_features_a = feature_cols

    X_train_a_bsel = X_train_a[selected_bayes_features_a]
    X_test_a_bsel = X_test_a[selected_bayes_features_a]

    xgb_model_a3 = train_xgboost_single_pass(X_train_a_bsel, y_train_a)
    preds_a3 = predict_xgboost(xgb_model_a3, X_test_a_bsel)
    proba_a3 = predict_proba_xgboost(xgb_model_a3, X_test_a_bsel)
    metrics_a3 = compute_metrics(y_test_a, preds_a3, y_proba=proba_a3)
    results["RandomSplit_XGBoost_BayesSelected"] = metrics_a3
    
    all_predictions["RandomSplit_XGBoost_BayesSelected"] = {
        "y_true": y_test_a,
        "y_pred": preds_a1
    }
    used_features["RandomSplit_XGBoost_BayesSelected"] = feature_cols 

    # ========== SPLIT B) Hospital-based ==========
    
    train_df_b = df_filtered[df_filtered['Hospital'] == 0]
    test_df_b = df_filtered[df_filtered['Hospital'] == 1]
    X_train_b = train_df_b[feature_cols]
    y_train_b = train_df_b['pathology']
    X_test_b = test_df_b[feature_cols]
    y_test_b = test_df_b['pathology']

    # --------------------------------------------------------------------
    # MODEL B1) Bayesian logistic regression
    # --------------------------------------------------------------------
    bayes_traces_b1 = train_bayes_single_pass(X_train_b, y_train_b)
    preds_b1, proba_b1 = predict_bayes_logistic_spike_slab(bayes_traces_b1, X_test_b, feature_cols)
    metrics_b1 = compute_metrics(y_test_b, preds_b1, y_proba=proba_b1)
    results["HospitalSplit_Bayes"] = metrics_b1
    
    all_predictions["HospitalSplit_Bayes"] = {
        "y_true": y_test_a,
        "y_pred": preds_a1
    }
    used_features["HospitalSplit_Bayes"] = feature_cols 

    # --------------------------------------------------------------------
    # MODEL B2) XGBoost (all features)
    # --------------------------------------------------------------------
    xgb_model_b2 = train_xgboost_single_pass(X_train_b, y_train_b)
    preds_b2 = predict_xgboost(xgb_model_b2, X_test_b)
    proba_b2 = predict_proba_xgboost(xgb_model_b2, X_test_b)
    metrics_b2 = compute_metrics(y_test_b, preds_b2, y_proba=proba_b2)
    results["HospitalSplit_XGBoost"] = metrics_b2
    
    all_predictions["HospitalSplit_XGBoost"] = {
        "y_true": y_test_b,
        "y_pred": preds_b2
    }
    used_features["HospitalSplit_XGBoost"] = feature_cols 

    # --------------------------------------------------------------------
    # MODEL B3) XGBoost w/ Bayes-selected features
    # --------------------------------------------------------------------
    selected_bayes_features_b = get_significant_features_from_bayes(
        bayes_traces_b1, feature_cols, 0.5
        )
    if len(selected_bayes_features_b) == 0:
        selected_bayes_features_b = feature_cols

    X_train_b_bsel = X_train_b[selected_bayes_features_b]
    X_test_b_bsel = X_test_b[selected_bayes_features_b]

    xgb_model_b3 = train_xgboost_single_pass(X_train_b_bsel, y_train_b)
    preds_b3 = predict_xgboost(xgb_model_b3, X_test_b_bsel)
    proba_b3 = predict_proba_xgboost(xgb_model_b3, X_test_b_bsel)
    metrics_b3 = compute_metrics(y_test_b, preds_b3, y_proba=proba_b3)
    results["HospitalSplit_XGBoost_BayesSelected"] = metrics_b3
    
    all_predictions["HospitalSplit_XGBoost_BayesSelected"] = {
        "y_true": y_test_a,
        "y_pred": preds_a1
    }
    used_features["HospitalSplit_XGBoost_BayesSelected"] = feature_cols 

    # ---------------------------------------------------------------------
    # 4) Print the 6 sets of metrics
    # ---------------------------------------------------------------------
    print("\nFINAL RESULTS (Accuracy, AUC, Sensitivity, Specificity, Balanced Accuracy):\n")
    for key, val in results.items():
        print(f"{key}:")
        print(f"  Accuracy         = {val['accuracy']:.4f}")
        print(f"  AUC              = {val['auc']:.4f}")
        print(f"  Sensitivity      = {val['sensitivity']:.4f}")
        print(f"  Specificity      = {val['specificity']:.4f}")
        print(f"  BalancedAccuracy = {val['balanced_accuracy']:.4f}\n")
        
        # output_dir = "/media/daniel/b6eaf548-4cbb-4781-8be0-fea0091c0087/EMCA_Study"
        # plot_all_confusion_matrices(all_predictions, output_dir=output_dir)
        
        # for experiment_name, feat_list in used_features.items():
        #     plot_feature_correlation(
        #         df_filtered, feat_list,
        #         title=f"{experiment_name} Features Correlation",
        #         output_dir=output_dir
        #     )


if __name__ == "__main__":
    main()
