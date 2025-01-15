#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 15:16:45 2025

@author: daniel

Implements XGBoost training and prediction, as well as a variant that uses
Bayesian model-selected features.
"""

import xgboost as xgb
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score


def train_xgboost(
    X_train: pd.DataFrame, 
    y_train: pd.Series,
    X_val: pd.DataFrame, 
    y_val: pd.Series,
    **kwargs
):
    """
    Train an XGBoost model on the training set and optionally evaluate on validation set.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training labels.
    X_val : pd.DataFrame
        Validation features.
    y_val : pd.Series
        Validation labels.
    kwargs : dict
        Extra parameters for xgb.XGBClassifier.

    Returns
    -------
    model : xgb.XGBClassifier
        Trained model.
    """
    model = xgb.XGBClassifier(eval_metric='logloss',
                              early_stopping_rounds=10,
                              **kwargs)
    model.fit(
        X_train, 
        y_train, 
        eval_set=[(X_val, y_val)]
    )
    return model

def train_xgboost_single_pass(
    X_train: pd.DataFrame, 
    y_train: pd.Series,
    X_val: pd.DataFrame = None, 
    y_val: pd.Series = None,
    scale_pos_weight=None,
    **kwargs
):
    """
    Train an XGBoost model on a single pass (no folds).
    If X_val, y_val are provided, we can do early stopping.
    We can pass scale_pos_weight to handle class imbalance.
    """
    import xgboost as xgb
    
    # 1) Apply SMOTE
    sm = SMOTE(random_state=42)
    X_resampled, y_resampled = sm.fit_resample(X_train, y_train)
    
    model = xgb.XGBClassifier(
        eval_metric='logloss', 
        **kwargs
    )

    if X_val is not None and y_val is not None:
        model.fit(X_resampled, y_resampled, eval_set=[(X_val, y_val)], 
                  early_stopping_rounds=10, verbose=False)
    else:
        model.fit(X_resampled, y_resampled)

    return model


def predict_xgboost(model, X_test: pd.DataFrame) -> np.ndarray:
    """
    Generate predictions for X_test using a trained XGBoost model.
    
    Parameters
    ----------
    model : xgb.XGBClassifier
        A trained XGBoost classifier.
    X_test : pd.DataFrame
        Test features.

    Returns
    -------
    predictions : np.ndarray
        Model predictions (0 or 1).
    """
    preds = model.predict(X_test)
    return preds


def predict_proba_xgboost(model, X_test: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X_test)[:, 1]

def evaluate_accuracy(model, X_test, y_test) -> float:
    """
    Evaluates the accuracy of a trained XGBoost model on a test set.

    Parameters
    ----------
    model : xgb.XGBClassifier
        Trained model.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        True labels.

    Returns
    -------
    accuracy : float
        Classification accuracy.
    """
    preds = predict_xgboost(model, X_test)
    return accuracy_score(y_test, preds)
