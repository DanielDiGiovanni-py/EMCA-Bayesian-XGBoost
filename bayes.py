#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 15:16:21 2025

@author: daniel

Implements a Bayesian logistic regression with spike-and-slab priors using PyMC.
"""

import pymc as pm
import arviz as az
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd

def bayesian_logistic_spike_slab(X: pd.DataFrame, y: pd.Series, 
                                 draws=1000, tune=1000, target_accept=0.99):
    """
    Fits a Bayesian logistic regression with spike-and-slab priors.
    Returns the PyMC model and its inference data (posterior).

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Binary target.
    draws : int
        Number of draws for MCMC sampling.
    tune : int
        Number of tuning steps.
    target_accept : float
        Target acceptance probability.

    Returns
    -------
    model : pm.Model
        The PyMC model object.
    trace : arviz.InferenceData
        Posterior samples from MCMC.
    """

    # Convert to numpy arrays
    X_ = X.values
    y_ = y.values
    n_features = X_.shape[1]

    with pm.Model() as model:
        
        # Inclusion indicator
        inclusion = pm.Bernoulli("inclusion", p=0.5, shape=n_features)
        
        # Slab prior for beta
        beta = pm.Normal("beta", mu=0.0, sigma=1.0, shape=n_features)

        # Effective coefficients = inclusion * beta
        coeff = pm.Deterministic("coeff", inclusion * beta)

        # Intercept
        intercept = pm.Normal("intercept", mu=0.0, sigma=1.0)

        # Logistic regression
        logits = intercept + pm.math.dot(X_, coeff)
        p = pm.invlogit(logits)

        # Likelihood
        pm.Bernoulli("likelihood", p=p, observed=y_)

        # Sample
        trace = pm.sample(
            draws=draws,
            tune=tune,
            # target_accept=target_accept,
            chains=2,
            random_seed=42,
            compute_convergence_checks=False
        )

    return model, trace


def bayesian_logistic_spike_slab_vi_relaxed(X, y, vi_iterations=int(1e+06)):
    """
    Bayesian logistic regression with a relaxed spike-and-slab prior using VI.

    Parameters
    ----------
    X : pd.DataFrame
        Training features.
    y : pd.Series
        Binary target.
    vi_iterations : int
        Number of optimization steps for VI.

    Returns
    -------
    model : pm.Model
        The PyMC model object.
    approx : pm.Approximation
        The variational approximation (e.g., MeanField).
    """
    X_ = X.values
    y_ = y.values
    n_features = X_.shape[1]

    with pm.Model() as model:
        # Relaxed inclusion variable using a Beta distribution
        inclusion = pm.Beta("inclusion", alpha=1.0, beta=1.0, shape=n_features)

        # Slab prior for beta coefficients
        beta = pm.Normal("beta", mu=0.0, sigma=1.0, shape=n_features)

        # Effective coefficients are scaled by inclusion
        coeff = pm.Deterministic("coeff", inclusion * beta)

        # Intercept
        intercept = pm.Normal("intercept", mu=0.0, sigma=1.0)

        # Logistic regression likelihood
        logits = intercept + pm.math.dot(X_, coeff)
        pm.Bernoulli("likelihood", p=pm.math.sigmoid(logits), observed=y_)

        # Variational inference
        approx = pm.fit(vi_iterations, method="advi")

    return model, approx

import pandas as pd

def bayesian_logistic_horseshoe(X: pd.DataFrame, y: pd.Series, 
                                draws=1000, tune=1000, target_accept=0.99):
    """
    Fits a Bayesian logistic regression with a horseshoe prior.
    Returns the PyMC model and its inference data (posterior).

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Binary target.
    draws : int
        Number of draws for MCMC sampling.
    tune : int
        Number of tuning steps.
    target_accept : float
        Target acceptance probability.

    Returns
    -------
    model : pm.Model
        The PyMC model object.
    trace : arviz.InferenceData
        Posterior samples from MCMC.
    """

    # Convert to numpy arrays
    X_ = X.values
    y_ = y.values
    n_features = X_.shape[1]

    with pm.Model() as model:
        
        # Global shrinkage parameter
        tau = pm.HalfCauchy("tau", beta=1.0)

        # Local shrinkage parameters
        lambda_ = pm.HalfCauchy("lambda", beta=1.0, shape=n_features)

        # Scale for each feature
        sigma = pm.Deterministic("sigma", tau * lambda_)

        # Coefficients with horseshoe prior
        beta = pm.Normal("beta", mu=0.0, sigma=sigma, shape=n_features)

        # Intercept
        intercept = pm.Normal("intercept", mu=0.0, sigma=1.0)

        # Logistic regression
        logits = intercept + pm.math.dot(X_, beta)
        p = pm.math.sigmoid(logits)

        # Likelihood
        pm.Bernoulli("likelihood", p=p, observed=y_)

        # Sample
        trace = pm.sample(
            draws=draws,
            tune=tune,
            target_accept=target_accept,
            chains=2,
            random_seed=42
        )

    return model, trace


def train_bayes_single_pass(X, y):
    """
    Train Bayesian logistic regression on a single pass of data (no folds).
    Returns the posterior trace.
    """
    
    trace = train_bayes_with_smote(X, y)
    return trace

def train_bayes_with_smote(X_train, y_train):
    """
    Train Bayesian logistic regression on a SMOTE-augmented dataset.
    """
    # 1) Apply SMOTE
    sm = SMOTE(random_state=42)
    X_resampled, y_resampled = sm.fit_resample(X_train, y_train)

    # 2) Train the Bayesian model on the oversampled dataset
    _, trace = bayesian_logistic_spike_slab_vi_relaxed(
        pd.DataFrame(X_resampled, columns=X_train.columns),
        pd.Series(y_resampled, name=y_train.name),
    )
    return trace



def stable_sigmoid(logits):
    """
    Numerically stable sigmoid function.
    """
    # Handle large positive logits
    return np.where(
        logits >= 0,
        1 / (1 + np.exp(-logits)),
        np.exp(logits) / (1 + np.exp(logits))
    )

def predict_bayes_logistic_spike_slab(trace, X_test, feature_names):
    """
    Predict from a single trace by averaging coefficients. 
    """

    X_ = X_test[feature_names].values
    posterior = trace.sample(10000)
    inclusion_samps = posterior.posterior["inclusion"].stack(draws=("chain", "draw")).values
    beta_samps = posterior.posterior["beta"].stack(draws=("chain", "draw")).values
    intercept_samps = posterior.posterior["intercept"].stack(draws=("chain", "draw")).values

    coeff_mean = (inclusion_samps * beta_samps).mean(axis=1)
    intercept_mean = intercept_samps.mean()

    logits = intercept_mean + X_.dot(coeff_mean)
    p = stable_sigmoid(logits)
    preds = (p > 0.5).astype(int)
    return preds, p

def combine_bayes_traces(traces):
    """
    Concatenates a list of ArviZ InferenceData objects along the 'chain' dimension.
    This effectively merges all posterior samples into one big multi-chain trace.
    """
    # By default, az.concat stacks them on a new chain dimension
    # so if each trace has 2 chains, 5 folds => 10 chains total
    combined_trace = az.concat(traces, dim="chain", copy=True)
    return combined_trace


def get_significant_features_from_bayes(trace, feature_names, inclusion_prob_threshold=0.9):
    """
    Given a PyMC trace from a spike-and-slab model, returns the feature names
    whose inclusion indicator is > inclusion_prob_threshold.

    Parameters
    ----------
    trace : arviz.InferenceData
        Posterior samples for the spike-and-slab model.
    feature_names : List[str]
        Names of features in order.
    inclusion_prob_threshold : float
        Probability threshold for deciding that a feature is 'included'.

    Returns
    -------
    List[str]
        The names of the features that appear to be included in the model 
        with probability above the threshold.
    """
    # The 'inclusion' variable is shape=(n_features,).
    # Posterior dimension is (chain, draw, feature).
    # We'll stack chain & draw for a single dimension of "samples".
    posterior = trace.sample(10000)
    inclusion_samples = posterior.posterior["inclusion"].stack(draws=("chain", "draw")).values
    inclusion_mean = inclusion_samples.mean(axis=1)  # average across draws
    
    # Identify features with mean inclusion prob > threshold
    selected_indices = np.where(inclusion_mean > inclusion_prob_threshold)[0]
    selected_features = [feature_names[i] for i in selected_indices] 
    
    return selected_features
