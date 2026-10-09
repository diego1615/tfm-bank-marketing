"""Feature availability, chronological partitions and decision metrics."""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, precision_score, recall_score, f1_score

RAW_FEATURES = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'pdays', 'previous', 'poutcome']
FORBIDDEN = ['duration', 'campaign', 'contact', 'month', 'day_of_week', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed', 'y', 'source_row']
NUMERIC = ['age', 'pdays', 'previous', 'previously_contacted']
CATEGORICAL = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'poutcome']
SEED = 1615

class BankFeatures(BaseEstimator, TransformerMixin):
    """Whitelist available client/history variables; encode 999 as no contact."""
    def fit(self, X, y=None):
        missing = set(RAW_FEATURES) - set(X.columns)
        if missing:
            raise ValueError(f'Missing client/history features: {sorted(missing)}')
        self.n_features_in_ = len(X.columns)
        return self

    def transform(self, X):
        missing = set(RAW_FEATURES) - set(X.columns)
        if missing:
            raise ValueError(f'Missing client/history features: {sorted(missing)}')
        result = X[RAW_FEATURES].copy()
        result['previously_contacted'] = (result['pdays'] != 999).astype(int)
        result['pdays'] = result['pdays'].replace(999, np.nan)
        return result

    def get_feature_names_out(self, input_features=None):
        return np.asarray(RAW_FEATURES + ['previously_contacted'], dtype=object)

def preprocessing():
    return ColumnTransformer([
        ('numeric', Pipeline([('impute', SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)), ('scale', StandardScaler())]), NUMERIC),
        ('categorical', Pipeline([('impute', SimpleImputer(strategy='most_frequent')), ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), CATEGORICAL)
    ], remainder='drop')

def pipeline(estimator):
    return Pipeline([('features', BankFeatures()), ('preprocess', preprocessing()), ('model', estimator)])

def chronological_splits(n):
    """60% train, 10% selection, 10% calibration/policy, 20% untouched test."""
    if n < 100:
        raise ValueError('At least 100 observations are required.')
    a, b, c = int(.60 * n), int(.70 * n), int(.80 * n)
    return {'train': np.arange(0, a), 'selection': np.arange(a, b), 'calibration': np.arange(b, c), 'test': np.arange(c, n)}

def top_k_mask(probabilities, fraction=.10):
    """Stable tie breaking by source order; exactly ceil(fraction*n) contacts."""
    p = np.asarray(probabilities)
    if len(p) == 0 or not 0 <= fraction <= 1:
        raise ValueError('Non-empty probabilities and fraction in [0, 1] required.')
    count = math.ceil(fraction * len(p))
    mask = np.zeros(len(p), dtype=bool)
    mask[np.argsort(-p, kind='stable')[:count]] = True
    return mask

def proxy_profit(y, contacted, benefit=100., cost=5.):
    """Counterfactual-free accounting proxy, NOT causal/incremental ROI."""
    y, contacted = np.asarray(y), np.asarray(contacted, dtype=bool)
    return float(benefit * y[contacted].sum() - cost * contacted.sum())

def evaluate(y, p, threshold=.5):
    y, p = np.asarray(y), np.asarray(p)
    pred = p >= threshold
    selected = top_k_mask(p)
    precision_top = float(y[selected].mean())
    prevalence = float(y.mean())
    return {'n': int(len(y)), 'prevalence': prevalence, 'roc_auc': float(roc_auc_score(y, p)),
            'average_precision': float(average_precision_score(y, p)),
            'brier': float(brier_score_loss(y, p)), 'precision': float(precision_score(y, pred, zero_division=0)),
            'recall': float(recall_score(y, pred, zero_division=0)), 'f1': float(f1_score(y, pred, zero_division=0)),
            'threshold': float(threshold), 'precision_at_10pct': precision_top,
            'recall_at_10pct': float(y[selected].sum()/max(y.sum(),1)),
            'lift_at_10pct': precision_top/prevalence if prevalence else 0.}
