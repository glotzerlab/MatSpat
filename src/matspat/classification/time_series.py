"""Time-series classifiers for spatial sequences (QUANT, MultiROCKET + Hydra).

These are alternatives to the class-dependent HMMs in :mod:`hmm`. Both are
designed for small training sets: they transform each sequence with a fixed
(non-learned) feature bank and fit a simple classifier on top, so there are
few trainable parameters.

- QUANT (Dempster et al. 2024): quantiles over dyadic intervals of each
  sequence and its derivatives, classified with extremely randomised trees.
- MultiROCKET + Hydra (Tan et al. 2022; Dempster et al. 2023): random
  convolutional kernels, pooled into features, classified with a
  cross-validated ridge classifier.

Requires the optional ``timeseries`` extra (``pip install "matspat[timeseries]"``),
which installs ``aeon`` and ``torch``.

Public API:
    classify_time_series   Train one classifier, classify test data, compute metrics.
    TimeSeriesResult       Dataclass container for classification outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix

from matspat.config import DEFAULT_CI_ALPHA, DEFAULT_N_BOOTSTRAPS, DEFAULT_RANDOM_STATE

from .hmm import _compute_metrics

METHODS = ("quant", "multirocket-hydra")

# MultiROCKET's dilations need at least this many quantile bins
MIN_LENGTH_MULTIROCKET = 10

# Probabilities are clipped before converting to log-odds
_PROB_EPS = 1e-6


@dataclass
class TimeSeriesResult:
    """Container for time-series classification outputs and evaluation metrics.

    ``scores`` holds one decision score per test sample; positive values
    favour ``classes[1]``. For QUANT it is the log-odds of the predicted
    probability, for MultiROCKET + Hydra the ridge decision function.
    """

    predicted_labels: dict[str, int]
    scores: dict[str, float]
    model: Any
    classes: np.ndarray
    method: str
    metrics: dict[str, float] | None = None
    confusion_matrix: np.ndarray | None = None
    confidence_intervals: dict[str, tuple[float, float]] = field(default_factory=dict)


class _MultiRocketHydra:
    """MultiROCKET and Hydra features, standardised, then a ridge classifier."""

    def __init__(self, random_state: int) -> None:
        from aeon.transformations.collection.convolution_based import (
            HydraTransformer,
            MultiRocket,
        )
        from sklearn.linear_model import RidgeClassifierCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        self.multirocket = MultiRocket(random_state=random_state)
        self.hydra = HydraTransformer(random_state=random_state)
        self.classifier = make_pipeline(
            StandardScaler(), RidgeClassifierCV(alphas=np.logspace(-3, 3, 13))
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> _MultiRocketHydra:
        F = np.hstack(
            [np.asarray(self.multirocket.fit_transform(X)), np.asarray(self.hydra.fit_transform(X))]
        )
        self.classifier.fit(F, y)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        F = np.hstack(
            [np.asarray(self.multirocket.transform(X)), np.asarray(self.hydra.transform(X))]
        )
        scores: np.ndarray = np.asarray(self.classifier.decision_function(F))
        return scores


def _to_panel(sequences: dict[str, np.ndarray], keys: list[str]) -> np.ndarray:
    """Stack ``(T, D)`` sequences into aeon's ``(n_samples, D, T)`` layout."""
    arrs = [sequences[k] if sequences[k].ndim == 2 else sequences[k].reshape(-1, 1) for k in keys]
    if len({a.shape for a in arrs}) > 1:
        raise ValueError("All sequences must have the same shape (T, D)")
    panel: np.ndarray = np.stack([a.T for a in arrs]).astype(np.float64)
    return panel


def classify_time_series(
    train_sequences: dict[str, np.ndarray],
    train_labels: dict[str, int],
    test_sequences: dict[str, np.ndarray],
    *,
    test_labels: dict[str, int] | None = None,
    method: str = "quant",
    random_state: int = DEFAULT_RANDOM_STATE,
    n_bootstraps: int = DEFAULT_N_BOOTSTRAPS,
    ci_alpha: float = DEFAULT_CI_ALPHA,
) -> TimeSeriesResult:
    """Train a QUANT or MultiROCKET + Hydra classifier and classify test sequences.

    Unlike :func:`hmm.classify`, which fits one generative model per class,
    this fits a single discriminative classifier on all training sequences.
    Missing values are not allowed; fill them first (e.g. with
    ``build_sequences``, which zero-fills).

    Args:
        train_sequences: Dict mapping sample key → array ``(T, D)``.
        train_labels: Dict mapping sample key → integer class label (0 or 1).
        test_sequences: Dict mapping sample key → array ``(T, D)``.
        test_labels: If provided, computes evaluation metrics with bootstrap CIs.
        method: ``"quant"`` or ``"multirocket-hydra"``.
        random_state: Seed for the random feature banks and trees.
        n_bootstraps: Number of bootstrap samples for CIs.
        ci_alpha: Significance level for CIs (0.05 → 95% CI).

    Returns:
        :class:`TimeSeriesResult` with predictions, scores, the fitted model,
        and optional metrics.

    Raises:
        ValueError: If labels are not binary, *method* is unknown, sequences
            differ in shape or contain NaN, or sequences are too short for
            MultiROCKET.
        ImportError: If ``aeon`` (and, for QUANT, ``torch``) is not installed.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}; got {method!r}")
    classes = np.unique(list(train_labels.values()))
    if len(classes) != 2:
        raise ValueError(f"Only binary classification supported; found classes {classes}")

    train_keys = list(train_sequences.keys())
    test_keys = list(test_sequences.keys())
    X_train = _to_panel(train_sequences, train_keys)
    X_test = _to_panel(test_sequences, test_keys)
    if X_train.shape[1:] != X_test.shape[1:]:
        raise ValueError("Train and test sequences must have the same shape (T, D)")
    if np.isnan(X_train).any() or np.isnan(X_test).any():
        raise ValueError("Sequences contain NaN; fill missing values first")
    if method == "multirocket-hydra" and X_train.shape[2] < MIN_LENGTH_MULTIROCKET:
        raise ValueError(
            f"MultiROCKET needs sequences of length >= {MIN_LENGTH_MULTIROCKET}; "
            f"got {X_train.shape[2]}. Use more quantile bins or method='quant'."
        )
    # Binary target: 1 for classes[1], 0 for classes[0]
    y_train = np.array([int(train_labels[k] == classes[1]) for k in train_keys])

    try:
        if method == "quant":
            from aeon.classification.interval_based import QUANTClassifier

            model: Any = QUANTClassifier(random_state=random_state).fit(X_train, y_train)
            prob = np.clip(model.predict_proba(X_test)[:, 1], _PROB_EPS, 1 - _PROB_EPS)
            raw = np.log(prob / (1 - prob))
        else:
            model = _MultiRocketHydra(random_state).fit(X_train, y_train)
            raw = model.decision_function(X_test)
    except ModuleNotFoundError as e:
        raise ImportError(
            f"{method} requires the optional 'timeseries' extra: "
            'pip install "matspat[timeseries]"'
        ) from e

    scores = {k: float(s) for k, s in zip(test_keys, raw)}
    preds = {k: int(classes[1] if s > 0 else classes[0]) for k, s in scores.items()}
    result = TimeSeriesResult(
        predicted_labels=preds,
        scores=scores,
        model=model,
        classes=classes,
        method=method,
    )

    if test_labels is not None:
        y_true = np.array([test_labels[k] for k in test_keys])
        y_pred = np.array([preds[k] for k in test_keys])
        metrics, ci = _compute_metrics(
            y_true,
            y_pred,
            raw,
            pos_label=int(classes[1]),
            n_bootstraps=n_bootstraps,
            ci_alpha=ci_alpha,
        )
        result.metrics = metrics
        result.confusion_matrix = confusion_matrix(y_true, y_pred)
        result.confidence_intervals = ci

    return result
