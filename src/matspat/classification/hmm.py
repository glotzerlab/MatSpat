"""Hidden Markov Model training and classification on spatial sequences.

Public API:
    train_class_hmm   Fit a GaussianHMM with multiple random restarts.
    score_sequence     Log-likelihood of one sequence under one model.
    classify           Train per-class HMMs, classify test data, compute metrics.
    ClassificationResult  Dataclass container for classification outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from hmmlearn import hmm as hmmlearn_hmm
from scipy.special import logsumexp
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class ClassificationResult:
    """Container for HMM classification outputs and evaluation metrics."""

    predicted_labels: dict[str, int]
    log_likelihoods: dict[str, dict[int, float]]
    models: dict[int, hmmlearn_hmm.GaussianHMM]
    classes: np.ndarray
    metrics: dict[str, float] | None = None
    confusion_matrix: np.ndarray | None = None
    confidence_intervals: dict[str, tuple[float, float]] = field(default_factory=dict)


def train_class_hmm(
    sequences: list[np.ndarray],
    *,
    n_states: int = 3,
    covariance_type: str = "full",
    max_iter: int = 100,
    n_replicates: int = 3,
    tol: float = 1e-4,
) -> hmmlearn_hmm.GaussianHMM:
    """Fit a GaussianHMM to a list of sequences, returning the best replicate.

    Concatenates all sequences and fits with *hmmlearn*, repeating with
    different random seeds and keeping the model with the highest
    log-likelihood.

    Args:
        sequences: List of arrays, each shape ``(T_i, D)``.
        n_states: Number of hidden states.
        covariance_type: One of ``"full"``, ``"diag"``, ``"tied"``,
            ``"spherical"``.
        max_iter: Maximum EM iterations per replicate.
        n_replicates: Number of random restarts.
        tol: EM convergence tolerance.

    Returns:
        Fitted ``GaussianHMM`` with the highest training log-likelihood.

    Raises:
        ValueError: If *sequences* is empty.
    """
    if not sequences:
        raise ValueError("sequences list is empty")

    xs = [s if s.ndim == 2 else s.reshape(-1, 1) for s in sequences]
    X = np.concatenate(xs, axis=0)
    lengths = [s.shape[0] for s in xs]

    best_ll: float = -np.inf
    best_model: hmmlearn_hmm.GaussianHMM | None = None

    for seed in range(n_replicates):
        model = hmmlearn_hmm.GaussianHMM(
            n_components=n_states,
            covariance_type=covariance_type,
            n_iter=max_iter,
            tol=tol,
            random_state=seed,
        )
        try:
            model.fit(X, lengths)
            ll = float(model.score(X, lengths))
        except Exception:
            ll = -np.inf
        if ll > best_ll:
            best_ll = ll
            best_model = model

    assert best_model is not None  # guaranteed by n_replicates >= 1
    return best_model


def score_sequence(
    sequence: np.ndarray,
    model: hmmlearn_hmm.GaussianHMM,
) -> float:
    """Log-likelihood of a single sequence under a fitted HMM.

    Args:
        sequence: Array of shape ``(T, D)``.
        model: Fitted ``GaussianHMM``.

    Returns:
        Log-likelihood (float).
    """
    X = sequence if sequence.ndim == 2 else sequence.reshape(-1, 1)
    return float(model.score(X))


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bootstraps: int = 1000,
    ci_alpha: float = 0.05,
) -> tuple[dict[str, float], dict[str, tuple[float, float]]]:
    """Compute classification metrics and bootstrap confidence intervals.

    Returns:
        Tuple of (metrics_dict, ci_dict).
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred)),
        "sensitivity": float(recall_score(y_true, y_pred)),
        "auroc": float(roc_auc_score(y_true, y_prob)),
    }

    rng = np.random.default_rng(seed=0)
    idxs = np.arange(len(y_true))
    boot: dict[str, list[float]] = {k: [] for k in metrics}

    for _ in range(n_bootstraps):
        bs = rng.choice(idxs, size=len(idxs), replace=True)
        yt, yp, ypr = y_true[bs], y_pred[bs], y_prob[bs]
        if np.unique(yt).size < 2:
            continue
        boot["accuracy"].append(float(accuracy_score(yt, yp)))
        boot["balanced_accuracy"].append(float(balanced_accuracy_score(yt, yp)))
        boot["f1_score"].append(float(f1_score(yt, yp)))
        boot["sensitivity"].append(float(recall_score(yt, yp)))
        boot["auroc"].append(float(roc_auc_score(yt, ypr)))

    lo = 100 * (ci_alpha / 2)
    hi = 100 * (1 - ci_alpha / 2)
    ci: dict[str, tuple[float, float]] = {}
    for k, samples in boot.items():
        if samples:
            ci[k] = (float(np.percentile(samples, lo)), float(np.percentile(samples, hi)))
        else:
            ci[k] = (np.nan, np.nan)

    return metrics, ci


def classify(
    train_sequences: dict[str, np.ndarray],
    train_labels: dict[str, int],
    test_sequences: dict[str, np.ndarray],
    *,
    test_labels: dict[str, int] | None = None,
    n_states: int | dict[int, int] = 3,
    covariance_type: str = "full",
    max_iter: int = 100,
    n_replicates: int = 3,
    tol: float = 1e-4,
    n_bootstraps: int = 1000,
    ci_alpha: float = 0.05,
) -> ClassificationResult:
    """Train class-dependent HMMs and classify test sequences.

    Trains one ``GaussianHMM`` per class on the training sequences, then
    scores each test sequence under both models.  Classification is by
    maximum log-likelihood.

    Args:
        train_sequences: Dict mapping sample key → array ``(T, D)``.
        train_labels: Dict mapping sample key → integer class label (0 or 1).
        test_sequences: Dict mapping sample key → array ``(T, D)``.
        test_labels: If provided, computes evaluation metrics with bootstrap CIs.
        n_states: Number of HMM states.  Either a single int (same for both
            classes) or a dict mapping class label → n_states.
        covariance_type: Covariance type for ``GaussianHMM``.
        max_iter: Max EM iterations.
        n_replicates: Random restarts per class model.
        tol: EM convergence tolerance.
        n_bootstraps: Number of bootstrap samples for CIs.
        ci_alpha: Significance level for CIs (0.05 → 95% CI).

    Returns:
        :class:`ClassificationResult` with predictions, models, and optional
        metrics.

    Raises:
        ValueError: If labels are not binary (exactly 2 unique classes).
    """
    classes = np.unique(list(train_labels.values()))
    if len(classes) != 2:
        raise ValueError(f"Only binary classification supported; found classes {classes}")

    # Resolve per-class n_states
    if isinstance(n_states, int):
        states_per_class = {int(c): n_states for c in classes}
    else:
        states_per_class = n_states

    # Train one HMM per class
    models: dict[int, hmmlearn_hmm.GaussianHMM] = {}
    for c in classes:
        c_int = int(c)
        seqs = [s for k, s in train_sequences.items() if train_labels[k] == c_int]
        models[c_int] = train_class_hmm(
            seqs,
            n_states=states_per_class[c_int],
            covariance_type=covariance_type,
            max_iter=max_iter,
            n_replicates=n_replicates,
            tol=tol,
        )

    # Score each test sequence under both class models
    log_lls: dict[str, dict[int, float]] = {}
    preds: dict[str, int] = {}
    for k, seq in test_sequences.items():
        ll = {int(c): score_sequence(seq, models[int(c)]) for c in classes}
        log_lls[k] = ll
        preds[k] = max(ll, key=lambda c: ll[c])

    result = ClassificationResult(
        predicted_labels=preds,
        log_likelihoods=log_lls,
        models=models,
        classes=classes,
    )

    # Compute metrics if test labels provided
    if test_labels is not None:
        keys = list(test_sequences.keys())
        y_true = np.array([test_labels[k] for k in keys])
        y_pred = np.array([preds[k] for k in keys])
        y_prob = np.array(
            [
                np.exp(
                    log_lls[k][int(classes[1])]
                    - logsumexp([log_lls[k][int(classes[0])], log_lls[k][int(classes[1])]])
                )
                for k in keys
            ]
        )

        metrics, ci = _compute_metrics(
            y_true,
            y_pred,
            y_prob,
            n_bootstraps=n_bootstraps,
            ci_alpha=ci_alpha,
        )
        result.metrics = metrics
        result.confusion_matrix = confusion_matrix(y_true, y_pred)
        result.confidence_intervals = ci

    return result
