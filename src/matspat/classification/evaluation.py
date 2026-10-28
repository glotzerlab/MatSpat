"""Train/test splitting and convenience wrappers for HMM classification.

Public functions:
    split_and_classify   Stratified split then classify in one call.
    classify_patients    Patient-level classification by aggregating image scores.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from matspat.config import (
    DEFAULT_CI_ALPHA,
    DEFAULT_N_BOOTSTRAPS,
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
)

from .hmm import ClassificationResult, _compute_metrics, classify
from .time_series import METHODS as TS_METHODS
from .time_series import TimeSeriesResult, classify_time_series

_AGGREGATORS: dict[str, Callable[[list[float]], float]] = {
    "mean": lambda v: float(np.mean(v)),
    "median": lambda v: float(np.median(v)),
}


@dataclass
class PatientClassificationResult:
    """Container for patient-level classification outputs and metrics.

    ``scores`` holds one aggregated decision score per test patient; positive
    values favour ``classes[1]``. ``image_result`` is the underlying
    image-level result for the test patients' images (metrics, but no CIs).
    """

    predicted_labels: dict[str, int]
    scores: dict[str, float]
    patient_labels: dict[str, int]
    image_result: ClassificationResult | TimeSeriesResult
    classes: np.ndarray
    metrics: dict[str, float] | None = None
    confusion_matrix: np.ndarray | None = None
    confidence_intervals: dict[str, tuple[float, float]] = field(default_factory=dict)


def split_and_classify(
    sequences: dict[str, np.ndarray],
    labels: dict[str, int],
    *,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
    stratify: bool = True,
    **classify_kwargs: object,
) -> ClassificationResult:
    """Stratified train/test split followed by HMM classification.

    Convenience wrapper that splits the data and calls :func:`hmm.classify`
    with ``test_labels`` automatically populated so that metrics are always
    computed.

    Args:
        sequences: Dict mapping sample key → array ``(T, D)``.
        labels: Dict mapping sample key → integer class label.
        test_size: Fraction of data held out for testing.
        random_state: Random seed for reproducibility.
        stratify: Whether to stratify the split by label.
        **classify_kwargs: Forwarded to :func:`hmm.classify` (e.g.
            ``n_states``, ``covariance_type``, ``n_bootstraps``).

    Returns:
        :class:`ClassificationResult` with metrics populated.
    """
    keys = list(sequences.keys())
    y = [labels[k] for k in keys]

    strat = y if stratify else None
    train_keys, test_keys = train_test_split(
        keys,
        test_size=test_size,
        stratify=strat,
        random_state=random_state,
    )

    train_seqs = {k: sequences[k] for k in train_keys}
    train_labs = {k: labels[k] for k in train_keys}
    test_seqs = {k: sequences[k] for k in test_keys}
    test_labs = {k: labels[k] for k in test_keys}

    return classify(
        train_seqs,
        train_labs,
        test_seqs,
        test_labels=test_labs,
        **classify_kwargs,  # type: ignore[arg-type]
    )


def classify_patients(
    sequences: dict[str, np.ndarray],
    labels: dict[str, int],
    patient_map: dict[str, str],
    *,
    method: str = "hmm",
    aggregate: str = "median",
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
    stratify: bool = True,
    n_bootstraps: int = DEFAULT_N_BOOTSTRAPS,
    ci_alpha: float = DEFAULT_CI_ALPHA,
    **classifier_kwargs: object,
) -> PatientClassificationResult:
    """Patient-level classification from image-level sequences.

    Patients (not images) are split into train and test sets, so no patient
    contributes images to both. An image-level classifier is trained on the
    training patients' images and scores every test image. Each test
    patient's image scores are then aggregated into one patient score, which
    is classified by its sign and used for patient-level metrics.

    Image scores are unbounded (HMM log-likelihood ratios grow with sequence
    length and feature count), so one extreme image can dominate a mean;
    the default ``"median"`` is robust to this, and its sign is the majority
    vote of the images. With few patients (e.g. ~10 in the test set) the
    patient-level bootstrap CIs are wide and can be degenerate; use repeated
    patient-grouped cross-validation for reliable estimates.

    Args:
        sequences: Dict mapping sample key → array ``(T, D)``.
        labels: Dict mapping sample key → integer class label. All images of
            one patient must share a label.
        patient_map: Dict mapping sample key → patient ID.
        method: Image-level classifier: ``"hmm"`` (:func:`hmm.classify`),
            ``"quant"`` or ``"multirocket-hydra"``
            (:func:`time_series.classify_time_series`).
        aggregate: How image scores are combined per patient: ``"median"``
            or ``"mean"``.
        test_size: Fraction of patients held out for testing.
        random_state: Random seed for the patient split, and for the
            time-series models (the HMMs use their own restart seeds).
        stratify: Whether to stratify the patient split by label.
        n_bootstraps: Number of bootstrap samples for CIs (resampling patients).
        ci_alpha: Significance level for CIs (0.05 → 95% CI).
        **classifier_kwargs: Forwarded to :func:`hmm.classify` (e.g.
            ``n_states``). Only allowed with ``method="hmm"``.

    Returns:
        :class:`PatientClassificationResult` with patient-level predictions,
        scores and metrics, plus the image-level result.

    Raises:
        ValueError: If a sequence has no patient, a patient's images have
            different labels, *method* or *aggregate* is unknown, extra
            keyword arguments are given for a time-series method, or the test
            patients do not include both classes.
    """
    if method not in ("hmm", *TS_METHODS):
        raise ValueError(f"method must be 'hmm' or one of {TS_METHODS}; got {method!r}")
    if method != "hmm" and classifier_kwargs:
        raise ValueError(
            f"Extra keyword arguments are only used by 'hmm': {list(classifier_kwargs)}"
        )
    if aggregate not in _AGGREGATORS:
        raise ValueError(f"aggregate must be one of {tuple(_AGGREGATORS)}; got {aggregate!r}")

    keys = list(sequences.keys())
    missing = [k for k in keys if k not in patient_map]
    if missing:
        raise ValueError(f"{len(missing)} sequences have no patient in patient_map: {missing[:5]}")

    patient_labels: dict[str, int] = {}
    for k in keys:
        pid, lab = patient_map[k], labels[k]
        if patient_labels.setdefault(pid, lab) != lab:
            raise ValueError(f"Patient {pid!r} has images with different labels")

    patients = list(patient_labels.keys())
    y_patients = [patient_labels[p] for p in patients]
    train_patients, test_patients = train_test_split(
        patients,
        test_size=test_size,
        stratify=y_patients if stratify else None,
        random_state=random_state,
    )
    if len({patient_labels[p] for p in test_patients}) < 2:
        raise ValueError("Test patients contain only one class; increase test_size or stratify")

    train_set = set(train_patients)
    train_keys = [k for k in keys if patient_map[k] in train_set]
    test_keys = [k for k in keys if patient_map[k] not in train_set]
    train_seqs = {k: sequences[k] for k in train_keys}
    train_labs = {k: labels[k] for k in train_keys}
    test_seqs = {k: sequences[k] for k in test_keys}
    test_labs = {k: labels[k] for k in test_keys}

    image_result: ClassificationResult | TimeSeriesResult
    if method == "hmm":
        image_result = classify(
            train_seqs,
            train_labs,
            test_seqs,
            test_labels=test_labs,
            n_bootstraps=0,
            **classifier_kwargs,  # type: ignore[arg-type]
        )
    else:
        image_result = classify_time_series(
            train_seqs,
            train_labs,
            test_seqs,
            test_labels=test_labs,
            method=method,
            random_state=random_state,
            n_bootstraps=0,
        )

    # Image-level CIs are skipped above: images of one patient are not independent.
    # Aggregate image decision scores (positive favours classes[1]) per patient
    per_patient: dict[str, list[float]] = {}
    for k in test_keys:
        per_patient.setdefault(patient_map[k], []).append(image_result.scores[k])
    agg = _AGGREGATORS[aggregate]
    scores = {p: float(agg(v)) for p, v in per_patient.items()}

    classes = image_result.classes
    preds = {p: int(classes[1] if s > 0 else classes[0]) for p, s in scores.items()}

    order = list(scores.keys())
    y_true = np.array([patient_labels[p] for p in order])
    y_pred = np.array([preds[p] for p in order])
    metrics, ci = _compute_metrics(
        y_true,
        y_pred,
        np.array([scores[p] for p in order]),
        pos_label=int(classes[1]),
        n_bootstraps=n_bootstraps,
        ci_alpha=ci_alpha,
    )

    return PatientClassificationResult(
        predicted_labels=preds,
        scores=scores,
        patient_labels={p: patient_labels[p] for p in order},
        image_result=image_result,
        classes=classes,
        metrics=metrics,
        confusion_matrix=confusion_matrix(y_true, y_pred),
        confidence_intervals=ci,
    )
