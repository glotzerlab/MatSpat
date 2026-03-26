"""Train/test splitting and convenience wrappers for HMM classification.

Public functions:
    split_and_classify   Stratified split then classify in one call.
    classify_patients    Patient-level classification (stub).
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split

from .hmm import ClassificationResult, classify


def split_and_classify(
    sequences: dict[str, np.ndarray],
    labels: dict[str, int],
    *,
    test_size: float = 0.3,
    random_state: int = 42,
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
    **kwargs: object,
) -> ClassificationResult:
    """Patient-level classification (not yet implemented).

    Args:
        sequences: Sample-level sequences.
        labels: Sample-level labels.
        patient_map: Dict mapping sample key → patient ID for aggregation.
        **kwargs: Passed to :func:`classify`.

    Raises:
        NotImplementedError: Always — this is a stub for future work.
    """
    raise NotImplementedError("Patient-level classification is not yet implemented")
