"""Tests for the matspat.classification module."""

import numpy as np
import pandas as pd
import pytest

from matspat.classification import (
    ClassificationResult,
    build_sequences,
    build_sequences_from_dict,
    classify,
    classify_patients,
    evaluation,
    extract_labels,
    score_sequence,
    split_and_classify,
    train_class_hmm,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)

N_IMAGES = 20
N_BINS = 5
PAIRS = ["CD8_Mac", "CD4_Mac"]


def _make_long_df() -> pd.DataFrame:
    """Create a synthetic long-format DataFrame for sequence tests."""
    rows = []
    for i in range(N_IMAGES):
        label = 0 if i < N_IMAGES // 2 else 1
        for b in range(N_BINS):
            for pair in PAIRS:
                rows.append(
                    {
                        "image_id": f"img_{i:03d}",
                        "quantile_bin": b,
                        "immune_pair": pair,
                        "lees_L": RNG.standard_normal(),
                        "label": label,
                    }
                )
    return pd.DataFrame(rows)


def _make_separable_sequences(
    n_per_class: int = 15,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Build well-separated synthetic sequences for classification tests.

    Class 0: ascending trajectory.  Class 1: descending trajectory.
    """
    seqs: dict[str, np.ndarray] = {}
    labels: dict[str, int] = {}
    for i in range(n_per_class):
        key0 = f"c0_{i}"
        key1 = f"c1_{i}"
        seqs[key0] = np.column_stack(
            [np.linspace(0, 1, N_BINS) + RNG.normal(0, 0.05, N_BINS) for _ in PAIRS]
        )
        seqs[key1] = np.column_stack(
            [np.linspace(1, 0, N_BINS) + RNG.normal(0, 0.05, N_BINS) for _ in PAIRS]
        )
        labels[key0] = 0
        labels[key1] = 1
    return seqs, labels


# ===========================================================================
# sequences.py tests
# ===========================================================================


class TestBuildSequences:
    def test_basic_shape(self) -> None:
        df = _make_long_df()
        seqs, features = build_sequences(df)
        assert len(seqs) == N_IMAGES
        assert features == sorted(PAIRS)
        for arr in seqs.values():
            assert arr.shape == (N_BINS, len(PAIRS))

    def test_missing_column_raises(self) -> None:
        df = _make_long_df().drop(columns=["lees_L"])
        with pytest.raises(KeyError, match="missing required columns"):
            build_sequences(df)

    def test_exclude_features(self) -> None:
        df = _make_long_df()
        seqs, features = build_sequences(df, exclude_features=["CD4_Mac"])
        assert features == ["CD8_Mac"]
        for arr in seqs.values():
            assert arr.shape[1] == 1

    def test_reverse_flips_order(self) -> None:
        df = _make_long_df()
        seqs_fwd, _ = build_sequences(df, reverse=False)
        seqs_rev, _ = build_sequences(df, reverse=True)
        key = list(seqs_fwd.keys())[0]
        np.testing.assert_array_almost_equal(
            seqs_fwd[key][::-1],
            seqs_rev[key],
        )

    def test_missing_values_filled_with_zero(self) -> None:
        df = _make_long_df()
        # Drop one pair for one image so it becomes missing
        mask = (df["image_id"] == "img_000") & (df["immune_pair"] == "CD4_Mac")
        df = df[~mask]
        seqs, _ = build_sequences(df)
        # The CD4_Mac column for img_000 should be all zeros
        idx = sorted(PAIRS).index("CD4_Mac")
        np.testing.assert_array_equal(seqs["img_000"][:, idx], 0.0)


class TestBuildSequencesFromDict:
    def test_basic_shape(self) -> None:
        trajs = {
            "img_0": {"A_B": [0.1, 0.2, 0.3], "C_D": [0.4, 0.5, 0.6]},
            "img_1": {"A_B": [0.7, 0.8, 0.9], "C_D": [1.0, 1.1, 1.2]},
        }
        seqs, features = build_sequences_from_dict(trajs)
        assert features == ["A_B", "C_D"]
        assert seqs["img_0"].shape == (3, 2)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            build_sequences_from_dict({})

    def test_feature_order(self) -> None:
        trajs = {"img_0": {"B": [1.0], "A": [2.0]}}
        seqs, features = build_sequences_from_dict(trajs, feature_order=["B", "A"])
        assert features == ["B", "A"]
        np.testing.assert_array_equal(seqs["img_0"], [[1.0, 2.0]])

    def test_reverse(self) -> None:
        trajs = {"img_0": {"A": [1.0, 2.0, 3.0]}}
        seqs_fwd, _ = build_sequences_from_dict(trajs, reverse=False)
        seqs_rev, _ = build_sequences_from_dict(trajs, reverse=True)
        np.testing.assert_array_equal(seqs_fwd["img_0"][::-1], seqs_rev["img_0"])


class TestExtractLabels:
    def test_integer_labels(self) -> None:
        df = _make_long_df()
        labels = extract_labels(df)
        assert len(labels) == N_IMAGES
        assert all(isinstance(k, str) for k in labels)
        assert all(v in (0, 1) for v in labels.values())

    def test_string_labels(self) -> None:
        df = _make_long_df()
        df["label"] = df["label"].map({0: "healthy", 1: "disease"})
        labels = extract_labels(df)
        # alphabetical: "disease"=0, "healthy"=1
        assert labels["img_000"] == 1  # was 0 → "healthy" → 1
        assert labels["img_010"] == 0  # was 1 → "disease" → 0

    def test_keys_match_build_sequences(self) -> None:
        df = _make_long_df()
        seqs, _ = build_sequences(df)
        labels = extract_labels(df)
        assert set(seqs.keys()) == set(labels.keys())

    def test_missing_column_raises(self) -> None:
        df = _make_long_df().drop(columns=["label"])
        with pytest.raises(KeyError, match="missing required columns"):
            extract_labels(df)


# ===========================================================================
# hmm.py tests
# ===========================================================================


class TestTrainClassHMM:
    def test_returns_fitted_model(self) -> None:
        seqs = [RNG.standard_normal((N_BINS, 2)) for _ in range(10)]
        model = train_class_hmm(seqs, n_states=2, n_replicates=2)
        assert hasattr(model, "score")
        assert model.n_components == 2

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            train_class_hmm([])

    def test_univariate(self) -> None:
        seqs = [RNG.standard_normal((N_BINS, 1)) for _ in range(5)]
        model = train_class_hmm(seqs, n_states=2, n_replicates=1)
        assert model.n_components == 2


class TestScoreSequence:
    def test_returns_float(self) -> None:
        seqs = [RNG.standard_normal((N_BINS, 2)) for _ in range(5)]
        model = train_class_hmm(seqs, n_states=2, n_replicates=1)
        ll = score_sequence(seqs[0], model)
        assert isinstance(ll, float)
        assert np.isfinite(ll)


class TestClassify:
    def test_binary_only(self) -> None:
        seqs = {"a": RNG.standard_normal((5, 2))}
        labels = {"a": 0}
        with pytest.raises(ValueError, match="binary"):
            classify(seqs, labels, seqs)

    def test_predictions_structure(self) -> None:
        seqs, labels = _make_separable_sequences(n_per_class=8)
        keys = list(seqs.keys())
        train_k, test_k = keys[:12], keys[12:]
        result = classify(
            {k: seqs[k] for k in train_k},
            {k: labels[k] for k in train_k},
            {k: seqs[k] for k in test_k},
            n_states=2,
            n_replicates=1,
        )
        assert isinstance(result, ClassificationResult)
        assert set(result.predicted_labels.keys()) == set(test_k)
        assert all(v in (0, 1) for v in result.predicted_labels.values())
        assert result.metrics is None  # no test_labels provided

    def test_with_metrics(self) -> None:
        seqs, labels = _make_separable_sequences(n_per_class=10)
        keys = list(seqs.keys())
        train_k, test_k = keys[:14], keys[14:]
        result = classify(
            {k: seqs[k] for k in train_k},
            {k: labels[k] for k in train_k},
            {k: seqs[k] for k in test_k},
            test_labels={k: labels[k] for k in test_k},
            n_states=2,
            n_replicates=1,
            n_bootstraps=50,
        )
        assert result.metrics is not None
        for metric in ("accuracy", "balanced_accuracy", "f1_score", "sensitivity", "auroc"):
            assert 0.0 <= result.metrics[metric] <= 1.0
        assert result.confusion_matrix is not None
        assert result.confusion_matrix.shape == (2, 2)
        assert len(result.confidence_intervals) == 5

    def test_per_class_n_states(self) -> None:
        seqs, labels = _make_separable_sequences(n_per_class=8)
        keys = list(seqs.keys())
        train_k, test_k = keys[:12], keys[12:]
        result = classify(
            {k: seqs[k] for k in train_k},
            {k: labels[k] for k in train_k},
            {k: seqs[k] for k in test_k},
            n_states={0: 2, 1: 3},
            n_replicates=1,
        )
        assert result.models[0].n_components == 2
        assert result.models[1].n_components == 3


# ===========================================================================
# evaluation.py tests
# ===========================================================================


class TestSplitAndClassify:
    def test_end_to_end(self) -> None:
        seqs, labels = _make_separable_sequences(n_per_class=15)
        result = split_and_classify(
            seqs,
            labels,
            test_size=0.3,
            n_states=2,
            n_replicates=1,
            n_bootstraps=50,
        )
        assert result.metrics is not None
        assert result.confusion_matrix is not None
        assert 0.0 <= result.metrics["accuracy"] <= 1.0

    def test_reproducibility(self) -> None:
        seqs, labels = _make_separable_sequences(n_per_class=15)
        r1 = split_and_classify(seqs, labels, n_states=2, n_replicates=1, n_bootstraps=10)
        r2 = split_and_classify(seqs, labels, n_states=2, n_replicates=1, n_bootstraps=10)
        assert r1.predicted_labels == r2.predicted_labels


def _make_patient_data(
    n_patients_per_class: int = 8, images_per_patient: int = 3
) -> tuple[dict[str, np.ndarray], dict[str, int], dict[str, str]]:
    """Separable sequences grouped into patients with several images each."""
    seqs, labels = _make_separable_sequences(n_patients_per_class * images_per_patient)
    patient_map = {}
    for k in seqs:
        cls, i = k.split("_")
        patient_map[k] = f"{cls}_p{int(i) // images_per_patient}"
    return seqs, labels, patient_map


class TestClassifyPatients:
    def test_end_to_end(self) -> None:
        seqs, labels, patient_map = _make_patient_data()
        result = classify_patients(
            seqs, labels, patient_map, n_states=2, n_replicates=1, n_bootstraps=10
        )
        assert result.metrics is not None
        assert result.metrics["auroc"] > 0.9
        assert set(result.predicted_labels) == set(result.scores) == set(result.patient_labels)
        # Every image of a test patient is scored at image level
        test_images = {k for k, p in patient_map.items() if p in result.scores}
        assert set(result.image_result.scores) == test_images

    def test_no_patient_in_both_splits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seqs, labels, patient_map = _make_patient_data()
        seen: dict[str, set[str]] = {}

        def spy(train_seqs, train_labs, test_seqs, **kwargs):
            seen["train"], seen["test"] = set(train_seqs), set(test_seqs)
            return classify(train_seqs, train_labs, test_seqs, **kwargs)

        monkeypatch.setattr(evaluation, "classify", spy)
        result = classify_patients(
            seqs, labels, patient_map, n_states=2, n_replicates=1, n_bootstraps=0
        )
        train_patients = {patient_map[k] for k in seen["train"]}
        test_patients = {patient_map[k] for k in seen["test"]}
        assert train_patients and test_patients
        assert not train_patients & test_patients
        assert test_patients == set(result.scores)
        # Every image of every test patient is in the test set
        assert seen["test"] == {k for k, p in patient_map.items() if p in test_patients}

    @pytest.mark.parametrize("aggregate,fn", [("mean", np.mean), ("median", np.median)])
    def test_patient_score_aggregates_image_scores(self, aggregate: str, fn) -> None:
        seqs, labels, patient_map = _make_patient_data()
        result = classify_patients(
            seqs,
            labels,
            patient_map,
            aggregate=aggregate,
            n_states=2,
            n_replicates=1,
            n_bootstraps=0,
        )
        for p, s in result.scores.items():
            imgs = [k for k, q in patient_map.items() if q == p]
            assert s == pytest.approx(fn([result.image_result.scores[k] for k in imgs]))
            assert result.predicted_labels[p] == int(s > 0)

    def test_non_01_labels(self) -> None:
        seqs, labels, patient_map = _make_patient_data()
        labels = {k: v + 1 for k, v in labels.items()}  # classes {1, 2}
        result = classify_patients(
            seqs, labels, patient_map, n_states=2, n_replicates=1, n_bootstraps=0
        )
        assert set(result.predicted_labels.values()) <= {1, 2}
        assert result.metrics is not None
        assert result.metrics["sensitivity"] == pytest.approx(1.0)
        assert result.metrics["auroc"] == pytest.approx(1.0)

    def test_one_class_in_test_raises(self) -> None:
        seqs, labels, patient_map = _make_patient_data()
        with pytest.raises(ValueError, match="one class"):
            classify_patients(seqs, labels, patient_map, test_size=1, stratify=False)

    def test_kwargs_rejected_for_time_series(self) -> None:
        seqs, labels, patient_map = _make_patient_data()
        with pytest.raises(ValueError, match="only used by 'hmm'"):
            classify_patients(seqs, labels, patient_map, method="quant", n_states=2)

    def test_mixed_labels_raise(self) -> None:
        seqs, labels, patient_map = _make_patient_data()
        patient_map["c1_0"] = patient_map["c0_0"]
        with pytest.raises(ValueError, match="different labels"):
            classify_patients(seqs, labels, patient_map)

    def test_missing_patient_raises(self) -> None:
        seqs, labels, patient_map = _make_patient_data()
        del patient_map["c0_0"]
        with pytest.raises(ValueError, match="no patient"):
            classify_patients(seqs, labels, patient_map)

    def test_unknown_method_raises(self) -> None:
        seqs, labels, patient_map = _make_patient_data()
        with pytest.raises(ValueError, match="method"):
            classify_patients(seqs, labels, patient_map, method="lstm")

    def test_unknown_aggregate_raises(self) -> None:
        seqs, labels, patient_map = _make_patient_data()
        with pytest.raises(ValueError, match="aggregate"):
            classify_patients(seqs, labels, patient_map, aggregate="max")


class TestClassifyScores:
    def test_scores_are_log_likelihood_ratios(self) -> None:
        seqs, labels = _make_separable_sequences()
        result = classify(seqs, labels, seqs, n_states=2, n_replicates=1)
        for k, ll in result.log_likelihoods.items():
            assert result.scores[k] == pytest.approx(ll[1] - ll[0])
            assert result.predicted_labels[k] == int(result.scores[k] > 0)

    def test_non_01_labels_metrics(self) -> None:
        seqs, labels = _make_separable_sequences()
        labels = {k: 2 * v for k, v in labels.items()}  # classes {0, 2}
        result = classify(seqs, labels, seqs, test_labels=labels, n_states=2, n_replicates=1)
        assert result.metrics is not None
        assert result.metrics["sensitivity"] == pytest.approx(1.0)
        assert result.metrics["f1_score"] == pytest.approx(1.0)


# ===========================================================================
# Integration: DataFrame → sequences → classify
# ===========================================================================


class TestIntegrationPipeline:
    def test_df_to_classification(self) -> None:
        """Full pipeline: long-format DataFrame → build_sequences → split_and_classify."""
        df = _make_long_df()
        seqs, _ = build_sequences(df)
        labels = extract_labels(df)

        result = split_and_classify(
            seqs,
            labels,
            n_states=2,
            n_replicates=1,
            n_bootstraps=20,
        )
        assert result.metrics is not None
        assert len(result.predicted_labels) > 0
