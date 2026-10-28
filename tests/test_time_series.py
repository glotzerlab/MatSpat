"""Tests for matspat.classification.time_series (requires the 'timeseries' extra)."""

import numpy as np
import pytest

pytest.importorskip("aeon")
pytest.importorskip("torch")

import sys  # noqa: E402

from matspat.classification import classify_patients, classify_time_series  # noqa: E402


def _make_sequences(
    n_per_class: int = 15, length: int = 10, n_features: int = 2, seed: int = 0
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Class 0: ascending trajectories. Class 1: descending trajectories."""
    rng = np.random.default_rng(seed)
    seqs: dict[str, np.ndarray] = {}
    labels: dict[str, int] = {}
    for i in range(n_per_class):
        for cls, (a, b) in enumerate([(0, 1), (1, 0)]):
            seqs[f"c{cls}_{i}"] = np.column_stack(
                [np.linspace(a, b, length) + rng.normal(0, 0.3, length) for _ in range(n_features)]
            )
            labels[f"c{cls}_{i}"] = cls
    return seqs, labels


@pytest.mark.parametrize("method", ["quant", "multirocket-hydra"])
class TestClassifyTimeSeries:
    def test_separable_data(self, method: str) -> None:
        seqs, labels = _make_sequences(seed=0)
        test_seqs, test_labels = _make_sequences(seed=1)  # held-out data
        result = classify_time_series(
            seqs, labels, test_seqs, test_labels=test_labels, method=method, n_bootstraps=10
        )
        assert result.method == method
        assert result.metrics is not None
        assert result.metrics["auroc"] > 0.9
        assert result.confusion_matrix is not None
        assert set(result.scores) == set(test_seqs)

    def test_non_01_labels(self, method: str) -> None:
        seqs, labels = _make_sequences(seed=0)
        test_seqs, test_labels = _make_sequences(seed=1)
        shift = lambda d: {k: v + 1 for k, v in d.items()}  # noqa: E731  classes {1, 2}
        result = classify_time_series(
            seqs,
            shift(labels),
            test_seqs,
            test_labels=shift(test_labels),
            method=method,
            n_bootstraps=0,
        )
        assert set(result.predicted_labels.values()) <= {1, 2}
        assert result.metrics is not None
        assert result.metrics["sensitivity"] > 0.8
        assert result.metrics["auroc"] > 0.9

    def test_predictions_follow_score_sign(self, method: str) -> None:
        seqs, labels = _make_sequences()
        result = classify_time_series(seqs, labels, seqs, method=method)
        assert result.metrics is None
        for k, s in result.scores.items():
            assert result.predicted_labels[k] == int(s > 0)

    def test_reproducible(self, method: str) -> None:
        seqs, labels = _make_sequences()
        r1 = classify_time_series(seqs, labels, seqs, method=method, random_state=3)
        r2 = classify_time_series(seqs, labels, seqs, method=method, random_state=3)
        assert r1.scores == pytest.approx(r2.scores)

    def test_patient_level(self, method: str) -> None:
        seqs, labels = _make_sequences(n_per_class=24)
        patient_map = {k: f"{k.split('_')[0]}_p{int(k.split('_')[1]) // 3}" for k in seqs}
        result = classify_patients(seqs, labels, patient_map, method=method, n_bootstraps=10)
        assert result.metrics is not None
        assert result.metrics["auroc"] > 0.9


class TestValidation:
    def test_missing_aeon_raises_import_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name in [m for m in sys.modules if m == "aeon" or m.startswith("aeon.")]:
            monkeypatch.delitem(sys.modules, name)
        monkeypatch.setitem(sys.modules, "aeon", None)
        seqs, labels = _make_sequences()
        with pytest.raises(ImportError, match="timeseries"):
            classify_time_series(seqs, labels, seqs, method="quant")

    def test_unknown_method(self) -> None:
        seqs, labels = _make_sequences()
        with pytest.raises(ValueError, match="method"):
            classify_time_series(seqs, labels, seqs, method="lstm")

    def test_non_binary(self) -> None:
        seqs, labels = _make_sequences()
        labels = {k: i % 3 for i, k in enumerate(labels)}
        with pytest.raises(ValueError, match="binary"):
            classify_time_series(seqs, labels, seqs)

    def test_multirocket_too_short(self) -> None:
        seqs, labels = _make_sequences(length=5)
        with pytest.raises(ValueError, match="length"):
            classify_time_series(seqs, labels, seqs, method="multirocket-hydra")

    def test_quant_short_sequences_ok(self) -> None:
        seqs, labels = _make_sequences(length=5)
        result = classify_time_series(seqs, labels, seqs, method="quant")
        assert len(result.scores) == len(seqs)

    def test_nan_raises(self) -> None:
        seqs, labels = _make_sequences()
        seqs["c0_0"][0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            classify_time_series(seqs, labels, seqs)

    def test_mismatched_shapes(self) -> None:
        seqs, labels = _make_sequences()
        seqs["c0_0"] = seqs["c0_0"][:-1]
        with pytest.raises(ValueError, match="same shape"):
            classify_time_series(seqs, labels, seqs)
