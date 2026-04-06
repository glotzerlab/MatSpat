import pandas as pd
import pytest

from matspat.sample_data import generate_sample_dataset


class TestGenerateSampleDataset:
    def test_returns_dataframe(self):
        df = generate_sample_dataset(n_per_class=2, seed=0)
        assert isinstance(df, pd.DataFrame)

    def test_expected_columns(self):
        df = generate_sample_dataset(n_per_class=2, seed=0)
        assert set(df.columns) == {"X", "Y", "cell_type", "image_id", "pathology"}

    def test_two_disease_classes(self):
        df = generate_sample_dataset(n_per_class=3, seed=0)
        assert set(df["pathology"].unique()) == {"inflamed", "excluded"}

    def test_n_per_class_images(self):
        n = 4
        df = generate_sample_dataset(n_per_class=n, seed=0)
        assert df["image_id"].nunique() == 2 * n

    def test_expected_cell_types(self):
        df = generate_sample_dataset(n_per_class=2, seed=0)
        assert set(df["cell_type"].unique()) == {"Tumor", "CD4", "CD8", "Macrophage"}

    def test_reproducible_with_same_seed(self):
        df1 = generate_sample_dataset(n_per_class=2, seed=7)
        df2 = generate_sample_dataset(n_per_class=2, seed=7)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self):
        df1 = generate_sample_dataset(n_per_class=2, seed=0)
        df2 = generate_sample_dataset(n_per_class=2, seed=1)
        assert not df1["X"].equals(df2["X"])
