"""Tests for Voronoi cell geometry."""

import numpy as np
import pandas as pd
import pytest

from matspat.descriptors.voronoi import compute_voronoi_geometry


def _lattice_df(n=8, spacing=1.0, image_id="img1", types=None):
    xs, ys = np.meshgrid(np.arange(n) * spacing, np.arange(n) * spacing)
    xy = np.column_stack([xs.ravel(), ys.ravel()])
    k = len(xy)
    return pd.DataFrame(
        {
            "cell_id": [f"{image_id}_{i}" for i in range(k)],
            "image_id": image_id,
            "x": xy[:, 0],
            "y": xy[:, 1],
            "cell_type": types if types is not None else ["A"] * k,
        }
    )


def _hex_df(n=10):
    pts = [(i + 0.5 * (j % 2), j * np.sqrt(3) / 2) for j in range(n) for i in range(n)]
    xy = np.array(pts)
    return pd.DataFrame(
        {
            "cell_id": np.arange(len(xy)),
            "image_id": "hex",
            "x": xy[:, 0],
            "y": xy[:, 1],
            "cell_type": "A",
        }
    )


class TestSquareLattice:
    def test_interior_geometry(self):
        out = compute_voronoi_geometry(_lattice_df(spacing=2.0))
        interior = out[~out["voronoi_boundary"]]
        assert len(interior) > 0
        assert np.allclose(interior["voronoi_area"], 4.0)
        assert np.allclose(interior["voronoi_perimeter"], 8.0)
        assert np.allclose(interior["voronoi_neighbors"], 4)
        assert np.allclose(interior["shape_index"], 4.0)

    def test_edge_cells_flagged_and_nan(self):
        out = compute_voronoi_geometry(_lattice_df())
        boundary = out[out["voronoi_boundary"]]
        assert len(boundary) >= 4 * 8 - 4  # at least the outer ring
        assert boundary["voronoi_area"].isna().all()
        assert boundary["shape_index"].isna().all()

    def test_output_columns(self):
        out = compute_voronoi_geometry(_lattice_df())
        assert list(out.columns) == [
            "cell_id",
            "image_id",
            "cell_type",
            "voronoi_area",
            "voronoi_perimeter",
            "voronoi_neighbors",
            "shape_index",
            "voronoi_boundary",
        ]
        assert len(out) == 64


class TestHexLattice:
    def test_hexagonal_shape_index(self):
        out = compute_voronoi_geometry(_hex_df())
        interior = out[~out["voronoi_boundary"]]
        assert np.allclose(interior["voronoi_neighbors"], 6)
        # regular hexagon: P / sqrt(A) = sqrt(8 sqrt(3)) ≈ 3.722
        assert np.allclose(interior["shape_index"], np.sqrt(8 * np.sqrt(3)), atol=1e-6)


class TestOptions:
    def test_focus_cell_type_uses_all_cells_for_tessellation(self):
        types = ["A", "B"] * 32
        df = _lattice_df(types=types)
        full = compute_voronoi_geometry(df)
        focus = compute_voronoi_geometry(df, focus_cell_type="B")
        assert (focus["cell_type"] == "B").all()
        merged = focus.merge(full, on="cell_id", suffixes=("", "_full"))
        assert np.allclose(merged["voronoi_area"], merged["voronoi_area_full"], equal_nan=True)

    def test_include_cell_types_changes_tessellation(self):
        types = ["A", "B"] * 32
        df = _lattice_df(types=types)
        only_a = compute_voronoi_geometry(df, include_cell_types=["A"])
        assert (only_a["cell_type"] == "A").all()
        assert only_a["voronoi_area"].dropna().mean() > 1.5  # sparser lattice -> bigger cells

    def test_max_area_flags_large_cells(self):
        out = compute_voronoi_geometry(_lattice_df(spacing=2.0), max_area=3.0)
        assert out["voronoi_boundary"].all()

    def test_invalid_max_area(self):
        with pytest.raises(ValueError, match="max_area"):
            compute_voronoi_geometry(_lattice_df(), max_area=0)

    def test_multiple_images(self):
        df = pd.concat([_lattice_df(image_id="a"), _lattice_df(image_id="b", spacing=3.0)])
        out = compute_voronoi_geometry(df)
        assert set(out["image_id"]) == {"a", "b"}
        assert out.groupby("image_id")["voronoi_area"].mean()["b"] == pytest.approx(9.0)

    def test_too_few_cells_skipped(self):
        df = _lattice_df(n=8)
        tiny = pd.concat([df, _lattice_df(n=1, image_id="tiny")])
        with pytest.warns(UserWarning):
            out = compute_voronoi_geometry(tiny, include_cell_types=["A"]).query(
                "image_id == 'tiny'"
            )
        assert out.empty


def test_duplicate_coordinates_flagged():
    df = _hex_df()
    i = len(df) // 2 + 5  # an interior cell
    dup = df.iloc[[i]].copy()
    dup["cell_id"] = -1
    out = compute_voronoi_geometry(pd.concat([df, dup], ignore_index=True))
    flagged = out[out["cell_id"].isin([df["cell_id"].iloc[i], -1])]
    assert flagged["voronoi_boundary"].all()
    assert flagged["shape_index"].isna().all()
