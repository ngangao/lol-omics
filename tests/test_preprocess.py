import numpy as np
import pandas as pd
import pytest

from lol.preprocess import detect_platform, normalize, select_highly_variable


def test_detect_platform_counts():
    counts = pd.DataFrame(np.random.poisson(10, size=(5, 20)))
    assert detect_platform(counts) == "rna_seq"


def test_detect_platform_intensities():
    intensities = pd.DataFrame(np.random.normal(8, 1, size=(5, 20)))
    assert detect_platform(intensities) == "microarray"


def test_normalize_rna_seq_nonnegative_and_finite():
    counts = pd.DataFrame(np.random.poisson(10, size=(6, 15)))
    out = normalize(counts, platform="rna_seq")
    assert np.isfinite(out.to_numpy()).all()
    assert (out.to_numpy() >= 0).all()


def test_normalize_microarray_shapes_and_matches_reference_distribution():
    intensities = pd.DataFrame(np.random.normal(8, 2, size=(6, 15)))
    out = normalize(intensities, platform="microarray")
    assert out.shape == intensities.shape
    # every sample should now have (approximately) the same set of values
    first_sorted = np.sort(out.iloc[0].to_numpy())
    second_sorted = np.sort(out.iloc[1].to_numpy())
    assert np.allclose(first_sorted, second_sorted, atol=1e-6)


def test_normalize_rejects_unknown_platform():
    df = pd.DataFrame(np.random.rand(3, 3))
    with pytest.raises(ValueError):
        normalize(df, platform="nonsense")


def test_select_highly_variable_keeps_requested_count_and_top_genes():
    df = pd.DataFrame(
        {
            "low_var": [1.0, 1.01, 0.99, 1.0],
            "high_var": [0.0, 10.0, -5.0, 20.0],
            "mid_var": [0.0, 2.0, -1.0, 3.0],
        }
    )
    out = select_highly_variable(df, n_genes=2)
    assert out.shape[1] == 2
    assert "high_var" in out.columns
    assert "low_var" not in out.columns


def test_select_highly_variable_caps_at_available_genes():
    df = pd.DataFrame(np.random.rand(5, 3))
    out = select_highly_variable(df, n_genes=100)
    assert out.shape[1] == 3
