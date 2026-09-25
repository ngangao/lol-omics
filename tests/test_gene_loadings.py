import numpy as np
import pandas as pd

from lol.datasets import make_synthetic_counts
from lol.pipeline import Pipeline


def test_gene_loadings_returns_series_indexed_by_gene():
    expr, labels = make_synthetic_counts(n_samples=30, n_genes=150, random_state=1)
    pipe = Pipeline(latent_method="pca", n_latent=6, n_hvg=100)
    pipe.fit(expr, labels)

    loadings = pipe.gene_loadings(component=0)
    assert isinstance(loadings, pd.Series)
    assert set(loadings.index).issubset(set(pipe.selected_genes_))
    assert loadings.is_monotonic_increasing  # sorted ascending


def test_gene_loadings_component_out_of_range_raises():
    expr, labels = make_synthetic_counts(n_samples=20, n_genes=100, random_state=2)
    pipe = Pipeline(latent_method="pca", n_latent=3, n_hvg=60)
    pipe.fit(expr, labels)
    try:
        pipe.gene_loadings(component=99)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_gene_loadings_requires_fit_first():
    pipe = Pipeline(latent_method="pca", n_latent=5, n_hvg=80)
    try:
        pipe.gene_loadings()
        raised = False
    except RuntimeError:
        raised = True
    assert raised


def test_gene_loadings_rejects_vae():
    expr, labels = make_synthetic_counts(n_samples=20, n_genes=100, random_state=3)
    pipe = Pipeline(
        latent_method="vae", n_latent=4, n_hvg=50,
        vae_kwargs={"epochs": 2, "n_hidden": 8, "batch_size": 4},
    )
    pipe.fit(expr, labels)
    try:
        pipe.gene_loadings()
        raised = False
    except NotImplementedError:
        raised = True
    assert raised


def test_gene_loadings_matches_a_known_driver_gene():
    # construct data where one gene obviously dominates PC1 variance
    rng = np.random.default_rng(0)
    n_samples, n_genes = 30, 50
    base = rng.normal(scale=1.0, size=(n_samples, n_genes))
    driver_signal = rng.normal(scale=20.0, size=n_samples)  # huge swing
    base[:, 0] += driver_signal  # gene "g0" carries most of the variance
    expr = pd.DataFrame(
        np.abs(base) * 10,
        index=[f"s{i}" for i in range(n_samples)],
        columns=[f"g{i}" for i in range(n_genes)],
    )
    labels = pd.Series(rng.integers(0, 2, size=n_samples), index=expr.index, name="label")

    pipe = Pipeline(latent_method="pca", n_latent=5, n_hvg=50)
    pipe.fit(expr, labels)
    loadings = pipe.gene_loadings(component=0)

    top_driver = loadings.abs().idxmax()
    assert top_driver == "g0"
