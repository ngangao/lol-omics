import numpy as np
import pandas as pd

from lol.datasets import make_synthetic_counts
from lol.pipeline import Pipeline


def test_permutation_test_on_strong_signal_rejects_null():
    # Strong, easy signal (many informative genes, big effect) -- the
    # real score should clearly beat every shuffled permutation, same
    # shape as the GSE16161 result during development.
    expression, labels = make_synthetic_counts(
        n_samples=40, n_genes=300, n_informative=60, random_state=1
    )
    pipe = Pipeline(latent_method="pca", n_latent=8, n_hvg=200)
    pipe.fit(expression, labels)

    result = pipe.permutation_test(metric="macro_f1", n_permutations=15)
    assert result.real_score > result.null_scores.max()
    assert result.p_value_estimate < 0.1


def test_permutation_test_on_pure_noise_does_not_reject_null():
    # No real signal at all -- the real score should land squarely
    # inside the null distribution, not clearly above it.
    rng = np.random.default_rng(0)
    n_samples, n_genes = 24, 200
    expression = pd.DataFrame(
        rng.poisson(20, size=(n_samples, n_genes)),
        index=[f"s{i}" for i in range(n_samples)],
        columns=[f"g{i}" for i in range(n_genes)],
    )
    labels = pd.Series(
        rng.integers(0, 2, size=n_samples), index=expression.index, name="label"
    )

    pipe = Pipeline(latent_method="pca", n_latent=5, n_hvg=200)
    pipe.fit(expression, labels)
    result = pipe.permutation_test(metric="macro_f1", n_permutations=15)

    # not a strict guarantee on random data, but with no real signal
    # the p-value should generally be unremarkable
    assert result.p_value_estimate > 0.1


def test_permutation_test_p_value_bounds():
    expression, labels = make_synthetic_counts(n_samples=30, n_genes=150, random_state=3)
    pipe = Pipeline(latent_method="pca", n_latent=5, n_hvg=100)
    pipe.fit(expression, labels)
    result = pipe.permutation_test(n_permutations=10)

    assert 0.0 < result.p_value_estimate <= 1.0
    # smallest reportable p-value with n_permutations=10 is 1/11
    assert result.p_value_estimate >= 1 / 11


def test_permutation_test_requires_fit_first():
    pipe = Pipeline(latent_method="pca", n_latent=5, n_hvg=100)
    try:
        pipe.permutation_test()
        raised = False
    except RuntimeError:
        raised = True
    assert raised


def test_permutation_test_rejects_vae():
    expression, labels = make_synthetic_counts(n_samples=20, n_genes=100, random_state=2)
    pipe = Pipeline(
        latent_method="vae",
        n_latent=4,
        n_hvg=50,
        vae_kwargs={"epochs": 2, "n_hidden": 8, "batch_size": 4},
    )
    pipe.fit(expression, labels)
    try:
        pipe.permutation_test()
        raised = False
    except NotImplementedError:
        raised = True
    assert raised


def test_permutation_test_rejects_unavailable_metric():
    expression, labels = make_synthetic_counts(n_samples=20, n_genes=100, random_state=4)
    pipe = Pipeline(latent_method="pca", n_latent=5, n_hvg=80)
    pipe.fit(expression, labels)
    try:
        # roc_auc is None for non-binary/degenerate cases in principle;
        # here we just check an unknown metric name fails loudly.
        pipe.permutation_test(metric="not_a_real_metric", n_permutations=3)
        raised = False
    except AttributeError:
        raised = True
    assert raised
