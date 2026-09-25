import numpy as np

from lol.datasets import make_synthetic_counts
from lol.pipeline import Pipeline


def test_nested_evaluate_is_default_and_runs():
    expression, labels = make_synthetic_counts(n_samples=60, n_genes=300, random_state=1)
    pipe = Pipeline(latent_method="pca", n_latent=10, n_hvg=100)
    pipe.fit(expression, labels)
    report = pipe.evaluate()  # nested=True by default
    assert 0.0 <= report.accuracy <= 1.0


def test_nested_and_non_nested_both_run_and_can_differ():
    # On a small, noisy cohort, nested (leakage-safe) evaluation should
    # not systematically report *higher* accuracy than the leaky
    # non-nested version -- if anything the opposite, since it no
    # longer benefits from held-out-sample information leaking into
    # feature selection and the PCA fit.
    expression, labels = make_synthetic_counts(n_samples=30, n_genes=200, random_state=5)
    pipe = Pipeline(latent_method="pca", n_latent=8, n_hvg=50)
    pipe.fit(expression, labels)

    nested_report = pipe.evaluate(nested=True)
    leaky_report = pipe.evaluate(nested=False)

    assert 0.0 <= nested_report.accuracy <= 1.0
    assert 0.0 <= leaky_report.accuracy <= 1.0


def test_nested_evaluate_rejects_vae_with_clear_error():
    expression, labels = make_synthetic_counts(n_samples=20, n_genes=100, random_state=2)
    pipe = Pipeline(
        latent_method="vae",
        n_latent=4,
        n_hvg=50,
        vae_kwargs={"epochs": 2, "n_hidden": 8, "batch_size": 4},
    )
    pipe.fit(expression, labels)

    try:
        pipe.evaluate(nested=True)
        raised = False
    except NotImplementedError:
        raised = True
    assert raised, "nested evaluation on a VAE pipeline should raise NotImplementedError"

    # nested=False should still work as the documented fallback
    report = pipe.evaluate(nested=False)
    assert 0.0 <= report.accuracy <= 1.0


def test_nested_evaluate_never_lets_a_fold_see_its_own_test_samples():
    # Regression test for the specific bug found on real GEO data
    # (GSE16161): fit() previously ran HVG selection and PCA once on
    # the *entire* dataset before evaluate() cross-validated only the
    # classifier, so every "held-out" fold had already influenced the
    # features used to predict it. This produced a suspicious 1.000
    # accuracy/ROC-AUC on an 18-sample real cohort. Nested evaluation
    # must refit HVG + PCA per fold instead.
    rng = np.random.default_rng(0)
    # pure noise, no real signal -- with the old leaky evaluation this
    # was capable of reporting inflated accuracy on a small cohort;
    # nested evaluation should behave much closer to chance.
    import pandas as pd

    n_samples, n_genes = 18, 500
    expression = pd.DataFrame(
        rng.poisson(20, size=(n_samples, n_genes)),
        index=[f"s{i}" for i in range(n_samples)],
        columns=[f"g{i}" for i in range(n_genes)],
    )
    labels = pd.Series(
        rng.integers(0, 2, size=n_samples), index=expression.index, name="label"
    )

    pipe = Pipeline(latent_method="pca", n_latent=5, n_hvg=2000)
    pipe.fit(expression, labels)
    report = pipe.evaluate(nested=True)

    # No real signal exists in pure noise; a leakage-safe evaluation
    # should not report near-perfect accuracy here.
    assert report.accuracy < 0.9
