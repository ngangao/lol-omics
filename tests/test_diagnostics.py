import numpy as np
import pandas as pd

from lol.datasets import make_synthetic_counts
from lol.diagnostics import pvca_lite
from lol.pipeline import Pipeline


def test_leakage_audit_runs_and_reports_deltas():
    expr, labels = make_synthetic_counts(n_samples=40, n_genes=250, random_state=1)
    pipe = Pipeline(latent_method="pca", n_latent=8, n_hvg=150)
    pipe.fit(expr, labels)

    audit = pipe.leakage_audit()
    assert audit.naive.accuracy >= 0
    assert audit.nested.accuracy >= 0
    assert isinstance(audit.delta_accuracy, float)
    assert isinstance(audit.delta_macro_f1, float)


def test_leakage_audit_requires_fit_first():
    pipe = Pipeline(latent_method="pca", n_latent=5, n_hvg=80)
    try:
        pipe.leakage_audit()
        raised = False
    except RuntimeError:
        raised = True
    assert raised


def test_leakage_audit_rejects_vae():
    expr, labels = make_synthetic_counts(n_samples=20, n_genes=100, random_state=2)
    pipe = Pipeline(
        latent_method="vae", n_latent=4, n_hvg=50,
        vae_kwargs={"epochs": 2, "n_hidden": 8, "batch_size": 4},
    )
    pipe.fit(expr, labels)
    try:
        pipe.leakage_audit()
        raised = False
    except NotImplementedError:
        raised = True
    assert raised


def test_pvca_lite_detects_a_known_confound():
    # construct latents where PC1 is driven almost entirely by a
    # "batch" covariate, unrelated to the phenotype label
    rng = np.random.default_rng(0)
    n = 60
    batch = pd.Series(rng.choice(["A", "B"], size=n), name="batch")
    phenotype = pd.Series(rng.choice(["case", "control"], size=n), name="phenotype")
    index = [f"s{i}" for i in range(n)]
    batch.index = phenotype.index = index

    pc1 = batch.map({"A": -5.0, "B": 5.0}).to_numpy() + rng.normal(scale=0.5, size=n)
    pc2 = rng.normal(scale=1.0, size=n)  # pure noise, unrelated to anything
    latents = pd.DataFrame({"pc1": pc1, "pc2": pc2}, index=index)
    metadata = pd.DataFrame({"batch": batch, "phenotype": phenotype})

    result = pvca_lite(latents, metadata, explained_variance_ratio=np.array([0.9, 0.1]))
    assert result.shares["batch"] > result.shares["phenotype"]
    assert result.shares["batch"] > 0.5


def test_pvca_lite_handles_continuous_covariate():
    rng = np.random.default_rng(1)
    n = 40
    age = pd.Series(rng.uniform(20, 80, size=n), name="age")
    index = [f"s{i}" for i in range(n)]
    age.index = index
    pc1 = age.to_numpy() * 0.5 + rng.normal(scale=2.0, size=n)
    latents = pd.DataFrame({"pc1": pc1}, index=index)
    metadata = pd.DataFrame({"age": age})

    result = pvca_lite(latents, metadata)
    assert result.shares["age"] > 0.3


def test_confound_diagnostic_via_pipeline():
    expr, labels = make_synthetic_counts(n_samples=30, n_genes=150, random_state=3)
    pipe = Pipeline(latent_method="pca", n_latent=5, n_hvg=100)
    pipe.fit(expr, labels)

    metadata = pd.DataFrame({"phenotype": labels})
    result = pipe.confound_diagnostic(metadata)
    assert "phenotype" in result.shares
    assert result.weighted is True  # PCA path has explained_variance_ratio_


def test_confound_diagnostic_requires_fit_first():
    pipe = Pipeline(latent_method="pca", n_latent=5, n_hvg=80)
    try:
        pipe.confound_diagnostic(pd.DataFrame({"x": [1, 2, 3]}))
        raised = False
    except RuntimeError:
        raised = True
    assert raised
