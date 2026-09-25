from lol.datasets import make_synthetic_counts
from lol.pipeline import Pipeline


def test_pipeline_pca_end_to_end():
    expression, labels = make_synthetic_counts(n_samples=60, n_genes=300, random_state=1)
    pipe = Pipeline(latent_method="pca", n_latent=10, n_hvg=100)
    pipe.fit(expression, labels)
    report = pipe.evaluate()
    assert 0.0 <= report.accuracy <= 1.0
    # the synthetic data has real signal, so this should clearly beat chance
    assert report.accuracy > 0.6


def test_pipeline_vae_end_to_end_runs():
    expression, labels = make_synthetic_counts(n_samples=40, n_genes=150, random_state=2)
    pipe = Pipeline(
        latent_method="vae",
        n_latent=6,
        n_hvg=80,
        vae_kwargs={"epochs": 5, "n_hidden": 16, "batch_size": 8},
    )
    pipe.fit(expression, labels)
    # Nested (leakage-safe) evaluation isn't available for the VAE path
    # yet -- see lol.pipeline.Pipeline.evaluate's docstring.
    report = pipe.evaluate(nested=False)
    assert 0.0 <= report.accuracy <= 1.0


def test_pipeline_pca_predict_on_new_samples():
    expression, labels = make_synthetic_counts(n_samples=50, n_genes=120, random_state=3)
    train_expr, train_labels = expression.iloc[:40], labels.iloc[:40]
    new_expr = expression.iloc[40:]

    pipe = Pipeline(latent_method="pca", n_latent=8, n_hvg=60)
    pipe.fit(train_expr, train_labels)
    preds = pipe.predict(new_expr)
    assert len(preds) == len(new_expr)
