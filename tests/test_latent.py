import numpy as np
import pandas as pd

from lol.latent import PCALatent, VAELatent


def _toy_expression(n_samples=20, n_genes=30, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(n_samples, n_genes))
    return pd.DataFrame(
        data,
        index=[f"s{i}" for i in range(n_samples)],
        columns=[f"g{i}" for i in range(n_genes)],
    )


def test_pca_latent_shape_and_index_preserved():
    expr = _toy_expression()
    pca = PCALatent(n_components=5)
    latents = pca.fit_transform(expr)
    assert latents.shape == (20, 5)
    assert list(latents.index) == list(expr.index)


def test_pca_latent_caps_components_to_data_size():
    expr = _toy_expression(n_samples=4, n_genes=30)
    pca = PCALatent(n_components=20)
    latents = pca.fit_transform(expr)
    assert latents.shape[1] <= 4


def test_vae_latent_shape_and_trains_without_error():
    expr = _toy_expression(n_samples=16, n_genes=25)
    vae = VAELatent(n_latent=4, n_hidden=8, epochs=3, batch_size=4)
    latents = vae.fit_transform(expr)
    assert latents.shape == (16, 4)
    assert np.isfinite(latents.to_numpy()).all()
    assert len(vae.history.train_loss) == 3


def test_vae_latent_loss_is_finite_throughout_training():
    expr = _toy_expression(n_samples=12, n_genes=20)
    vae = VAELatent(n_latent=3, n_hidden=8, epochs=5, batch_size=6)
    vae.fit_transform(expr)
    assert all(np.isfinite(v) for v in vae.history.train_loss)
