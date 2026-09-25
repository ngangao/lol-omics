import matplotlib

matplotlib.use("Agg")

from lol import Pipeline
from lol.datasets import make_synthetic_counts
from lol.visualize import plot_latent_space, plot_permutation_null


def test_plot_latent_space_runs():
    expr, labels = make_synthetic_counts(n_samples=30, n_genes=150, random_state=1)
    pipe = Pipeline(latent_method="pca", n_latent=5, n_hvg=100)
    pipe.fit(expr, labels)
    ax = plot_latent_space(pipe.latents_, labels)
    assert ax is not None
    assert ax.get_xlabel() == pipe.latents_.columns[0]


def test_plot_permutation_null_runs():
    expr, labels = make_synthetic_counts(n_samples=25, n_genes=120, random_state=2)
    pipe = Pipeline(latent_method="pca", n_latent=4, n_hvg=80)
    pipe.fit(expr, labels)
    result = pipe.permutation_test(n_permutations=5)
    ax = plot_permutation_null(result)
    assert ax is not None


def test_plot_latent_space_accepts_existing_axes():
    import matplotlib.pyplot as plt

    expr, labels = make_synthetic_counts(n_samples=20, n_genes=100, random_state=3)
    pipe = Pipeline(latent_method="pca", n_latent=3, n_hvg=60)
    pipe.fit(expr, labels)
    _fig, ax = plt.subplots()
    returned = plot_latent_space(pipe.latents_, labels, ax=ax)
    assert returned is ax
