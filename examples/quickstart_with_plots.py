"""Example: end-to-end pipeline with visualization, on synthetic data.

Runs offline (no GEO download needed) so it works as a first thing to
try. Swap `make_synthetic_counts(...)` for `pd.read_csv(...)` or
`lol.fetch.load_series_matrix(...)` to use it on real data.
"""

import matplotlib.pyplot as plt

from lol import Pipeline
from lol.datasets import make_synthetic_counts
from lol.visualize import plot_latent_space, plot_permutation_null

expression, labels = make_synthetic_counts(n_samples=80, n_genes=500, random_state=0)

pipe = Pipeline(latent_method="pca", n_latent=10, n_hvg=2000)
pipe.fit(expression, labels)

print(pipe.evaluate())
result = pipe.permutation_test(metric="macro_f1", n_permutations=30)
print(result)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
plot_latent_space(pipe.latents_, labels, ax=axes[0], title="Latent space (PC1 vs PC2)")
plot_permutation_null(result, ax=axes[1], title="Permutation test")
plt.tight_layout()
plt.savefig("example_output.png", dpi=200, bbox_inches="tight")
print("Saved example_output.png")
