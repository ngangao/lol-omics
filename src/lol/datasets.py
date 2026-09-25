"""Synthetic data generation for tests, examples, and docs.

Not a substitute for real GEO data -- purely so the test suite and
quickstart run offline, fast, and without depending on NCBI being
reachable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_counts(
    n_samples: int = 80,
    n_genes: int = 500,
    n_informative: int = 30,
    random_state: int = 0,
) -> tuple[pd.DataFrame, pd.Series]:
    """Simulate an RNA-seq-like count matrix with a binary phenotype.

    A handful of genes carry a real signal correlated with the label;
    the rest are noise. Useful for smoke-testing the whole pipeline.
    """
    rng = np.random.default_rng(random_state)
    labels = pd.Series(
        rng.integers(0, 2, size=n_samples),
        index=[f"sample_{i}" for i in range(n_samples)],
        name="phenotype",
    )

    base_rate = rng.gamma(shape=2.0, scale=5.0, size=n_genes)
    counts = rng.poisson(lam=base_rate, size=(n_samples, n_genes)).astype(float)

    informative = rng.choice(n_genes, size=n_informative, replace=False)
    effect = rng.uniform(2.0, 5.0, size=n_informative)
    counts[:, informative] += (labels.to_numpy()[:, None] * effect[None, :])

    genes = [f"gene_{i}" for i in range(n_genes)]
    expression = pd.DataFrame(counts, index=labels.index, columns=genes)
    return expression, labels
