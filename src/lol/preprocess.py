"""Preprocessing utilities: platform detection, normalization, feature selection.

Kept deliberately simple. LOL targets bulk RNA-seq and microarray data
(not single-cell), where a GEO cohort is typically tens to a few
hundred samples and a handful of thousand genes after filtering --
comfortably within reach of an 8GB-RAM laptop.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def detect_platform(expression: pd.DataFrame) -> str:
    """Heuristically classify a samples x genes matrix as counts or intensities.

    Returns ``"rna_seq"`` for integer-like, non-negative count data, or
    ``"microarray"`` for continuous intensity-like data. This is a
    heuristic, not a guarantee -- if you know the platform, skip this
    and pass it explicitly to :func:`normalize`.
    """
    values = expression.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Expression matrix has no finite values to inspect.")

    looks_integer = np.allclose(finite, np.round(finite), atol=1e-6)
    non_negative = finite.min() >= -1e-6

    if looks_integer and non_negative:
        return "rna_seq"
    return "microarray"


def normalize(expression: pd.DataFrame, platform: str | None = None) -> pd.DataFrame:
    """Normalize a samples x genes matrix ready for latent extraction.

    - ``"rna_seq"``: log-CPM (counts per million, log1p transformed).
    - ``"microarray"``: per-sample quantile normalization against the
      mean distribution across samples (a standard, simple stand-in
      for RMA-style normalization when starting from GEO-processed
      intensities rather than raw CEL files).

    Parameters
    ----------
    platform:
        ``"rna_seq"`` or ``"microarray"``. If omitted, inferred with
        :func:`detect_platform`.
    """
    if platform is None:
        platform = detect_platform(expression)

    if platform == "rna_seq":
        counts = expression.clip(lower=0)
        library_size = counts.sum(axis=1).replace(0, np.nan)
        cpm = counts.div(library_size, axis=0) * 1e6
        return np.log1p(cpm).fillna(0.0)

    if platform == "microarray":
        # Standard quantile normalization: sort each sample, average across
        # samples at each rank position, then map each sample's original
        # values back onto that averaged reference distribution.
        arr = expression.to_numpy(dtype=float)  # samples x genes
        sorted_arr = np.sort(arr, axis=1)
        ref_distribution = sorted_arr.mean(axis=0)

        ranks = np.argsort(np.argsort(arr, axis=1), axis=1)
        normalized_arr = ref_distribution[ranks]
        return pd.DataFrame(normalized_arr, index=expression.index, columns=expression.columns)

    raise ValueError(f"Unknown platform '{platform}'; expected 'rna_seq' or 'microarray'.")


def select_highly_variable(
    expression: pd.DataFrame, n_genes: int = 2000
) -> pd.DataFrame:
    """Keep the top-``n_genes`` most variable genes (by variance).

    This is the main lever for keeping both memory use and VAE
    training time small on modest hardware.
    """
    n_genes = min(n_genes, expression.shape[1])
    variances = expression.var(axis=0, ddof=1).sort_values(ascending=False)
    keep = variances.index[:n_genes]
    return expression.loc[:, keep]
