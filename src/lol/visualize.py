"""Plotting helpers for latent representations and permutation tests.

Kept deliberately small: two plot types, both directly tied to what
the rest of the package already computes (Pipeline.latents_ and
PermutationTestResult), not a general-purpose plotting library.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lol.diagnostics import PVCALiteResult
from lol.predict import LeakageAudit, PermutationTestResult


def plot_latent_space(
    latents: pd.DataFrame,
    labels: pd.Series,
    dims: tuple[int, int] = (0, 1),
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Axes:
    """Scatter the first two (or chosen) latent dimensions, colored by label.

    Parameters
    ----------
    latents:
        A samples x latent-dims DataFrame, e.g. ``pipe.latents_``.
    labels:
        Phenotype label per sample, aligned to ``latents``' index.
    dims:
        Which two latent-dimension column positions to plot (0-indexed).
    ax:
        Existing matplotlib axes to draw into; a new figure/axes is
        created if omitted.

    This is a diagnostic plot, not a claim of separation -- a clean
    visual split here does not substitute for
    :meth:`~lol.pipeline.Pipeline.permutation_test`, and a lack of
    visible separation in 2D does not mean there is no signal in the
    full latent space. Pair this with the permutation test result
    rather than in place of it.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4.5))

    labels = labels.loc[latents.index]
    x_col, y_col = latents.columns[dims[0]], latents.columns[dims[1]]

    for label_value in sorted(labels.unique()):
        mask = labels == label_value
        ax.scatter(
            latents.loc[mask, x_col],
            latents.loc[mask, y_col],
            label=str(label_value),
            alpha=0.75,
            edgecolors="white",
            linewidths=0.4,
        )

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.legend(frameon=False, title=labels.name)
    if title:
        ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def plot_permutation_null(
    result: PermutationTestResult,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Axes:
    """Histogram of the permutation-test null distribution, with the
    real (unshuffled) score marked.

    Parameters
    ----------
    result:
        The object returned by
        :meth:`~lol.pipeline.Pipeline.permutation_test`.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4.5))

    ax.hist(
        result.null_scores,
        bins=min(15, max(5, result.n_permutations // 2)),
        color="#9aa5b1",
        edgecolor="white",
        label=f"null (n={result.n_permutations} shuffles)",
    )
    ax.axvline(
        result.real_score,
        color="#c0392b",
        linewidth=2,
        label=f"real score = {result.real_score:.3f}",
    )
    ax.set_xlabel(result.metric_name)
    ax.set_ylabel("count")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(title or f"p \u2248 {result.p_value_estimate:.3f}")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def plot_leakage_audit(
    audit: LeakageAudit,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Axes:
    """Grouped bar chart of naive vs. nested scores per metric (Delta-LII).

    Parameters
    ----------
    audit:
        The object returned by
        :meth:`~lol.pipeline.Pipeline.leakage_audit`.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4.5))

    metrics, naive_vals, nested_vals = [], [], []
    for name, naive_val, nested_val in (
        ("accuracy", audit.naive.accuracy, audit.nested.accuracy),
        ("macro-F1", audit.naive.macro_f1, audit.nested.macro_f1),
        ("ROC-AUC", audit.naive.roc_auc, audit.nested.roc_auc),
    ):
        if naive_val is not None and nested_val is not None:
            metrics.append(name)
            naive_vals.append(naive_val)
            nested_vals.append(nested_val)

    x = range(len(metrics))
    width = 0.35
    ax.bar([i - width / 2 for i in x], naive_vals, width, label="naive", color="#c0392b")
    ax.bar([i + width / 2 for i in x], nested_vals, width, label="nested", color="#2c7fb8")
    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False)
    ax.set_title(title or "Leakage Inflation Index (naive vs. nested)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def plot_pvca_lite(
    result: PVCALiteResult,
    ax: plt.Axes | None = None,
    title: str | None = None,
    highlight: str | None = None,
) -> plt.Axes:
    """Horizontal bar chart of per-covariate variance association (PVCA-lite).

    Parameters
    ----------
    result:
        The object returned by
        :meth:`~lol.pipeline.Pipeline.confound_diagnostic`.
    highlight:
        Optional covariate name to color differently (e.g. the
        phenotype label itself), so it's visually easy to compare
        against technical covariates like tissue or batch.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 0.5 * len(result.shares) + 1.5))

    items = sorted(
        ((k, v) for k, v in result.shares.items() if not np.isnan(v)),
        key=lambda kv: kv[1],
    )
    names = [k for k, _ in items]
    values = [v for _, v in items]
    colors = ["#c0392b" if highlight and n == highlight else "#7f8c8d" for n in names]

    ax.barh(names, values, color=colors, edgecolor="white")
    ax.set_xlabel("variance association (R\u00b2, marginal)")
    ax.set_xlim(0, max(1.0, max(values, default=1.0) * 1.1))
    ax.set_title(title or "PVCA-lite: confound diagnostic")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax
