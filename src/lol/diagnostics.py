"""Confound diagnostics: how much of the latent space associates with
each metadata covariate (tissue, batch, patient, phenotype, ...).

This is a deliberately simplified relative of Principal Variance
Component Analysis (PVCA). Full PVCA fits a mixed-effects model per
principal component with all covariates simultaneously and partitions
each PC's variance among them plus a residual, so the shares sum to
100% per PC. That requires a mixed-model fitting library and is a much
heavier piece of statistics than this package otherwise needs.

What's implemented here instead: for each covariate, independently,
the fraction of variance it explains in each latent dimension
(R-squared from a one-way ANOVA for categorical covariates, or squared
correlation for continuous ones), averaged across latent dimensions
and weighted by each dimension's own share of explained variance (from
PCA) where available. Because covariates are handled one at a time
rather than jointly, their shares are not adjusted for overlap with
each other and will not generally sum to 100% -- two correlated
covariates (e.g. tissue and diagnosis, as in the GSE75214/GSE66407
case studies) can each show a substantial, overlapping association.
Read this as "how much does X alone associate with the latent space",
not "how much of the latent space is uniquely attributable to X". For
a rigorous joint decomposition, use a proper PVCA/mixed-model
implementation (e.g. the R pvca package) on the same latent matrix.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

_CATEGORICAL_MAX_UNIQUE_FRACTION = 0.5


def _looks_categorical(series: pd.Series) -> bool:
    if series.dtype == object or series.dtype.name == "category":
        return True
    if pd.api.types.is_bool_dtype(series):
        return True
    if pd.api.types.is_numeric_dtype(series):
        return series.nunique(dropna=True) <= max(2, len(series) * _CATEGORICAL_MAX_UNIQUE_FRACTION * 0.2)
    return True


def _r_squared_categorical(values: np.ndarray, covariate: pd.Series) -> float:
    df = pd.DataFrame({"value": values, "group": covariate.to_numpy()}).dropna()
    if df["group"].nunique() < 2 or len(df) < 3:
        return float("nan")
    grand_mean = df["value"].mean()
    ss_total = ((df["value"] - grand_mean) ** 2).sum()
    if ss_total == 0:
        return float("nan")
    group_stats = df.groupby("group")["value"].agg(["mean", "count"])
    ss_between = (group_stats["count"] * (group_stats["mean"] - grand_mean) ** 2).sum()
    return float(ss_between / ss_total)


def _r_squared_continuous(values: np.ndarray, covariate: pd.Series) -> float:
    df = pd.DataFrame({"value": values, "cov": pd.to_numeric(covariate, errors="coerce")}).dropna()
    if len(df) < 3 or df["cov"].std() == 0 or df["value"].std() == 0:
        return float("nan")
    r = np.corrcoef(df["value"], df["cov"])[0, 1]
    return float(r**2)


@dataclass
class PVCALiteResult:
    """Per-covariate variance association, sorted descending.

    ``shares`` maps covariate name to its weighted-average R-squared
    against the latent space (see module docstring for what this does
    and does not mean). ``n_dims_used`` and ``weighted`` record how the
    average across latent dimensions was computed, for transparency.
    """

    shares: dict[str, float]
    n_dims_used: int
    weighted: bool

    def __str__(self) -> str:  # pragma: no cover - display only
        lines = [
            ("PVCA-lite: variance association per covariate "
             f"({self.n_dims_used} latent dims, "
             f"{'PCA-eigenvalue-weighted' if self.weighted else 'unweighted'} average)"),
        ]
        for name, share in sorted(self.shares.items(), key=lambda kv: -kv[1] if not np.isnan(kv[1]) else 1):
            if np.isnan(share):
                lines.append(f"  {name}: insufficient variation to assess")
            else:
                lines.append(f"  {name}: {share:.1%}")
        lines.append(
            "  (Shares are independent per covariate and will not sum to "
            "100% -- see PVCALiteResult / lol.diagnostics docstring.)"
        )
        return "\n".join(lines)


def pvca_lite(
    latents: pd.DataFrame,
    metadata: pd.DataFrame,
    explained_variance_ratio: np.ndarray | None = None,
) -> PVCALiteResult:
    """Estimate how much each metadata covariate associates with the latent space.

    Parameters
    ----------
    latents:
        Samples x latent-dims DataFrame, e.g. ``pipe.latents_``.
    metadata:
        Samples x covariates DataFrame (tissue, batch, phenotype,
        patient ID, age, ...), aligned to ``latents`` by index. Columns
        that look categorical (object/category/bool dtype, or numeric
        with few unique values) use a one-way-ANOVA R-squared; other
        numeric columns use squared correlation.
    explained_variance_ratio:
        Per-latent-dimension explained variance ratio, e.g.
        ``PCALatent.explained_variance_ratio_``. When given, the
        per-dimension R-squared values are averaged weighted by this
        (so a covariate that only associates with a minor, low-variance
        PC scores lower than one associating with PC1). When omitted
        (e.g. for the VAE path, which has no such ratio), dimensions
        are weighted equally -- a real approximation, not a refinement.
    """
    metadata = metadata.loc[latents.index]
    n_dims = latents.shape[1]

    if explained_variance_ratio is not None:
        weights = np.asarray(explained_variance_ratio, dtype=float)[:n_dims]
        weights = weights / weights.sum()
        weighted = True
    else:
        weights = np.full(n_dims, 1.0 / n_dims)
        weighted = False

    shares: dict[str, float] = {}
    for col in metadata.columns:
        covariate = metadata[col]
        r2_per_dim = np.empty(n_dims)
        for j in range(n_dims):
            pc_values = latents.iloc[:, j].to_numpy()
            if _looks_categorical(covariate):
                r2_per_dim[j] = _r_squared_categorical(pc_values, covariate)
            else:
                r2_per_dim[j] = _r_squared_continuous(pc_values, covariate)

        valid = ~np.isnan(r2_per_dim)
        if not valid.any():
            shares[col] = float("nan")
            continue
        w = weights[valid] / weights[valid].sum()
        shares[col] = float(np.sum(r2_per_dim[valid] * w))

    return PVCALiteResult(shares=shares, n_dims_used=n_dims, weighted=weighted)
