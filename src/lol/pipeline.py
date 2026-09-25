"""The end-to-end LOL pipeline: preprocess -> latent extraction -> predict."""

from __future__ import annotations

import pandas as pd

from lol.diagnostics import PVCALiteResult, pvca_lite
from lol.latent import PCALatent, VAELatent
from lol.predict import (
    EvaluationReport,
    LatentClassifier,
    LeakageAudit,
    PermutationTestResult,
    nested_cross_validate,
    permutation_test,
)
from lol.preprocess import normalize, select_highly_variable


class Pipeline:
    """Fit a latent representation and a downstream classifier.

    Parameters
    ----------
    latent_method:
        ``"pca"`` (default, instant) or ``"vae"`` (trains a small
        PyTorch VAE; slower but potentially better at small latent
        dimensions -- see the docs for the trade-off).
    n_latent:
        Number of latent dimensions.
    n_hvg:
        Number of highly-variable genes to keep before latent
        extraction. The main lever for memory/speed on a laptop.
    platform:
        ``"rna_seq"``, ``"microarray"``, or ``None`` to auto-detect.
    vae_kwargs:
        Extra keyword arguments forwarded to :class:`~lol.latent.VAELatent`
        when ``latent_method="vae"`` (e.g. ``epochs``, ``n_hidden``).
    """

    def __init__(
        self,
        latent_method: str = "pca",
        n_latent: int = 20,
        n_hvg: int = 2000,
        platform: str | None = None,
        random_state: int = 0,
        vae_kwargs: dict | None = None,
        class_weight: str | dict | None = "balanced",
    ):
        if latent_method not in ("pca", "vae"):
            raise ValueError("latent_method must be 'pca' or 'vae'")
        self.latent_method = latent_method
        self.n_latent = n_latent
        self.n_hvg = n_hvg
        self.platform = platform
        self.random_state = random_state
        self.vae_kwargs = vae_kwargs or {}
        self.class_weight = class_weight

        self._extractor = None
        self._normalized_: pd.DataFrame | None = None
        self.latents_: pd.DataFrame | None = None
        self.classifier_: LatentClassifier | None = None
        self.labels_: pd.Series | None = None
        self.groups_: pd.Series | None = None
        self.selected_genes_: pd.Index | None = None

    def _make_extractor(self):
        if self.latent_method == "pca":
            return PCALatent(n_components=self.n_latent, random_state=self.random_state)
        return VAELatent(
            n_latent=self.n_latent, random_state=self.random_state, **self.vae_kwargs
        )

    def fit(
        self,
        expression: pd.DataFrame,
        labels: pd.Series,
        groups: pd.Series | None = None,
    ) -> Pipeline:
        """Preprocess, extract latent features, and fit a classifier.

        Stores the normalized (but not yet gene-selected) matrix as
        well, so :meth:`evaluate` can rerun HVG selection and latent
        extraction per cross-validation fold rather than reusing
        features that were fit on the full dataset -- see the
        docstring of :meth:`evaluate` for why that distinction matters.

        Parameters
        ----------
        groups:
            Optional per-sample group/subject ID (e.g. patient ID),
            aligned to ``expression``'s index. If given, it's used by
            :meth:`evaluate` and :meth:`permutation_test` to keep every
            sample from one subject on the same side of every
            cross-validation fold -- important for repeated-measures
            cohorts (multiple biopsies/samples per patient), where
            without this a model can partly learn "which patient is
            this" rather than the phenotype of interest. See their
            docstrings for details.
        """
        self._normalized_ = normalize(expression, platform=self.platform)
        reduced = select_highly_variable(self._normalized_, n_genes=self.n_hvg)
        self.selected_genes_ = reduced.columns

        self._extractor = self._make_extractor()
        self.latents_ = self._extractor.fit_transform(reduced)
        self.labels_ = labels.loc[self.latents_.index]
        self.groups_ = groups.loc[self.latents_.index] if groups is not None else None

        self.classifier_ = LatentClassifier(
            random_state=self.random_state, class_weight=self.class_weight
        )
        self.classifier_.fit(self.latents_, self.labels_)
        return self

    def evaluate(self, nested: bool = True) -> EvaluationReport:
        """Cross-validated evaluation report on the fitted data.

        Call :meth:`fit` first.

        Parameters
        ----------
        nested:
            If ``True`` (default, and strongly recommended), HVG
            selection and the latent extractor are **refit inside each
            cross-validation fold**, so a held-out sample never
            influences the features used to predict it. This is the
            methodologically correct way to evaluate a pipeline like
            this one, and matters most exactly when it's tempting to
            skip it -- small cohorts, where a handful of samples can
            shift both HVG selection and a PCA/VAE fit noticeably.

            If ``False``, evaluation reuses the single latent
            representation fit on the *entire* dataset in :meth:`fit`,
            and only cross-validates the classifier on top of it. This
            is faster and was LOL's original default, but it lets
            information from held-out samples leak into the features
            (unsupervised leakage, milder than a label leak, but real)
            and can inflate reported accuracy -- this is exactly what
            produced a suspicious 1.000 accuracy/ROC-AUC on an 18-sample
            real GEO cohort during development, which is why the
            default changed. Kept only for quick iteration or very
            large cohorts where nested refitting is expensive.

        Notes
        -----
        Nested evaluation currently only supports ``latent_method="pca"``,
        because the VAE has no out-of-sample transform yet (see
        :meth:`predict`). Calling this with ``nested=True`` on a
        VAE-based pipeline raises ``NotImplementedError``; pass
        ``nested=False`` to get the older (leakier) VAE evaluation in
        the meantime.
        """
        if self.latents_ is None or self.labels_ is None:
            raise RuntimeError("Call fit() before evaluate().")

        if not nested:
            evaluator = LatentClassifier(
                random_state=self.random_state, class_weight=self.class_weight
            )
            return evaluator.evaluate(self.latents_, self.labels_)

        if self.latent_method != "pca":
            raise NotImplementedError(
                "Nested (leakage-safe) evaluation currently only supports "
                "latent_method='pca', since the VAE has no out-of-sample "
                "transform yet. Pass evaluate(nested=False) for the "
                "original (leakier) behavior on a VAE pipeline."
            )

        return nested_cross_validate(
            normalized_expression=self._normalized_,
            labels=self.labels_,
            n_hvg=self.n_hvg,
            n_latent=self.n_latent,
            random_state=self.random_state,
            class_weight=self.class_weight,
            groups=self.groups_,
        )

    def permutation_test(
        self,
        metric: str = "macro_f1",
        n_permutations: int = 30,
    ) -> PermutationTestResult:
        """Check whether :meth:`evaluate`'s score is distinguishable from chance.

        Reruns the full leakage-safe pipeline (HVG selection + PCA +
        classifier, refit per fold) on ``n_permutations`` randomly
        shuffled copies of the labels, and reports where the real
        score falls relative to that null distribution.

        Call :meth:`fit` first. PCA-only, like the nested evaluation
        this builds on -- see :meth:`evaluate`.

        This is not a formality to run once and forget: a strong
        accuracy or ROC-AUC can arise from a confound (batch effects,
        a tissue-type mismatch between groups, etc.) as easily as from
        real biology, and the only way to tell the difference is to
        check whether shuffled labels could plausibly produce a
        similar score. Treat a result you haven't permutation-tested
        as provisional.
        """
        if self._normalized_ is None or self.labels_ is None:
            raise RuntimeError("Call fit() before permutation_test().")
        if self.latent_method != "pca":
            raise NotImplementedError(
                "permutation_test currently only supports latent_method='pca' "
                "(same limitation as evaluate(nested=True) -- see its docstring)."
            )
        return permutation_test(
            normalized_expression=self._normalized_,
            labels=self.labels_,
            n_hvg=self.n_hvg,
            n_latent=self.n_latent,
            metric=metric,
            n_permutations=n_permutations,
            random_state=self.random_state,
            class_weight=self.class_weight,
            groups=self.groups_,
        )

    def leakage_audit(self) -> LeakageAudit:
        """Compare naive vs. leakage-safe evaluation (the Delta-LII diagnostic).

        Runs both ``evaluate(nested=False)`` (features fit once on the
        full dataset) and ``evaluate(nested=True)`` (features refit
        per fold) and returns both reports plus their difference per
        metric. Call :meth:`fit` first.

        This makes explicit, as a named and reported quantity, exactly
        the comparison that caught the fold-safety leak on GSE16161
        and demonstrated its real-world impact on GSE66407 during
        development (see ``docs/case-studies.md``) -- rather than
        leaving it as something a user has to think to check by
        calling ``evaluate()`` twice themselves.
        """
        if self.latents_ is None or self.labels_ is None:
            raise RuntimeError("Call fit() before leakage_audit().")
        naive = self.evaluate(nested=False)
        nested = self.evaluate(nested=True)
        return LeakageAudit(naive=naive, nested=nested)

    def confound_diagnostic(self, metadata: pd.DataFrame) -> PVCALiteResult:
        """Estimate how much each metadata covariate associates with the latent space.

        A simplified (per-covariate, not jointly-adjusted) relative of
        Principal Variance Component Analysis -- see
        :func:`lol.diagnostics.pvca_lite` for exactly what this does
        and does not claim. Call :meth:`fit` first.

        Parameters
        ----------
        metadata:
            Samples x covariates DataFrame (tissue, batch, patient ID,
            the phenotype label itself, age, ...), aligned to the
            samples ``fit`` was called on. Including the phenotype
            label as one of the covariates is often informative: a
            technical covariate (e.g. tissue) scoring as high or
            higher than the phenotype itself is a warning sign worth
            investigating before trusting :meth:`evaluate`'s score --
            this is exactly the check that would have caught the
            GSE75214/GSE66407 tissue confounds earlier, had it existed then.
        """
        if self.latents_ is None:
            raise RuntimeError("Call fit() before confound_diagnostic().")
        weights = (
            self._extractor.explained_variance_ratio_
            if isinstance(self._extractor, PCALatent)
            else None
        )
        return pvca_lite(self.latents_, metadata, explained_variance_ratio=weights)

    def gene_loadings(self, component: int = 0) -> pd.Series:
        """Which highly-variable genes drive one PCA latent dimension.

        Returns the fitted PCA loadings for ``component`` (0-indexed;
        ``0`` = PC1), indexed by gene and sorted ascending -- the genes
        with the largest-magnitude loadings (``.head(n)`` for the most
        negative, ``.tail(n)`` for the most positive) are the ones most
        strongly associated with that latent dimension. Call :meth:`fit`
        first. PCA-only: there is no equivalent single-gene attribution
        for the VAE path.
        """
        if self.latents_ is None:
            raise RuntimeError("Call fit() before gene_loadings().")
        if not isinstance(self._extractor, PCALatent):
            raise NotImplementedError(
                "gene_loadings() is only available for latent_method='pca'; "
                "the VAE path has no direct per-gene loading equivalent."
            )
        return self._extractor.gene_loadings(component)

    def predict(self, expression: pd.DataFrame):
        """Predict phenotype labels for new samples using the fitted pipeline.

        New samples are restricted to the same genes selected as
        highly-variable during :meth:`fit` (missing genes are filled
        with 0 post-normalization) -- the extractor was fit on that
        exact gene set and expects the same features back.
        """
        if self._extractor is None or self.classifier_ is None:
            raise RuntimeError("Call fit() before predict().")
        normalized = normalize(expression, platform=self.platform)
        aligned = normalized.reindex(columns=self.selected_genes_, fill_value=0.0)
        latents = self._transform_new(aligned)
        return self.classifier_.predict(latents)

    def _transform_new(self, normalized: pd.DataFrame) -> pd.DataFrame:
        if isinstance(self._extractor, PCALatent):

            x = normalized.to_numpy(dtype=float)
            latents = self._extractor.model.transform(x)
            columns = [f"pc{i + 1}" for i in range(latents.shape[1])]
            return pd.DataFrame(latents, index=normalized.index, columns=columns)
        raise NotImplementedError(
            "Transforming new samples through a fitted VAE is not yet implemented; "
            "retrain on the combined dataset for now. Tracked as a roadmap item."
        )
