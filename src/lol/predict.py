"""Downstream prediction: train a classifier on latent features, evaluate it.

Deliberately conventional -- scikit-learn RandomForest by default. The
scientific interest of LOL is in the latent extraction and the
GEO-to-prediction workflow, not in the classifier itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict


@dataclass
class EvaluationReport:
    accuracy: float
    roc_auc: float | None
    confusion: np.ndarray
    classification_report: str
    labels: list = field(default_factory=list)
    macro_f1: float | None = None

    def __str__(self) -> str:  # pragma: no cover - display only
        lines = [
            f"Accuracy: {self.accuracy:.3f}",
        ]
        if self.macro_f1 is not None:
            lines.append(f"Macro-F1: {self.macro_f1:.3f}")
        if self.roc_auc is not None:
            lines.append(f"ROC-AUC:  {self.roc_auc:.3f}")
        lines.append("")
        lines.append(self.classification_report)
        return "\n".join(lines)


class LatentClassifier:
    """Train and cross-validate a classifier on latent features.

    Parameters
    ----------
    class_weight:
        Forwarded to the default ``RandomForestClassifier`` (ignored if
        you pass your own ``estimator``). Defaults to ``"balanced"`` --
        on an imbalanced cohort (e.g. 11 cases vs 74 controls), plain
        accuracy can look good from a classifier that just predicts the
        majority class every time; balanced class weights and the
        macro-F1 now reported alongside accuracy are both aimed at
        catching that rather than rewarding it.
    """

    def __init__(
        self,
        estimator=None,
        n_splits: int = 5,
        random_state: int = 0,
        class_weight: str | dict | None = "balanced",
    ):
        self.estimator = estimator or RandomForestClassifier(
            n_estimators=300, random_state=random_state, class_weight=class_weight
        )
        self.n_splits = n_splits
        self.random_state = random_state

    def evaluate(self, latents: pd.DataFrame, labels: pd.Series) -> EvaluationReport:
        """Stratified k-fold cross-validated evaluation.

        Uses cross-validated *out-of-fold* predictions for all metrics,
        so the report reflects generalization rather than training fit.
        """
        labels = labels.loc[latents.index]
        n_splits = min(self.n_splits, labels.value_counts().min())
        n_splits = max(n_splits, 2)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        x = latents.to_numpy(dtype=float)
        y = labels.to_numpy()

        y_pred = cross_val_predict(self.estimator, x, y, cv=cv)

        roc_auc = None
        unique_labels = sorted(pd.unique(y))
        if len(unique_labels) == 2:
            try:
                y_proba = cross_val_predict(
                    self.estimator, x, y, cv=cv, method="predict_proba"
                )[:, 1]
                roc_auc = roc_auc_score(y, y_proba)
            except (AttributeError, ValueError):
                roc_auc = None

        accuracy = float((y_pred == y).mean())
        report = EvaluationReport(
            accuracy=accuracy,
            roc_auc=roc_auc,
            confusion=confusion_matrix(y, y_pred, labels=unique_labels),
            classification_report=classification_report(y, y_pred),
            labels=unique_labels,
            macro_f1=float(f1_score(y, y_pred, average="macro")),
        )
        return report

    def fit(self, latents: pd.DataFrame, labels: pd.Series) -> LatentClassifier:
        labels = labels.loc[latents.index]
        self.estimator.fit(latents.to_numpy(dtype=float), labels.to_numpy())
        return self

    def predict(self, latents: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict(latents.to_numpy(dtype=float))


def nested_cross_validate(
    normalized_expression: pd.DataFrame,
    labels: pd.Series,
    n_hvg: int,
    n_latent: int,
    n_splits: int = 5,
    random_state: int = 0,
    class_weight: str | dict | None = "balanced",
    groups: pd.Series | None = None,
) -> EvaluationReport:
    """Leakage-safe cross-validation: refit HVG selection and PCA per fold.

    Unlike :meth:`LatentClassifier.evaluate`, which cross-validates only
    the classifier on top of features that were fit once on the full
    dataset, this refits highly-variable-gene selection and the PCA
    latent extractor inside each training fold, then transforms the
    held-out fold with that fold's fitted PCA. No held-out sample
    influences the features used to predict it.

    Optional group-aware splitting
    ------------------------------
    If ``groups`` is given (e.g. a patient/subject ID per sample),
    uses :class:`~sklearn.model_selection.StratifiedGroupKFold` instead
    of plain :class:`~sklearn.model_selection.StratifiedKFold`, so every
    sample from one group (patient) stays on the same side of every
    fold. This matters for repeated-measures cohorts (multiple biopsies
    per patient): without it, a model can partly learn "which patient
    is this" rather than the phenotype of interest, since a held-out
    sample's own patient may have other samples sitting in the training
    set. This was found to be a real risk on GSE66407 during
    development, where individual patients contributed up to 9
    biopsies each (see CHANGELOG).

    PCA-only for now: there's no out-of-sample transform for the VAE
    path yet (see ``Pipeline.predict``), so this can't be used for
    ``latent_method="vae"``.
    """
    # local imports to avoid a module-level circular import between
    # lol.predict and lol.latent/lol.preprocess
    from lol.latent import PCALatent
    from lol.preprocess import select_highly_variable

    labels = labels.loc[normalized_expression.index]

    if groups is not None:
        groups = groups.loc[normalized_expression.index]
        # cap n_splits by the number of *groups* in the smallest class,
        # not the number of samples -- a class with few distinct
        # patients can't support more folds than it has patients,
        # regardless of how many biopsies each contributes
        min_groups_per_class = (
            pd.DataFrame({"label": labels, "group": groups})
            .groupby("label")["group"]
            .nunique()
            .min()
        )
        n_splits = max(min(n_splits, min_groups_per_class), 2)
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_args = (np.zeros(len(labels)), labels.to_numpy(), groups.to_numpy())
    else:
        n_splits = max(min(n_splits, labels.value_counts().min()), 2)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_args = (np.zeros(len(labels)), labels.to_numpy())

    y = labels.to_numpy()
    y_pred = np.empty(len(y), dtype=y.dtype)
    y_proba = np.full(len(y), np.nan)
    unique_labels = sorted(pd.unique(y))

    for train_idx, test_idx in cv.split(*split_args):
        train_expr = normalized_expression.iloc[train_idx]
        test_expr = normalized_expression.iloc[test_idx]
        train_labels = labels.iloc[train_idx]

        hvg_genes = select_highly_variable(train_expr, n_genes=n_hvg).columns
        train_reduced = train_expr[hvg_genes]
        test_reduced = test_expr[hvg_genes]

        extractor = PCALatent(n_components=n_latent, random_state=random_state)
        train_latents = extractor.fit_transform(train_reduced)
        test_latents_arr = extractor.model.transform(test_reduced.to_numpy(dtype=float))

        clf = RandomForestClassifier(
            n_estimators=300, random_state=random_state, class_weight=class_weight
        )
        clf.fit(train_latents.to_numpy(dtype=float), train_labels.to_numpy())

        y_pred[test_idx] = clf.predict(test_latents_arr)
        if len(unique_labels) == 2 and hasattr(clf, "predict_proba"):
            y_proba[test_idx] = clf.predict_proba(test_latents_arr)[:, 1]

    roc_auc = None
    if len(unique_labels) == 2 and not np.isnan(y_proba).any():
        try:
            roc_auc = roc_auc_score(y, y_proba)
        except ValueError:
            roc_auc = None

    accuracy = float((y_pred == y).mean())
    return EvaluationReport(
        accuracy=accuracy,
        roc_auc=roc_auc,
        confusion=confusion_matrix(y, y_pred, labels=unique_labels),
        classification_report=classification_report(y, y_pred),
        labels=unique_labels,
        macro_f1=float(f1_score(y, y_pred, average="macro")),
    )


@dataclass
class PermutationTestResult:
    """Result of comparing a real evaluation score against a null
    distribution built from shuffled labels.

    Generalizes the ad-hoc permutation-test scripts used during LOL's
    own development (see CHANGELOG) into a first-class feature -- the
    permutation test is what actually made GSE16161, GSE22619, and
    GSE75214's results trustworthy rather than merely plausible, so it
    belongs in the package rather than in one-off scripts.
    """

    metric_name: str
    real_score: float
    null_scores: np.ndarray
    n_permutations: int

    @property
    def p_value_estimate(self) -> float:
        """(# null scores >= real score + 1) / (n_permutations + 1).

        The "+1" in both numerator and denominator accounts for the
        real (unshuffled) observation itself as one more draw under
        the null -- this is the standard, slightly conservative way to
        avoid ever reporting p=0.0 from a finite number of permutations.
        With ``n_permutations`` shuffles, the smallest reportable
        p-value is 1/(n_permutations + 1); use more permutations
        (e.g. 100-500) to get a tighter estimate if this one lands
        close to your significance threshold.
        """
        n_at_or_above = int((self.null_scores >= self.real_score).sum())
        return (n_at_or_above + 1) / (self.n_permutations + 1)

    def __str__(self) -> str:  # pragma: no cover - display only
        null = self.null_scores
        return (
            f"Permutation test on {self.metric_name}\n"
            f"  Real score:        {self.real_score:.3f}\n"
            f"  Null distribution: mean={null.mean():.3f}, "
            f"min={null.min():.3f}, max={null.max():.3f} "
            f"({self.n_permutations} shuffles)\n"
            f"  p-value estimate:  {self.p_value_estimate:.3f}\n"
            f"  (smallest reportable p-value at this permutation count: "
            f"{1 / (self.n_permutations + 1):.3f})"
        )


def permutation_test(
    normalized_expression: pd.DataFrame,
    labels: pd.Series,
    n_hvg: int,
    n_latent: int,
    metric: str = "macro_f1",
    n_permutations: int = 30,
    n_splits: int = 5,
    random_state: int = 0,
    class_weight: str | dict | None = "balanced",
    groups: pd.Series | None = None,
) -> PermutationTestResult:
    """Compare the real nested-CV score against a null of shuffled labels.

    Runs :func:`nested_cross_validate` once on the real labels, then
    ``n_permutations`` more times on randomly shuffled labels, and
    reports where the real score falls relative to that null
    distribution.

    Parameters
    ----------
    metric:
        Which field of :class:`EvaluationReport` to compare --
        ``"macro_f1"`` (default), ``"accuracy"``, or ``"roc_auc"``.
        ``macro_f1`` is the recommended default: on an imbalanced
        cohort, accuracy and ROC-AUC can both look deceptively strong
        even when a classifier is just leaning on the majority class
        (this is exactly what happened on GSE75214 during development
        -- see CHANGELOG). Macro-F1 is far less forgiving of that.
    n_permutations:
        More permutations narrow the smallest reportable p-value
        (``1 / (n_permutations + 1)``) and reduce noise in the null
        estimate, at the cost of runtime -- each permutation reruns
        the full nested cross-validation. 30 is a reasonable default
        for a quick check; use 100+ if you need a tighter estimate
        near a significance threshold.
    groups:
        Passed through to :func:`nested_cross_validate` for
        leakage-safe splitting. Just as importantly, when ``groups``
        is given, shuffling happens **at the group level**: each
        group's (e.g. patient's) real label is looked up once, the
        group-to-label mapping is shuffled, and that shuffled label is
        then applied to every sample belonging to that group. This is
        deliberately different from shuffling every sample's label
        independently -- a patient has one true diagnosis, not a
        separate coin-flip per biopsy, so the null distribution needs
        to reflect that same structure or it won't be a fair null.
        Raises ``ValueError`` if any group has more than one distinct
        label in the real data (which would mean labels aren't
        actually constant within a group, e.g. patient).

    PCA-only, like :func:`nested_cross_validate` -- there's no
    out-of-sample transform for the VAE path yet.
    """
    real_report = nested_cross_validate(
        normalized_expression=normalized_expression,
        labels=labels,
        n_hvg=n_hvg,
        n_latent=n_latent,
        n_splits=n_splits,
        random_state=random_state,
        class_weight=class_weight,
        groups=groups,
    )
    real_score = getattr(real_report, metric)
    if real_score is None:
        raise ValueError(
            f"metric={metric!r} was None on the real (unshuffled) data "
            "(e.g. roc_auc is only computed for binary labels) -- "
            "choose a different metric."
        )

    rng = np.random.default_rng(random_state)

    if groups is not None:
        groups = groups.loc[labels.index]
        group_label_counts = labels.groupby(groups).nunique()
        if (group_label_counts > 1).any():
            bad_groups = group_label_counts[group_label_counts > 1].index.tolist()
            raise ValueError(
                f"groups={bad_groups[:5]}{'...' if len(bad_groups) > 5 else ''} "
                "each have more than one distinct label -- group-level "
                "permutation requires one consistent label per group "
                "(e.g. one diagnosis per patient)."
            )
        per_group_labels = labels.groupby(groups).first()
        unique_groups = per_group_labels.index.to_numpy()

    null_scores = np.empty(n_permutations)
    for i in range(n_permutations):
        if groups is not None:
            shuffled_group_labels = pd.Series(
                rng.permutation(per_group_labels.to_numpy()),
                index=unique_groups,
            )
            shuffled = groups.map(shuffled_group_labels)
            shuffled.index = labels.index
        else:
            shuffled = pd.Series(
                rng.permutation(labels.to_numpy()), index=labels.index, name=labels.name
            )
        report_i = nested_cross_validate(
            normalized_expression=normalized_expression,
            labels=shuffled,
            n_hvg=n_hvg,
            n_latent=n_latent,
            n_splits=n_splits,
            random_state=random_state + i + 1,
            class_weight=class_weight,
            groups=groups,
        )
        null_scores[i] = getattr(report_i, metric)

    return PermutationTestResult(
        metric_name=metric,
        real_score=real_score,
        null_scores=null_scores,
        n_permutations=n_permutations,
    )


@dataclass
class LeakageAudit:
    """Naive vs. nested evaluation, and the gap between them (Delta-LII).

    Delta-LII ("leakage inflation index") is naive_score - nested_score
    for a given metric: how much fitting features on the whole dataset
    before cross-validation inflates the reported score relative to
    refitting them inside each fold. A value near zero means the naive
    shortcut happened not to matter for this dataset (as on GSE16161);
    a large positive value means it did (as on GSE66407, where nested
    accuracy was 0.651 vs. 0.721 naive). Either outcome is a legitimate
    finding -- Delta-LII is a diagnostic, not something to be minimized
    by construction.
    """

    naive: EvaluationReport
    nested: EvaluationReport

    @property
    def delta_accuracy(self) -> float:
        return self.naive.accuracy - self.nested.accuracy

    @property
    def delta_macro_f1(self) -> float | None:
        if self.naive.macro_f1 is None or self.nested.macro_f1 is None:
            return None
        return self.naive.macro_f1 - self.nested.macro_f1

    @property
    def delta_roc_auc(self) -> float | None:
        if self.naive.roc_auc is None or self.nested.roc_auc is None:
            return None
        return self.naive.roc_auc - self.nested.roc_auc

    def __str__(self) -> str:  # pragma: no cover - display only
        lines = ["Leakage Inflation Index (naive - nested)"]
        lines.append(
            f"  Accuracy:  naive={self.naive.accuracy:.3f}  "
            f"nested={self.nested.accuracy:.3f}  "
            f"\u0394LII={self.delta_accuracy:+.3f}"
        )
        if self.delta_macro_f1 is not None:
            lines.append(
                f"  Macro-F1:  naive={self.naive.macro_f1:.3f}  "
                f"nested={self.nested.macro_f1:.3f}  "
                f"\u0394LII={self.delta_macro_f1:+.3f}"
            )
        if self.delta_roc_auc is not None:
            lines.append(
                f"  ROC-AUC:   naive={self.naive.roc_auc:.3f}  "
                f"nested={self.nested.roc_auc:.3f}  "
                f"\u0394LII={self.delta_roc_auc:+.3f}"
            )
        max_delta = max(
            abs(d) for d in (self.delta_accuracy, self.delta_macro_f1, self.delta_roc_auc)
            if d is not None
        )
        if max_delta > 0.05:
            lines.append(
                "  Notable gap: the naive (non-nested) score is meaningfully "
                "higher than the leakage-safe one. Report the nested score."
            )
        else:
            lines.append(
                "  Small gap: naive and nested scores agree closely here -- "
                "not evidence that nested evaluation is unnecessary in "
                "general, only that it didn't change the answer this time."
            )
        return "\n".join(lines)
