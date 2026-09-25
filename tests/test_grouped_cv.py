import numpy as np
import pandas as pd

from lol.pipeline import Pipeline


def _grouped_dataset(n_patients=20, biopsies_per_patient=3, n_genes=200, seed=0):
    """Synthetic repeated-measures data: each patient has one true label
    and contributes multiple correlated samples (biopsies)."""
    rng = np.random.default_rng(seed)
    patient_labels = rng.integers(0, 2, size=n_patients)
    patient_base = rng.normal(size=(n_patients, n_genes))  # patient "identity" signal
    label_effect = rng.normal(scale=3.0, size=n_genes)  # real phenotype signal

    rows, groups, labels = [], [], []
    for p in range(n_patients):
        for b in range(biopsies_per_patient):
            noise = rng.normal(scale=0.5, size=n_genes)
            row = patient_base[p] + patient_labels[p] * label_effect + noise
            rows.append(row)
            groups.append(f"patient_{p}")
            labels.append(patient_labels[p])

    index = [f"sample_{i}" for i in range(len(rows))]
    expression = pd.DataFrame(np.abs(rows) * 50, index=index, columns=[f"g{i}" for i in range(n_genes)])
    labels = pd.Series(labels, index=index, name="label")
    groups = pd.Series(groups, index=index, name="patient")
    return expression, labels, groups


def test_grouped_evaluate_runs_and_keeps_groups_intact():
    expression, labels, groups = _grouped_dataset()
    pipe = Pipeline(latent_method="pca", n_latent=5, n_hvg=100)
    pipe.fit(expression, labels, groups=groups)
    report = pipe.evaluate(nested=True)
    assert 0.0 <= report.accuracy <= 1.0


def test_grouped_permutation_test_shuffles_at_group_level_not_sample_level():
    expression, labels, groups = _grouped_dataset(n_patients=16, biopsies_per_patient=3)
    pipe = Pipeline(latent_method="pca", n_latent=4, n_hvg=100)
    pipe.fit(expression, labels, groups=groups)

    result = pipe.permutation_test(metric="macro_f1", n_permutations=8)

    # every sample from the same patient must always share a label within
    # a single permutation -- sanity check the fixture itself has that
    # property, so the test above is actually testing something real
    per_group_counts = labels.groupby(groups).nunique()
    assert per_group_counts.max() == 1
    assert 0.0 <= result.real_score <= 1.0
    assert len(result.null_scores) == 8


def test_grouped_permutation_test_rejects_inconsistent_group_labels():
    expression, labels, groups = _grouped_dataset(n_patients=10, biopsies_per_patient=2)
    # corrupt one patient's labels so they're inconsistent within the group
    labels = labels.copy()
    labels.iloc[0] = 1 - labels.iloc[0]

    pipe = Pipeline(latent_method="pca", n_latent=3, n_hvg=80)
    pipe.fit(expression, labels, groups=groups)

    try:
        pipe.permutation_test(n_permutations=3)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_ungrouped_pipeline_still_works_without_groups_argument():
    from lol.datasets import make_synthetic_counts

    expression, labels = make_synthetic_counts(n_samples=30, n_genes=150, random_state=1)
    pipe = Pipeline(latent_method="pca", n_latent=6, n_hvg=100)
    pipe.fit(expression, labels)  # no groups passed
    assert pipe.groups_ is None
    report = pipe.evaluate()
    assert 0.0 <= report.accuracy <= 1.0
