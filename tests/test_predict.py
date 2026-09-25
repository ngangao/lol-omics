import numpy as np
import pandas as pd

from lol.predict import LatentClassifier


def _separable_latents(n_per_class=15, seed=0):
    rng = np.random.default_rng(seed)
    class0 = rng.normal(loc=-3, scale=0.5, size=(n_per_class, 2))
    class1 = rng.normal(loc=3, scale=0.5, size=(n_per_class, 2))
    x = np.vstack([class0, class1])
    y = np.array([0] * n_per_class + [1] * n_per_class)
    index = [f"s{i}" for i in range(len(y))]
    latents = pd.DataFrame(x, index=index, columns=["z1", "z2"])
    labels = pd.Series(y, index=index, name="label")
    return latents, labels


def test_evaluate_recovers_easily_separable_classes():
    latents, labels = _separable_latents()
    clf = LatentClassifier(n_splits=5, random_state=0)
    report = clf.evaluate(latents, labels)
    assert report.accuracy > 0.85
    assert report.roc_auc is not None
    assert report.roc_auc > 0.85


def test_fit_predict_roundtrip():
    latents, labels = _separable_latents()
    clf = LatentClassifier(random_state=0)
    clf.fit(latents, labels)
    preds = clf.predict(latents)
    assert preds.shape == (len(labels),)


def test_evaluate_handles_small_class_counts_without_crashing():
    latents = pd.DataFrame(
        np.random.default_rng(0).normal(size=(6, 3)),
        index=[f"s{i}" for i in range(6)],
        columns=["z1", "z2", "z3"],
    )
    labels = pd.Series([0, 0, 1, 1, 0, 1], index=latents.index, name="label")
    clf = LatentClassifier(n_splits=5, random_state=0)
    report = clf.evaluate(latents, labels)
    assert 0.0 <= report.accuracy <= 1.0
