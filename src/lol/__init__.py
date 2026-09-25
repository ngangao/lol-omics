"""LOL: Latent Omics Learning.

A small, documented pipeline for turning GEO gene expression data
(bulk RNA-seq or microarray) into low-dimensional latent representations,
and using those representations to predict a phenotype of interest.

Typical usage:

    from lol import Pipeline

    pipe = Pipeline(latent_method="pca", n_latent=20)
    pipe.fit(expression_df, labels)
    report = pipe.evaluate()
    perm_result = pipe.permutation_test()  # is that score distinguishable from chance?

See the README and docs/ for a full walkthrough, including how to go
straight from a GEO accession to a trained model.
"""

from lol.diagnostics import PVCALiteResult
from lol.latent import PCALatent, VAELatent
from lol.pipeline import Pipeline
from lol.predict import EvaluationReport, LeakageAudit, PermutationTestResult
from lol.preprocess import detect_platform, normalize, select_highly_variable

__all__ = [
    "EvaluationReport",
    "LeakageAudit",
    "PCALatent",
    "PVCALiteResult",
    "PermutationTestResult",
    "Pipeline",
    "VAELatent",
    "detect_platform",
    "normalize",
    "select_highly_variable",
]

__version__ = "0.1.0"
