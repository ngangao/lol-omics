# LOL — Latent Omics Learning

[![tests](https://img.shields.io/badge/tests-pytest-informational)](https://github.com/ngangao/lol-omics/actions)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)

**LOL takes a GEO gene expression cohort (bulk RNA-seq or microarray) and turns it into a trained phenotype predictor in a few lines of code.**

```
GEO accession / local matrix  →  normalize  →  select variable genes
   →  latent extraction (PCA or a small VAE)  →  classifier  →  evaluation report
```

## What this is (and isn't)

LOL is a **reproducibility and accessibility tool**, not a new machine learning method. Latent representation learning on gene expression data (PCA and VAE-based) is well established. See [Way & Greene 2018 ("Tybalt")](https://doi.org/10.1101/174474) for the foundational VAE approach this package's VAE module is inspired by, and [Way et al. 2020](https://doi.org/10.1186/s13059-020-02021-3) for why latent dimensionality shouldn't be fixed to a single choice. What LOL adds is a small, well-documented, laptop-friendly pipeline that goes from a **GEO accession straight through to a cross-validated phenotype predictor**, with both a linear (PCA) and non-linear (VAE) latent method available side by side so you can see which one actually helps on *your* dataset rather than assuming one is better.

LOL deliberately does **not** target single-cell data. That's a much larger-scale, better-served niche (see [scvi-tools](https://scvi-tools.org/)); LOL stays in the bulk RNA-seq / microarray regime, where cohorts are small enough (tens to a few hundred samples) to train comfortably on a laptop with 8GB of RAM.

## Install

```bash
pip install lol            # core pipeline, local files only
pip install lol [fetch]     # + geofetch, for pulling GEO accessions directly
```

## Quickstart

```python
from lol import Pipeline
from lol.datasets import make_synthetic_counts

# swap this for your own samples x genes DataFrame + label Series
expression, labels = make_synthetic_counts(n_samples=80, n_genes=500)

pipe = Pipeline(latent_method="pca", n_latent=20, n_hvg=2000)
pipe.fit(expression, labels)

print(pipe.evaluate())
print(pipe.permutation_test())  # is that score distinguishable from chance?
```

```
Here is an example of an output from a sample GEO dataset
Accuracy: 0.912
Macro-F1: 0.905
ROC-AUC:  0.967

              precision    recall  f1-score   support
           0       0.90      0.93      0.91        40
           1       0.93      0.90      0.91        40

Permutation test on macro_f1
  Real score:        0.905
  Null distribution: mean=0.501, min=0.350, max=0.680 (30 shuffles)
  p-value estimate:  0.032
```

From the command line, on your own CSVs:

```bash
lol run --matrix expression.csv --labels metadata.csv --target disease_state
```

## Diagnostics: don't just run the pipeline, audit it

Two checks are built in, motivated directly by real confounds and a
real leakage bug caught during development (see
[`docs/case-studies.md`](docs/case-studies.md)):

```python
print(pipe.leakage_audit())        # naive vs. nested score, and the gap (Delta-LII)
print(pipe.confound_diagnostic(metadata))  # how much of the latent space each covariate explains
```

`confound_diagnostic` takes a samples x covariates DataFrame (tissue,
batch, patient ID, the phenotype label itself) and reports each
covariate's association with the latent space — a technical covariate
scoring as high or higher than the phenotype is a warning sign worth
investigating before trusting `evaluate()`.

## Validated on real data

Six real GEO cohorts were run through LOL during development. They include
a strong positive result, a clean null, two cases where a
confound (class imbalance, tissue-type mismatch, repeated samples per
patient) had to be caught and controlled for, a multi-class cohort
with an important label-circularity caveat, and a case where a real
confound (age, entangled with cohort by recruitment design) had no
clean fix and is reported as a genuine limitation rather than argued
around. Along the way: several parsing/leakage bugs, two `geofetch`
command-line bugs, and a general RNA-seq counts-table loader were
found and fixed, not by inspection but by running real data through
the pipeline. See [`docs/case-studies.md`](docs/case-studies.md) for
the full account, including what changed in the code as a result.

## Why both PCA and a VAE?

Short answer: the literature doesn't give a clean winner at the sample sizes GEO cohorts typically offer. VAEs tend to reconstruct expression data better than PCA only at small latent dimensions (roughly <20) before overfitting overtakes them, while PCA's own quality degrades as sample size shrinks. LOL's default is PCA (instant, deterministic, no hyperparameters to tune); the VAE is there as an option to try and compare, not a claimed upgrade. See [`docs/architecture.md`](docs/architecture.md) for the full reasoning and citations.

## Why not GEOparse for fetching?

Because it looks unmaintained: no recent PyPI releases, a backlog of unresolved GitHub issues, and issue creation disabled on the repo as of this writing. LOL instead shells out to [geofetch](https://github.com/pepkit/geofetch) (actively maintained, published in *Bioinformatics* 2023) for downloading, and ships its own small series-matrix parser in `lol/fetch.py` rather than depending on GEOparse's parsing internals. See [`docs/architecture.md`](docs/architecture.md#why-not-geoparse-for-fetching) for details.

## Scope and honest limitations

- Bulk RNA-seq and microarray only — no single-cell support.
- The VAE is intentionally small (1 hidden layer by default) and CPU-trainable; it is not competitive with large single-cell foundation models, and isn't trying to be.
- `Pipeline.predict()` on new samples is currently only implemented for the PCA path; VAE out-of-sample transform is on the roadmap (see [`CHANGELOG.md`](CHANGELOG.md)).
- The microarray normalization is a standard quantile-normalization step operating on GEO-processed intensities, not a full RMA pipeline from raw CEL files.

## Documentation

Full docs (architecture, tutorials on real GEO series, API reference) live in [`docs/`](docs/) and are published at the project's GitHub Pages site. Start with [`docs/quickstart.md`](docs/quickstart.md).

## Development

```bash
git clone https://github.com/martinnganga/lol
cd lol
pip install -e ".[dev]"
pytest
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## AI-assistance disclosure

Claude (Antropic) was used to assist in debugging errors in this codebase and and its documentation. Claude was also used to review logical errors. After this, all fixes were reviewed, tested, and modified by the author in line with GEO datasets and known biology.

## License

MIT — see [`LICENSE`](LICENSE).
