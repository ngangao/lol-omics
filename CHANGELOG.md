# Changelog

## [Unreleased]

### Added
- `EvaluationReport.macro_f1` and `class_weight="balanced"` (default)
  on the classifier -- found necessary after GSE75214 (97 UC vs. 11
  control) showed strong accuracy/ROC-AUC while quietly failing to
  detect the minority class.
- `Pipeline.permutation_test()`: reruns the full leakage-safe pipeline
  on shuffled labels and reports where the real score falls relative
  to that null. Generalizes the ad-hoc validation scripts used during
  development (on GSE16161, GSE22619, GSE75214) into a first-class
  method, since checking a result against a permutation null is what
  actually made each of those three results trustworthy.
- Group-aware cross-validation (`groups=` on `Pipeline.fit`, used
  automatically by `evaluate()` and `permutation_test()`): uses
  `StratifiedGroupKFold` so every sample from one subject stays on one
  side of a fold, and shuffles permutation-test labels at the group
  level (one shuffle per subject, not per sample). Found necessary on
  GSE66407, a repeated-measures cohort where individual patients
  contributed up to 9 biopsies each, and again on GSE161731's
  longitudinally-resampled COVID-19/CoV-other arms.
- `lol.visualize`: `plot_latent_space`, `plot_permutation_null`,
  `plot_leakage_audit`, `plot_pvca_lite`.
- `Pipeline.leakage_audit()`: runs naive and nested evaluation together
  and reports the gap per metric (Delta-LII), formalizing the
  comparison that caught the GSE66407 leakage impact as a named,
  reported quantity instead of something computed ad hoc.
- `lol.diagnostics.pvca_lite` / `Pipeline.confound_diagnostic()`: a
  simplified (per-covariate, not jointly-adjusted) relative of PVCA --
  reports how much of the latent space associates with each metadata
  covariate. Dependency-free (ANOVA R-squared for categorical
  covariates, squared correlation for continuous ones, weighted by
  each latent dimension's own explained-variance share). Verified
  against a synthetic reconstruction of the GSE75214 tissue-confound
  scenario (correctly attributed 50.4% of variance to an injected
  tissue confound vs. 0.9% to phenotype), and against two real
  datasets: GSE39582, where it showed the cleanest possible profile
  (phenotype dominant, batch negligible), and GSE161731, where it
  caught a genuine, unfixable age/cohort confound before that result
  could be over-interpreted as pure infection-type signal.
- `Pipeline.gene_loadings()` / `PCALatent.gene_loadings()`: exposes the
  fitted PCA loadings per gene for a given latent dimension. Added
  after reviewing a manuscript-figure script that reimplemented GEO
  parsing from scratch (reintroducing the repeated-characteristics-key
  risk already fixed in `lol.fetch`) and reached into a private
  attribute with a redundant second PCA refit to get loadings the
  package's own fitted model already had.
- `lol.fetch.load_counts_table`: loads a plain genes x samples (or
  samples x genes) counts CSV/TSV, optionally joined with a separate
  metadata/key file by sample ID. Many RNA-seq GEO series -- GSE161731
  among them -- don't embed a processed data table in the series
  matrix at all (unlike most microarray series); their processed data
  is supplementary files instead. `load_series_matrix` correctly
  raises `FetchError` on these (no `!series_matrix_table` block
  exists), which is what surfaced the need for this function.
- `examples/`: `quickstart_with_plots.py` (offline, synthetic data) and
  `geo_accession_with_groups.py` (real GEO workflow template with
  metadata inspection, group-aware fitting, and plotting).

### Fixed
- `fetch.load_series_matrix` silently dropped repeated `!Sample_*` keys
  (e.g. multiple `!Sample_characteristics_ch1` lines, one per
  characteristic) -- found while parsing real GEO series (GSE16161),
  where it caused every sample's disease-state label to collapse to a
  single repeated value. Now de-duplicates by suffixing repeated keys
  instead of dropping them.
- `Pipeline.evaluate()` fit HVG selection and the latent extractor once
  on the *entire* dataset in `fit()`, then only cross-validated the
  classifier on top -- an unsupervised leakage that produced a
  suspicious 1.000 accuracy/ROC-AUC on an 18-sample real cohort
  (GSE16161). Added `nested_cross_validate`, which refits HVG selection
  and PCA inside each fold; this is now the default (`evaluate(nested=True)`).
- `fetch.fetch_series_matrix` called the `geofetch` CLI as
  `geofetch -i <accession> -p <dest>`. `-p`/`--processed` is a boolean
  flag with no argument of its own, and `-m` (metadata output folder)
  is required but was never passed -- geofetch's argument parser
  rejected the call outright. This had gone unnoticed because every
  dataset used in development up to this point was downloaded by hand
  rather than through this function; it was only caught when first
  exercised against a real geofetch install, attempting GSE161731.
  Now calls `geofetch -i <accession> --processed --geo-folder <dest>
  -m <dest>`, verified to reach past argument parsing and attempt a
  real network request.
- Same function was missing `--data-source all`. geofetch's
  `--processed` defaults to downloading sample-level processed files
  only; GSE161731 registers its series matrix at the series level, so
  the default silently downloaded nothing (geofetch reported "No files
  found" rather than erroring) even after the fix above. Added
  `--data-source all` so both levels are checked.

## [0.1.0] - initial scaffold

- Core pipeline: normalize -> select highly-variable genes -> PCA/VAE
  extraction -> RandomForest classifier -> cross-validated evaluation report.
- `lol.fetch`: series-matrix downloading via `geofetch`, plus a small
  dependency-light parser (deliberately not using GEOparse — see README).
- CLI (`lol run`) for local CSV matrix + labels files.
- Synthetic data generator (`lol.datasets`) for offline testing/demos.
- Test suite covering preprocessing, latent extraction, prediction, and
  end-to-end pipeline behavior.

### Roadmap
- VAE out-of-sample transform for `Pipeline.predict()` (currently PCA-only).
- VAE support for nested evaluation / permutation testing (currently PCA-only,
  since the VAE has no out-of-sample transform yet -- same blocker as above).
- Direct `Pipeline.from_accession("GSExxxxx")` convenience constructor wrapping
  `lol.fetch`.
- Latent-dimension sweep helper (train once across several `n_latent`
  values and report which captures the most signal), following the
  reasoning in Way et al. 2020 that a single fixed dimensionality limits
  what's discoverable.
- Optional gene-level attribution for VAE latent dimensions (`gene_loadings`
  currently only covers the PCA path).
- GSEA integration (via gseapy) on top of `gene_loadings`, so classifier-
  relevant genes can be traced to pathways, not just ranked individually.
- Cross-study / out-of-domain benchmarking: fit on one GEO accession,
  evaluate on another, with a distribution-shift diagnostic (e.g. MMD)
  between the two latent distributions.
- A full (jointly-adjusted, mixed-model) PVCA implementation as an
  alternative to the current per-covariate PVCA-lite approximation.

