"""Fetching and parsing GEO data.

Design note
-----------
We deliberately do NOT depend on GEOparse for parsing. As of this
writing GEOparse's own maintenance health looks poor (no recent
releases, a growing list of unresolved issues, issue creation disabled
on the repo), so we use it for nothing. Downloading is delegated to
``geofetch`` (actively maintained, published in Bioinformatics 2023,
under the pepkit organisation), and we ship a small, dependency-light
parser here for turning a downloaded GEO series matrix file into a
samples x genes DataFrame. Keeping this parser small and ours means
a future upstream break is a few lines to fix, not a blocked pipeline.
"""

from __future__ import annotations

import gzip
import io
import subprocess
from pathlib import Path

import pandas as pd


class FetchError(RuntimeError):
    """Raised when a GEO accession cannot be downloaded or parsed."""


def fetch_series_matrix(accession: str, dest: str | Path = "./geo_data") -> Path:
    """Download a GEO series (GSE) using geofetch and return the local path.

    Parameters
    ----------
    accession:
        A GEO series accession, e.g. ``"GSE12345"``.
    dest:
        Local directory to download into. Created if it does not exist.

    Returns
    -------
    Path to the directory geofetch downloaded the accession into.

    Notes
    -----
    Requires the optional ``geofetch`` dependency
    (``pip install lol-omics[fetch]``) and network access to GEO.
    This function shells out to the ``geofetch`` CLI rather than
    importing it, so a broken import in an optional dependency never
    breaks the core pipeline for users who only work from local files.

    Passes ``--processed`` (download processed GEO files, e.g. the
    series matrix) with both ``--geo-folder`` and ``-m`` pointed at
    ``dest`` -- geofetch's ``-m`` (metadata output folder) is
    required, and ``-p``/``--processed`` is a boolean flag with no
    argument of its own, not a destination path. An earlier version of
    this function called ``geofetch -i <accession> -p <dest>``, which
    geofetch's argument parser rejects for exactly that reason (``-p``
    consumes no value, so ``<dest>`` is left as an unrecognized
    positional argument) -- caught when first exercised against a real
    geofetch install rather than a hand-downloaded file, on GSE161731.

    Also passes ``--data-source all``: geofetch's ``--processed``
    defaults to sample-level processed files only, but many series
    (GSE161731 among them) register their processed data -- including
    the series matrix -- at the series level instead. Defaulting to
    samples-only silently downloads nothing for those series (geofetch
    reports "No files found" without erroring), rather than failing
    loudly; ``all`` checks both levels.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run(
            [
                "geofetch", "-i", accession,
                "--processed", "--data-source", "all",
                "--geo-folder", str(dest),
                "-m", str(dest),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise FetchError(
            "geofetch is not installed. Run `pip install lol-omics[fetch]`, "
            "or download the series matrix yourself and pass a local path "
            "to `load_series_matrix` instead."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise FetchError(
            f"geofetch failed to download {accession}:\n{exc.stderr}"
        ) from exc

    return dest


def load_series_matrix(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse a GEO series matrix (``*_series_matrix.txt`` or ``.txt.gz``) file.

    Returns
    -------
    (expression, metadata):
        ``expression`` is a samples x genes/probes DataFrame of the
        ``!series_matrix_table`` block. ``metadata`` is a samples x
        fields DataFrame built from the ``!Sample_*`` header lines
        (e.g. title, characteristics, platform).

    This handles the single most common GEO download format. It does
    not attempt to handle every historical GEO format variant -- for
    anything unusual, read the file yourself and hand LOL a plain
    samples x genes DataFrame directly.
    """
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open

    sample_fields: dict[str, list[str]] = {}
    table_lines: list[str] = []
    in_table = False

    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped.startswith("!Sample_"):
                key, *values = stripped.split("\t")
                key = key.lstrip("!").replace('"', "")
                values = [v.strip('"') for v in values]

                # GEO series matrix files routinely repeat the same key on
                # multiple lines -- most commonly several
                # "!Sample_characteristics_ch1" lines in a row, one per
                # characteristic (tissue, disease state, age, sex, ...).
                # A plain dict would silently keep only the first and drop
                # the rest, which is exactly how a disease-state line can
                # vanish while a same-named tissue/group line survives.
                # De-duplicate the key instead of overwriting/dropping.
                unique_key = key
                suffix = 0
                while unique_key in sample_fields:
                    suffix += 1
                    unique_key = f"{key}_{suffix}"
                sample_fields[unique_key] = values
            elif stripped.startswith("!series_matrix_table_begin"):
                in_table = True
                continue
            elif stripped.startswith("!series_matrix_table_end"):
                in_table = False
                continue
            elif in_table:
                table_lines.append(line)

    if not table_lines:
        raise FetchError(
            f"No !series_matrix_table block found in {path}. "
            "Is this a GEO series matrix file?"
        )

    expression = pd.read_csv(
        io.StringIO("".join(table_lines)), sep="\t", index_col=0
    ).transpose()
    expression.columns = [str(c).strip('"') for c in expression.columns]
    expression.index = [str(i).strip('"') for i in expression.index]

    if sample_fields:
        metadata = pd.DataFrame(sample_fields)
        metadata.index = expression.index[: len(metadata)]
    else:
        metadata = pd.DataFrame(index=expression.index)

    return expression, metadata


def load_counts_table(
    counts_path: str | Path,
    metadata_path: str | Path | None = None,
    sample_id_col: str | None = None,
    gene_axis: str = "rows",
    sep: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load a plain counts table, optionally joined with a separate metadata file.

    Many RNA-seq GEO series don't embed a processed data table in the
    series matrix at all (unlike most microarray series) -- their
    processed data is submitted as separate supplementary files
    instead, commonly a genes x samples (or samples x genes) counts
    CSV/TSV plus a separate metadata/key file mapping sample IDs to
    clinical or demographic fields. GSE161731 is one example
    (``*_counts.csv.gz`` + ``*_key.csv.gz``), found to need this
    during development after ``load_series_matrix`` correctly raised
    ``FetchError`` on it (no ``!series_matrix_table`` block exists in
    that series at all). This function handles that shape generally,
    not just GSE161731's exact file names.

    Parameters
    ----------
    counts_path:
        Path to the counts file (``.csv``, ``.tsv``, or gzipped
        variants of either). Its first column is treated as the gene
        identifier.
    metadata_path:
        Optional path to a separate metadata/key CSV or TSV, with one
        row per sample. If given, it's read and returned as the
        ``metadata`` output, aligned to the expression matrix's sample
        index.
    sample_id_col:
        Which column in the metadata file holds sample IDs matching
        the counts file's sample columns (i.e. the expression matrix's
        row index after loading). Required if ``metadata_path`` is
        given. Inspect the metadata file's columns yourself first --
        this is not auto-detected, the same "don't guess a label
        column" discipline as everywhere else in this package.
    gene_axis:
        ``"rows"`` (default) if genes are rows and samples are columns
        in the counts file (the common convention for STAR/featureCounts
        -style output, as in GSE161731) -- the loaded table is
        transposed to the samples x genes shape the rest of LOL
        expects. ``"columns"`` if the file is already samples x genes.
    sep:
        Field separator. Inferred from the file extension (comma for
        ``.csv``, tab for ``.tsv``/``.txt``) if not given.

    Returns
    -------
    (expression, metadata): expression is samples x genes. metadata is
    indexed the same way as expression if ``metadata_path`` was given,
    else an empty-columns DataFrame with just the sample index (same
    shape of return as :func:`load_series_matrix`, for consistency).
    """
    counts_path = Path(counts_path)
    if sep is None:
        sep = "\t" if counts_path.name.replace(".gz", "").endswith((".tsv", ".txt")) else ","

    counts = pd.read_csv(counts_path, sep=sep, index_col=0)
    expression = counts.T if gene_axis == "rows" else counts
    expression.index = [str(i) for i in expression.index]
    expression.columns = [str(c) for c in expression.columns]

    if metadata_path is None:
        return expression, pd.DataFrame(index=expression.index)

    if sample_id_col is None:
        raise ValueError(
            "sample_id_col is required when metadata_path is given -- "
            "inspect the metadata file's columns first and pass the "
            "one containing sample IDs matching the counts file's "
            "sample columns."
        )

    meta_sep = "\t" if str(metadata_path).replace(".gz", "").endswith((".tsv", ".txt")) else ","
    metadata = pd.read_csv(metadata_path, sep=meta_sep)
    metadata[sample_id_col] = metadata[sample_id_col].astype(str)
    metadata = metadata.set_index(sample_id_col)

    common = expression.index.intersection(metadata.index)
    if len(common) == 0:
        raise FetchError(
            f"No overlap between counts sample columns and "
            f"metadata['{sample_id_col}'] values -- check sample_id_col "
            "is the right column, and that IDs match exactly (e.g. no "
            "extra whitespace or a type mismatch)."
        )

    return expression.loc[common], metadata.loc[common]
