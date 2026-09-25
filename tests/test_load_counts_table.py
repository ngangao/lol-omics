import gzip

import numpy as np
import pandas as pd
import pytest

from lol.fetch import FetchError, load_counts_table


def _write_counts_csv(path, genes, samples, rng):
    """Genes-as-rows, samples-as-columns counts CSV (STAR/featureCounts
    convention), matching GSE161731's actual file shape."""
    data = rng.integers(0, 500, size=(len(genes), len(samples)))
    df = pd.DataFrame(data, index=genes, columns=samples)
    df.to_csv(path)


def test_load_counts_table_transposes_genes_as_rows_by_default(tmp_path):
    rng = np.random.default_rng(0)
    genes = [f"ENSG{i:08d}" for i in range(20)]
    samples = [f"s{i}" for i in range(6)]
    counts_path = tmp_path / "counts.csv"
    _write_counts_csv(counts_path, genes, samples, rng)

    expression, metadata = load_counts_table(counts_path)

    assert expression.shape == (6, 20)  # samples x genes after transpose
    assert list(expression.index) == samples
    assert list(expression.columns) == genes
    assert metadata.shape[1] == 0  # no metadata_path given


def test_load_counts_table_with_metadata_join(tmp_path):
    rng = np.random.default_rng(1)
    genes = [f"ENSG{i:08d}" for i in range(15)]
    samples = ["94189", "DU09-03S0000604", "105920"]
    counts_path = tmp_path / "counts.csv"
    _write_counts_csv(counts_path, genes, samples, rng)

    meta_path = tmp_path / "key.csv"
    pd.DataFrame({
        "rna_id": samples,
        "cohort": ["Bacterial", "Influenza", "Influenza"],
        "age": [57, 19, 21],
    }).to_csv(meta_path, index=False)

    expression, metadata = load_counts_table(
        counts_path, metadata_path=meta_path, sample_id_col="rna_id"
    )

    assert expression.shape == (3, 15)
    assert "cohort" in metadata.columns
    assert list(metadata["cohort"]) == ["Bacterial", "Influenza", "Influenza"]
    assert list(expression.index) == list(metadata.index)  # aligned


def test_load_counts_table_handles_gzip_and_tsv(tmp_path):
    rng = np.random.default_rng(2)
    genes = [f"g{i}" for i in range(10)]
    samples = ["a", "b", "c"]
    data = rng.integers(0, 100, size=(len(genes), len(samples)))
    df = pd.DataFrame(data, index=genes, columns=samples)

    tsv_gz_path = tmp_path / "counts.tsv.gz"
    with gzip.open(tsv_gz_path, "wt") as f:
        df.to_csv(f, sep="\t")

    expression, _ = load_counts_table(tsv_gz_path)
    assert expression.shape == (3, 10)


def test_load_counts_table_metadata_requires_sample_id_col(tmp_path):
    rng = np.random.default_rng(3)
    genes = [f"g{i}" for i in range(5)]
    samples = ["a", "b"]
    counts_path = tmp_path / "counts.csv"
    _write_counts_csv(counts_path, genes, samples, rng)
    meta_path = tmp_path / "meta.csv"
    pd.DataFrame({"id": samples, "label": ["x", "y"]}).to_csv(meta_path, index=False)

    with pytest.raises(ValueError, match="sample_id_col is required"):
        load_counts_table(counts_path, metadata_path=meta_path)


def test_load_counts_table_raises_on_no_overlap(tmp_path):
    rng = np.random.default_rng(4)
    genes = [f"g{i}" for i in range(5)]
    samples = ["a", "b"]
    counts_path = tmp_path / "counts.csv"
    _write_counts_csv(counts_path, genes, samples, rng)
    meta_path = tmp_path / "meta.csv"
    pd.DataFrame({"rna_id": ["x", "y"], "label": ["p", "q"]}).to_csv(meta_path, index=False)

    with pytest.raises(FetchError, match="No overlap"):
        load_counts_table(counts_path, metadata_path=meta_path, sample_id_col="rna_id")


def test_load_counts_table_gene_axis_columns(tmp_path):
    rng = np.random.default_rng(5)
    # samples as rows, genes as columns -- the non-default orientation
    samples = ["s1", "s2", "s3", "s4"]
    genes = [f"g{i}" for i in range(8)]
    data = rng.integers(0, 50, size=(len(samples), len(genes)))
    df = pd.DataFrame(data, index=samples, columns=genes)
    counts_path = tmp_path / "counts.csv"
    df.to_csv(counts_path)

    expression, _ = load_counts_table(counts_path, gene_axis="columns")
    assert expression.shape == (4, 8)
    assert list(expression.index) == samples
