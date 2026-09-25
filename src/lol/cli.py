"""Command-line interface: `lol run --matrix path.csv --labels path.csv --target col`."""

from __future__ import annotations

import click
import pandas as pd

from lol.pipeline import Pipeline


@click.group()
def main():
    """LOL: Latent Omics Learning."""


@main.command()
@click.option("--matrix", required=True, type=click.Path(exists=True), help="Samples x genes CSV (samples as rows).")
@click.option("--labels", required=True, type=click.Path(exists=True), help="CSV with a sample-id column and a label column.")
@click.option("--target", required=True, help="Column name in --labels to predict.")
@click.option("--sample-id-col", default=None, help="ID column in --labels matching the matrix's row index. Defaults to the first column.")
@click.option("--method", type=click.Choice(["pca", "vae"]), default="pca", show_default=True)
@click.option("--n-latent", default=20, show_default=True)
@click.option("--n-hvg", default=2000, show_default=True)
def run(matrix, labels, target, sample_id_col, method, n_latent, n_hvg):
    """Run the full pipeline on local files and print an evaluation report."""
    expression = pd.read_csv(matrix, index_col=0)
    labels_df = pd.read_csv(labels)
    id_col = sample_id_col or labels_df.columns[0]
    labels_series = labels_df.set_index(id_col)[target]

    pipe = Pipeline(latent_method=method, n_latent=n_latent, n_hvg=n_hvg)
    pipe.fit(expression, labels_series)
    report = pipe.evaluate()
    click.echo(str(report))


if __name__ == "__main__":
    main()
