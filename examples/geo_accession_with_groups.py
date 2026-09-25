"""Example: fetch, parse, and analyze a real GEO series, with
patient-grouped cross-validation for repeated-measures cohorts.

Adapted from the GSE66407 case study in docs/case-studies.md. Point
MATRIX_GLOB at any downloaded series matrix and adjust the label/group
extraction to match that series' actual metadata (inspect it first --
see docs/case-studies.md for why that matters).
"""

import glob
import re
from pathlib import Path

import matplotlib.pyplot as plt

from lol.fetch import load_series_matrix
from lol.pipeline import Pipeline
from lol.visualize import plot_latent_space, plot_permutation_null

MATRIX_GLOB = "geo_data/*/**/*series_matrix*"

candidates = glob.glob(MATRIX_GLOB, recursive=True)
if not candidates:
    raise SystemExit(
        f"No series matrix found matching {MATRIX_GLOB!r} -- "
        "download one first, e.g. lol.fetch.fetch_series_matrix('GSExxxxx')"
    )

expr, meta = load_series_matrix(Path(candidates[0]))
print(f"Loaded {candidates[0]}: {expr.shape}")

char_cols = [c for c in meta.columns if "characteristics" in c.lower()]
combined_text = meta[char_cols].astype(str).agg(" | ".join, axis=1)
combined_text.index = expr.index


def extract(field, text):
    m = re.search(rf"{field}:\s*([^|]+)", text)
    return m.group(1).strip() if m else None


# --- inspect before trusting: print the raw combinations first ---
print("\nUnique characteristics combinations (first 10):")
print(combined_text.value_counts().head(10).to_string())

# --- adjust these two lines to match what you actually see above ---
labels = combined_text.apply(lambda t: extract("diagnosis", t))
groups = combined_text.apply(lambda t: extract("patient", t))

keep_mask = labels.notna()
expr, labels, groups = expr.loc[keep_mask], labels.loc[keep_mask], groups.loc[keep_mask]

pipe = Pipeline(latent_method="pca", n_latent=10, n_hvg=2000)
pipe.fit(expr, labels, groups=groups if groups.notna().any() else None)

print("\n" + str(pipe.evaluate(nested=True)))
result = pipe.permutation_test(metric="macro_f1", n_permutations=30)
print(result)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
plot_latent_space(pipe.latents_, labels, ax=axes[0])
plot_permutation_null(result, ax=axes[1])
plt.tight_layout()
plt.savefig("geo_example_output.png", dpi=200, bbox_inches="tight")
print("Saved geo_example_output.png")
