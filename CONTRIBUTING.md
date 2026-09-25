# Contributing to LOL

Thanks for considering a contribution. This project is small and welcomes
small, focused pull requests.

## Setup

```bash
git clone https://github.com/martinnganga/lol
cd lol
pip install -e ".[dev]"
pytest
ruff check src/ tests/
```

## Guidelines

- Keep the scope narrow: bulk RNA-seq and microarray data from GEO. Single-cell
  support is explicitly out of scope for now (see README).
- New features need tests. `tests/` mirrors `src/lol/` module-for-module.
- Run `ruff check --fix` before committing.
- Avoid adding heavy dependencies without discussion first — part of the
  point of this project is that it stays runnable on an 8GB-RAM laptop.

## AI assistance

This project was built with assistance from Claude (Anthropic), used for:
drafting initial module implementations, generating and reviewing test
cases, and drafting documentation. All AI-assisted code was reviewed, run,
and in several cases corrected by the author before being committed — for
example, an early version of `Pipeline.predict()` did not restrict new
samples to the same highly-variable gene set selected at fit time; this was
caught by the test suite and fixed (see commit history).

If you use AI tools to help with a contribution, please disclose it in your
pull request description, along with what you did to verify the output
(tests run, manual checks, etc.).

## Reporting issues

Please include: your Python version, the platform data you're working with
(RNA-seq or microarray), and a minimal reproducible example where possible.
