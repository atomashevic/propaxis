# Testing

## Automated tests

```bash
uv venv
uv pip install -e '.[dev]'
uv run --no-sync pytest -q
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync python -m build
uv run --no-sync python examples/offline_demo.py
```

The default tests use deterministic encoders and check:

- The centroid-difference formula and independence from proposition ordering.
- Word caps and overlapping passage construction.
- The scoring formula against an independently calculated example.
- Consistent scores and evidence when a document is scored alone or in a batch.
- Save/load round trips, backend identity, and checksums.
- Component-specific passage selection and NLI hypothesis matching.
- Binary thresholds, tied-score metrics, and validation of labels.
- Notebook form editing, validation, and JSON construction.
- Importing the library without initializing neural models or notebook tools.

GitHub Actions runs these checks and builds the package on Linux, macOS, and Windows with Python 3.10 and 3.12. These jobs make no model downloads.

## Real-model integration check

```bash
uv pip install -e '.[models]'
OMP_NUM_THREADS=2 uv run --no-sync python examples/integration_check.py
```

`OMP_NUM_THREADS` is an optional limit on Linux/macOS. On Windows, set the environment variable using the syntax for your shell or omit it.

The script downloads E5-small-v2 and BART-large-MNLI if absent, fits a small tutorial measure, checks expected ordering for an explicit supportive and opposing statement, compares single-document and batch scoring, and checks offline reload. It prints the resolved model revisions. It needs several gigabytes of disk space for the model cache and enough memory for the two models.

This is an integration test, not a scientific benchmark. It does not establish performance on a new construct.

## Initial CPU verification

On 30 September 2026, the local integration check passed with Python 3.12.14, NumPy 2.5.3, PyTorch 2.14.0+cpu, and Transformers 4.57.6. The model revisions were:

| Model | Revision |
| --- | --- |
| `intfloat/e5-small-v2` | `ffb93f3bd4047442299a41ebb6fa998a38507c52` |
| `facebook/bart-large-mnli` | `d7645e127eaf1aefc7862fd59a17a5aa8558b8ce` |

E5-large-v2, CUDA, and Apple MPS were not exercised in this initial real-model check. The default constructors resolve their requested revision on first use and save the resolved commit in the fitted instrument. For exact reproduction, use the recorded revisions or reload a saved instrument.

## Notebook

Install a Jupyter frontend in the same environment, then open the example:

```bash
uv pip install jupyterlab ipykernel
uv run --no-sync jupyter lab examples/construct_a_measure.ipynb
```

The form supports specification editing, validation, and JSON preview. Running all cells also exercises fitting, scoring, and offline reload. Widget rendering requires a frontend with ipywidgets support; the Python tests validate form behavior without a browser.
