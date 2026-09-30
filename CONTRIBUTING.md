# Contributing to propaxis

Use an issue to describe the measurement problem, a reproducible failure, or a proposed change to the API.

```bash
git clone https://github.com/atomashevic/propaxis.git
cd propaxis
uv venv
uv pip install -e '.[dev]'
uv run --no-sync pytest -q
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
```

The default test suite uses deterministic test encoders and makes no model downloads. Optional integration tests with real models are described in `docs/testing.md`.

Keep NumPy as the only required inference-independent dependency. Import neural and notebook dependencies inside the functions that need them. A scoring change needs numerical tests, an explanation of its effect on saved instruments, and a schema version change if old parameters no longer mean the same thing.

Do not commit user texts, caches, model weights, credentials, or research datasets. Small artificial examples are welcome. Keep public examples executable and distinguish score scaling, label calibration, and model training.

The planned R interface is **propRaxis**. Keep the Python API serializable and usable through reticulate; the R interface is not implemented in this release.
