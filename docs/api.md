# Python API

The top-level API imports without PyTorch, Transformers, pandas, or ipywidgets. Neural and notebook dependencies are loaded only when needed.

## Public objects

| Object | Purpose |
| --- | --- |
| `Component(name, positive, contrast, hypotheses=(), description="")` | One direction specified by two lists of propositions. |
| `Construct(name, components, window=1, top_k=1, max_words=60, combine="geometric_mean", temperature=.25)` | Immutable measurement specification. |
| `Encoder(model="e5-small-v2", revision="main", device="cpu", batch_size=32, max_length=256, local_files_only=False)` | Local, mean-pooled E5 backend. Also supports E5-large-v2. |
| `CallableEncoder(function, name)` | Adapter for a custom function returning a two-dimensional embedding array. |
| `Entailment(model="facebook/bart-large-mnli", revision="main", device="cpu", batch_size=16, max_length=320, local_files_only=False)` | Optional NLI backend. Alternate checkpoints must identify their three NLI labels explicitly. |
| `Instrument` | Fitted axes and reference scaling parameters, with model configuration. |
| `ScoreResult` | Score records, selected passage evidence, and instrument identifier. |
| `BinaryClassifier` | A threshold bound to one score column and instrument. |

## Construct

```python
instrument = construct.fit(reference_texts, encoder=encoder, claim_check=None)
construct.save("construct.json")
construct = Construct.load("construct.json")
payload = construct.to_dict()
construct = Construct.from_dict(payload)
identifier = construct.fingerprint
```

`reference_texts` must contain at least three nonempty strings. Fitting updates no neural weights. Component names are unique Python-style identifiers beginning with a letter; `combined` is reserved. The construct and its component sequences are immutable.

## Instrument

```python
result = instrument.score(texts, ids=None)
diagnostics = instrument.diagnostics
instrument.save("instrument.json")
instrument = Instrument.load(
    "instrument.json",
    encoder=None,
    claim_check=None,
    device="cpu",
    local_files_only=False,
)
```

`score` accepts one string or a sequence. Optional IDs must be unique strings. Loading a custom backend requires explicitly supplying it again. For built-in backends, the saved model revision is used. Scoring rejects changes to the stored construct, parameters, or backend configuration.

## Results

`result.scores` contains dictionaries with these fields:

| Field | Meaning |
| --- | --- |
| `id` | Supplied document ID or zero-based string index. |
| `n_windows` | Number of passages produced before selection. |
| `axis` | Aggregate composite activation before document standardization. |
| `score` | Standardized axis score, or equal-weight axis/NLI combination. |
| `<name>_axis` | Component activation aggregated over that component's selected passages. |
| `<name>_score` | Component score using its own reference scale and optional NLI fusion. |
| `claim_check` | Composite NLI aggregate; present only when enabled. |
| `<name>_claim_check` | Component NLI aggregate; present only when enabled. |

`result.evidence` contains one record per document. `targets` maps component names and `combined` to their selected passages. Each passage has a document-relative `window_index`, `text`, and `components` mapping with raw projection, standardized projection, activation, and optional entailment value and matched hypothesis.

```python
frame = result.to_pandas()  # optional pandas dependency
result.to_json("scores.json")  # includes passage text
```

## Binary calibration

```python
classifier = calibrate(
    calibration_result,
    binary_labels,
    column="score",
    objective="balanced_accuracy",
)
predictions = classifier.predict(test_result)
metrics = classifier.evaluate(test_result, test_labels)
classifier.save("classifier.json")
classifier = BinaryClassifier.load("classifier.json")
```

The only objectives in v0.1 are `balanced_accuracy` and `f1`. Input labels must be 0/1. Continuous scores and binary predictions are distinct outputs; no probability calibration is implied.

## Models and notebooks

```python
download_models("e5-small-v2", claim_check=False)

from propaxis.notebook import construct_editor

editor = construct_editor(construct=None)
display(editor)
construct = editor.build()

from propaxis.presets import populism

construct = populism()
```

Use a notebook with ipywidgets support for the editor. It validates specification fields and generates JSON without invoking any model. A general-purpose Streamlit application and the propRaxis R wrapper are outside v0.1.
