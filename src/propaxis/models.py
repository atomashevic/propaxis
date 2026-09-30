"""Local Hugging Face models, loaded only when first used."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .spec import positive_int, texts

ENCODERS = {
    "e5-small-v2": "intfloat/e5-small-v2",
    "e5-large-v2": "intfloat/e5-large-v2",
}


def _libraries():
    try:
        import torch
        from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as error:
        raise ImportError(
            "Local models require the models extra: pip install 'propaxis[models]'"
        ) from error
    return torch, AutoModel, AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class Encoder:
    """Mean-pooled E5 sentence encoder. Models download on first encode().

    Both propositions and documents use the same ``query: `` prefix. Raw pooled
    embeddings are returned; propaxis normalizes anchors and centers documents.
    A saved Instrument records the resolved checkpoint revision. CPU is default.
    """

    model: str = "e5-small-v2"
    revision: str = "main"
    device: str = "cpu"
    batch_size: int = 32
    max_length: int = 256
    local_files_only: bool = False
    _model: object = field(default=None, init=False, repr=False)
    _tokenizer: object = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self.model not in ENCODERS and self.model not in ENCODERS.values():
            raise ValueError(
                "Encoder supports e5-small-v2 and e5-large-v2; use CallableEncoder for other models"
            )
        self.model = ENCODERS.get(self.model, self.model)
        for name in ("batch_size", "max_length"):
            positive_int(getattr(self, name), name)

    def _load(self):
        if self._model is not None:
            return
        _, auto_model, _, tokenizer = _libraries()
        model = (
            auto_model.from_pretrained(
                self.model, revision=self.revision, local_files_only=self.local_files_only
            )
            .eval()
            .to(self.device)
        )
        self.revision = getattr(model.config, "_commit_hash", None) or self.revision
        self._tokenizer = tokenizer.from_pretrained(
            self.model, revision=self.revision, local_files_only=self.local_files_only
        )
        self._model = model

    def encode(self, items: Sequence[str]) -> np.ndarray:
        items = texts(items, "encoder input")
        self._load()
        torch, *_ = _libraries()
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(items), self.batch_size):
                batch = self._tokenizer(
                    ["query: " + x for x in items[start : start + self.batch_size]],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                hidden = self._model(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                outputs.append(pooled.float().cpu().numpy())
        return np.concatenate(outputs)

    def metadata(self) -> dict:
        return {
            "kind": "e5",
            "model": self.model,
            "revision": self.revision,
            "max_length": self.max_length,
            "pooling": "mean",
            "prefix": "query: ",
        }


@dataclass
class CallableEncoder:
    """Adapter for encode(texts) -> a finite, two-dimensional NumPy array.

    Use a stable name/version identifying the callable's model and preprocessing.
    Loading its saved instrument requires passing the same encoder explicitly.
    """

    function: object
    name: str

    def __post_init__(self):
        if not callable(self.function) or not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("provide an encoding callable and a nonempty identifying name")

    def encode(self, items):
        return self.function(items)

    def metadata(self):
        return {"kind": "custom", "name": self.name}


@dataclass
class Entailment:
    """BART MNLI check returning entailment probabilities, including the neutral class.

    Each document window is a premise; user-written claims are hypotheses. These
    outputs are model scores, not calibrated probabilities of the research label.
    """

    model: str = "facebook/bart-large-mnli"
    revision: str = "main"
    device: str = "cpu"
    batch_size: int = 16
    max_length: int = 320
    local_files_only: bool = False
    _model: object = field(default=None, init=False, repr=False)
    _tokenizer: object = field(default=None, init=False, repr=False)
    _entailment_index: int = field(default=2, init=False, repr=False)

    def __post_init__(self):
        positive_int(self.batch_size, "batch_size")
        positive_int(self.max_length, "max_length")

    def _load(self):
        if self._model is not None:
            return
        _, _, classifier, tokenizer = _libraries()
        model = (
            classifier.from_pretrained(
                self.model, revision=self.revision, local_files_only=self.local_files_only
            )
            .eval()
            .to(self.device)
        )
        labels = {str(label).lower(): int(i) for i, label in model.config.id2label.items()}
        if set(labels) != {"contradiction", "neutral", "entailment"}:
            raise ValueError(
                "NLI checkpoint must identify contradiction, neutral and entailment labels"
            )
        self._entailment_index = labels["entailment"]
        self.revision = getattr(model.config, "_commit_hash", None) or self.revision
        self._tokenizer = tokenizer.from_pretrained(
            self.model, revision=self.revision, local_files_only=self.local_files_only
        )
        self._model = model

    def predict(self, premises: Sequence[str], hypotheses: Sequence[str]) -> np.ndarray:
        premises = texts(premises, "premises")
        hypotheses = texts(hypotheses, "hypotheses")
        self._load()
        torch, *_ = _libraries()
        result = np.empty(len(premises) * len(hypotheses), dtype=np.float64)
        with torch.inference_mode():
            for start in range(0, len(result), self.batch_size):
                positions = range(start, min(start + self.batch_size, len(result)))
                batch = self._tokenizer(
                    [premises[i // len(hypotheses)] for i in positions],
                    [hypotheses[i % len(hypotheses)] for i in positions],
                    padding=True,
                    truncation="only_first",
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                logits = self._model(**batch).logits
                probs = torch.softmax(logits.float(), dim=-1)[:, self._entailment_index]
                result[start : start + len(probs)] = probs.cpu().numpy()
        return result.reshape(len(premises), len(hypotheses))

    def metadata(self) -> dict:
        return {
            "kind": "nli",
            "model": self.model,
            "revision": self.revision,
            "max_length": self.max_length,
        }


def download_models(encoder: str = "e5-small-v2", *, claim_check: bool = False) -> dict:
    """Download and load supported models once; report the cached checkpoint revisions.

    Hugging Face cache settings, including HF_HOME, are respected. No text is sent
    to an inference service. Reuse the cache with local_files_only=True offline.
    """
    backend = Encoder(encoder)
    backend._load()
    result = {"encoder": backend.metadata()}
    del backend
    if claim_check:
        checker = Entailment()
        checker._load()
        result["claim_check"] = checker.metadata()
    return result
