"""Written constructs: immutable, serializable specifications with no model imports."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


def texts(value: Sequence[str], field: str, minimum: int = 1) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{field} must be a sequence of texts, not one string")
    items = tuple(value)
    if len(items) < minimum:
        raise ValueError(f"{field} requires at least {minimum} text(s)")
    if any(not isinstance(item, str) or not item.strip() for item in items):
        raise ValueError(f"{field} must contain nonempty strings; remove missing/blank rows")
    return tuple(item.strip() for item in items)


def positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class Component:
    """One conceptual direction, from contrast propositions to positive propositions.

    Poles need not have the same number of propositions or index-paired sentences.
    Hypotheses are optional explicit claims for an entailment check. When absent,
    all positive propositions are used as hypotheses.
    """

    name: str
    positive: tuple[str, ...]
    contrast: tuple[str, ...]
    hypotheses: tuple[str, ...] = ()
    description: str = ""

    def __post_init__(self):
        if not isinstance(self.name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", self.name):
            raise ValueError(
                "component name must start with a letter and contain letters, digits or _"
            )
        if self.name == "combined":
            raise ValueError("combined is reserved for the composite score")
        for field in ("positive", "contrast", "hypotheses"):
            clean = texts(getattr(self, field), field, 0 if field == "hypotheses" else 1)
            if len(set(clean)) != len(clean):
                raise ValueError(f"duplicate proposition in {field}")
            object.__setattr__(self, field, clean)
        if set(self.positive) & set(self.contrast):
            raise ValueError("a proposition cannot belong to both poles")

    @property
    def claims(self) -> tuple[str, ...]:
        return self.hypotheses or self.positive


@dataclass(frozen=True)
class Construct:
    """A reusable measurement recipe. Call :meth:`fit` on reference documents.

    ``combine`` accepts geometric_mean (soft conjunction), mean, or minimum.
    ``top_k=None`` averages all windows; otherwise it averages the strongest k.
    The reference corpus defines the score scale and is never refitted by score().
    """

    name: str
    components: tuple[Component, ...]
    window: int = 1
    top_k: int | None = 1
    max_words: int = 60
    combine: str = "geometric_mean"
    temperature: float = 0.25

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("construct name must be nonempty")
        components = tuple(self.components)
        if not components or any(not isinstance(c, Component) for c in components):
            raise ValueError("components must contain at least one Component")
        if len({c.name for c in components}) != len(components):
            raise ValueError("component names must be unique")
        object.__setattr__(self, "components", components)
        positive_int(self.window, "window")
        positive_int(self.max_words, "max_words")
        if self.top_k is not None:
            positive_int(self.top_k, "top_k")
        if self.combine not in {"geometric_mean", "mean", "minimum"}:
            raise ValueError("combine must be geometric_mean, mean or minimum")
        import math

        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> Construct:
        value = dict(value)
        value["components"] = tuple(Component(**c) for c in value["components"])
        return cls(**value)

    @property
    def fingerprint(self) -> str:
        raw = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False).encode()
        return hashlib.sha256(raw).hexdigest()

    def save(self, path: str | Path) -> None:
        """Save the editable recipe as UTF-8 JSON (no models or corpus text)."""
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> Construct:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def fit(self, reference: Sequence[str], *, encoder=None, claim_check=None):
        """Fit reference statistics without labels or updating neural model weights.

        Pass an Encoder or any object implementing encode(texts) and metadata().
        An optional Entailment checker adds reference-scaled, equal-weight fusion.
        """
        from .instrument import Instrument
        from .models import Encoder

        return Instrument.fit(
            self, reference, encoder=encoder or Encoder(), claim_check=claim_check
        )
