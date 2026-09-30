"""Fit a measurement to a reference dataset, then score without changing its scale."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import (
    activation,
    combine,
    matrix,
    pole_axis,
    pool,
    select_windows,
    split_windows,
    unit_rows,
)
from .models import Encoder, Entailment
from .spec import Construct, texts

SCHEMA = "propaxis.instrument.v1"


@dataclass
class ScoreResult:
    """Document scores, supporting passages, and the scoring configuration ID."""

    scores: list[dict]
    evidence: list[dict]
    instrument_id: str

    def to_pandas(self):
        """Return scores as a DataFrame. Requires pandas (the notebook extra)."""
        try:
            import pandas as pd
        except ImportError as error:
            raise ImportError("Install pandas or use result.scores directly") from error
        return pd.DataFrame(self.scores)

    def to_json(self, path: str | Path) -> None:
        """Export results, including supplied text excerpts; review before sharing."""
        payload = {
            "instrument_id": self.instrument_id,
            "scores": self.scores,
            "evidence": self.evidence,
        }
        Path(path).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )


class Instrument:
    """A fitted Construct with saved axes, model revisions and scaling parameters.

    Create with ``construct.fit(reference_texts, encoder=...)`` or load a saved
    instrument. ``score`` never estimates statistics from the new documents.
    """

    @classmethod
    def fit(cls, spec: Construct, reference, *, encoder, claim_check=None):
        reference = texts(reference, "reference dataset", minimum=3)
        self = cls()
        self.spec = spec
        self.encoder = encoder
        self.claim_check = claim_check
        axes = []
        self._diagnostics = {}
        dimension = None
        for component in spec.components:
            count = len(component.positive)
            anchors = component.positive + component.contrast
            embedded = matrix(encoder.encode(anchors), len(anchors), "anchor embeddings")
            if dimension is not None and embedded.shape[1] != dimension:
                raise ValueError("encoder dimensions changed between components")
            dimension = embedded.shape[1]
            direction, gap = pole_axis(embedded[:count], embedded[count:])
            axes.append(direction)
            normalized = unit_rows(embedded)
            similarities = normalized @ direction
            stability = []
            for index in range(len(anchors)):
                positive, contrast = normalized[:count], normalized[count:]
                if index < count and count > 1:
                    positive = np.delete(positive, index, axis=0)
                elif index >= count and len(contrast) > 1:
                    contrast = np.delete(contrast, index - count, axis=0)
                else:
                    continue
                try:
                    alternative, _ = pole_axis(positive, contrast)
                    stability.append(float(direction @ alternative))
                except ValueError:
                    stability.append(0.0)
            self._diagnostics[component.name] = {
                "centroid_distance": gap,
                "positive_projection_mean": float(similarities[:count].mean()),
                "contrast_projection_mean": float(similarities[count:].mean()),
                "min_leave_one_out_cosine": min(stability) if stability else None,
                "anchor_projections": [
                    {
                        "text": anchor,
                        "pole": "positive" if i < count else "contrast",
                        "projection": float(similarities[i]),
                    }
                    for i, anchor in enumerate(anchors)
                ],
            }
        self.axes = np.column_stack(axes)
        self.reference_size = len(reference)
        self.reference_hash = hashlib.sha256(
            json.dumps(reference, ensure_ascii=False).encode()
        ).hexdigest()
        self._compute(reference, fitting=True)
        self.encoder_config = self.encoder.metadata()
        self.checker_config = None if claim_check is None else claim_check.metadata()
        # Identity describes the complete saved recipe and fitted scale, not the text being scored.
        self.instrument_id = self._identity(self._payload())
        return self

    @staticmethod
    def _identity(payload):
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()

    @property
    def diagnostics(self) -> dict:
        """Anchor projections, leave-one-out stability and cosine between axes.

        These are geometry checks, not evidence of substantive construct validity.
        Returns a detached copy so callers cannot alter the fitted instrument.
        """
        names = [c.name for c in self.spec.components]
        gram = self.axes.T @ self.axes
        return json.loads(
            json.dumps(
                {
                    "components": self._diagnostics,
                    "axis_cosines": {
                        name: {other: float(gram[i, j]) for j, other in enumerate(names)}
                        for i, name in enumerate(names)
                    },
                }
            )
        )

    def _compute(self, documents, fitting=False):
        windows, doc_ids = [], []
        for i, document in enumerate(documents):
            pieces = split_windows(document, self.spec.window, self.spec.max_words)
            windows.extend(pieces)
            doc_ids.extend([i] * len(pieces))
        doc_ids = np.asarray(doc_ids)
        embedded = matrix(self.encoder.encode(windows), len(windows), "window embeddings")
        if embedded.shape[1] != self.axes.shape[0]:
            raise ValueError("encoder dimension does not match the saved axes")
        if fitting:
            self.center = embedded.mean(axis=0)
        projections = unit_rows(embedded - self.center) @ self.axes
        if fitting:
            self.projection_mean = projections.mean(axis=0)
            self.projection_std = projections.std(axis=0)
            if (self.projection_std < 1e-12).any():
                raise ValueError(
                    "reference dataset has no variation on an axis; use more varied reference texts"
                )
        z = (projections - self.projection_mean) / self.projection_std
        activated = activation(z, self.spec.temperature)
        if fitting:
            self.activation_scale = np.maximum(np.median(activated, axis=0), 1e-12)
        activated = activated / self.activation_scale
        window_axis = np.column_stack([activated, combine(activated, self.spec.combine)])
        selections = [
            select_windows(window_axis[:, j], doc_ids, len(documents), self.spec.top_k)
            for j in range(window_axis.shape[1])
        ]
        axis = np.column_stack(
            [pool(window_axis[:, j], selections[j]) for j in range(window_axis.shape[1])]
        )
        if fitting:
            self.axis_mean = axis.mean(axis=0)
            self.axis_std = np.maximum(axis.std(axis=0), 1e-12)
        axis_z = (axis - self.axis_mean) / self.axis_std
        nli = nli_z = support = matches = None
        if self.claim_check is not None:
            # Score the union of selected windows. Each component uses its own top windows.
            selected = np.unique(
                np.concatenate([idx for per_target in selections for idx in per_target])
            )
            hypotheses = [h for c in self.spec.components for h in c.claims]
            probabilities = matrix(
                self.claim_check.predict([windows[i] for i in selected], hypotheses),
                len(selected),
                "entailment output",
            )
            if (
                probabilities.shape[1] != len(hypotheses)
                or ((probabilities < 0) | (probabilities > 1)).any()
            ):
                raise ValueError(
                    "entailment output must have one probability in [0, 1] per premise and hypothesis"
                )
            support = np.zeros((len(windows), len(self.spec.components)))
            matches = np.full(support.shape, -1, dtype=int)
            start = 0
            for j, component in enumerate(self.spec.components):
                stop = start + len(component.claims)
                support[selected, j] = probabilities[:, start:stop].max(axis=1)
                matches[selected, j] = probabilities[:, start:stop].argmax(axis=1)
                start = stop
            window_nli = np.column_stack([support, combine(support, self.spec.combine)])
            nli = np.column_stack(
                [pool(window_nli[:, j], selections[j]) for j in range(window_nli.shape[1])]
            )
            if fitting:
                self.nli_mean = nli.mean(axis=0)
                self.nli_std = np.maximum(nli.std(axis=0), 1e-12)
            nli_z = (nli - self.nli_mean) / self.nli_std
        scores = axis_z if nli_z is None else (axis_z + nli_z) / 2
        if not np.isfinite(scores).all():
            raise ValueError("non-finite scores; check the reference data and encoder")
        return dict(
            windows=windows,
            doc_ids=doc_ids,
            projections=projections,
            z=z,
            activated=activated,
            selections=selections,
            axis=axis,
            scores=scores,
            nli=nli,
            support=support,
            matches=matches,
        )

    def score(self, documents, *, ids=None) -> ScoreResult:
        """Score a string or a sequence of nonempty documents; retain input order.

        A score is standardized relative to the reference dataset. It is neither
        a class probability nor an estimate of the fraction of text in a class.
        """
        if self.encoder.metadata() != self.encoder_config:
            raise ValueError("encoder settings changed; fit a new instrument")
        if self.claim_check is not None and self.claim_check.metadata() != self.checker_config:
            raise ValueError("claim checker settings changed; fit a new instrument")
        if self._identity(self._payload()) != self.instrument_id:
            raise ValueError("fitted parameters changed; fit a new instrument")
        documents = texts([documents] if isinstance(documents, str) else documents, "documents")
        if ids is None:
            ids = [str(i) for i in range(len(documents))]
        else:
            ids = list(ids)
        if (
            len(ids) != len(documents)
            or any(not isinstance(i, str) for i in ids)
            or len(set(ids)) != len(ids)
        ):
            raise ValueError("ids must be unique strings with one ID per document")
        data = self._compute(documents)
        scores, evidence = [], []
        for d, identifier in enumerate(ids):
            row = {
                "id": identifier,
                "score": float(data["scores"][d, -1]),
                "axis": float(data["axis"][d, -1]),
                "n_windows": int((data["doc_ids"] == d).sum()),
            }
            if data["nli"] is not None:
                row["claim_check"] = float(data["nli"][d, -1])
            for j, component in enumerate(self.spec.components):
                row[f"{component.name}_score"] = float(data["scores"][d, j])
                row[f"{component.name}_axis"] = float(data["axis"][d, j])
                if data["nli"] is not None:
                    row[f"{component.name}_claim_check"] = float(data["nli"][d, j])
            scores.append(row)
            targets = {}
            for j, name in enumerate([c.name for c in self.spec.components] + ["combined"]):
                rows = []
                for index in data["selections"][j][d]:
                    item = {
                        "text": data["windows"][index],
                        "window_index": int(index - np.flatnonzero(data["doc_ids"] == d)[0]),
                        "components": {},
                    }
                    for k, component in enumerate(self.spec.components):
                        details = {
                            "projection": float(data["projections"][index, k]),
                            "standardized_projection": float(data["z"][index, k]),
                            "activation": float(data["activated"][index, k]),
                        }
                        if data["support"] is not None:
                            details["entailment"] = float(data["support"][index, k])
                            details["matched_hypothesis"] = component.claims[
                                data["matches"][index, k]
                            ]
                        item["components"][component.name] = details
                    rows.append(item)
                targets[name] = rows
            evidence.append({"id": identifier, "targets": targets})
        return ScoreResult(scores, evidence, self.instrument_id)

    def _payload(self):
        array_names = [
            "axes",
            "center",
            "projection_mean",
            "projection_std",
            "activation_scale",
            "axis_mean",
            "axis_std",
        ]
        if self.claim_check is not None:
            array_names += ["nli_mean", "nli_std"]
        return {
            "schema": SCHEMA,
            "package_version": "0.1.0",
            "construct": self.spec.to_dict(),
            "encoder": self.encoder_config,
            "claim_check": self.checker_config,
            "reference_size": self.reference_size,
            "reference_hash": self.reference_hash,
            "diagnostics": self._diagnostics,
            "parameters": {name: getattr(self, name).tolist() for name in array_names},
        }

    def save(self, path: str | Path) -> None:
        """Save JSON with axes, scales, propositions and model revisions. No raw reference texts or model weights."""
        payload = self._payload()
        if self._identity(payload) != self.instrument_id:
            raise ValueError("fitted parameters have changed; fit a new instrument before saving")
        payload["instrument_id"] = self.instrument_id
        Path(path).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        encoder=None,
        claim_check=None,
        device="cpu",
        local_files_only=False,
    ):
        """Load saved parameters. Custom backends must be supplied explicitly.

        Standard backends load lazily from their exact saved model revision.
        ``local_files_only=True`` requires those models to be cached already.
        """
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema") != SCHEMA:
            raise ValueError("unsupported instrument schema")
        identity = payload.pop("instrument_id")
        if cls._identity(payload) != identity:
            raise ValueError("instrument checksum mismatch; fit again after editing a construct")
        enc = payload["encoder"]
        if encoder is None:
            if enc["kind"] != "e5":
                raise ValueError("supply the original custom encoder to Instrument.load")
            encoder = Encoder(
                enc["model"],
                revision=enc["revision"],
                max_length=enc["max_length"],
                device=device,
                local_files_only=local_files_only,
            )
        if encoder.metadata() != enc:
            raise ValueError("encoder does not match the saved model and preprocessing")
        checker = payload["claim_check"]
        if checker is not None:
            if claim_check is None:
                if checker["kind"] != "nli":
                    raise ValueError("supply the original custom claim checker to Instrument.load")
                claim_check = Entailment(
                    checker["model"],
                    revision=checker["revision"],
                    max_length=checker["max_length"],
                    device=device,
                    local_files_only=local_files_only,
                )
            if claim_check.metadata() != checker:
                raise ValueError("claim checker does not match the saved model")
        elif claim_check is not None:
            raise ValueError("this instrument was fitted without a claim checker")
        self = cls()
        self.spec = Construct.from_dict(payload["construct"])
        self.encoder, self.claim_check = encoder, claim_check
        self.encoder_config, self.checker_config = enc, checker
        self.reference_size, self.reference_hash = (
            payload["reference_size"],
            payload["reference_hash"],
        )
        self._diagnostics, self.instrument_id = payload["diagnostics"], identity
        expected = {
            "axes",
            "center",
            "projection_mean",
            "projection_std",
            "activation_scale",
            "axis_mean",
            "axis_std",
        }
        if checker is not None:
            expected |= {"nli_mean", "nli_std"}
        if set(payload["parameters"]) != expected:
            raise ValueError("invalid saved parameter set")
        for name, value in payload["parameters"].items():
            array = np.asarray(value, dtype=np.float64)
            if not np.isfinite(array).all():
                raise ValueError(f"non-finite saved parameter: {name}")
            setattr(self, name, array)
        n = len(self.spec.components)
        if (
            self.axes.ndim != 2
            or self.axes.shape[1] != n
            or self.center.shape != (self.axes.shape[0],)
        ):
            raise ValueError("invalid axis or centering dimensions")
        for name in expected - {"axes", "center"}:
            size = n if name in {"projection_mean", "projection_std", "activation_scale"} else n + 1
            if getattr(self, name).shape != (size,):
                raise ValueError(f"invalid dimensions for {name}")
            if (name.endswith("std") or name == "activation_scale") and (
                getattr(self, name) <= 0
            ).any():
                raise ValueError(f"invalid scale: {name}")
        return self
