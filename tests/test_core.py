import json

import numpy as np
import pytest

from propaxis import CallableEncoder, Component, Construct, Instrument
from propaxis.geometry import combine, pole_axis, split_windows
from propaxis.presets import populism


def encode(texts):
    # Stable, nonlinear vectors for arbitrary text without neural dependencies.
    import hashlib

    return np.array(
        [[1.0] + [b / 255 for b in hashlib.sha256(t.encode()).digest()[:7]] for t in texts]
    )


@pytest.fixture
def encoder():
    return CallableEncoder(encode, "test-hash-v1")


@pytest.fixture
def spec():
    return Construct(
        "test",
        [
            Component("P", ["people decide", "people rule"], ["experts decide", "officials rule"]),
            Component(
                "E",
                ["elites obstruct", "insiders obstruct"],
                ["institutions serve", "officials serve"],
            ),
        ],
        window=1,
        top_k=2,
    )


@pytest.fixture
def reference():
    return [
        "A first sentence. And another.",
        "A different claim. Something else.",
        "Text about government. News today.",
        "One more document. With different words.",
    ]


class Checker:
    def predict(self, premises, hypotheses):
        return np.asarray(
            [
                [((sum(p.encode()) + sum(h.encode())) % 91 + 1) / 100 for h in hypotheses]
                for p in premises
            ]
        )

    def metadata(self):
        return {"kind": "custom", "name": "test-checker-v1"}


def test_geometry_matches_centroid_definition_and_pair_order():
    p, c = (
        np.array([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]]),
        np.array([[0.0, 0.0, 5.0], [0.0, 1.0, 1.0]]),
    )
    axis, _ = pole_axis(p, c)
    expected = (p / np.linalg.norm(p, axis=1)[:, None]).mean(0) - (
        c / np.linalg.norm(c, axis=1)[:, None]
    ).mean(0)
    np.testing.assert_allclose(axis, expected / np.linalg.norm(expected))
    np.testing.assert_allclose(axis, pole_axis(p[::-1], c)[0])
    with pytest.raises(ValueError, match="zero-length"):
        pole_axis(p, p)


def test_windowing_caps_preserves_words_and_overlaps():
    long = "one two three four, five six seven eight; nine ten eleven."
    windows = split_windows(long, max_words=3)
    assert " ".join(windows).split() == long.split()
    assert all(len(w.split()) <= 3 for w in windows)
    assert split_windows("First. Second. Third. Fourth.", 3) == [
        "First. Second. Third.",
        "Second. Third. Fourth.",
    ]


@pytest.mark.parametrize("checker", [None, Checker()])
def test_batch_invariance_and_save_load(spec, reference, encoder, checker, tmp_path):
    instrument = spec.fit(reference, encoder=encoder, claim_check=checker)
    docs = ["A new sentence. Another statement.", "Something unrelated."]
    result = instrument.score(docs, ids=["x", "y"])
    single = instrument.score(docs[0], ids=["x"])
    assert result.scores[0] == single.scores[0]
    assert result.evidence[0] == single.evidence[0]
    assert len(result.evidence[0]["targets"]["combined"]) == 2
    path = tmp_path / "instrument.json"
    instrument.save(path)
    assert reference[0] not in path.read_text()
    loaded = Instrument.load(path, encoder=encoder, claim_check=checker)
    assert loaded.score(docs, ids=["x", "y"]).scores == result.scores
    assert loaded.instrument_id == instrument.instrument_id
    assert loaded.diagnostics == instrument.diagnostics
    result.to_json(tmp_path / "scores.json")
    assert (
        json.loads((tmp_path / "scores.json").read_text())["instrument_id"]
        == instrument.instrument_id
    )


def test_scaling_is_estimated_once(spec, reference, encoder):
    instrument = spec.fit(reference, encoder=encoder)
    result = instrument.score(reference)
    scores = np.array([r["score"] for r in result.scores])
    np.testing.assert_allclose(scores.mean(), 0, atol=1e-12)
    np.testing.assert_allclose(scores.std(), 1, atol=1e-12)
    original = instrument.center.copy()
    instrument.score(["Brand new corpus.", "Other examples."])
    np.testing.assert_array_equal(original, instrument.center)


def test_direct_numerical_pipeline_matches_saved_parameters(spec, reference, encoder):
    instrument = spec.fit(reference, encoder=encoder)
    window = "An unseen test sentence."
    vector = encode([window])[0] - instrument.center
    projections = (vector / np.linalg.norm(vector)) @ instrument.axes
    z = (projections - instrument.projection_mean) / instrument.projection_std
    activated = np.logaddexp(0, z / 0.25) ** 2 / instrument.activation_scale
    raw = np.sqrt(np.prod(activated))
    expected = (raw - instrument.axis_mean[-1]) / instrument.axis_std[-1]
    np.testing.assert_allclose(instrument.score(window).scores[0]["score"], expected)


def test_independent_component_windows_and_hypotheses(spec, reference, encoder):
    instrument = spec.fit(reference, encoder=encoder, claim_check=Checker())
    result = instrument.score("Sentence one. Sentence two. Sentence three.")
    for target in ["P", "E", "combined"]:
        for window in result.evidence[0]["targets"][target]:
            for component in spec.components:
                detail = window["components"][component.name]
                assert detail["matched_hypothesis"] in component.claims
                expected = Checker().predict([window["text"]], component.claims).max()
                assert detail["entailment"] == expected


def test_construct_roundtrip_and_immutable_lists(spec, tmp_path):
    path = tmp_path / "construct.json"
    spec.save(path)
    assert Construct.load(path) == spec
    assert Construct.load(path).fingerprint == spec.fingerprint
    original = ["first", "second"]
    component = Component("test", original, ["third"])
    original.append("fourth")
    assert component.positive == ("first", "second")


@pytest.mark.parametrize("values", [[], [""], [None], "a string", ["same", "same"]])
def test_invalid_propositions(values):
    with pytest.raises((ValueError, TypeError)):
        Component("test", values, ["contrast"])


def test_invalid_specs():
    c = Component("test", ["positive"], ["contrast"])
    for kwargs in [
        {"window": 0},
        {"top_k": -1},
        {"max_words": True},
        {"temperature": float("nan")},
        {"combine": "invalid"},
    ]:
        with pytest.raises(ValueError):
            Construct("test", [c], **kwargs)
    with pytest.raises(ValueError):
        Construct("test", [c, c])
    with pytest.raises(ValueError):
        Component("test", ["same"], ["same"])


def test_invalid_encoder_and_degenerate_reference(spec, reference):
    with pytest.raises(ValueError, match="matrix"):
        spec.fit(reference, encoder=CallableEncoder(lambda x: [1, 2], "bad"))
    with pytest.raises(ValueError, match="non-finite"):
        spec.fit(reference, encoder=CallableEncoder(lambda x: np.full((len(x), 3), np.nan), "bad"))
    with pytest.raises(ValueError, match="variation"):
        spec.fit(["same"] * 3, encoder=CallableEncoder(encode, "hash"))


def test_blank_input_ids_and_checker_validation(spec, reference, encoder):
    instrument = spec.fit(reference, encoder=encoder)
    for documents in [[], [None], [""], ["  "]]:
        with pytest.raises(ValueError):
            instrument.score(documents)
    with pytest.raises(ValueError):
        instrument.score(["one", "two"], ids=["same", "same"])

    class BadChecker(Checker):
        def predict(self, premises, hypotheses):
            return np.full((len(premises), len(hypotheses)), 2.0)

    with pytest.raises(ValueError, match="probability"):
        spec.fit(reference, encoder=encoder, claim_check=BadChecker())


def test_load_rejects_edits_and_backend_changes(spec, reference, encoder, tmp_path):
    instrument = spec.fit(reference, encoder=encoder)
    path = tmp_path / "instrument.json"
    instrument.save(path)
    with pytest.raises(ValueError, match="custom encoder"):
        Instrument.load(path)
    with pytest.raises(ValueError, match="does not match"):
        Instrument.load(path, encoder=CallableEncoder(encode, "other"))
    payload = json.loads(path.read_text())
    payload["construct"]["window"] = 10
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="checksum"):
        Instrument.load(path, encoder=encoder)


@pytest.mark.parametrize("rule", ["geometric_mean", "mean", "minimum"])
def test_combinations_and_all_window_pooling(rule, spec, reference, encoder):
    alternative = Construct("test", spec.components, combine=rule, top_k=None)
    fitted = alternative.fit(reference, encoder=encoder)
    assert len(fitted.score("First. Second. Third.").evidence[0]["targets"]["combined"]) == 3
    assert combine(np.array([[0.0, 1.0]]), "geometric_mean")[0] == 0


def test_populism_preset():
    spec = populism()
    assert len(spec.components) == 2
    assert sum(len(c.positive) + len(c.contrast) for c in spec.components) == 40
    assert sum(len(c.claims) for c in spec.components) == 10
    assert spec.window == 3 and spec.top_k == 8
