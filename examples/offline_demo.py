"""Run with only NumPy. The toy encoder demonstrates the API, not semantic validity."""

import json
import re

import numpy as np

from propaxis import CallableEncoder, Component, Construct, Instrument, calibrate


def toy_encode(texts):
    """Three hand-coded features, used only to make this example download-free."""
    return np.array(
        [
            [
                1,
                len(re.findall(r"\b(support|expand|invest)\b", text.lower())),
                len(re.findall(r"\b(oppose|reduce|cut)\b", text.lower())),
            ]
            for text in texts
        ],
        dtype=float,
    )


def main():
    encoder = CallableEncoder(toy_encode, name="tutorial-counts-v1")
    component = Component(
        "transit_support",
        positive=["We support public transit.", "We should expand and invest in transit."],
        contrast=["We oppose public transit.", "We should reduce and cut transit spending."],
    )
    construct = Construct("transit_stance", [component])
    reference = [
        "We support transit.",
        "Expand and invest in transit.",
        "We oppose transit.",
        "Reduce and cut spending.",
        "We discussed the transit budget.",
        "Support expansion but cut other spending.",
    ]
    instrument = construct.fit(reference, encoder=encoder)
    calibration = instrument.score(
        ["Support and invest.", "Expand transit.", "Oppose transit.", "Reduce spending."]
    )
    classifier = calibrate(calibration, [1, 1, 0, 0])
    result = instrument.score(
        ["We support and expand transit.", "We oppose and cut transit funding."]
    )
    print(json.dumps(result.scores, indent=2))
    print("Predicted labels:", classifier.predict(result).tolist())
    from pathlib import Path
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as directory:
        path = Path(directory) / "instrument.json"
        instrument.save(path)
        restored = Instrument.load(path, encoder=encoder)
        assert (
            restored.score("We support and expand transit.").scores[0]["score"]
            == result.scores[0]["score"]
        )
    print("Saved parameters reproduce the same score.")


if __name__ == "__main__":
    main()
