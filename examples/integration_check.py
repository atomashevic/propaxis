"""Optional real-model check. Downloads E5-small and BART on first use."""

import gc
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from quickstart import REFERENCE, TRANSIT

from propaxis import Construct, Encoder, Entailment, Instrument


def main():
    construct = Construct("transit_stance", [TRANSIT])
    instrument = construct.fit(
        REFERENCE,
        encoder=Encoder("e5-small-v2"),
        claim_check=Entailment(batch_size=4),
    )
    documents = [
        "The city should invest much more in buses and trains.",
        "Public transportation should receive less government funding.",
    ]
    result = instrument.score(documents)
    assert result.scores[0]["score"] > result.scores[1]["score"]
    single = instrument.score(documents[0])
    np.testing.assert_allclose(result.scores[0]["score"], single.scores[0]["score"], atol=1e-4)
    assert 0 <= result.scores[0]["claim_check"] <= 1
    metadata = {"encoder": instrument.encoder_config, "claim_check": instrument.checker_config}
    for spec in metadata.values():
        assert len(spec["revision"]) == 40
    with TemporaryDirectory() as directory:
        path = Path(directory) / "instrument.json"
        instrument.save(path)
        del instrument
        gc.collect()
        restored = Instrument.load(path, local_files_only=True)
        repeat = restored.score(documents)
        np.testing.assert_allclose(
            [r["score"] for r in result.scores],
            [r["score"] for r in repeat.scores],
            atol=1e-4,
        )
    print(json.dumps({"models": metadata, "scores": result.scores, "checks": "passed"}, indent=2))


if __name__ == "__main__":
    main()
