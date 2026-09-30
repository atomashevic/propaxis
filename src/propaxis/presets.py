"""Editable construct definitions; fitted parameters must come from your reference data."""

import json
from importlib.resources import files

from .spec import Construct


def populism() -> Construct:
    """Forty P/E propositions and ten hypotheses from the Populism Axis project.

    Three-sentence windows and top-eight averaging follow the source configuration.
    Use Encoder('e5-large-v2') and Entailment() for the source model families.
    Package defaults and new reference data do not reproduce published benchmarks.
    """
    payload = files("propaxis").joinpath("populism.json").read_text(encoding="utf-8")
    return Construct.from_dict(json.loads(payload))
