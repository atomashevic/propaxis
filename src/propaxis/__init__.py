"""Proposition-based text measurement and classification."""

from .calibration import BinaryClassifier, calibrate
from .instrument import Instrument, ScoreResult
from .models import CallableEncoder, Encoder, Entailment, download_models
from .spec import Component, Construct

__version__ = "0.1.0"
__all__ = [
    "Component",
    "Construct",
    "Encoder",
    "CallableEncoder",
    "Entailment",
    "Instrument",
    "ScoreResult",
    "BinaryClassifier",
    "calibrate",
    "download_models",
]
