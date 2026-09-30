import subprocess
import sys

import pytest

from propaxis import Encoder


def test_import_does_not_load_models_or_notebooks():
    command = "import sys, propaxis; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules; assert 'ipywidgets' not in sys.modules"
    subprocess.run([sys.executable, "-c", command], check=True)


def test_encoder_metadata_and_validation():
    encoder = Encoder("e5-large-v2", revision="test-revision")
    assert encoder.metadata()["model"] == "intfloat/e5-large-v2"
    assert encoder.metadata()["prefix"] == "query: "
    assert encoder.metadata()["revision"] == "test-revision"
    assert encoder._model is None
    with pytest.raises(ValueError):
        Encoder("unknown")
    with pytest.raises(ValueError):
        Encoder(batch_size=0)
