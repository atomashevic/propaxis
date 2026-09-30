import numpy as np
import pytest

from propaxis import BinaryClassifier, ScoreResult, calibrate


def result(values, identity="test"):
    return ScoreResult([{"score": v} for v in values], [], identity)


def test_threshold_uses_calibration_and_predicts_held_out():
    classifier = calibrate(result([-2, -1, 1, 2]), [0, 0, 1, 1])
    assert classifier.threshold == 1
    heldout = result([-3, 3, -2, 4])
    assert classifier.predict(heldout).tolist() == [0, 1, 0, 1]
    metrics = classifier.evaluate(heldout, [0, 1, 0, 1])
    for key in ["balanced_accuracy", "f1", "auroc", "average_precision"]:
        assert metrics[key] == 1


def test_ties_auc_and_average_precision():
    classifier = calibrate(result([0, 0, 0, 0]), [0, 1, 0, 1])
    assert classifier.predict(result([0])).tolist() == [0]
    metrics = classifier.evaluate(result([0, 0, 0, 0]), [0, 1, 0, 1])
    assert metrics["auroc"] == 0.5
    assert metrics["average_precision"] == 0.5
    assert metrics["f1"] == 0
    other = calibrate(result([1, 2, 3, 4]), [0, 1, 0, 1], objective="f1")
    metrics = other.evaluate(result([1, 2, 3, 4]), [0, 1, 0, 1])
    assert metrics["auroc"] == 0.75
    np.testing.assert_allclose(metrics["average_precision"], 5 / 6)


def test_serialization_and_instrument_binding(tmp_path):
    classifier = calibrate(result([0, 1, 2]), [0, 0, 1])
    path = tmp_path / "classifier.json"
    classifier.save(path)
    assert BinaryClassifier.load(path) == classifier
    with pytest.raises(ValueError, match="different instrument"):
        classifier.predict(result([1], "another"))


@pytest.mark.parametrize("labels", [[0, 0], [1, 1], [0, 2], [0], [[0], [1]]])
def test_invalid_labels(labels):
    with pytest.raises(ValueError):
        calibrate(result([0, 1]), labels)


def test_invalid_scores_and_objective():
    with pytest.raises(ValueError):
        calibrate(result([float("nan"), 1]), [0, 1])
    with pytest.raises(ValueError):
        calibrate(result([0, 1]), [0, 1], objective="accuracy")
