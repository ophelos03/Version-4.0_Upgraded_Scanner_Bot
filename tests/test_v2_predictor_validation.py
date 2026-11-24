import json
from pathlib import Path

from scannerbot.prediction.predictor import UpsidePredictor


def test_predictor_refuses_model_set_without_validation_metadata(tmp_path):
    for key in ("2x", "5x", "10x"):
        (tmp_path / f"xgb_{key}.joblib").write_bytes(b"not loaded")
    predictor = UpsidePredictor(str(tmp_path))
    assert predictor.ready is False


def test_predictor_refuses_incomplete_validation_metadata(tmp_path):
    (tmp_path / "metadata.json").write_text(json.dumps({
        "validation_passed": True, "targets": {"2x": {}, "5x": {}}
    }))
    predictor = UpsidePredictor(str(tmp_path))
    assert predictor.ready is False
