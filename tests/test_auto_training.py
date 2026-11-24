from types import SimpleNamespace

from scannerbot.auto_training import AutoTrainingManager


def manager(tmp_path, **overrides):
    settings = SimpleNamespace(
        model_dir=str(tmp_path / "models"),
        min_training_rows=100,
        auto_train_min_tokens=100,
        auto_train_min_test_positives=2,
        auto_train_min_test_negatives=2,
        auto_train_require_baseline_beat=True,
        **overrides,
    )
    return AutoTrainingManager(settings, predictor=None, telegram=None)


def good_report():
    return {
        "rows": 1000,
        "distinct_tokens": 150,
        "targets": {
            key: {
                "test_positives": 5,
                "test_negatives": 15,
                "challenger_beats_incumbent": True,
                "average_precision": 0.25,
                "positive_rate": 0.10,
                "brier_score": 0.08,
                "baseline_brier": 0.09,
            }
            for key in ("2x", "5x", "10x")
        },
    }


def test_candidate_passes_when_all_targets_beat_baseline(tmp_path):
    ok, reason = manager(tmp_path)._candidate_passes(good_report())
    assert ok
    assert "passed" in reason


def test_validation_retries_when_positive_count_is_too_low(tmp_path):
    report = good_report()
    report["targets"]["10x"]["test_positives"] = 1
    ok, reason = manager(tmp_path)._validation_data_ready(report)
    assert not ok
    assert "positive validation examples" in reason


def test_validation_retries_when_negative_count_is_too_low(tmp_path):
    report = good_report()
    report["targets"]["5x"]["test_negatives"] = 1
    ok, reason = manager(tmp_path)._validation_data_ready(report)
    assert not ok
    assert "negative validation examples" in reason


def test_candidate_rejected_when_incumbent_performs_better(tmp_path):
    report = good_report()
    report["targets"]["5x"]["challenger_beats_incumbent"] = False
    ok, reason = manager(tmp_path)._candidate_passes(report)
    assert not ok
    assert "did not outperform" in reason


def test_candidate_rejected_when_dataset_is_too_small(tmp_path):
    report = good_report()
    report["rows"] = 10
    ok, reason = manager(tmp_path)._candidate_passes(report)
    assert not ok
    assert "rows" in reason
