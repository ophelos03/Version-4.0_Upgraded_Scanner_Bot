import tempfile
import unittest
from pathlib import Path

from scannerbot.database import Database
from scannerbot.models import Candidate
from scripts.train_xgboost import archive_matured_examples, load_archive


class EligibleTrainingArchiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "scannerbot.db"
        self.db = Database(str(self.db_path), outcome_horizon_hours=2, training_archive_path=str(self.db_path))

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def test_discovery_is_deduplicated_globally_and_per_source(self):
        candidate = Candidate("solana", "TokenMint111111111111111111111111111111111", "shrine", {})
        self.assertEqual(self.db.register_candidate(candidate), (True, True))
        self.assertEqual(self.db.register_candidate(candidate), (False, False))
        same_token_other_source = Candidate("solana", candidate.token_address, "pumpportal", {})
        self.assertEqual(self.db.register_candidate(same_token_other_source), (False, True))

    def test_one_baseline_and_future_prices_label_independently_without_liquidity_gate(self):
        token = "TokenMint111111111111111111111111111111111"
        baseline_at = 1_000.0
        features = {"price_usd": 1.0, "liquidity_usd": 8_500.0}
        self.assertTrue(self.db.create_training_baseline(
            "solana", token, "pool", baseline_at, 1.0, 1.0, features,
        ))
        self.assertFalse(self.db.create_training_baseline(
            "solana", token, "pool", baseline_at + 1, 1.1, 1.1, features,
        ))
        # The observation API deliberately accepts price only: later liquidity
        # drops cannot disqualify a token or suppress its future price high.
        for offset in range(60, 7_200, 300):
            price = 9.0 if offset == 6_960 else (7.0 if offset == 3_600 else (2.5 if offset == 1_800 else 1.5))
            self.db.record_training_observation("solana", token, baseline_at + offset, price)
        self.db.record_training_observation("solana", token, baseline_at + 7_199, 9.0)

        import sqlite3
        con = sqlite3.connect(str(self.db_path))
        try:
            added = archive_matured_examples(con, 2)
            self.assertEqual(added, 1)
            frame = load_archive(con)
        finally:
            con.close()

        self.assertEqual(len(frame), 1)
        self.assertEqual(int(frame.iloc[0]["y_2x"]), 1)
        self.assertEqual(int(frame.iloc[0]["y_5x"]), 1)
        self.assertEqual(int(frame.iloc[0]["y_10x"]), 0)
        self.assertEqual(float(frame.iloc[0]["max_multiple_in_window"]), 9.0)
        self.assertTrue(self.db.has_training_baseline("solana", token))


def test_continuously_observed_crash_is_kept_as_a_negative_example(tmp_path):
    import sqlite3
    import time

    db_path = Path(tmp_path) / "scannerbot.db"
    db = Database(str(db_path), outcome_horizon_hours=2, training_archive_path=str(db_path))
    token = "TokenMintCrash1111111111111111111111111111111"
    baseline_at = time.time() - 7_300
    db.create_training_baseline("solana", token, "pool", baseline_at, 1.0, 1.0, {"price_usd": 1.0})
    for offset in range(60, 7_200, 300):
        db.record_training_observation("solana", token, baseline_at + offset, 1.1)
    db.record_training_observation("solana", token, baseline_at + 7_199, 0.0)
    db.close()

    con = sqlite3.connect(str(db_path))
    try:
        assert archive_matured_examples(con, 2) == 1
        row = con.execute(
            "SELECT max_multiple,y_2x,y_5x,y_10x FROM eligible_training_examples"
        ).fetchone()
        assert row == (1.1, 0, 0, 0)
    finally:
        con.close()


def test_training_archive_survives_recreating_ephemeral_scanner_database(tmp_path):
    scanner_db = Path(tmp_path) / "scannerbot.db"
    archive_db = Path(tmp_path) / "models" / "training_archive.sqlite"
    token = "TokenMintPersistent111111111111111111111111111"

    first = Database(str(scanner_db), outcome_horizon_hours=2, training_archive_path=str(archive_db))
    assert first.create_training_baseline(
        "solana", token, "pool", 10_000.0, 1.0, 1.0, {"price_usd": 1.0},
    )
    first.close()

    # Simulate a new Railway deployment: scannerbot.db may be recreated,
    # while the training archive remains on the persistent model volume.
    scanner_db.unlink()
    second = Database(str(scanner_db), outcome_horizon_hours=2, training_archive_path=str(archive_db))
    try:
        assert second.has_training_baseline("solana", token)
        assert second.training_example_count() == 0
        assert archive_db.exists()
    finally:
        second.close()


if __name__ == "__main__":
    unittest.main()
