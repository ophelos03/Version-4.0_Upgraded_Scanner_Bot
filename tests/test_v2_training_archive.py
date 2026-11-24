import importlib.util
import json
import sqlite3
import time
from pathlib import Path


def load_training_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "train_xgboost.py"
    spec = importlib.util.spec_from_file_location("scannerbot_train_xgboost", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_matured_examples_are_archived_once(tmp_path):
    module = load_training_module()
    con = sqlite3.connect(tmp_path / "scanner.db")
    now = time.time()
    baseline_at = now - 25 * 3600
    features = json.dumps({"price_usd": 1.0, "liquidity_usd": 10000})
    con.execute("""CREATE TABLE training_baselines(
        chain_id TEXT NOT NULL, token_address TEXT NOT NULL, pair_address TEXT,
        captured_at REAL NOT NULL, pair_age_minutes REAL NOT NULL, baseline_price REAL NOT NULL,
        features_json TEXT NOT NULL, outcome_complete INTEGER NOT NULL DEFAULT 0,
        max_multiple REAL, observed_count INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY(chain_id,token_address))""")
    con.execute("""CREATE TABLE training_observations(
        chain_id TEXT NOT NULL, token_address TEXT NOT NULL, captured_at REAL NOT NULL,
        price_usd REAL NOT NULL, PRIMARY KEY(chain_id,token_address,captured_at))""")
    con.execute(
        """INSERT INTO training_baselines
           (chain_id,token_address,pair_address,captured_at,pair_age_minutes,baseline_price,features_json)
           VALUES('solana','mint','pair',?,60,1,?)""",
        (baseline_at, features),
    )
    outcome_end = baseline_at + 24 * 3600
    observations = []
    for offset in range(60, 24 * 3600, 300):
        price = 2.5 if offset == 43_260 else 1.2
        observations.append(("solana","mint",baseline_at+offset,price))
    observations.append(("solana","mint",outcome_end-1,1.5))
    con.executemany(
        "INSERT INTO training_observations(chain_id,token_address,captured_at,price_usd) VALUES(?,?,?,?)",
        observations,
    )
    con.commit()

    inserted = module.archive_matured_examples(con, 24)
    assert inserted == 1
    row = con.execute(
        "SELECT max_multiple,y_2x,y_5x,y_10x FROM eligible_training_examples"
    ).fetchone()
    assert row == (2.5, 1, 0, 0)
    watermark = con.execute(
        "SELECT processed_before,horizon_seconds FROM eligible_training_archive_state WHERE id=1"
    ).fetchone()
    assert watermark is not None and watermark[1] == 24 * 3600
    assert module.archive_matured_examples(con, 24) == 0
    con.close()


def test_sliding_window_maxima_matches_brute_force():
    import random

    module = load_training_module()
    rng = random.Random(20261117)
    for _ in range(100):
        count = rng.randint(2, 80)
        # Duplicate timestamps exercise the strict "later than this snapshot" rule.
        times = []
        current = 1_000.0
        for _ in range(count):
            current += rng.choice([0.0, 0.0, 1.0, 5.0, 30.0])
            times.append(current)
        prices = [rng.choice([0.0, 0.1, 1.0, 2.0, 5.0, 10.0]) for _ in range(count)]
        liquidities = [rng.choice([0.0, 5_000.0, 8_000.0, 20_000.0]) for _ in range(count)]
        horizon = rng.choice([1.0, 5.0, 30.0, 60.0])
        floor = 8_000.0

        actual = module._future_window_maxima(times, prices, liquidities, horizon, floor)
        expected = []
        for i, captured_at in enumerate(times):
            valid = [
                prices[j] for j in range(i + 1, count)
                if captured_at < times[j] <= captured_at + horizon
                and prices[j] > 0 and liquidities[j] >= floor
            ]
            expected.append(max(valid) if valid else None)
        assert actual == expected


def test_sliding_window_maxima_respects_inclusive_horizon_and_liquidity():
    module = load_training_module()
    times = [100.0, 101.0, 105.0, 105.0, 106.0]
    prices = [1.0, 3.0, 9.0, 20.0, 7.0]
    liquidities = [10_000.0, 10_000.0, 7_999.0, 10_000.0, 10_000.0]

    actual = module._future_window_maxima(times, prices, liquidities, 5.0, 8_000.0)
    assert actual == [20.0, 20.0, 7.0, 7.0, None]



def test_chronological_splits_keep_new_matured_batch_unseen():
    import pandas as pd

    module = load_training_module()
    rows = []
    for i in range(20):
        rows.append({
            "chain_id": "solana",
            "token_address": f"token-{i}",
            "captured_at": float(i * 10_000),
            "archived_at": 500.0 if i < 10 else 1_500.0,
            "feature": 1.0,
        })
    df = pd.DataFrame(rows)

    train, calibration, test = module.chronological_token_splits(
        df, horizon_hours=2, since_timestamp=1_000.0,
    )
    train_tokens = set(train.token_address)
    calibration_tokens = set(calibration.token_address)
    test_tokens = set(test.token_address)

    assert train_tokens.isdisjoint(calibration_tokens)
    assert train_tokens.isdisjoint(test_tokens)
    assert calibration_tokens.isdisjoint(test_tokens)
    assert test_tokens == {f"token-{i}" for i in range(10, 20)}
    assert len(train) + len(calibration) == 10
