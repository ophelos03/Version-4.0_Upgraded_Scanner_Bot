import json

from scannerbot.early_momentum import EarlyMomentumArchive, classify_early_momentum


def test_momentum_requires_independent_signals_and_wallet_evidence():
    features = {
        "price_return_since_snapshot_pct": 5,
        "volume_change_since_snapshot_pct": 30,
        "liquidity_change_since_snapshot_pct": 2,
        "buy_share_m5": 0.7,
        "transactions_m5": 12,
        "market_cap_usd": 20000,
    }
    result = classify_early_momentum(features)
    assert result["classification"] == "INSUFFICIENT EVIDENCE"
    result = classify_early_momentum(features, wallet_evidence={"low_manipulation": True})
    assert result["classification"] == "EARLY MOMENTUM"


def test_early_archive_records_short_horizon_excursions(tmp_path):
    db = EarlyMomentumArchive(str(tmp_path / "training_archive.sqlite"))
    features = {"price_usd": 1.0, "liquidity_usd": 9000}
    classification = {"classification": "INSUFFICIENT EVIDENCE", "signals": [], "cautions": []}
    db.record_observation("solana", "mint", 1000, 1.0, 9000, 15000, features, classification)
    db.record_observation("solana", "mint", 1120, 1.5, 9500, 22000, features, classification)
    row = db.conn.execute(
        "SELECT max_price_5m,min_price_5m,max_price_15m,mature_5m,mature_15m FROM early_momentum_samples ORDER BY captured_at LIMIT 1"
    ).fetchone()
    assert tuple(row) == (1.5, 1.0, 1.5, 0, 0)
    db.record_observation("solana", "mint", 1310, 1.2, 9500, 18000, features, classification)
    row = db.conn.execute(
        "SELECT mature_5m,mature_15m FROM early_momentum_samples ORDER BY captured_at LIMIT 1"
    ).fetchone()
    assert tuple(row) == (1, 0)
    db.record_observation("solana", "mint", 1900, 1.4, 10000, 51000, features, classification)
    assert db.market_cap_progression("solana", "mint") == [15000.0, 30000.0, 50000.0]
    db.close()


def test_dev_wallet_requires_five_5x_calls_and_two_rugs_disqualify(tmp_path):
    db = EarlyMomentumArchive(str(tmp_path / "training_archive.sqlite"))
    for i in range(5):
        assert db.record_dev_call("creator", "solana", f"token-{i}", 1.0, {}, 1000 + i)
        db.observe_dev_call("solana", f"token-{i}", 5.0)
    assert db.developer_wallet("creator")["status"] == "verified"
    db.record_dev_call("creator", "solana", "rug-a", 1.0, {}, 2000)
    db.observe_dev_call("solana", "rug-a", 0.9)
    db.record_dev_call("creator", "solana", "rug-b", 1.0, {}, 3000)
    db.observe_dev_call("solana", "rug-b", 0.9)
    assert db.developer_wallet("creator")["status"] == "disqualified"
    assert not db.record_dev_call("creator", "solana", "new-token", 1.0, {}, 4000)
    db.close()


def test_wallet_storage_limit_refuses_new_records(tmp_path):
    db = EarlyMomentumArchive(str(tmp_path / "training_archive.sqlite"), max_wallet_mb=1)
    db.max_wallet_bytes = 5
    assert not db.record_dev_call("creator", "solana", "token", 1.0, {"large": "payload"})
    db.close()


def test_zero_price_is_recorded_as_crash_not_missing_data(tmp_path):
    db = EarlyMomentumArchive(str(tmp_path / "training_archive.sqlite"))
    features = {"price_usd": 1.0}
    classification = {"classification": "INSUFFICIENT EVIDENCE", "signals": [], "cautions": []}
    db.record_observation("solana", "mint", 1000, 1.0, 9000, 15000, features, classification)
    db.record_outcome_observation("solana", "mint", 1060, 1.1, 5000)
    row = db.conn.execute(
        "SELECT max_price_5m,min_liquidity_5m,mature_5m FROM early_momentum_samples"
    ).fetchone()
    assert tuple(row) == (1.1, 5000.0, 0)
    db.record_zero_price_observation("solana", "mint", 1120)
    row = db.conn.execute(
        "SELECT min_price_5m,mature_5m FROM early_momentum_samples"
    ).fetchone()
    assert tuple(row) == (0.0, 0)
    db.close()



def test_early_entry_requires_target_band_confirmation_and_independent_signals():
    from scannerbot.early_momentum import evaluate_early_entry

    features = {
        "price_change_m5_pct": 3.0,
        "price_return_since_snapshot_pct": 1.5,
        "volume_acceleration_5m_vs_hour": 1.8,
        "transaction_acceleration_5m_vs_hour": 1.7,
        "buy_share_m5": 0.68,
        "transactions_m5": 20,
        "liquidity_change_since_snapshot_pct": 1.0,
        "snapshot_interval_seconds": 60,
    }
    history = [{"price_usd": 1.0}]
    result = evaluate_early_entry(features, 17500, history, {"low_manipulation": True})
    assert result["qualified"] is True
    assert result["score"] >= 60

    assert evaluate_early_entry(features, 22000, history, {})["qualified"] is False
    assert evaluate_early_entry(features, 17500, [], {})["qualified"] is False
    assert evaluate_early_entry(
        features, 17500, history, {"suspicious": True, "reasons": ["suspicious holder flow"]}
    )["qualified"] is False


def test_early_entry_requires_price_buy_pressure_and_acceleration():
    from scannerbot.early_momentum import evaluate_early_entry

    features = {
        "price_change_m5_pct": 3.0,
        "volume_acceleration_5m_vs_hour": 1.8,
        "buy_share_m5": 0.68,
        "transactions_m5": 20,
        "snapshot_interval_seconds": 60,
    }
    result = evaluate_early_entry(features, 17500, [{"price_usd": 1.0}], {})
    assert result["qualified"] is True

    features["buy_share_m5"] = 0.49
    assert evaluate_early_entry(features, 17500, [{"price_usd": 1.0}], {})["qualified"] is False


def test_early_lookup_priority_is_limited_to_low_market_caps():
    from scannerbot.early_momentum import early_lookup_priority_signal_count

    features = {
        "price_change_m5_pct": 3.0,
        "volume_acceleration_5m_vs_hour": 1.8,
        "buy_share_m5": 0.68,
        "transactions_m5": 20,
    }
    assert early_lookup_priority_signal_count(features, 17500) >= 2
    assert early_lookup_priority_signal_count(features, 30000) == 0
