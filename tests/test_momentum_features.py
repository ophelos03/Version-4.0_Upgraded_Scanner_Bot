from scannerbot.features import build_features


def test_short_interval_acceleration_uses_elapsed_snapshot_interval():
    pair = {
        "priceUsd": "1.10",
        "liquidity": {"usd": 12000},
        "volume": {"m5": 900, "h1": 6000, "h24": 10000},
        "txns": {
            "m5": {"buys": 12, "sells": 4},
            "h1": {"buys": 60, "sells": 36},
        },
        "priceChange": {"m5": 12, "h1": 20},
        "marketCap": 20000,
        "fdv": 25000,
    }
    previous = {
        "price_usd": 1.0,
        "liquidity_usd": 10000,
        "volume_m5_usd": 600,
        "transactions_m5": 10,
        "buy_share_m5": 0.5,
        "_interval_seconds": 60,
    }
    features = build_features(pair, 10, previous)
    assert abs(features["price_return_since_snapshot_pct"] - 10) < 1e-9
    assert abs(features["price_return_1m_pct"] - 10) < 1e-9
    assert features["snapshot_interval_seconds"] == 60
    assert features["volume_acceleration_5m_vs_hour"] > 1
    assert features["transaction_acceleration_5m_vs_hour"] > 1
    assert abs(features["transaction_change_since_snapshot_pct"] - 60) < 1e-9
