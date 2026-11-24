from scannerbot.dexpaprika import normalize_pool
from scannerbot.features import build_features


def test_normalize_pool_uses_requested_base_token_and_preserves_metrics():
    mint = "MintABC"
    pool = {
        "id": "PoolABC",
        "base_token_id": mint,
        "dex_id": "raydium",
        "created_at": "2026-11-20T00:00:00Z",
        "price_usd": 0.002,
        "liquidity_usd": 12000,
        "volume_usd_24h": 90000,
        "price_change_percentage_5m": 3.5,
        "price_change_percentage_1h": 12.0,
        "tokens": [
            {
                "id": mint, "name": "Example", "symbol": "EX",
                "fdv": 75000,
                "5m": {"volume_usd": 600, "buys": 9, "sells": 2},
                "1h": {"volume_usd": 12000, "buys": 50, "sells": 20},
                "24h": {"volume_usd": 90000, "buys": 500, "sells": 200},
            },
            {"id": "QuoteMint", "name": "Wrapped SOL", "symbol": "SOL"},
        ],
    }
    pair = normalize_pool(pool, mint)
    assert pair is not None
    assert pair["baseToken"]["address"] == mint
    assert pair["priceUsd"] == "0.002"
    assert pair["liquidity"]["usd"] == 12000
    assert pair["volume"]["m5"] == 600
    assert pair["txns"]["m5"] == {"buys": 9, "sells": 2}
    assert pair["marketCap"] == 75000  # conservative FDV upper-bound proxy
    assert pair["_scanner_market_cap_source"] == "dexpaprika_fdv_proxy"
    assert pair["url"].endswith("/solana/pools/PoolABC")


def test_normalize_pool_refuses_quote_token_price_mismatch():
    pool = {
        "id": "PoolABC", "base_token_id": "OtherMint", "price_usd": 1,
        "tokens": [{"id": "OtherMint"}, {"id": "MintABC"}],
    }
    assert normalize_pool(pool, "MintABC") is None


def test_fdv_proxy_is_not_mislabeled_as_market_cap_feature():
    pair = {
        "priceUsd": "0.002",
        "liquidity": {"usd": 12000},
        "volume": {"m5": 600, "h1": 12000, "h24": 90000},
        "txns": {"m5": {"buys": 9, "sells": 2}, "h1": {"buys": 50, "sells": 20}},
        "marketCap": 75000,
        "fdv": 75000,
        "_scanner_market_cap_source": "dexpaprika_fdv_proxy",
    }
    features = build_features(pair, 2.0)
    assert features["market_cap_usd"] == 0.0
    assert features["fdv_usd"] == 75000.0
