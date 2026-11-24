from scannerbot.features import build_features

def test_build_features_has_expected_ratios():
    pair={"priceUsd":"0.01","liquidity":{"usd":10000},"volume":{"m5":500,"h1":5000,"h24":20000},
          "txns":{"m5":{"buys":8,"sells":2},"h1":{"buys":40,"sells":10}},
          "priceChange":{"m5":10,"h1":20,"h6":30,"h24":40},"fdv":100000,"marketCap":90000}
    f=build_features(pair,30)
    assert f["buy_share_m5"]==0.8
    assert f["transactions_m5"]==10
    assert f["liquidity_to_fdv"]==0.1

def test_previous_snapshot_deltas():
    pair={"priceUsd":"2","liquidity":{"usd":120},"volume":{"m5":30},"txns":{},"priceChange":{}}
    f=build_features(pair,40,{"price_usd":1,"liquidity_usd":100,"volume_m5_usd":20})
    assert f["price_return_since_snapshot_pct"]==100
    assert abs(f["liquidity_change_since_snapshot_pct"]-20)<1e-9
    assert abs(f["volume_change_since_snapshot_pct"]-50)<1e-9


def test_wash_trading_flags_require_multiple_independent_signals():
    from scannerbot.features import wash_trading_flags

    assert wash_trading_flags({
        "transactions_m5": 10, "volume_m5_usd": 30000,
        "liquidity_usd": 5000, "buy_share_m5": 0.98,
    })
    assert not wash_trading_flags({
        "transactions_m5": 10, "volume_m5_usd": 1000,
        "liquidity_usd": 5000, "buy_share_m5": 0.98,
    })
