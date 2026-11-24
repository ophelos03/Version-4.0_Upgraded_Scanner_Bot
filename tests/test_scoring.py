from scannerbot.scoring import analyse_pair

def test_score_and_token_fields():
    pair = {
        "baseToken": {"address": "0xabc", "symbol": "TEST", "name": "Test Token"},
        "pairAddress": "0xpool", "dexId": "uniswap", "url": "https://dexscreener.com/test",
        "priceUsd": "0.01", "liquidity": {"usd": 60000},
        "volume": {"h1": 20000, "h24": 100000},
        "txns": {"h1": {"buys": 80, "sells": 40}, "h24": {"buys": 400, "sells": 250}},
        "priceChange": {"h1": 12}, "pairCreatedAt": 1794300000000,
        "marketCap": 1000000, "fdv": 1200000,
        "info": {"socials": [{"platform": "x", "handle": "test"}]}
    }
    result = analyse_pair("ethereum", "0xabc", pair, {}, 5000, 1000, 10, 72)
    assert result.symbol == "TEST"
    assert result.score > 50
    assert result.liquidity_usd == 60000



def test_pair_created_at_accepts_seconds_and_milliseconds():
    import time

    now = time.time()
    common = {
        "baseToken": {"address": "mint", "symbol": "TEST", "name": "Test"},
        "priceUsd": "1",
        "liquidity": {"usd": 10000},
        "volume": {"h1": 1000, "h24": 2000},
        "txns": {"h1": {"buys": 10, "sells": 5}, "h24": {"buys": 20, "sells": 10}},
    }
    seconds_pair = {**common, "pairCreatedAt": now - 3600}
    millis_pair = {**common, "pairCreatedAt": (now - 3600) * 1000}

    seconds = analyse_pair("solana", "mint", seconds_pair, {}, 5000, 100, 5, 72)
    millis = analyse_pair("solana", "mint", millis_pair, {}, 5000, 100, 5, 72)

    assert seconds.age_hours is not None
    assert millis.age_hours is not None
    assert abs(seconds.age_hours - millis.age_hours) < 1e-6
    assert abs(seconds.age_hours - 1.0) < 1e-4
