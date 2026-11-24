import asyncio

from scannerbot.models import TokenAnalysis
from scannerbot.telegram import Telegram


def analysis():
    return TokenAnalysis(
        chain_id="solana", token_address="MintABC", symbol="ABC", name="Example Token",
        pair_address="PairABC", dex_id="pumpswap", url="https://dexscreener.com/solana/PairABC",
        price_usd=0.00008809, liquidity_usd=21601, volume_h1=300768, volume_h24=500000,
        buys_h1=3959, sells_h1=3106, buys_h24=5000, sells_h24=4000,
        price_change_h1=118.0, market_cap=88025, fdv=90000, age_hours=0.3,
        score=69, decision="WATCHLIST",
        reasons=["Moderate liquidity (≥$20k)", "Buy activity currently exceeds sells"],
        risk_flags=["Very sharp 1h move; elevated chase risk"],
    )


def test_premodel_alert_is_watchlist_and_has_no_ml_probability():
    bot = Telegram("token", "chat")
    sent = []
    async def fake_send(message):
        sent.append(message)
    bot.send = fake_send

    asyncio.run(bot.signal_alert(
        analysis(),
        {"buys_h1": 3959, "sells_h1": 3106},
        "PASS", 12.0, [],
    ))
    message = sent[0]
    assert "DexScreener: https://dexscreener.com/solana/PairABC" in message
    assert "DexPaprika: https://dexpaprika.com/solana/pools/PairABC" in message
    assert "PRE-MODEL / WATCHLIST" in message
    assert "Score: 69/100" in message
    assert "Price: $0.00008809" in message
    assert "MODEL PROBABILITIES" not in message


def test_model_active_alert_appends_three_probabilities():
    bot = Telegram("token", "chat")
    sent = []
    async def fake_send(message):
        sent.append(message)
    bot.send = fake_send

    asyncio.run(bot.signal_alert(
        analysis(),
        {"buys_h1": 3959, "sells_h1": 3106},
        "PASS", 12.0, [],
        {"2x": 0.7, "5x": 0.2, "10x": 0.05}, "release-1",
    ))
    message = sent[0]
    assert "MODEL-ACTIVE" in message
    assert "2x: 70.0%" in message
    assert "5x: 20.0%" in message
    assert "10x: 5.0%" in message
    assert "Model version: release-1" in message
