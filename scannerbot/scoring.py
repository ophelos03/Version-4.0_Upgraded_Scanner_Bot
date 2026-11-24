from __future__ import annotations
from datetime import datetime, timezone
from .models import TokenAnalysis


def _num(value, default=0.0) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _tx(pair: dict, window: str, side: str) -> int:
    try:
        return int((pair.get("txns") or {}).get(window, {}).get(side, 0) or 0)
    except (TypeError, ValueError):
        return 0


def analyse_pair(chain_id: str, token_address: str, pair: dict, metadata: dict,
                 min_liquidity: float, min_volume: float, min_txns: int,
                 max_age_hours: float) -> TokenAnalysis:
    base = pair.get("baseToken") or {}
    liquidity = _num((pair.get("liquidity") or {}).get("usd"))
    vol = pair.get("volume") or {}
    volume_h1 = _num(vol.get("h1"))
    volume_h24 = _num(vol.get("h24"))
    buys_h1, sells_h1 = _tx(pair, "h1", "buys"), _tx(pair, "h1", "sells")
    buys_h24, sells_h24 = _tx(pair, "h24", "buys"), _tx(pair, "h24", "sells")
    txns_h1 = buys_h1 + sells_h1
    price_change_h1 = _num((pair.get("priceChange") or {}).get("h1"))
    price = _num(pair.get("priceUsd"))
    market_cap = pair.get("marketCap")
    fdv = pair.get("fdv")

    age_hours = None
    created = pair.get("pairCreatedAt")
    if created:
        try:
            created_ts = float(created)
            # DEX Screener timestamps are usually milliseconds, but tolerate seconds too.
            if created_ts > 10_000_000_000:
                created_ts /= 1000
            age_hours = max(0.0, (datetime.now(timezone.utc).timestamp() - created_ts) / 3600)
        except (TypeError, ValueError):
            pass

    score = 0.0
    reasons: list[str] = []
    flags: list[str] = []

    # Liquidity: up to 25 points. A deeper pool is generally easier to exit.
    if liquidity >= 100_000: score += 25; reasons.append("Deep liquidity (≥$100k)")
    elif liquidity >= 50_000: score += 21; reasons.append("Good liquidity (≥$50k)")
    elif liquidity >= 20_000: score += 16; reasons.append("Moderate liquidity (≥$20k)")
    elif liquidity >= min_liquidity: score += 10; reasons.append("Liquidity clears minimum")
    else: flags.append("Liquidity below configured minimum")

    # Activity: up to 25 points.
    if volume_h1 >= 50_000: score += 15
    elif volume_h1 >= 10_000: score += 12
    elif volume_h1 >= min_volume: score += 8
    else: flags.append("Hourly volume below minimum")
    if txns_h1 >= 200: score += 10
    elif txns_h1 >= 75: score += 8
    elif txns_h1 >= min_txns: score += 5
    else: flags.append("Hourly transaction count below minimum")
    if txns_h1:
        buy_ratio = buys_h1 / txns_h1
        if 0.52 <= buy_ratio <= 0.75:
            score += 10; reasons.append("Buy activity currently exceeds sells")
        elif buy_ratio > 0.75:
            score += 5; flags.append("Buy flow unusually one-sided; verify sellability")
        elif buy_ratio < 0.35:
            flags.append("Sell transactions dominate recent flow")
    else:
        flags.append("No recent transactions reported")

    # Momentum: up to 15 points. Extreme pumps are not automatically rewarded.
    if 5 <= price_change_h1 <= 40:
        score += 15; reasons.append("Positive but not extreme 1h momentum")
    elif 0 < price_change_h1 < 5:
        score += 8; reasons.append("Slight positive 1h momentum")
    elif price_change_h1 > 80:
        score += 3; flags.append("Very sharp 1h move; elevated chase risk")
    elif price_change_h1 < -15:
        flags.append("Sharp negative 1h momentum")

    # Longevity and basic data quality: up to 15 points.
    if age_hours is not None:
        if 1 <= age_hours <= max_age_hours:
            score += 10; reasons.append(f"Pair age {age_hours:.1f} hours")
        elif age_hours < 1:
            score += 5; flags.append("Very new pool; metrics may be unstable")
        else:
            score += 3
    else:
        flags.append("Pool age unavailable")
    if price > 0 and base.get("address"):
        score += 5
    else:
        flags.append("Missing price or token address data")

    # Metadata presence is not proof of legitimacy; paid boosts are deliberately not scored.
    info = pair.get("info") or {}
    links = metadata.get("links") or []
    socials = info.get("socials") or []
    websites = info.get("websites") or []
    if links or socials or websites:
        score += 5; reasons.append("Some public project metadata is present")

    if liquidity < min_liquidity or volume_h1 < min_volume or txns_h1 < min_txns:
        decision = "WATCH"
    elif score >= 75:
        decision = "HIGH PRIORITY"
    elif score >= 60:
        decision = "WATCHLIST"
    else:
        decision = "LOW PRIORITY"

    return TokenAnalysis(
        chain_id=chain_id, token_address=token_address,
        symbol=str(base.get("symbol") or "UNKNOWN"), name=str(base.get("name") or "Unknown token"),
        pair_address=str(pair.get("pairAddress") or ""), dex_id=str(pair.get("dexId") or "unknown"),
        url=str(pair.get("url") or metadata.get("url") or ""), price_usd=price,
        liquidity_usd=liquidity, volume_h1=volume_h1, volume_h24=volume_h24,
        buys_h1=buys_h1, sells_h1=sells_h1, buys_h24=buys_h24, sells_h24=sells_h24,
        price_change_h1=price_change_h1,
        market_cap=_num(market_cap, None) if market_cap is not None else None,
        fdv=_num(fdv, None) if fdv is not None else None, age_hours=age_hours,
        score=min(100.0, score), decision=decision, reasons=reasons, risk_flags=flags,
        metadata=metadata,
        dexpaprika_url=f"https://dexpaprika.com/solana/pools/{pair.get('pairAddress')}" if pair.get("pairAddress") else "",
        market_cap_source=str(pair.get("_scanner_market_cap_source") or "market_cap"),
    )
