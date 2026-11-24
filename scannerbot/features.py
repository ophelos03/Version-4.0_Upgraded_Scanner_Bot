from __future__ import annotations
import math, time
from typing import Any

def n(v: Any, default: float=0.0) -> float:
    try:
        x=float(v)
        return x if math.isfinite(x) else default
    except (TypeError,ValueError): return default

def tx(pair: dict, window: str, side: str) -> int:
    try: return int((pair.get("txns") or {}).get(window,{}).get(side,0) or 0)
    except (TypeError,ValueError): return 0

def build_features(pair: dict, age_minutes: float, previous: dict | None=None) -> dict[str,float]:
    vol=pair.get("volume") or {}; liq=n((pair.get("liquidity") or {}).get("usd")); price=n(pair.get("priceUsd"))
    txns=pair.get("txns") or {}; pc=pair.get("priceChange") or {}
    b5,s5=tx(pair,"m5","buys"),tx(pair,"m5","sells")
    b1,s1=tx(pair,"h1","buys"),tx(pair,"h1","sells")
    buys5vol=n(vol.get("m5")); buys1vol=n(vol.get("h1"))
    total5=b5+s5; total1=b1+s1
    f={
      "pair_age_minutes":age_minutes,"price_usd":price,"liquidity_usd":liq,
      "volume_m5_usd":buys5vol,"volume_h1_usd":buys1vol,"volume_h24_usd":n(vol.get("h24")),
      "buys_m5":b5,"sells_m5":s5,"transactions_m5":total5,
      "buys_h1":b1,"sells_h1":s1,"transactions_h1":total1,
      "buy_share_m5":b5/max(1,total5),"buy_share_h1":b1/max(1,total1),
      "price_change_m5_pct":n(pc.get("m5")),"price_change_h1_pct":n(pc.get("h1")),
      "price_change_h6_pct":n(pc.get("h6")),"price_change_h24_pct":n(pc.get("h24")),
      "market_cap_usd":0.0 if pair.get("_scanner_market_cap_source") == "dexpaprika_fdv_proxy" else n(pair.get("marketCap")),"fdv_usd":n(pair.get("fdv")),
      "liquidity_to_fdv":liq/max(1,n(pair.get("fdv"))),
      "volume_to_liquidity_5m":buys5vol/max(1,liq),
      "volume_to_liquidity_1h":buys1vol/max(1,liq),
      "buy_sell_count_ratio_5m":b5/max(1,s5),
      "buy_sell_count_ratio_1h":b1/max(1,s1),
    }
    # Rate-normalized acceleration compares the current 5m window with the 1h baseline.
    f["volume_acceleration_5m_vs_hour"] = buys5vol / max(1.0, buys1vol / 12.0)
    f["transaction_acceleration_5m_vs_hour"] = total5 / max(1.0, total1 / 12.0)
    f["buy_pressure_change_m5"] = f["buy_share_m5"] - 0.5
    if previous:
        old=json_num(previous.get("price_usd")); oldliq=json_num(previous.get("liquidity_usd")); oldvol=json_num(previous.get("volume_m5_usd"))
        oldtx=json_num(previous.get("transactions_m5")); oldbuy=json_num(previous.get("buy_share_m5"),0.5)
        f["price_return_since_snapshot_pct"]=(price/old-1)*100 if old>0 else 0
        f["liquidity_change_since_snapshot_pct"]=(liq/oldliq-1)*100 if oldliq>0 else 0
        f["volume_change_since_snapshot_pct"]=(buys5vol/oldvol-1)*100 if oldvol>0 else 0
        f["transaction_change_since_snapshot_pct"]=(total5/oldtx-1)*100 if oldtx>0 else 0
        f["buy_share_change_since_snapshot"] = f["buy_share_m5"] - oldbuy
        elapsed = max(0.0, n(previous.get("_interval_seconds"), 0.0))
        f["snapshot_interval_seconds"] = elapsed
        f["price_return_1m_pct"] = f["price_return_since_snapshot_pct"] if 30 <= elapsed <= 180 else 0.0
    else:
        f.update({"price_return_since_snapshot_pct":0.0,"liquidity_change_since_snapshot_pct":0.0,
                  "volume_change_since_snapshot_pct":0.0,"transaction_change_since_snapshot_pct":0.0,
                  "buy_share_change_since_snapshot":0.0,"snapshot_interval_seconds":0.0,"price_return_1m_pct":0.0})
    return f

def json_num(v, default=0.0): return n(v, default)


def wash_trading_flags(
    features: dict[str, float],
    volume_per_tx_threshold: float = 2500.0,
    volume_to_liquidity_threshold: float = 3.0,
    one_sided_share_threshold: float = 0.92,
) -> list[str]:
    """Conservative aggregate-metric wash-trading heuristic; not wallet-level proof."""
    tx_count = n(features.get("transactions_m5"))
    volume = n(features.get("volume_m5_usd"))
    liquidity = n(features.get("liquidity_usd"))
    buy_share = n(features.get("buy_share_m5"), 0.5)
    signals = []
    if tx_count >= 8 and volume / max(1.0, tx_count) >= volume_per_tx_threshold:
        signals.append("unusually high 5m volume per transaction")
    if liquidity > 0 and volume / liquidity >= volume_to_liquidity_threshold:
        signals.append("5m volume is unusually large relative to pool liquidity")
    if tx_count >= 8 and (buy_share >= one_sided_share_threshold or buy_share <= 1.0 - one_sided_share_threshold):
        signals.append("extremely one-sided 5m buy/sell flow")
    # Require multiple independent warning signs to avoid blocking a token on one noisy metric.
    return signals if len(signals) >= 2 else []
