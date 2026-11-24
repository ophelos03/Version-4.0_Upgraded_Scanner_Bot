from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from typing import Any

import httpx

log = logging.getLogger(__name__)
BASE = "https://api.dexpaprika.com"


def _timestamp_ms(value: str | None) -> int | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return int(parsed.timestamp() * 1000)
    except (TypeError, ValueError):
        return None


def normalize_pool(pool: dict[str, Any], token_address: str) -> dict[str, Any] | None:
    """Convert a detailed DexPaprika pool result into ScannerBot's pair shape."""
    tokens = pool.get("tokens") or []
    token = next(
        (item for item in tokens if str(item.get("id") or "") == token_address),
        None,
    )
    if not token:
        return None

    other = next((item for item in tokens if item is not token), {})
    # Detailed pool-search results embed per-token interval metrics directly.
    intervals = {key: token.get(key) or (token.get("summary") or {}).get(key) or {} for key in ("5m", "1h", "24h")}
    base_token_id = str(pool.get("base_token_id") or (tokens[0].get("id") if tokens else ""))
    # pool.price_usd is the base-token price; never apply it to the quote token.
    if base_token_id and base_token_id != token_address:
        return None
    fdv = token.get("fdv")
    # DexPaprika exposes FDV, not circulating market cap. Using FDV as the
    # market-cap gate value is deliberately conservative: it can reject more
    # tokens, but cannot make the existing market-cap ceiling more permissive.
    market_cap_proxy = float(fdv) if fdv is not None else None
    price = pool.get("price_usd")

    created_at = _timestamp_ms(pool.get("created_at"))
    pool_id = str(pool.get("id") or "")
    if not pool_id or price is None:
        return None

    return {
        "chainId": "solana",
        "dexId": str(pool.get("dex_id") or pool.get("dex_name") or "dexpaprika"),
        "pairAddress": pool_id,
        "url": f"https://dexpaprika.com/solana/pools/{pool_id}",
        "baseToken": {
            "address": token_address,
            "name": str(token.get("name") or "Unknown token"),
            "symbol": str(token.get("symbol") or "UNKNOWN"),
        },
        "quoteToken": {
            "address": str(other.get("id") or ""),
            "name": str(other.get("name") or "Unknown token"),
            "symbol": str(other.get("symbol") or "UNKNOWN"),
        },
        "priceUsd": str(price),
        "liquidity": {"usd": float(pool.get("liquidity_usd") or 0)},
        "volume": {
            "m5": float(intervals["5m"].get("volume_usd") or 0),
            "h1": float(intervals["1h"].get("volume_usd") or 0),
            "h24": float(intervals["24h"].get("volume_usd") or pool.get("volume_usd_24h") or 0),
        },
        "txns": {
            "m5": {
                "buys": int(intervals["5m"].get("buys") or 0),
                "sells": int(intervals["5m"].get("sells") or 0),
            },
            "h1": {
                "buys": int(intervals["1h"].get("buys") or 0),
                "sells": int(intervals["1h"].get("sells") or 0),
            },
            "h24": {
                "buys": int(intervals["24h"].get("buys") or 0),
                "sells": int(intervals["24h"].get("sells") or 0),
            },
        },
        "priceChange": {
            "m5": float(pool.get("price_change_percentage_5m") or 0),
            "h1": float(pool.get("price_change_percentage_1h") or 0),
            "h24": float(pool.get("price_change_percentage_24h") or 0),
        },
        "marketCap": market_cap_proxy,
        "fdv": market_cap_proxy,
        "pairCreatedAt": created_at,
        "_scanner_market_cap_source": "dexpaprika_fdv_proxy",
        "_scanner_provider": "DexPaprika",
    }


class DexPaprikaClient:
    """Free-tier DexPaprika fallback for Solana mint-to-pool resolution."""

    def __init__(self, api_key: str = "") -> None:
        self.api_key = api_key.strip()
        self.client = httpx.AsyncClient(
            timeout=15,
            headers={
                "User-Agent": "ScannerBot/1.0",
                **({"Authorization": self.api_key} if self.api_key else {}),
            },
        )
        self._rate_lock = asyncio.Lock()
        self._next_request_at = 0.0
        self._quota_exhausted = False
        # Current free-key limit is 30 req/min; keyless is 15 req/min.
        # Preserve the 100K free-key / 10K keyless monthly credit budgets.
        self._request_interval = 30.0 if self.api_key else 300.0

    async def close(self) -> None:
        await self.client.aclose()

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        if self._quota_exhausted:
            return None
        async with self._rate_lock:
            wait = self._next_request_at - time.monotonic()
            if wait > 0:
                # Never stall the scanner loop waiting for a free-tier slot.
                return None
            self._next_request_at = time.monotonic() + self._request_interval
        response = await self.client.get(f"{BASE}{path}", params=params)
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            try:
                delay = max(1.0, float(retry_after or 60))
            except ValueError:
                delay = 60.0
            self._next_request_at = max(self._next_request_at, time.monotonic() + delay)
            log.warning("DexPaprika rate limited; retry_after_seconds=%.0f", delay)
            return None
        if response.status_code == 402:
            self._quota_exhausted = True
            log.warning("DexPaprika free-tier credit allowance exhausted; fallback disabled until process restart/quota renewal")
            return None
        response.raise_for_status()
        return response.json()

    async def token_pair(self, token_address: str) -> dict[str, Any] | None:
        payload = await self._get(
            "/networks/solana/pools/search",
            params={
                "token_address": token_address,
                "limit": 10,
                "order_by": "liquidity_usd",
                "sort": "desc",
                "detailed": "true",
            },
        )
        results = payload.get("results", []) if isinstance(payload, dict) else []
        matches = [
            normalize_pool(pool, token_address)
            for pool in results
            if isinstance(pool, dict)
        ]
        matches = [pair for pair in matches if pair is not None]
        return max(matches, key=lambda pair: float((pair.get("liquidity") or {}).get("usd") or 0), default=None)
