from __future__ import annotations
import asyncio
import logging
import time
from typing import Any
import httpx
from .models import Candidate, canonical_token_key

log = logging.getLogger(__name__)
BASE = "https://api.dexscreener.com"

class DexScreenerClient:
    def __init__(self) -> None:
        self.client = httpx.AsyncClient(timeout=15, headers={"User-Agent": "ScannerBot/1.0"})
        self._batch_rate_lock = asyncio.Lock()
        self._next_batch_request_at = 0.0

    async def close(self) -> None:
        await self.client.aclose()

    async def _get(self, path: str) -> Any:
        response = await self.client.get(f"{BASE}{path}")
        response.raise_for_status()
        return response.json()

    async def discover(self) -> list[Candidate]:
        """Use latest profiles and latest boosts as discovery feeds, not quality endorsements."""
        found: dict[tuple[str, str], Candidate] = {}
        endpoints = (("/token-profiles/latest/v1", "profile"),
                     ("/token-boosts/latest/v1", "boost"))
        for path, source in endpoints:
            try:
                payload = await self._get(path)
                if not isinstance(payload, list):
                    continue
                for item in payload:
                    chain = str(item.get("chainId", "")).lower()
                    address = str(item.get("tokenAddress", "")).strip()
                    if chain and address:
                        key = (chain, canonical_token_key(chain, address))
                        if key not in found:
                            found[key] = Candidate(chain, address, source, item)
                        else:
                            found[key].metadata.update(item)
                            found[key].source = "profile+boost"
            except (httpx.HTTPError, ValueError) as exc:
                log.warning("Discovery endpoint %s failed: %s", path, exc)
        return list(found.values())

    async def token_pairs(self, chain_id: str, token_address: str) -> list[dict[str, Any]]:
        data = await self._get(f"/token-pairs/v1/{chain_id}/{token_address}")
        return data if isinstance(data, list) else []

    async def token_pairs_batch(self, chain_id: str, token_addresses: list[str]) -> list[dict[str, Any]]:
        """Fetch pair data for up to 30 token mints in one API request."""
        if not token_addresses:
            return []
        if len(token_addresses) > 30:
            raise ValueError("DEX Screener batch endpoint accepts at most 30 token addresses")
        # Smooth batch requests to 240/minute, leaving headroom below DexScreener's
        # documented 300/minute limit for other callers and newly discovered tokens.
        async with self._batch_rate_lock:
            now = time.monotonic()
            wait = self._next_batch_request_at - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._next_batch_request_at = time.monotonic() + 0.25
        path=f"/tokens/v1/{chain_id}/"+",".join(token_addresses)
        data=await self._get(path)
        return data if isinstance(data,list) else []
