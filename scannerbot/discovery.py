from __future__ import annotations
import asyncio, json, logging, os, threading, time
from datetime import datetime, timezone
from urllib.parse import urlencode
from typing import Any, Awaitable, Callable
import socketio
import websockets
import httpx
from .models import Candidate

log=logging.getLogger(__name__)
Handler=Callable[[Candidate], Awaitable[None]]


class DiscoveryCheckpointStore:
    """Tiny atomic checkpoint file on the persistent model/state volume."""
    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        self._lock = threading.Lock()
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            self._values = loaded if isinstance(loaded, dict) else {}
        except (OSError, ValueError, TypeError):
            self._values = {}

    def get(self, provider: str) -> int:
        with self._lock:
            try:
                return int(self._values.get(provider, 0))
            except (TypeError, ValueError):
                return 0

    def set(self, provider: str, value: int) -> None:
        if value <= 0:
            return
        with self._lock:
            self._values[provider] = max(int(value), int(self._values.get(provider, 0) or 0))
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            temporary = self.path + ".tmp"
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump(self._values, handle, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)

class ShrineDiscovery:
    """Free Socket.IO new-pool feed. Protocol list follows Shrine's documented API."""
    def __init__(self, handler: Handler, protocols: list[str]):
        self.handler,self.protocols=handler,protocols
        self.sio=socketio.AsyncClient(reconnection=True,reconnection_attempts=0,reconnection_delay=1,reconnection_delay_max=30)
        @self.sio.on("new_token")
        async def on_new(data: dict[str,Any]):
            try: await self.handler(self._candidate(data,"shrine"))
            except Exception: log.exception("Failed to process Shrine event")

        @self.sio.event
        async def connect():
            await self.sio.emit("subscribe_new_tokens", {"protocols": self.protocols})
            log.info("Subscribed to Shrine new-token stream: %s", self.protocols)

    @staticmethod
    def _candidate(d: dict[str,Any], source: str) -> Candidate:
        # Provider payloads vary by venue; keep raw metadata for future decoder updates.
        mint_fields=("mint", "token_address", "tokenAddress", "base_mint", "address")
        mint_field=next((key for key in mint_fields if d.get(key)), "missing")
        mint=str(d.get(mint_field) or "").strip() if mint_field != "missing" else ""
        protocol=str(d.get("protocol") or d.get("dex") or "unknown").upper()
        created=d.get("created_at") or d.get("timestamp") or d.get("createdAt")
        try:
            created=float(created)
            if created>10_000_000_000: created/=1000
        except (TypeError,ValueError): created=None
        return Candidate("solana",mint,source,{**d,"protocol":protocol,"_scanner_mint_field":mint_field},str(d.get("pool") or d.get("pool_address") or d.get("pair") or ""),created)

    async def run(self):
        delay=2
        while True:
            try:
                if not self.sio.connected:
                    await self.sio.connect("https://sol.shrine.trade",wait_timeout=20)
                    log.info("Connected to Shrine new-token stream")
                await self.sio.sleep(20)
            except asyncio.CancelledError: raise
            except Exception as exc:
                log.warning("Shrine stream disconnected: %s",exc)
                try: await self.sio.disconnect()
                except Exception: pass
                await asyncio.sleep(delay); delay=min(delay*2,30)

    async def close(self):
        if self.sio.connected: await self.sio.disconnect()


class ShrineAdvancedDiscovery:
    """Keyed all-launchpad stream with durable slot-based resume after reconnect."""
    def __init__(self, handler: Handler, checkpoints: DiscoveryCheckpointStore, api_key: str, protocols: list[str]):
        self.handler, self.checkpoints, self.api_key, self.protocols = handler, checkpoints, api_key, protocols

    async def run(self):
        if not self.api_key:
            log.info("Shrine advanced stream disabled: SHRINE_API_KEY is not configured")
            return
        delay = 2
        while True:
            try:
                # Replay a small overlap to cover out-of-order neighboring slots; DB dedup makes it safe.
                since_slot = max(0, self.checkpoints.get("shrine_advanced") - 64)
                params = urlencode({
                    "key": self.api_key,
                    "actions": "create,createPool,migrate,curveComplete",
                    **({"sinceSlot": str(since_slot)} if since_slot else {}),
                })
                url = "wss://sol.shrine.trade/api/stream/ws?" + params
                async with websockets.connect(url, ping_interval=20, ping_timeout=20, max_size=2**20) as ws:
                    log.info("Connected to Shrine advanced discovery stream; resume_slot=%s", since_slot or "latest")
                    delay = 2
                    async for raw in ws:
                        event = json.loads(raw)
                        if not isinstance(event, dict):
                            continue
                        if event.get("type"):
                            if event.get("type") in {"lagged", "error"}:
                                log.warning("Shrine advanced stream control message: %s; reconnecting from durable slot", event)
                                raise RuntimeError(f"Shrine stream control message: {event.get('type')}")
                            continue
                        action = str(event.get("action") or "")
                        if action not in {"create", "createPool", "migrate", "curveComplete"}:
                            continue
                        candidate = ShrineDiscovery._candidate(event, "shrine_advanced")
                        if candidate.token_address:
                            await self.handler(candidate)
                        slot = int(event.get("block") or 0)
                        if slot:
                            self.checkpoints.set("shrine_advanced", slot)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("Shrine advanced stream disconnected; will resume from saved slot: %s", exc)
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30)

    async def close(self):
        return None


class ShrineReplayBackfill:
    """Replay completed hourly Shrine launch events after downtime using the public archive."""
    BASE = "https://replay.shrine.trade/pump"

    def __init__(self, handler: Handler, checkpoints: DiscoveryCheckpointStore):
        self.handler, self.checkpoints = handler, checkpoints
        self._startup_backfill = True
        self.client = httpx.AsyncClient(timeout=90, headers={"User-Agent": "ScannerBot/1.0"})

    async def close(self):
        await self.client.aclose()

    async def _process_line(self, line: bytes):
        if not line.strip():
            return
        try:
            event = json.loads(line)
            if not isinstance(event, dict):
                return
            if event.get("action") not in {"create", "createPool", "migrate", "curveComplete"}:
                return
            candidate = ShrineDiscovery._candidate(event, "shrine_replay")
            if candidate.token_address:
                await self.handler(candidate)
        except (ValueError, TypeError):
            log.warning("Skipping malformed event in Shrine hourly replay")

    async def _replay_hour(self, hour: str):
        url = f"{self.BASE}/{hour}.jsonl.zst"
        import zstandard
        decompressor = zstandard.ZstdDecompressor().decompressobj()
        pending = bytearray()
        async with self.client.stream("GET", url) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes():
                pending.extend(decompressor.decompress(chunk))
                while True:
                    split_at = pending.find(b"\n")
                    if split_at < 0:
                        break
                    line = bytes(pending[:split_at])
                    del pending[:split_at + 1]
                    await self._process_line(line)
        pending.extend(decompressor.flush())
        if pending.strip():
            await self._process_line(bytes(pending))

    async def run(self):
        delay = 300
        while True:
            try:
                response = await self.client.get(f"{self.BASE}/index.json")
                response.raise_for_status()
                index = response.json()
                hours = index.get("hours", []) if isinstance(index, dict) else []
                parsed = []
                for hour in hours:
                    try:
                        stamp = datetime.strptime(hour, "%Y/%m/%d/%H").replace(tzinfo=timezone.utc).timestamp()
                        if stamp + 3600 <= time.time() - 180:
                            parsed.append((stamp, hour))
                    except (TypeError, ValueError):
                        continue
                checkpoint = self.checkpoints.get("shrine_replay")
                if self._startup_backfill:
                    # SQLite is intentionally ephemeral: replay the last two complete hours
                    # after every process start to repopulate recent active candidates.
                    selected = {stamp: hour for stamp, hour in parsed if stamp > checkpoint}
                    selected.update({stamp: hour for stamp, hour in parsed[-2:]})
                    pending_hours = sorted(selected.items())
                    if parsed:
                        self._startup_backfill = False
                else:
                    pending_hours = [(stamp, hour) for stamp, hour in parsed if stamp > checkpoint]
                for stamp, hour in pending_hours:
                    log.info("Replaying Shrine discovery archive hour %s", hour)
                    await self._replay_hour(hour)
                    self.checkpoints.set("shrine_replay", int(stamp))
                    log.info("Completed Shrine replay hour %s", hour)
                delay = 300
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("Shrine hourly replay/backfill failed; will retry: %s", exc)
                delay = min(max(delay, 60), 300)
            await asyncio.sleep(delay)

class PumpPortalDiscovery:
    """PumpPortal new-token and migration stream; no private key or trade subscription."""
    def __init__(self, handler: Handler, api_key: str=""):
        self.handler,self.api_key=handler,api_key

    async def run(self):
        delay=2
        url="wss://pumpportal.fun/api/data"+(f"?api-key={self.api_key}" if self.api_key else "")
        while True:
            try:
                async with websockets.connect(url,ping_interval=20,ping_timeout=20,max_size=2**20) as ws:
                    await ws.send('{"method":"subscribeNewToken"}')
                    await ws.send('{"method":"subscribeMigration"}')
                    log.info("Connected to PumpPortal launch/migration stream")
                    log.info("PumpPortal does not expose a historical replay cursor; reconnect gaps can only be recovered by overlapping discovery providers.")
                    delay=2
                    async for raw in ws:
                        try:
                            import json
                            d=json.loads(raw)
                            mint_field="mint" if d.get("mint") else ("tokenAddress" if d.get("tokenAddress") else "missing")
                            mint=str(d.get(mint_field) or "").strip() if mint_field != "missing" else ""
                            created=d.get("timestamp")
                            try:
                                created=float(created)
                                if created>10_000_000_000: created/=1000
                            except (TypeError,ValueError): created=None
                            await self.handler(Candidate("solana",mint,"pumpportal",{**d,"_scanner_mint_field":mint_field},str(d.get("pool") or ""),created))
                        except Exception: log.exception("Bad PumpPortal event")
            except asyncio.CancelledError: raise
            except Exception as exc:
                log.warning("PumpPortal stream disconnected: %s",exc)
                await asyncio.sleep(delay); delay=min(delay*2,30)
