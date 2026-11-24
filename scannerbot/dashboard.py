from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Callable

log = logging.getLogger(__name__)

_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ScannerBot Discovery</title>
<style>
body{font:16px system-ui,sans-serif;background:#0b1220;color:#e5e7eb;margin:0;padding:28px}
main{max-width:900px;margin:auto}h1{font-size:26px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px}
section{background:#151f32;border:1px solid #29364d;border-radius:12px;padding:18px}
.metric{display:flex;justify-content:space-between;padding:10px 0;border-bottom:1px solid #29364d}
.metric:last-child{border:0}.value{font-variant-numeric:tabular-nums;font-weight:700}
small{color:#9ca3af}#status{color:#9ca3af;margin:12px 0}
</style></head><body><main><h1>ScannerBot discovery</h1>
<p id="status">Waiting for scanner metrics…</p><div class="grid">
<section><h2>Token discovery</h2><div id="discovery"></div></section>
<section><h2>Stage 1 eligibility</h2><div id="eligibility"></div></section>
</div></main>
<script>
function row(label,value){return '<div class="metric"><span>'+label+'</span><span class="value">'+value+'</span></div>'}
async function refresh(){try{const r=await fetch('/metrics',{cache:'no-store'});const d=await r.json();
document.getElementById('discovery').innerHTML=row('Shrine',d.discovery.Shrine||0)+row('PumpPortal',d.discovery.PumpPortal||0)+row('DexScreener',d.discovery.DexScreener||0)+row('Unique tokens discovered',d.unique_discovered||0);
document.getElementById('eligibility').innerHTML=row('Unique tokens evaluated',d.stage1.evaluated||0)+row('Passed',d.stage1.passed||0)+row('Failed',d.stage1.failed||0);
document.getElementById('status').textContent='Period: '+(d.period_seconds||0)+' seconds · Updated '+new Date(d.updated_at*1000).toLocaleTimeString();
}catch(e){document.getElementById('status').textContent='Metrics temporarily unavailable';}}
refresh();setInterval(refresh,5000);
</script></body></html>"""


class DiscoveryDashboard:
    """Small read-only HTTP dashboard limited to discovery and Stage 1 eligibility."""
    def __init__(self, metrics: Callable[[], dict], host: str = "0.0.0.0", port: int | None = None):
        self.metrics = metrics
        self.host = host
        self.port = int(port or os.getenv("PORT", "8080"))
        self.server: asyncio.AbstractServer | None = None

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        try:
            request = await asyncio.wait_for(reader.readline(), timeout=5)
            parts = request.decode("ascii", "ignore").strip().split()
            path = parts[1] if len(parts) >= 2 else "/"
            while True:
                header = await asyncio.wait_for(reader.readline(), timeout=5)
                if header in (b"\r\n", b"\n", b""):
                    break
            if path == "/metrics":
                body = json.dumps(self.metrics(), separators=(",", ":")).encode()
                content_type = "application/json; charset=utf-8"
            elif path == "/":
                body = _PAGE.encode()
                content_type = "text/html; charset=utf-8"
            else:
                body = b"Not found"
                content_type = "text/plain; charset=utf-8"
                status = "404 Not Found"
                response = (f"HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n").encode() + body
                writer.write(response)
                await writer.drain()
                return
            response = (f"HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nCache-Control: no-store\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n").encode() + body
            writer.write(response)
            await writer.drain()
        except (asyncio.TimeoutError, ConnectionError, BrokenPipeError):
            pass
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def run(self):
        try:
            self.server = await asyncio.start_server(self._handle, self.host, self.port)
            log.info("Discovery dashboard listening on %s:%s", self.host, self.port)
            async with self.server:
                await self.server.serve_forever()
        except asyncio.CancelledError:
            raise
        except OSError:
            log.exception("Could not start discovery dashboard on port %s", self.port)

    async def close(self):
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            self.server = None
