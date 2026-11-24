from __future__ import annotations

import logging
import httpx
from .models import TokenAnalysis

log = logging.getLogger(__name__)


def _money(value: float | None) -> str:
    if value is None:
        return "N/A"
    value = float(value)
    if value == 0:
        return "$0"
    if abs(value) < 0.0001:
        return f"${value:.10f}".rstrip("0").rstrip(".")
    if abs(value) < 1:
        return f"${value:.8f}".rstrip("0").rstrip(".")
    return f"${value:,.2f}" if value % 1 else f"${value:,.0f}"


class Telegram:
    def __init__(self, token: str, chat_id: str):
        self.token, self.chat_id = token, chat_id
        self.base = f"https://api.telegram.org/bot{token}"
        self.client = httpx.AsyncClient(timeout=20)
        self.offset = 0

    def _safe_detail(self, value: object, limit: int = 300) -> str:
        detail = str(value)
        if self.token:
            detail = detail.replace(self.token, "<redacted>")
        detail = detail.replace(self.base, "https://api.telegram.org/<redacted>")
        return detail[:limit] or "<empty>"

    async def close(self):
        await self.client.aclose()

    async def send(self, text: str):
        try:
            response = await self.client.post(
                f"{self.base}/sendMessage",
                json={"chat_id": self.chat_id, "text": text, "disable_web_page_preview": True},
            )
        except httpx.TimeoutException as exc:
            log.warning("Telegram send failed: category=timeout exception_type=%s detail=%s",
                        type(exc).__name__, self._safe_detail(exc))
            raise
        except httpx.RequestError as exc:
            log.warning("Telegram send failed: category=connection_or_transport exception_type=%s detail=%s",
                        type(exc).__name__, self._safe_detail(exc))
            raise

        if response.is_error:
            log.warning("Telegram send rejected: category=http_error status=%d", response.status_code)
            response.raise_for_status()
        try:
            data = response.json()
        except ValueError as exc:
            log.warning("Telegram send failed: category=invalid_response status=%d exception_type=%s",
                        response.status_code, type(exc).__name__)
            raise

        if not data.get("ok"):
            error_code = data.get("error_code", response.status_code)
            description = self._safe_detail(data.get("description", "Telegram returned ok=false"))
            log.warning("Telegram send rejected: category=telegram_rejection error_code=%s description=%s",
                        error_code, description)
            raise RuntimeError(f"Telegram rejected message (error_code={error_code}): {description}")

        result = data.get("result") or {}
        log.info("Telegram send succeeded: message_id=%s status=%d",
                 result.get("message_id", "unknown"), response.status_code)
        return result

    async def signal_alert(
        self, a: TokenAnalysis, features: dict, security_status: str,
        risk_score: float | None, risk_reasons: list[str],
        probabilities: dict[str, float] | None = None, model_version: str | None = None,
        alert_timestamp: float | None = None, momentum: dict | None = None,
    ):
        mode = (
            "PRE-MODEL / WATCHLIST — SECURITY UNVERIFIED"
            if security_status == "UNKNOWN" and not probabilities
            else ("MODEL-ACTIVE" if probabilities else "PRE-MODEL / WATCHLIST")
        )
        title = a.symbol or "UNKNOWN"
        buy_total = int(features.get("buys_h1", a.buys_h1) or 0)
        sell_total = int(features.get("sells_h1", a.sells_h1) or 0)
        total = buy_total + sell_total
        buy_share = (buy_total / total * 100) if total else None
        age = f"{a.age_hours:.1f}h" if a.age_hours is not None else "N/A"
        lines = [
            f"🚨 {mode} — {title} ({a.chain_id})",
            f"Name: {a.name or 'N/A'}",
            f"Score: {a.score:.0f}/100 (market-analysis score, not probability)",
            f"Price: {_money(a.price_usd)}",
            f"Liquidity: {_money(a.liquidity_usd)}",
            f"Volume (1h): {_money(a.volume_h1)} | (24h): {_money(a.volume_h24)}",
            f"Buys/Sells (1h): {buy_total}/{sell_total} | Buy share: {f'{buy_share:.0f}%' if buy_share is not None else 'N/A'}",
            f"1h price change: {a.price_change_h1:+.1f}%",
            f"Market cap: {_money(a.market_cap)}" + (" (FDV upper-bound proxy)" if a.market_cap_source == "dexpaprika_fdv_proxy" else ""),
            f"Alert time (UTC): {__import__('datetime').datetime.fromtimestamp(alert_timestamp, __import__('datetime').timezone.utc).isoformat(timespec='seconds') if alert_timestamp is not None else 'N/A'}",
            f"Alert reference price: {_money(a.price_usd)}",
            f"Pool age: {age}",
            "",
            "Positive signals:",
        ]
        lines.extend([f"• {reason}" for reason in (a.reasons[:6] or ["No strong positive signals identified"])])
        if momentum:
            lines.extend([
                "",
                f"Early momentum: {momentum.get('early_momentum_classification', 'INSUFFICIENT EVIDENCE')}",
                f"Market-cap band: {momentum.get('early_momentum_market_cap_tier', 'unknown')}",
                "Market-cap milestones: " + (", ".join(f"$ {int(x):,}" for x in momentum.get("market_cap_progression", [])) or "none reached yet"),
                f"On-chain evidence: {momentum.get('wallet_evidence_quality', 'not available')}",
                "Momentum evidence: " + (", ".join(momentum.get("early_momentum_signals") or []) or "not yet established"),
                "Momentum cautions: " + (", ".join(momentum.get("early_momentum_cautions") or []) or "none reported"),
                (f"Top holder-owner share: {float(momentum.get('top1_holder_owner_share', 0)):.1%}; "
                 f"top five: {float(momentum.get('top5_holder_owner_share', 0)):.1%}; "
                 f"sampled owners: {int(momentum.get('sampled_holder_owner_count', 0))}"
                 if momentum.get("sampled_holder_owner_count") else "Holder distribution: insufficient data"),
            ])
        lines.append("Cautions:")
        cautions = list(a.risk_flags)
        cautions.extend(risk_reasons[:4])
        if security_status != "PASS":
            cautions.append(f"RugCheck status: {security_status}")
        lines.extend([f"⚠️ {reason}" for reason in (cautions[:8] or ["No additional cautions from current metrics"])])
        lines.extend([
            f"RugCheck: {security_status}" + (f" | risk score: {risk_score:.1f}/100" if risk_score is not None else ""),
            f"DEX: {a.dex_id or 'N/A'}",
            f"Token: {a.token_address}",
        ])
        if a.pair_address:
            # Always show both provider chart links, independent of the lookup source.
            lines.append(f"DexScreener: https://dexscreener.com/solana/{a.pair_address}")
            lines.append(f"DexPaprika: {a.dexpaprika_url or 'https://dexpaprika.com/solana/pools/' + a.pair_address}")
        elif a.url:
            lines.append(f"Chart: {a.url}")
        if probabilities:
            lines.extend([
                "",
                f"MODEL PROBABILITIES — NEXT {24}H",
                f"2x: {probabilities['2x']:.1%}",
                f"5x: {probabilities['5x']:.1%}",
                f"10x: {probabilities['10x']:.1%}",
                f"Model version: {model_version or 'unknown'}",
            ])
        lines.append("\nREAD-ONLY research signal, not a buy instruction. Probabilities are experimental and do not guarantee executable returns.")
        return await self.send("\n".join(lines))

    async def prediction_alert(self, a: TokenAnalysis, p: dict, risk_score: float | None,
                              risk_reasons: list[str], version: str):
        features = {"buys_h1": a.buys_h1, "sells_h1": a.sells_h1}
        await self.signal_alert(a, features, "PASS", risk_score, risk_reasons, p, version)

    async def alert(self, a: TokenAnalysis):
        links = (
            f"\nDexScreener: https://dexscreener.com/solana/{a.pair_address}"
            f"\nDexPaprika: {a.dexpaprika_url or 'https://dexpaprika.com/solana/pools/' + a.pair_address}"
            if a.pair_address else f"\nChart: {a.url}"
        )
        await self.send(f"{a.decision}: {a.symbol}\nToken: {a.token_address}{links}\nRead-only heuristic alert.")

    async def poll_commands(self):
        r = await self.client.get(f"{self.base}/getUpdates", params={"timeout": 1, "offset": self.offset})
        r.raise_for_status()
        for update in r.json().get("result", []):
            self.offset = max(self.offset, int(update["update_id"]) + 1)
            msg = update.get("message") or {}
            chat = msg.get("chat") or {}
            if str(chat.get("id")) != str(self.chat_id):
                continue
            command = (msg.get("text") or "").strip().split()[0].lower() if msg.get("text") else ""
            if command in ("/start", "/help"):
                await self.send("ScannerBot v2.0 is online. Commands: /status")
            elif command == "/status":
                await self.send("ScannerBot is running read-only. No wallet keys or trade execution are enabled.")
