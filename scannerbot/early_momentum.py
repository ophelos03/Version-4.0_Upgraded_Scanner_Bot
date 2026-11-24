from __future__ import annotations

import asyncio
import base64
import math
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

import httpx

log = logging.getLogger(__name__)


def classify_early_momentum(features: dict[str, Any], history: list[dict[str, Any]] | None = None,
                            wallet_evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    """Conservative momentum label; never calls momentum genuine without wallet evidence."""
    history = history or []
    wallet_evidence = wallet_evidence or {}
    signals: list[str] = []
    cautions: list[str] = []
    price_delta = float(features.get("price_return_since_snapshot_pct") or 0)
    volume_delta = float(features.get("volume_change_since_snapshot_pct") or 0)
    price_m5 = float(features.get("price_change_m5_pct") or 0)
    volume_accel = float(features.get("volume_acceleration_5m_vs_hour") or 0)
    tx_accel = float(features.get("transaction_acceleration_5m_vs_hour") or 0)
    liquidity_delta = float(features.get("liquidity_change_since_snapshot_pct") or 0)
    buy_share = float(features.get("buy_share_m5") or 0)
    txns = int(features.get("transactions_m5") or 0)
    mcap = float(features.get("market_cap_usd") or 0)

    if price_delta > 0 or float(features.get("price_change_m5_pct") or 0) > 0:
        signals.append("positive short-interval price movement")
    if float(features.get("price_change_m5_pct") or 0) > 0:
        signals.append("positive 5m price momentum")
    if volume_delta >= 20 or volume_accel >= 1.5:
        signals.append("5m volume accelerating")
    if tx_accel >= 1.5:
        signals.append("5m transaction acceleration")
    if float(features.get("buy_share_change_since_snapshot") or 0) > 0.05:
        signals.append("buy pressure strengthening")
    if buy_share >= 0.60:
        signals.append("buy-side pressure")
    if txns >= 8:
        signals.append("active transaction flow")
    if liquidity_delta > 0:
        signals.append("pool liquidity increasing")
    elif liquidity_delta < -5:
        cautions.append("pool liquidity declining")
    if history:
        recent = history[-1]
        if float(recent.get("price_usd") or 0) > 0 and float(features.get("price_usd") or 0) > float(recent["price_usd"]):
            signals.append("price above recent archived observation")
    if mcap > 0:
        tier = "15k-30k" if 15000 <= mcap < 30000 else "30k-50k" if 30000 <= mcap < 50000 else "50k-80k" if 50000 <= mcap <= 80000 else "below-15k" if mcap < 15000 else "above-80k"
    else:
        tier = "unknown"
    if wallet_evidence.get("suspicious"):
        label = "SUSPICIOUS MOMENTUM"
        cautions.extend(wallet_evidence.get("reasons") or ["on-chain wallet/flow pattern is suspicious"])
    elif len(signals) >= 3 and wallet_evidence.get("low_manipulation") is True:
        label = "EARLY MOMENTUM"
    else:
        label = "INSUFFICIENT EVIDENCE"
        if not wallet_evidence:
            cautions.append("wallet concentration, bundles and independent buyers have not been verified")
    if wallet_evidence and not wallet_evidence.get("transaction_flow_verified"):
        cautions.append(
            "Trade-level buyer independence, repeated counterparties, bundles, shared funders and developer selling are not yet verified"
        )
    evidence_quality = (
        "holder-concentration-only" if wallet_evidence.get("holder_concentration", {}).get("available")
        else "creator-candidate-only" if wallet_evidence.get("creator_candidate")
        else "market-data-only"
    )
    return {"classification": label, "signals": signals, "cautions": cautions,
            "market_cap_tier": tier, "evidence_quality": evidence_quality}


def early_entry_signal_reasons(features: dict[str, Any]) -> list[str]:
    """Return independent short-window signals suitable for early-entry evaluation."""
    price_m5 = float(features.get("price_change_m5_pct") or 0)
    price_delta = float(features.get("price_return_since_snapshot_pct") or 0)
    volume_delta = float(features.get("volume_change_since_snapshot_pct") or 0)
    volume_accel = float(features.get("volume_acceleration_5m_vs_hour") or 0)
    tx_delta = float(features.get("transaction_change_since_snapshot_pct") or 0)
    tx_accel = float(features.get("transaction_acceleration_5m_vs_hour") or 0)
    buy_share = float(features.get("buy_share_m5") or 0)
    txns = int(features.get("transactions_m5") or 0)
    liquidity_delta = float(features.get("liquidity_change_since_snapshot_pct") or 0)
    interval = float(features.get("snapshot_interval_seconds") or 0)

    signals: list[str] = []
    if price_m5 >= 2.0 or price_delta >= 1.0:
        signals.append("positive price momentum")
    if volume_accel >= 1.5 or volume_delta >= 20.0:
        signals.append("accelerating 5m volume")
    if tx_accel >= 1.5 or tx_delta >= 20.0:
        signals.append("accelerating transaction flow")
    if buy_share >= 0.60 and txns >= 8:
        signals.append("buy pressure at or above 60%")
    if 30 <= interval <= 180 and liquidity_delta >= -2.5:
        signals.append("liquidity stable or increasing")
    return signals


def evaluate_early_entry(
    features: dict[str, Any],
    market_cap_usd: float | None,
    history: list[dict[str, Any]] | None = None,
    wallet_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Strict 15k–20k entry trigger; it supplements, never replaces, Stage 1/security gates."""
    history = history or []
    wallet_evidence = wallet_evidence or {}
    try:
        market_cap = float(market_cap_usd or 0)
    except (TypeError, ValueError):
        market_cap = 0.0

    signals = early_entry_signal_reasons(features)
    names = set(signals)
    interval = float(features.get("snapshot_interval_seconds") or 0)
    reasons: list[str] = []
    qualified = True

    if not 15000 <= market_cap <= 20000:
        qualified = False
        reasons.append("market cap is outside the $15k–$20k target band")
    if not history or not 30 <= interval <= 180:
        qualified = False
        reasons.append("a recent confirming snapshot is not available")
    if wallet_evidence.get("suspicious"):
        qualified = False
        reasons.extend(wallet_evidence.get("reasons") or ["suspicious wallet evidence"])
    concentration = wallet_evidence.get("holder_concentration") or {}
    if concentration.get("available") and wallet_evidence.get("low_manipulation") is False:
        qualified = False
        reasons.append("available holder distribution does not pass the existing low-manipulation screen")

    independent_confirmation = (
        "positive price momentum" in names
        and "buy pressure at or above 60%" in names
        and bool({"accelerating 5m volume", "accelerating transaction flow"} & names)
        and len(names) >= 3
    )
    if not independent_confirmation:
        qualified = False
        reasons.append("requires price momentum, buy pressure, and at least one independent acceleration signal")

    if qualified:
        reasons.append("early-entry momentum confirmed by at least three independent signals")
    return {
        "qualified": qualified,
        "score": min(100, len(signals) * 20),
        "signals": signals,
        "reasons": reasons,
    }


def early_lookup_priority_signal_count(features: dict[str, Any], market_cap_usd: float | None) -> int:
    """Count early signals for faster refresh of promising $10k–$25k candidates."""
    try:
        market_cap = float(market_cap_usd or 0)
    except (TypeError, ValueError):
        return 0
    if not 10000 <= market_cap <= 25000:
        return 0
    return len(early_entry_signal_reasons(features))


class EarlyMomentumArchive:
    """Additive tables in the existing training_archive.sqlite; no new database or volume."""
    def __init__(self, archive_path: str, max_wallet_mb: int = 10):
        path = Path(archive_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.max_wallet_bytes = max(1, int(max_wallet_mb)) * 1024 * 1024
        self.conn = sqlite3.connect(str(path), timeout=30, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA busy_timeout=30000")
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS early_momentum_samples(
          id INTEGER PRIMARY KEY, chain_id TEXT NOT NULL, token_address TEXT NOT NULL,
          captured_at REAL NOT NULL, baseline_price REAL NOT NULL, features_json TEXT NOT NULL,
          classification TEXT NOT NULL, signals_json TEXT NOT NULL, cautions_json TEXT NOT NULL,
          max_price_5m REAL, min_price_5m REAL, max_price_15m REAL, min_price_15m REAL,
          max_price_30m REAL, min_price_30m REAL,
          mature_5m INTEGER NOT NULL DEFAULT 0, mature_15m INTEGER NOT NULL DEFAULT 0,
          mature_30m INTEGER NOT NULL DEFAULT 0,
          baseline_liquidity REAL NOT NULL DEFAULT 0,
          min_liquidity_5m REAL, min_liquidity_15m REAL, min_liquidity_30m REAL,
          UNIQUE(chain_id,token_address,captured_at));
        CREATE INDEX IF NOT EXISTS idx_early_samples_token_time
          ON early_momentum_samples(chain_id,token_address,captured_at);
        CREATE TABLE IF NOT EXISTS early_momentum_observations(
          chain_id TEXT NOT NULL, token_address TEXT NOT NULL, captured_at REAL NOT NULL,
          price_usd REAL NOT NULL, liquidity_usd REAL, market_cap_usd REAL,
          PRIMARY KEY(chain_id,token_address,captured_at));
        CREATE TABLE IF NOT EXISTS market_cap_milestones(
          chain_id TEXT NOT NULL, token_address TEXT NOT NULL, threshold_usd REAL NOT NULL,
          first_reached_at REAL NOT NULL, observed_market_cap REAL NOT NULL,
          PRIMARY KEY(chain_id,token_address,threshold_usd));
        CREATE TABLE IF NOT EXISTS holder_concentration_snapshots(
          chain_id TEXT NOT NULL, token_address TEXT NOT NULL, captured_at REAL NOT NULL,
          sampled_owner_count INTEGER NOT NULL, top1_owner_share REAL NOT NULL,
          top5_owner_share REAL NOT NULL, top10_owner_share REAL NOT NULL,
          provider TEXT NOT NULL,
          PRIMARY KEY(chain_id,token_address,captured_at));
        CREATE TABLE IF NOT EXISTS developer_wallets(
          wallet TEXT PRIMARY KEY, status TEXT NOT NULL, qualifying_calls INTEGER NOT NULL DEFAULT 0,
          confirmed_rugs INTEGER NOT NULL DEFAULT 0, payload_json TEXT NOT NULL,
          created_at REAL NOT NULL, updated_at REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS developer_calls(
          wallet TEXT NOT NULL, chain_id TEXT NOT NULL, token_address TEXT NOT NULL,
          call_price REAL NOT NULL, called_at REAL NOT NULL, max_multiple REAL NOT NULL DEFAULT 1,
          rug_confirmed INTEGER NOT NULL DEFAULT 0, evidence_json TEXT NOT NULL,
          PRIMARY KEY(wallet,chain_id,token_address));
        """)
        columns = {row[1] for row in self.conn.execute("PRAGMA table_info(early_momentum_samples)")}
        migrations = {
            "mature_5m": "INTEGER NOT NULL DEFAULT 0",
            "mature_15m": "INTEGER NOT NULL DEFAULT 0",
            "mature_30m": "INTEGER NOT NULL DEFAULT 0",
            "baseline_liquidity": "REAL NOT NULL DEFAULT 0",
            "min_liquidity_5m": "REAL",
            "min_liquidity_15m": "REAL",
            "min_liquidity_30m": "REAL",
        }
        for name, declaration in migrations.items():
            if name not in columns:
                self.conn.execute(f"ALTER TABLE early_momentum_samples ADD COLUMN {name} {declaration}")
        self.conn.commit()

    def _wallet_bytes(self) -> int:
        wallets = self.conn.execute(
            "SELECT COALESCE(SUM(LENGTH(wallet)+LENGTH(payload_json)),0) FROM developer_wallets"
        ).fetchone()
        calls = self.conn.execute(
            """SELECT COALESCE(SUM(LENGTH(wallet)+LENGTH(chain_id)+LENGTH(token_address)+
               LENGTH(evidence_json)+64),0) FROM developer_calls"""
        ).fetchone()
        return int(wallets[0] or 0) + int(calls[0] or 0)

    def record_dev_call(self, wallet: str, chain: str, token: str, price: float,
                        evidence: dict[str, Any] | None = None, captured_at: float | None = None) -> bool:
        if not wallet or price <= 0:
            return False
        existing = self.developer_wallet(wallet)
        if existing and existing.get("status") == "disqualified":
            return False
        now = captured_at or time.time()
        evidence_json = json.dumps(evidence or {}, sort_keys=True, separators=(",", ":"))
        estimated_increment = (
            len(wallet.encode("utf-8")) + len(chain.encode("utf-8")) +
            len(token.encode("utf-8")) + len(evidence_json.encode("utf-8")) + 64
        )
        if existing is None:
            estimated_increment += len(wallet.encode("utf-8")) + 16
        if self._wallet_bytes() + estimated_increment > self.max_wallet_bytes:
            log.warning("Developer wallet storage cap reached; refusing new call record")
            return False
        inserted = self.conn.execute(
            "INSERT OR IGNORE INTO developer_calls(wallet,chain_id,token_address,call_price,called_at,evidence_json) VALUES(?,?,?,?,?,?)",
            (wallet, chain, token, price, now, evidence_json),
        ).rowcount
        if not inserted:
            return False
        self.conn.execute(
            """INSERT INTO developer_wallets(wallet,status,payload_json,created_at,updated_at)
               VALUES(?,?,?,?,?) ON CONFLICT(wallet) DO UPDATE SET updated_at=excluded.updated_at""",
            (wallet, "candidate", "{}", now, now),
        )
        self.conn.commit()
        self._refresh_wallet(wallet, now)
        return True

    def observe_dev_call(self, chain: str, token: str, price: float | None) -> None:
        if price is None or not isinstance(price, (int, float)) or not __import__("math").isfinite(float(price)) or price < 0:
            return
        rows = self.conn.execute(
            "SELECT wallet,call_price,max_multiple,rug_confirmed FROM developer_calls WHERE chain_id=? AND token_address=?",
            (chain, token),
        ).fetchall()
        for row in rows:
            multiple = max(float(row["max_multiple"]), price / float(row["call_price"]))
            rug = int(bool(row["rug_confirmed"]) or price <= float(row["call_price"]) * 0.90)
            self.conn.execute(
                "UPDATE developer_calls SET max_multiple=?,rug_confirmed=? WHERE wallet=? AND chain_id=? AND token_address=?",
                (multiple, rug, row["wallet"], chain, token),
            )
            self._refresh_wallet(row["wallet"], time.time())
        self.conn.commit()

    def _refresh_wallet(self, wallet: str, now: float) -> None:
        rows = self.conn.execute(
            "SELECT token_address,max_multiple,rug_confirmed FROM developer_calls WHERE wallet=?",
            (wallet,),
        ).fetchall()
        rugs = sum(int(r["rug_confirmed"]) for r in rows)
        qualified_calls = sum(1 for r in rows if float(r["max_multiple"]) >= 5.0 and not int(r["rug_confirmed"]))
        qualified = qualified_calls >= 5
        status = "disqualified" if rugs >= 2 else "verified" if qualified else "candidate"
        if status == "disqualified":
            # User requested retaining the disqualification, not a historical Dev Call evidence archive.
            self.conn.execute("DELETE FROM developer_calls WHERE wallet=?", (wallet,))
            payload = json.dumps({"reason": "two distinct Dev Calls reached 90% rug threshold"}, separators=(",", ":"))
        else:
            payload = json.dumps({"qualifying_calls": qualified_calls, "tracked_calls": len(rows),
                                  "confirmed_rugs": rugs}, separators=(",", ":"))
        self.conn.execute(
            """UPDATE developer_wallets SET status=?,qualifying_calls=?,confirmed_rugs=?,
               payload_json=?,updated_at=? WHERE wallet=?""",
            (status, qualified_calls, rugs, payload, now, wallet),
        )
        self.conn.commit()

    def developer_wallet(self, wallet: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM developer_wallets WHERE wallet=?", (wallet,)).fetchone()
        return dict(row) if row else None

    def record_observation(self, chain: str, token: str, captured: float, price: float,
                           liquidity: float, market_cap: float, features: dict[str, Any],
                           classification: dict[str, Any]) -> list[float]:
        if price <= 0:
            return self.market_cap_progression(chain, token)
        key = token
        self.conn.execute(
            """INSERT INTO early_momentum_observations VALUES(?,?,?,?,?,?)
               ON CONFLICT(chain_id,token_address,captured_at) DO UPDATE SET
               price_usd=excluded.price_usd,
               liquidity_usd=COALESCE(excluded.liquidity_usd,early_momentum_observations.liquidity_usd),
               market_cap_usd=COALESCE(excluded.market_cap_usd,early_momentum_observations.market_cap_usd)""",
            (chain, key, captured, price, liquidity, market_cap),
        )
        payload = json.dumps(features, sort_keys=True, default=str)
        self.conn.execute(
            """INSERT OR IGNORE INTO early_momentum_samples
               (chain_id,token_address,captured_at,baseline_price,features_json,classification,signals_json,cautions_json,
                max_price_5m,min_price_5m,max_price_15m,min_price_15m,max_price_30m,min_price_30m,
                baseline_liquidity,min_liquidity_5m,min_liquidity_15m,min_liquidity_30m)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (chain, key, captured, price, payload, classification["classification"],
             json.dumps(classification["signals"]), json.dumps(classification["cautions"]),
             price, price, price, price, price, price,
             liquidity, liquidity, liquidity, liquidity),
        )
        for threshold in (15000.0, 30000.0, 50000.0, 80000.0):
            if market_cap >= threshold:
                self.conn.execute(
                    """INSERT OR IGNORE INTO market_cap_milestones
                       (chain_id,token_address,threshold_usd,first_reached_at,observed_market_cap)
                       VALUES(?,?,?,?,?)""",
                    (chain, key, threshold, captured, market_cap),
                )
        # Label each snapshot independently over 5m, 15m and 30m, without future leakage.
        for sample in self.conn.execute(
            "SELECT id,captured_at,baseline_price FROM early_momentum_samples WHERE chain_id=? AND token_address=? AND captured_at<?",
            (chain, key, captured),
        ).fetchall():
            elapsed = captured - float(sample["captured_at"])
            for minutes in (5, 15, 30):
                mature_col = f"mature_{minutes}m"
                if elapsed <= minutes * 60:
                    col_max, col_min = f"max_price_{minutes}m", f"min_price_{minutes}m"
                    self.conn.execute(
                        f"""UPDATE early_momentum_samples SET {col_max}=MAX({col_max},?),
                             {col_min}=MIN({col_min},?) WHERE id=?""",
                        (price, price, sample["id"]),
                    )
                    if liquidity > 0:
                        liq_col = f"min_liquidity_{minutes}m"
                        self.conn.execute(
                            f"UPDATE early_momentum_samples SET {liq_col}=MIN(CASE WHEN {liq_col} IS NULL OR {liq_col}=0 THEN ? ELSE {liq_col} END,?) WHERE id=?",
                            (liquidity, liquidity, sample["id"]),
                        )
                if elapsed >= minutes * 60:
                    self.conn.execute(
                        f"UPDATE early_momentum_samples SET {mature_col}=1 WHERE id=?",
                        (sample["id"],),
                    )
        self.conn.commit()
        return self.market_cap_progression(chain, token)

    def record_outcome_observation(self, chain: str, token: str, captured: float,
                                   price: float, liquidity: float | None) -> None:
        """Update prior eligible snapshots even when the token later fails live filters."""
        self.conn.execute(
            "INSERT OR IGNORE INTO early_momentum_observations VALUES(?,?,?,?,?,?)",
            (chain, token, captured, price, liquidity, None),
        )
        for sample in self.conn.execute(
            "SELECT id,captured_at FROM early_momentum_samples WHERE chain_id=? AND token_address=? AND captured_at<?",
            (chain, token, captured),
        ).fetchall():
            elapsed = captured - float(sample["captured_at"])
            for minutes in (5, 15, 30):
                if elapsed <= minutes * 60:
                    self.conn.execute(
                        f"""UPDATE early_momentum_samples SET
                            max_price_{minutes}m=MAX(max_price_{minutes}m,?),
                            min_price_{minutes}m=MIN(min_price_{minutes}m,?)
                            WHERE id=?""",
                        (price, price, sample["id"]),
                    )
                    if liquidity is not None and liquidity >= 0:
                        col = f"min_liquidity_{minutes}m"
                        self.conn.execute(
                            f"UPDATE early_momentum_samples SET {col}=CASE WHEN {col} IS NULL THEN ? ELSE MIN({col},?) END WHERE id=?",
                            (liquidity, liquidity, sample["id"]),
                        )
                if elapsed >= minutes * 60:
                    self.conn.execute(
                        f"UPDATE early_momentum_samples SET mature_{minutes}m=1 WHERE id=?",
                        (sample["id"],),
                    )
        self.conn.commit()

    def record_liquidity_observation(self, chain: str, token: str, captured: float,
                                     liquidity: float | None) -> None:
        """Track observed pool-liquidity drawdown even after a token fails a live gate."""
        if liquidity is None or not isinstance(liquidity, (int, float)) or not math.isfinite(float(liquidity)) or liquidity < 0:
            return
        for sample in self.conn.execute(
            "SELECT id,captured_at FROM early_momentum_samples WHERE chain_id=? AND token_address=? AND captured_at<?",
            (chain, token, captured),
        ).fetchall():
            elapsed = captured - float(sample["captured_at"])
            for minutes in (5, 15, 30):
                if elapsed <= minutes * 60:
                    col = f"min_liquidity_{minutes}m"
                    self.conn.execute(
                        f"UPDATE early_momentum_samples SET {col}=CASE WHEN {col} IS NULL THEN ? ELSE MIN({col},?) END WHERE id=?",
                        (float(liquidity), float(liquidity), sample["id"]),
                    )
        self.conn.commit()

    def record_holder_concentration(self, chain: str, token: str, captured: float,
                                    concentration: dict[str, Any], provider: str) -> dict[str, Any] | None:
        if not concentration.get("available"):
            return None
        row = self.conn.execute(
            """SELECT top1_owner_share,top5_owner_share,top10_owner_share,captured_at
               FROM holder_concentration_snapshots WHERE chain_id=? AND token_address=?
               ORDER BY captured_at DESC LIMIT 1""",
            (chain, token),
        ).fetchone()
        previous = dict(row) if row else None
        self.conn.execute(
            """INSERT OR IGNORE INTO holder_concentration_snapshots
               (chain_id,token_address,captured_at,sampled_owner_count,top1_owner_share,
                top5_owner_share,top10_owner_share,provider) VALUES(?,?,?,?,?,?,?,?)""",
            (chain, token, captured, int(concentration.get("sampled_owner_count") or 0),
             float(concentration.get("top1_owner_share") or 0),
             float(concentration.get("top5_owner_share") or 0),
             float(concentration.get("top10_owner_share") or 0), provider),
        )
        self.conn.commit()
        return previous

    def record_zero_price_observation(self, chain: str, token: str, captured: float,
                                      liquidity: float | None = None) -> None:
        """A reported zero is a valid crash observation, not missing price data."""
        key = token
        self.conn.execute(
            "INSERT OR IGNORE INTO early_momentum_observations VALUES(?,?,?,?,?,?)",
            (chain, key, captured, 0.0, liquidity, None),
        )
        for sample in self.conn.execute(
            "SELECT id,captured_at FROM early_momentum_samples WHERE chain_id=? AND token_address=? AND captured_at<?",
            (chain, key, captured),
        ).fetchall():
            elapsed = captured - float(sample["captured_at"])
            for minutes in (5, 15, 30):
                mature_col = f"mature_{minutes}m"
                if elapsed <= minutes * 60:
                    min_col = f"min_price_{minutes}m"
                    self.conn.execute(
                        f"UPDATE early_momentum_samples SET {min_col}=0 WHERE id=?",
                        (sample["id"],),
                    )
                    if liquidity is not None and liquidity >= 0:
                        liq_col = f"min_liquidity_{minutes}m"
                        self.conn.execute(
                            f"UPDATE early_momentum_samples SET {liq_col}=CASE WHEN {liq_col} IS NULL THEN ? ELSE MIN({liq_col},?) END WHERE id=?",
                            (liquidity, liquidity, sample["id"]),
                        )
                if elapsed >= minutes * 60:
                    self.conn.execute(
                        f"UPDATE early_momentum_samples SET {mature_col}=1 WHERE id=?",
                        (sample["id"],),
                    )
        self.conn.commit()

    def market_cap_progression(self, chain: str, token: str) -> list[float]:
        rows = self.conn.execute(
            "SELECT threshold_usd FROM market_cap_milestones WHERE chain_id=? AND token_address=? ORDER BY threshold_usd",
            (chain, token),
        ).fetchall()
        return [float(row[0]) for row in rows]

    def recent_history(self, chain: str, token: str, limit: int = 10) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT captured_at,baseline_price AS price_usd,features_json FROM early_momentum_samples WHERE chain_id=? AND token_address=? ORDER BY captured_at DESC LIMIT ?",
            (chain, token, limit),
        ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def close(self) -> None:
        self.conn.close()


def _base58_encode(raw: bytes) -> str:
    alphabet = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
    zeroes = len(raw) - len(raw.lstrip(bytes([0])))
    number = int.from_bytes(raw, "big")
    encoded = ""
    while number:
        number, rem = divmod(number, 58)
        encoded = alphabet[rem] + encoded
    return "1" * zeroes + encoded


class SolanaEvidenceClient:
    """Optional Helius-enhanced, public-RPC-fallback evidence lookup.

    Creator identity is a candidate only. Holder concentration is reported as
    token-account-owner evidence and is not treated as proof of manipulation.
    """
    def __init__(self, helius_api_key: str = "", rpc_url: str = ""):
        self.helius_api_key = helius_api_key.strip()
        self.provider = "helius" if self.helius_api_key or "helius" in rpc_url.lower() else "public-rpc"
        self.provider = "helius" if "helius" in rpc_url.lower() or (self.helius_api_key and not rpc_url.strip()) else "public-rpc"
        primary = rpc_url.strip() or (
            f"https://mainnet.helius-rpc.com/?api-key={self.helius_api_key}"
            if self.helius_api_key else "https://solana-rpc.publicnode.com"
        )
        self.rpc_urls = list(dict.fromkeys([
            primary,
            "https://solana-rpc.publicnode.com",
            "https://public.rpc.solanavibestation.com",
            "https://solana.leorpc.com/?api_key=FREE",
            "https://public-solana-mainnet.rpcnode.dev",
            "https://rpc.nodeflare.app/solana/public",
            "https://api.mainnet-beta.solana.com",
        ]))
        self.rpc_url = self.rpc_urls[0]
        self.client = httpx.AsyncClient(timeout=15)
        self.cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self.last_request: dict[str, float] = {}
        self.endpoint_cooldowns: dict[str, float] = {}
        self.lock = asyncio.Lock()

    async def _rpc(self, method: str, params: list[Any]) -> dict[str, Any] | None:
        for endpoint_index, endpoint in enumerate(self.rpc_urls):
            if self.endpoint_cooldowns.get(endpoint, 0.0) > time.time():
                continue
            for attempt in range(2):
                async with self.lock:
                    wait = 0.5 - (time.monotonic() - self.last_request.get(endpoint, 0.0))
                    if wait > 0:
                        await asyncio.sleep(wait)
                    self.last_request[endpoint] = time.monotonic()
                    try:
                        response = await self.client.post(endpoint, json={
                            "jsonrpc": "2.0", "id": 1, "method": method, "params": params,
                        })
                        if response.status_code == 429:
                            retry = response.headers.get("Retry-After")
                            try:
                                cooldown = max(30.0, float(retry)) if retry else 60.0
                            except ValueError:
                                cooldown = 60.0
                            self.endpoint_cooldowns[endpoint] = time.time() + min(cooldown, 900.0)
                            log.warning("Solana RPC endpoint rate limited: method=%s endpoint_index=%s cooldown_seconds=%s",
                                        method, endpoint_index + 1, int(min(cooldown, 900.0)))
                            break
                        if response.status_code == 403:
                            self.endpoint_cooldowns[endpoint] = time.time() + 600.0
                            log.warning("Solana RPC endpoint blocked this service IP: method=%s endpoint_index=%s",
                                        method, endpoint_index + 1)
                            break
                        if response.status_code >= 500:
                            if attempt == 0:
                                await asyncio.sleep(1.0)
                                continue
                            self.endpoint_cooldowns[endpoint] = time.time() + 30.0
                            log.warning("Solana RPC endpoint unavailable: method=%s endpoint_index=%s status=%s",
                                        method, endpoint_index + 1, response.status_code)
                            break
                        if response.status_code >= 400:
                            self.endpoint_cooldowns[endpoint] = time.time() + 300.0
                            log.warning("Solana RPC HTTP error: method=%s endpoint_index=%s status=%s",
                                        method, endpoint_index + 1, response.status_code)
                            break
                        payload = response.json()
                        if payload.get("error"):
                            error = payload.get("error") or {}
                            code = error.get("code")
                            if code == -32603:
                                self.endpoint_cooldowns[endpoint] = time.time() + 60.0
                            log.warning("Solana RPC returned error code=%s: method=%s endpoint_index=%s",
                                        code, method, endpoint_index + 1)
                            break
                        result = payload.get("result")
                        if result is None:
                            break
                        self.endpoint_cooldowns.pop(endpoint, None)
                        return result
                    except (httpx.HTTPError, ValueError) as exc:
                        self.endpoint_cooldowns[endpoint] = time.time() + 30.0
                        log.warning("Solana RPC request failed: method=%s endpoint_index=%s exception=%s",
                                    method, endpoint_index + 1, type(exc).__name__)
                        break
        return None

    async def _holder_concentration(self, mint: str) -> dict[str, Any]:
        largest = await self._rpc("getTokenLargestAccounts", [mint])
        supply = await self._rpc("getTokenSupply", [mint])
        if not largest or not supply:
            return {"available": False, "quality": "insufficient"}
        accounts = (largest.get("value") or [])[:20]
        supply_info = supply.get("value") or {}
        supply_value = supply_info.get("uiAmount")
        if supply_value is None:
            supply_value = supply_info.get("uiAmountString")
        try:
            total_supply = float(supply_value)
        except (TypeError, ValueError):
            total_supply = 0.0
        addresses = [str(item.get("address")) for item in accounts if item.get("address")]
        if total_supply <= 0 or not addresses:
            return {"available": False, "quality": "insufficient"}
        details = await self._rpc("getMultipleAccounts", [addresses, {"encoding": "base64"}])
        values = (details or {}).get("value") or []
        owner_balances: dict[str, float] = {}
        for item, account in zip(accounts, values):
            try:
                data = account["data"][0]
                raw = base64.b64decode(data)
                if len(raw) < 72:
                    continue
                owner = _base58_encode(raw[32:64])
                amount = int.from_bytes(raw[64:72], "little")
                decimals = int((supply.get("value") or {}).get("decimals") or 0)
                ui_amount = amount / (10 ** decimals)
                owner_balances[owner] = owner_balances.get(owner, 0.0) + ui_amount
            except (KeyError, TypeError, ValueError, IndexError):
                continue
        if not owner_balances:
            return {"available": False, "quality": "insufficient"}
        shares = sorted((value / total_supply for value in owner_balances.values()), reverse=True)
        top1 = shares[0]
        top5 = sum(shares[:5])
        top10 = sum(shares[:10])
        return {
            "available": True, "quality": "top20_token_account_owners",
            "sampled_owner_count": len(owner_balances), "top1_owner_share": top1,
            "top5_owner_share": top5, "top10_owner_share": top10,
            "warning": "Largest token-account owners can include LP, vault, escrow and burn accounts; not a wallet-risk verdict.",
        }

    async def creator_for_mint(self, mint: str) -> dict[str, Any] | None:
        cached = self.cache.get(mint)
        if cached and time.time() - cached[0] < 300:
            return cached[1]
        signatures = await self._rpc("getSignaturesForAddress", [mint, {"limit": 100}])
        creator = None
        if signatures:
            earliest = min(signatures, key=lambda item: item.get("blockTime") or float("inf"))
            tx = await self._rpc("getTransaction", [
                earliest["signature"],
                {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 1},
            ])
            message = ((tx or {}).get("transaction") or {}).get("message") or {}
            instructions = message.get("instructions") or []
            init = any(
                (ins.get("parsed") or {}).get("type") in {"initializeMint", "initializeMint2"}
                for ins in instructions if isinstance(ins, dict)
            )
            keys = message.get("accountKeys") or []
            payer = keys[0].get("pubkey") if keys and isinstance(keys[0], dict) else (keys[0] if keys else None)
            if init and payer:
                creator = {"wallet": str(payer), "provider": self.provider,
                           "evidence": "fee payer on observed mint-initialization transaction",
                           "confidence": "candidate_only"}
        concentration = await self._holder_concentration(mint)
        result = {"creator": creator, "holder_concentration": concentration,
                  "provider": self.provider, "evidence_quality": "partial"}
        self.cache[mint] = (time.time(), result)
        return result

    async def close(self) -> None:
        await self.client.aclose()
