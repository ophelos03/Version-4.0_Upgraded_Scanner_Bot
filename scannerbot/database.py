from __future__ import annotations
import json, math, sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any
from .models import TokenAnalysis, Candidate, canonical_token_key

class Database:
    def __init__(self, path: str, outcome_horizon_hours: int = 24, training_archive_path: str | None = None):
        p = Path(path)
        if str(p.parent) not in (".", ""): p.parent.mkdir(parents=True, exist_ok=True)
        self.outcome_horizon_hours = max(1, int(outcome_horizon_hours))
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS token_snapshots(
          id INTEGER PRIMARY KEY, chain_id TEXT, token_address TEXT, symbol TEXT,
          price_usd REAL, liquidity_usd REAL, volume_h1 REAL, buys_h1 INTEGER,
          sells_h1 INTEGER, score REAL, decision TEXT, captured_at TEXT);
        CREATE TABLE IF NOT EXISTS alert_state(
          chain_id TEXT, token_address TEXT, last_alert_at TEXT,
          PRIMARY KEY(chain_id, token_address));
        CREATE TABLE IF NOT EXISTS alert_events(
          chain_id TEXT NOT NULL, token_address TEXT NOT NULL, alerted_at REAL NOT NULL,
          alert_price_usd REAL NOT NULL, market_cap_usd REAL, liquidity_usd REAL,
          features_json TEXT NOT NULL, telegram_message_id INTEGER,
          max_price_since_alert REAL NOT NULL, min_price_since_alert REAL NOT NULL,
          max_multiple_since_alert REAL NOT NULL DEFAULT 1,
          first_2x_at REAL, first_5x_at REAL, first_10x_at REAL,
          last_observed_at REAL NOT NULL,
          PRIMARY KEY(chain_id, token_address));
        CREATE INDEX IF NOT EXISTS idx_alert_events_time ON alert_events(alerted_at);
        CREATE INDEX IF NOT EXISTS idx_snapshots_token ON token_snapshots(chain_id, token_address, captured_at);
        CREATE TABLE IF NOT EXISTS discovery_checkpoints(
          provider TEXT PRIMARY KEY, last_slot INTEGER NOT NULL, updated_at REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS candidate_sources(
          chain_id TEXT NOT NULL, token_address TEXT NOT NULL, source TEXT NOT NULL,
          first_seen REAL NOT NULL, PRIMARY KEY(chain_id,token_address,source));
        CREATE TABLE IF NOT EXISTS candidates(
          chain_id TEXT NOT NULL, token_address TEXT NOT NULL, source TEXT,
          pair_address TEXT, first_seen REAL NOT NULL, pair_created_at REAL,
          metadata_json TEXT, last_seen REAL NOT NULL,
          PRIMARY KEY(chain_id, token_address));
        CREATE TABLE IF NOT EXISTS market_snapshots(
          id INTEGER PRIMARY KEY, chain_id TEXT NOT NULL, token_address TEXT NOT NULL,
          pair_address TEXT, captured_at REAL NOT NULL, pair_age_minutes REAL,
          price_usd REAL, liquidity_usd REAL, market_cap REAL, fdv REAL,
          features_json TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS idx_market_token_time ON market_snapshots(chain_id,token_address,captured_at);
        CREATE TABLE IF NOT EXISTS predictions(
          id INTEGER PRIMARY KEY, chain_id TEXT, token_address TEXT, captured_at REAL,
          pair_age_minutes REAL, p2 REAL, p5 REAL, p10 REAL, model_version TEXT,
          UNIQUE(chain_id,token_address,captured_at));
        CREATE TABLE IF NOT EXISTS security_checks(
          chain_id TEXT, token_address TEXT, checked_at REAL, provider TEXT,
          score REAL, status TEXT, details_json TEXT,
          PRIMARY KEY(chain_id,token_address,provider));
        CREATE TABLE IF NOT EXISTS training_examples(
          snapshot_id INTEGER PRIMARY KEY, chain_id TEXT NOT NULL, token_address TEXT NOT NULL,
          captured_at REAL NOT NULL, archived_at REAL NOT NULL DEFAULT 0,
          features_json TEXT NOT NULL, max_multiple REAL NOT NULL,
          y_2x INTEGER NOT NULL, y_5x INTEGER NOT NULL, y_10x INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS training_archive_state(
          id INTEGER PRIMARY KEY CHECK(id=1), processed_before REAL NOT NULL,
          completed_at REAL NOT NULL, horizon_seconds REAL NOT NULL DEFAULT 86400);
        CREATE INDEX IF NOT EXISTS idx_training_time ON training_examples(captured_at);
        """)
        columns = {row[1] for row in self.conn.execute("PRAGMA table_info(training_examples)")}
        if "archived_at" not in columns:
            self.conn.execute("ALTER TABLE training_examples ADD COLUMN archived_at REAL NOT NULL DEFAULT 0")
        archive_columns = {row[1] for row in self.conn.execute("PRAGMA table_info(training_archive_state)")}
        if "horizon_seconds" not in archive_columns:
            self.conn.execute("ALTER TABLE training_archive_state ADD COLUMN horizon_seconds REAL NOT NULL DEFAULT 86400")
        self.conn.commit()
        archive_path = training_archive_path or str(Path(path).with_name("training_archive.sqlite"))
        archive_file = Path(archive_path)
        archive_file.parent.mkdir(parents=True, exist_ok=True)
        self.training_conn = sqlite3.connect(str(archive_file), check_same_thread=False)
        self.training_conn.row_factory = sqlite3.Row
        self.training_conn.execute("PRAGMA journal_mode=WAL")
        self.training_conn.execute("PRAGMA synchronous=NORMAL")
        self.training_conn.executescript("""
                CREATE TABLE IF NOT EXISTS training_baselines(
          chain_id TEXT NOT NULL, token_address TEXT NOT NULL, pair_address TEXT,
          captured_at REAL NOT NULL, pair_age_minutes REAL NOT NULL, baseline_price REAL NOT NULL,
          features_json TEXT NOT NULL, outcome_complete INTEGER NOT NULL DEFAULT 0,
          max_multiple REAL, observed_count INTEGER NOT NULL DEFAULT 0,
          PRIMARY KEY(chain_id,token_address));
        CREATE TABLE IF NOT EXISTS training_observations(
          chain_id TEXT NOT NULL, token_address TEXT NOT NULL, captured_at REAL NOT NULL,
          price_usd REAL NOT NULL,
          PRIMARY KEY(chain_id,token_address,captured_at));
        CREATE INDEX IF NOT EXISTS idx_training_observations_window
          ON training_observations(chain_id,token_address,captured_at);
        CREATE TABLE IF NOT EXISTS eligible_training_examples(
          snapshot_id INTEGER PRIMARY KEY, chain_id TEXT NOT NULL, token_address TEXT NOT NULL,
          captured_at REAL NOT NULL, archived_at REAL NOT NULL DEFAULT 0,
          features_json TEXT NOT NULL, max_multiple REAL NOT NULL,
          y_2x INTEGER NOT NULL, y_5x INTEGER NOT NULL, y_10x INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS eligible_training_archive_state(
          id INTEGER PRIMARY KEY CHECK(id=1), processed_before REAL NOT NULL,
          completed_at REAL NOT NULL, horizon_seconds REAL NOT NULL DEFAULT 7200);
        CREATE INDEX IF NOT EXISTS idx_eligible_training_time ON eligible_training_examples(captured_at);

        """)
        self.training_conn.commit()

    def discovery_checkpoint(self, provider: str) -> int:
        row = self.conn.execute(
            "SELECT last_slot FROM discovery_checkpoints WHERE provider=?", (provider,)
        ).fetchone()
        return int(row[0]) if row else 0

    def save_discovery_checkpoint(self, provider: str, slot: int) -> None:
        if slot <= 0:
            return
        self.conn.execute(
            """INSERT INTO discovery_checkpoints(provider,last_slot,updated_at) VALUES(?,?,?)
               ON CONFLICT(provider) DO UPDATE SET
               last_slot=MAX(discovery_checkpoints.last_slot,excluded.last_slot),
               updated_at=excluded.updated_at""",
            (provider, int(slot), datetime.now(timezone.utc).timestamp()),
        )
        self.conn.commit()

    def register_candidate(self, c: Candidate, now: float | None = None) -> tuple[bool, bool]:
        now = now or datetime.now(timezone.utc).timestamp()
        key = canonical_token_key(c.chain_id, c.token_address)
        cur = self.conn.execute(
            """INSERT OR IGNORE INTO candidates
               (chain_id,token_address,source,pair_address,first_seen,pair_created_at,metadata_json,last_seen)
               VALUES(?,?,?,?,?,?,?,?)""",
            (c.chain_id,key,c.source,c.pair_address,now,c.created_at,json.dumps(c.metadata,default=str),now),
        )
        is_new_token = cur.rowcount == 1
        if not is_new_token:
            self.conn.execute(
                """UPDATE candidates SET source=?,pair_address=COALESCE(NULLIF(?,''),pair_address),
                   pair_created_at=COALESCE(pair_created_at,?),metadata_json=?,last_seen=?
                   WHERE chain_id=? AND token_address=?""",
                (c.source,c.pair_address,c.created_at,json.dumps(c.metadata,default=str),now,c.chain_id,key),
            )
        source = c.source.split("+")[0]
        if source in {"profile", "boost"}: source = "DexScreener"
        if source.startswith("shrine"): source = "Shrine"
        if source == "pumpportal": source = "PumpPortal"
        source_cur = self.conn.execute(
            "INSERT OR IGNORE INTO candidate_sources(chain_id,token_address,source,first_seen) VALUES(?,?,?,?)",
            (c.chain_id,key,source,now),
        )
        self.conn.commit()
        return is_new_token, source_cur.rowcount == 1

    def active_candidates(self, max_age_minutes: int, limit: int) -> list[dict[str, Any]]:
        cutoff = datetime.now(timezone.utc).timestamp() - max_age_minutes*60
        rows = self.conn.execute("SELECT * FROM candidates WHERE last_seen>=? ORDER BY first_seen DESC LIMIT ?",(cutoff,limit)).fetchall()
        return [dict(r) for r in rows]

    def save_market_snapshot(self, chain: str, token: str, pair: str, captured: float,
                             age: float, price: float, liquidity: float, mcap: float | None,
                             fdv: float | None, features: dict[str, Any]) -> None:
        self.conn.execute("""INSERT INTO market_snapshots(chain_id,token_address,pair_address,captured_at,pair_age_minutes,
        price_usd,liquidity_usd,market_cap,fdv,features_json) VALUES(?,?,?,?,?,?,?,?,?,?)""",
        (chain,canonical_token_key(chain,token),pair,captured,age,price,liquidity,mcap,fdv,json.dumps(features,default=str)))
        self.conn.commit()
        self._update_alert_outcome(chain, token, captured, price)

    def has_training_baseline(self, chain: str, token: str) -> bool:
        row = self.training_conn.execute(
            "SELECT 1 FROM training_baselines WHERE chain_id=? AND token_address=?",
            (chain, canonical_token_key(chain, token)),
        ).fetchone()
        return row is not None

    def record_training_observation(self, chain: str, token: str, captured: float, price: float) -> None:
        if not isinstance(price, (int, float)) or not math.isfinite(float(price)) or price < 0:
            return
        key = canonical_token_key(chain, token)
        baseline = self.training_conn.execute(
            "SELECT captured_at FROM training_baselines WHERE chain_id=? AND token_address=? AND outcome_complete=0",
            (chain, key),
        ).fetchone()
        if not baseline:
            return
        start = float(baseline[0])
        if captured <= start or captured > start + self.outcome_horizon_hours * 3600:
            return
        self.training_conn.execute(
            "INSERT OR IGNORE INTO training_observations(chain_id,token_address,captured_at,price_usd) VALUES(?,?,?,?)",
            (chain, key, captured, price),
        )
        self.training_conn.commit()

    def create_training_baseline(self, chain: str, token: str, pair: str, captured: float,
                                 age: float, price: float, features: dict[str, Any]) -> bool:
        if price <= 0:
            return False
        key = canonical_token_key(chain, token)
        cur = self.training_conn.execute(
            """INSERT OR IGNORE INTO training_baselines
               (chain_id,token_address,pair_address,captured_at,pair_age_minutes,baseline_price,features_json)
               VALUES(?,?,?,?,?,?,?)""",
            (chain, key, pair, captured, age, price, json.dumps(features, default=str)),
        )
        self.training_conn.commit()
        return cur.rowcount == 1

    def _update_alert_outcome(self, chain: str, token: str, captured: float, price: float) -> None:
        if price <= 0:
            return
        key = canonical_token_key(chain, token)
        row = self.conn.execute(
            "SELECT * FROM alert_events WHERE chain_id=? AND token_address=?",
            (chain, key),
        ).fetchone()
        if not row or captured < float(row["alerted_at"]):
            return
        # Measure post-call outcomes only inside the configured forward window.
        if captured > float(row["alerted_at"]) + self.outcome_horizon_hours * 3600:
            return
        alert_price = float(row["alert_price_usd"])
        maximum = max(float(row["max_price_since_alert"]), price)
        minimum = min(float(row["min_price_since_alert"]), price)
        multiple = maximum / alert_price if alert_price > 0 else 1.0
        first = {}
        for target, column in ((2.0, "first_2x_at"), (5.0, "first_5x_at"), (10.0, "first_10x_at")):
            first[column] = row[column] if row[column] is not None else (captured if multiple >= target else None)
        self.conn.execute(
            """UPDATE alert_events SET max_price_since_alert=?, min_price_since_alert=?,
               max_multiple_since_alert=?, first_2x_at=?, first_5x_at=?, first_10x_at=?,
               last_observed_at=? WHERE chain_id=? AND token_address=?""",
            (maximum, minimum, multiple, first["first_2x_at"], first["first_5x_at"],
             first["first_10x_at"], captured, chain, key),
        )
        self.conn.commit()

    def recent_snapshots(self, chain: str, token: str, limit: int = 2) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM market_snapshots WHERE chain_id=? AND token_address=? ORDER BY captured_at DESC LIMIT ?",(chain,canonical_token_key(chain,token),limit)).fetchall()
        return [dict(r) for r in rows]

    def save_prediction(self, chain: str, token: str, captured: float, age: float,
                        p2: float, p5: float, p10: float, version: str) -> None:
        self.conn.execute("INSERT OR IGNORE INTO predictions(chain_id,token_address,captured_at,pair_age_minutes,p2,p5,p10,model_version) VALUES(?,?,?,?,?,?,?,?)",
                          (chain,canonical_token_key(chain,token),captured,age,p2,p5,p10,version)); self.conn.commit()

    def should_alert(self, chain: str, token: str, cooldown_minutes: int = 0) -> bool:
        # A called token is permanently deduplicated; cooldown no longer permits repeats.
        row = self.conn.execute(
            "SELECT 1 FROM alert_state WHERE chain_id=? AND token_address=?",
            (chain, canonical_token_key(chain, token)),
        ).fetchone()
        return row is None

    def record_alert(
        self, chain: str, token: str, alerted_at: float | None = None,
        price_usd: float | None = None, market_cap_usd: float | None = None,
        liquidity_usd: float | None = None, features: dict[str, Any] | None = None,
        telegram_message_id: int | None = None,
    ) -> None:
        alerted_at = float(alerted_at if alerted_at is not None else datetime.now(timezone.utc).timestamp())
        now_iso = datetime.fromtimestamp(alerted_at, timezone.utc).isoformat()
        key = canonical_token_key(chain, token)
        self.conn.execute(
            "INSERT OR IGNORE INTO alert_state VALUES(?,?,?)", (chain, key, now_iso)
        )
        if price_usd is not None and float(price_usd) > 0:
            self.conn.execute(
                """INSERT OR IGNORE INTO alert_events
                   (chain_id,token_address,alerted_at,alert_price_usd,market_cap_usd,
                    liquidity_usd,features_json,telegram_message_id,max_price_since_alert,
                    min_price_since_alert,max_multiple_since_alert,last_observed_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,1,?)""",
                (chain, key, alerted_at, float(price_usd), market_cap_usd,
                 liquidity_usd, json.dumps(features or {}, default=str), telegram_message_id,
                 float(price_usd), float(price_usd), alerted_at),
            )
        self.conn.commit()

    def alert_event(self, chain: str, token: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM alert_events WHERE chain_id=? AND token_address=?",
            (chain, canonical_token_key(chain, token)),
        ).fetchone()
        return dict(row) if row else None

    def save_snapshot(self, a: TokenAnalysis) -> None:
        self.conn.execute("INSERT INTO token_snapshots(chain_id,token_address,symbol,price_usd,liquidity_usd,volume_h1,buys_h1,sells_h1,score,decision,captured_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
          (a.chain_id,canonical_token_key(a.chain_id,a.token_address),a.symbol,a.price_usd,a.liquidity_usd,a.volume_h1,a.buys_h1,a.sells_h1,a.score,a.decision,datetime.now(timezone.utc).isoformat())); self.conn.commit()

    def save_security(self, chain: str, token: str, provider: str, score: float | None, status: str, details: dict) -> None:
        self.conn.execute("INSERT OR REPLACE INTO security_checks VALUES(?,?,?,?,?,?,?)",(chain,canonical_token_key(chain,token),datetime.now(timezone.utc).timestamp(),provider,score,status,json.dumps(details,default=str))); self.conn.commit()

    def count_snapshots(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM token_snapshots").fetchone()[0])

    def prune_old_data(self, retention_hours: int, horizon_hours: int = 24) -> dict[str, int]:
        """Bound raw telemetry while keeping the compact labelled replay archive."""
        cutoff = datetime.now(timezone.utc).timestamp() - max(1, retention_hours) * 3600
        # A non-empty replay table alone does not prove that the latest archive
        # sweep processed every matured raw snapshot. Only prune after a full
        # successful sweep has covered the retention cutoff.
        state = self.training_conn.execute(
            "SELECT processed_before,horizon_seconds FROM eligible_training_archive_state WHERE id=1"
        ).fetchone()
        archive_ready = (
            state is not None
            and int(float(state[1])) == int(horizon_hours * 3600)
        )
        before_market = self.conn.total_changes
        if archive_ready:
            safe_cutoff = min(cutoff, float(state[0]))
            self.conn.execute("DELETE FROM market_snapshots WHERE captured_at < ?", (safe_cutoff,))
        market_deleted = self.conn.total_changes - before_market
        before_tokens = self.conn.total_changes
        iso_cutoff = datetime.fromtimestamp(cutoff, timezone.utc).isoformat()
        self.conn.execute("DELETE FROM token_snapshots WHERE captured_at < ?", (iso_cutoff,))
        token_deleted = self.conn.total_changes - before_tokens
        self.training_conn.execute(
            """DELETE FROM training_observations WHERE captured_at < ? AND EXISTS (
                 SELECT 1 FROM training_baselines b
                 WHERE b.chain_id=training_observations.chain_id
                   AND b.token_address=training_observations.token_address
                   AND b.outcome_complete=1)""", (cutoff,))
        self.conn.execute("DELETE FROM candidates WHERE last_seen < ?", (cutoff,))
        self.conn.commit()
        self.training_conn.commit()
        # WAL checkpoint is best-effort; VACUUM is deliberately not run on the live DB.
        try:
            self.conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
        except sqlite3.Error:
            pass
        return {"market_snapshots": market_deleted, "token_snapshots": token_deleted}

    def training_example_count(self) -> int:
        return int(self.training_conn.execute("SELECT COUNT(*) FROM eligible_training_examples").fetchone()[0])

    def close(self) -> None:
        self.conn.close()
        self.training_conn.close()
