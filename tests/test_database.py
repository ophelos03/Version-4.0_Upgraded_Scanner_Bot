from scannerbot.database import Database
from scannerbot.models import Candidate

def test_candidate_registration_and_snapshot_storage(tmp_path):
    db=Database(str(tmp_path/"test.db"))
    db.register_candidate(Candidate("solana","MintABC","test",{"symbol":"ABC"}))
    rows=db.active_candidates(60,10)
    assert len(rows)==1
    assert rows[0]["token_address"]=="MintABC"
    db.save_market_snapshot("solana","MintABC","PairABC",1000,30,0.01,12000,100000,110000,{"volume_m5_usd":500})
    saved=db.recent_snapshots("solana","MintABC",1)
    assert saved[0]["pair_age_minutes"]==30
    assert '"volume_m5_usd": 500' in saved[0]["features_json"]
    db.close()



def test_existing_v1_database_is_extended_without_losing_snapshots(tmp_path):
    import sqlite3

    path = tmp_path / "existing-v1.db"
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE token_snapshots(
          id INTEGER PRIMARY KEY, chain_id TEXT, token_address TEXT, symbol TEXT,
          price_usd REAL, liquidity_usd REAL, volume_h1 REAL, buys_h1 INTEGER,
          sells_h1 INTEGER, score REAL, decision TEXT, captured_at TEXT
        );
        CREATE TABLE candidates(
          chain_id TEXT NOT NULL, token_address TEXT NOT NULL, source TEXT,
          pair_address TEXT, first_seen REAL NOT NULL, pair_created_at REAL,
          metadata_json TEXT, last_seen REAL NOT NULL,
          PRIMARY KEY(chain_id, token_address)
        );
        CREATE TABLE market_snapshots(
          id INTEGER PRIMARY KEY, chain_id TEXT NOT NULL, token_address TEXT NOT NULL,
          pair_address TEXT, captured_at REAL NOT NULL, pair_age_minutes REAL,
          price_usd REAL, liquidity_usd REAL, market_cap REAL, fdv REAL,
          features_json TEXT NOT NULL
        );
        INSERT INTO token_snapshots VALUES(
          7,'solana','mint','OLD',0.01,12000,5000,40,20,75,'WATCHLIST','2026-11-17T10:00:00+00:00'
        );
        INSERT INTO market_snapshots VALUES(
          9,'solana','mint','pair',1794900000,60,0.01,12000,100000,110000,'{}'
        );
    """)
    con.commit()
    con.close()

    db = Database(str(path))
    assert tuple(db.conn.execute("SELECT id,symbol FROM token_snapshots").fetchone()) == (7, "OLD")
    assert tuple(db.conn.execute("SELECT id,token_address FROM market_snapshots").fetchone()) == (9, "mint")
    tables = {
        row[0] for row in db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    assert {"alert_state", "predictions", "security_checks", "training_examples", "training_archive_state"} <= tables
    indexes = {
        row[0] for row in db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        )
    }
    assert "idx_market_token_time" in indexes
    db.close()


def test_alert_is_permanently_deduplicated_and_tracks_from_call_price(tmp_path):
    db = Database(str(tmp_path / "alerts.db"))
    assert db.should_alert("solana", "MintABC")
    db.record_alert(
        "solana", "MintABC", alerted_at=1000, price_usd=0.01,
        market_cap_usd=50000, liquidity_usd=12000,
        features={"transactions_m5": 10},
    )
    assert not db.should_alert("solana", "MintABC", cooldown_minutes=0)
    db.save_market_snapshot("solana", "MintABC", "PairABC", 1060, 31, 0.02, 12000, 100000, 110000, {})
    event = db.alert_event("solana", "MintABC")
    assert event["alerted_at"] == 1000
    assert event["alert_price_usd"] == 0.01
    assert event["max_multiple_since_alert"] == 2
    assert event["first_2x_at"] == 1060
    db.save_market_snapshot("solana", "MintABC", "PairABC", 1120, 32, 0.005, 11000, 50000, 60000, {})
    event = db.alert_event("solana", "MintABC")
    assert event["max_multiple_since_alert"] == 2
    assert event["min_price_since_alert"] == 0.005
    assert not db.should_alert("solana", "MintABC", cooldown_minutes=0)
    db.close()
