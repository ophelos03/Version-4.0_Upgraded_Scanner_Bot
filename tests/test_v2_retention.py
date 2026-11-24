import time

from scannerbot.database import Database


def test_raw_market_history_is_not_pruned_before_a_completed_archive_sweep(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    old = time.time() - 48 * 3600
    db.save_market_snapshot("solana", "mint", "pair", old, 60, 1, 10000, None, None, {})
    # Even a non-empty replay table is not enough: no completed-sweep watermark exists.
    db.conn.execute(
        "INSERT INTO training_examples(snapshot_id,chain_id,token_address,captured_at,archived_at,features_json,max_multiple,y_2x,y_5x,y_10x) VALUES(1,'solana','mint',?,?,?,1,0,0,0)",
        (old, time.time(), "{}"),
    )
    db.conn.commit()
    deleted = db.prune_old_data(30)
    assert deleted["market_snapshots"] == 0
    assert db.conn.execute("SELECT COUNT(*) FROM market_snapshots").fetchone()[0] == 1
    db.close()


def test_raw_market_history_is_pruned_after_completed_sweep_covers_cutoff(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    old = time.time() - 48 * 3600
    db.save_market_snapshot("solana", "mint", "pair", old, 60, 1, 10000, None, None, {})
    cutoff = time.time() - 30 * 3600
    db.training_conn.execute(
        "INSERT INTO eligible_training_archive_state(id,processed_before,completed_at,horizon_seconds) VALUES(1,?,?,86400)",
        (cutoff + 1, time.time()),
    )
    db.conn.commit()
    deleted = db.prune_old_data(30, horizon_hours=24)
    assert deleted["market_snapshots"] == 1
    db.close()


def test_raw_market_history_is_not_pruned_when_label_horizon_changed(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    old = time.time() - 48 * 3600
    db.save_market_snapshot("solana", "mint", "pair", old, 60, 1, 10000, None, None, {})
    cutoff = time.time() - 30 * 3600
    db.training_conn.execute(
        "INSERT INTO eligible_training_archive_state(id,processed_before,completed_at,horizon_seconds) VALUES(1,?,?,86400)",
        (cutoff + 1, time.time()),
    )
    db.conn.commit()
    deleted = db.prune_old_data(30, horizon_hours=48)
    assert deleted["market_snapshots"] == 0
    db.close()
