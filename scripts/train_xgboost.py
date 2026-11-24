from __future__ import annotations

import argparse
import json
import os
from collections import deque
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    precision_score, recall_score, roc_auc_score,
)
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scannerbot.prediction.predictor import FEATURES

TARGETS = {"2x": 2.0, "5x": 5.0, "10x": 10.0}


def ensure_archive(con: sqlite3.Connection) -> None:
    con.execute("""
        CREATE TABLE IF NOT EXISTS eligible_training_examples(
          snapshot_id INTEGER PRIMARY KEY, chain_id TEXT NOT NULL, token_address TEXT NOT NULL,
          captured_at REAL NOT NULL, archived_at REAL NOT NULL DEFAULT 0,
          features_json TEXT NOT NULL, max_multiple REAL NOT NULL,
          y_2x INTEGER NOT NULL, y_5x INTEGER NOT NULL, y_10x INTEGER NOT NULL)
    """)
    columns = {row[1] for row in con.execute("PRAGMA table_info(eligible_training_examples)")}
    if "archived_at" not in columns:
        con.execute("ALTER TABLE eligible_training_examples ADD COLUMN archived_at REAL NOT NULL DEFAULT 0")
    con.execute("CREATE INDEX IF NOT EXISTS idx_eligible_training_time ON eligible_training_examples(captured_at)")
    con.execute("""
        CREATE TABLE IF NOT EXISTS eligible_training_archive_state(
          id INTEGER PRIMARY KEY CHECK(id=1), processed_before REAL NOT NULL,
          completed_at REAL NOT NULL, horizon_seconds REAL NOT NULL DEFAULT 7200)
    """)
    archive_columns = {row[1] for row in con.execute("PRAGMA table_info(eligible_training_archive_state)")}
    if "horizon_seconds" not in archive_columns:
        con.execute("ALTER TABLE eligible_training_archive_state ADD COLUMN horizon_seconds REAL NOT NULL DEFAULT 7200")
    con.commit()


def _future_window_maxima(
    times: list[float], prices: list[float], liquidities: list[float],
    horizon_seconds: float, min_liquidity: float,
) -> list[float | None]:
    """Legacy market-snapshot helper; new eligible-baseline labels do not use a future liquidity floor."""
    maxima: list[float | None] = [None] * len(times)
    candidates: deque[int] = deque()
    right = 0
    count = len(times)
    for i, captured_at in enumerate(times):
        if right < i:
            right = i
        upper_bound = captured_at + horizon_seconds
        while right + 1 < count and times[right + 1] <= upper_bound:
            right += 1
            candidate_price = prices[right]
            if (times[right] > captured_at and candidate_price > 0
                    and liquidities[right] >= min_liquidity):
                while candidates and prices[candidates[-1]] <= candidate_price:
                    candidates.pop()
                candidates.append(right)
        while candidates and (candidates[0] <= i or times[candidates[0]] <= captured_at):
            candidates.popleft()
        if candidates:
            maxima[i] = prices[candidates[0]]
    return maxima



def archive_matured_examples(con: sqlite3.Connection, horizon_hours: int) -> int:
    """Create one label per eligible baseline after a complete future observation window."""
    if horizon_hours <= 0:
        raise ValueError("horizon_hours must be positive")
    ensure_archive(con)
    now = datetime.now(timezone.utc).timestamp()
    horizon = horizon_hours * 3600
    inserted_count = 0
    # Each eligible token contributes exactly one immutable baseline example.
    baselines = con.execute(
        """SELECT rowid,chain_id,token_address,captured_at,pair_age_minutes,baseline_price,
                  features_json FROM training_baselines
           WHERE outcome_complete=0 AND captured_at + ? <= ?""",
        (horizon, now),
    ).fetchall()
    for baseline_id, chain, token, captured_at, age, baseline_price, features_json in baselines:
        end_at = float(captured_at) + horizon
        observations = con.execute(
            """SELECT captured_at,price_usd FROM training_observations
               WHERE chain_id=? AND token_address=? AND captured_at>? AND captured_at<=?
               ORDER BY captured_at""",
            (chain, token, captured_at, end_at),
        ).fetchall()
        # A sparse/no-observation window is unknown, not a negative label.
        # Require coverage near both ends and multiple observations before labelling.
        if len(observations) < 3:
            continue
        first_at, last_at = float(observations[0][0]), float(observations[-1][0])
        if first_at > float(captured_at) + 300 or last_at < end_at - 300:
            continue
        if any(float(right[0]) - float(left[0]) > 300 for left, right in zip(observations, observations[1:])):
            continue
        valid_prices = [float(row[1]) for row in observations if row[1] is not None and np.isfinite(float(row[1])) and float(row[1]) >= 0]
        if len(valid_prices) < 3 or float(baseline_price or 0) <= 0:
            continue
        maximum = max(valid_prices)
        multiple = maximum / float(baseline_price)
        con.execute(
            """INSERT OR IGNORE INTO eligible_training_examples
               (snapshot_id,chain_id,token_address,captured_at,archived_at,features_json,
                max_multiple,y_2x,y_5x,y_10x)
               VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (int(baseline_id),chain,token,float(captured_at),now,features_json,multiple,
             int(multiple >= 2),int(multiple >= 5),int(multiple >= 10)),
        )
        con.execute(
            """UPDATE training_baselines SET outcome_complete=1,max_multiple=?,observed_count=?
               WHERE chain_id=? AND token_address=?""",
            (multiple,len(observations),chain,token),
        )
        inserted_count += 1
    con.commit()
    # Keep the full eligible historical archive; do not discard older training examples.
    con.execute(
        """INSERT INTO eligible_training_archive_state(id,processed_before,completed_at,horizon_seconds)
           VALUES(1,?,?,?) ON CONFLICT(id) DO UPDATE SET
           processed_before=excluded.processed_before,completed_at=excluded.completed_at,
           horizon_seconds=excluded.horizon_seconds""",
        (now-horizon,now,horizon),
    )
    con.commit()
    return inserted_count


def load_archive(con: sqlite3.Connection) -> pd.DataFrame:
    rows = con.execute(
        """SELECT chain_id,token_address,captured_at,archived_at,features_json,max_multiple,y_2x,y_5x,y_10x
           FROM eligible_training_examples ORDER BY captured_at"""
    ).fetchall()
    if not rows:
        return pd.DataFrame()
    records = []
    for chain, token, captured_at, archived_at, raw_features, max_multiple, y2, y5, y10 in rows:
        try:
            features = json.loads(raw_features)
        except (TypeError, ValueError):
            continue
        record = {k: float(features.get(k, 0) or 0) for k in FEATURES}
        record.update({
            "chain_id": chain, "token_address": token, "captured_at": float(captured_at), "archived_at": float(archived_at),
            "max_multiple_in_window": float(max_multiple), "y_2x": int(y2),
            "y_5x": int(y5), "y_10x": int(y10),
        })
        records.append(record)
    return pd.DataFrame.from_records(records)



def chronological_token_splits(
    df: pd.DataFrame, horizon_hours: int, since_timestamp: float = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Keep the newest batch unseen; train/calibrate on all earlier accumulated examples."""
    if since_timestamp > 0:
        test = df[df.archived_at > since_timestamp].copy()
        development = df[df.archived_at <= since_timestamp].copy()
        if test.empty or development.empty:
            raise RuntimeError("Need prior historical examples and a new held-out batch.")
        if test[["chain_id","token_address"]].drop_duplicates().shape[0] < 10:
            raise RuntimeError("Not enough new distinct tokens for an unseen challenger test cohort.")
        launch = development.groupby(["chain_id","token_address"])["captured_at"].min().sort_values()
        if len(launch) < 10:
            raise RuntimeError("Not enough historical tokens for train/calibration cohorts.")
        calibration_end = max(1, int(len(launch) * 0.15))
        calibration_keys = set(launch.index[-calibration_end:].tolist())
        train_keys = set(launch.index[:-calibration_end].tolist())
        row_keys = list(zip(development.chain_id, development.token_address))
        train = development[[key in train_keys for key in row_keys]].copy()
        calibration = development[[key in calibration_keys for key in row_keys]].copy()
        if train.empty or calibration.empty:
            raise RuntimeError("Historical train/calibration cohort is empty.")
        return train, calibration, test

    launch = df.groupby(["chain_id","token_address"])["captured_at"].min().sort_values()
    n = len(launch)
    if n < 10:
        raise RuntimeError("Need at least 10 distinct matured tokens for disjoint cohorts.")
    train_end = max(1, int(n * 0.70))
    calibration_end = max(train_end + 1, int(n * 0.80))
    if calibration_end >= n:
        raise RuntimeError("Not enough distinct tokens for train/calibration/test cohorts.")
    keys = list(launch.index)
    train_keys, calibration_keys, test_keys = set(keys[:train_end]), set(keys[train_end:calibration_end]), set(keys[calibration_end:])
    row_keys = list(zip(df.chain_id, df.token_address))
    train = df[[key in train_keys for key in row_keys]].copy()
    calibration = df[[key in calibration_keys for key in row_keys]].copy()
    test = df[[key in test_keys for key in row_keys]].copy()
    if train.empty or calibration.empty or test.empty:
        raise RuntimeError("Chronological train/calibration/test cohort is empty.")
    return train, calibration, test


def logit(values: np.ndarray) -> np.ndarray:
    eps = 1e-6
    clipped = np.clip(values, eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped)).reshape(-1, 1)



def training_database_diagnostics(db_path: Path, horizon_hours: int, since_timestamp: float) -> dict:
    """Inspect the live training database without creating or modifying it."""
    volume_path = Path(os.environ.get("RAILWAY_VOLUME_MOUNT_PATH", "/data")).expanduser()
    volume_files = []
    if volume_path.is_dir():
        for candidate in sorted(volume_path.iterdir()):
            try:
                stat = candidate.stat()
                volume_files.append({"name": candidate.name, "is_file": candidate.is_file(), "bytes": stat.st_size if candidate.is_file() else None})
            except OSError:
                volume_files.append({"name": candidate.name, "unreadable": True})
    report = {
        "volume_mount_path": str(volume_path),
        "volume_directory_exists": volume_path.is_dir(),
        "volume_root_files": volume_files,
        "database_path": str(db_path.resolve()),
        "database_exists": db_path.is_file(),
        "database_bytes": db_path.stat().st_size if db_path.is_file() else None,
        "horizon_hours": horizon_hours,
    }
    if not db_path.is_file():
        report["error"] = "Database file does not exist; refusing to create an empty replacement."
        return report

    uri = f"file:{db_path.resolve().as_posix()}?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    try:
        tables = {r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        report["tables_present"] = {
            name: name in tables for name in
            ("training_baselines", "training_observations", "eligible_training_examples", "eligible_training_archive_state")
        }
        if "training_baselines" not in tables or "training_observations" not in tables:
            report["error"] = "Persistent eligible training archive tables are missing."
            return report

        now = datetime.now(timezone.utc).timestamp()
        horizon = horizon_hours * 3600
        baseline_row = con.execute(
            """SELECT COUNT(*),MIN(captured_at),MAX(captured_at),
                      COUNT(DISTINCT chain_id || ':' || token_address),
                      SUM(CASE WHEN outcome_complete=1 THEN 1 ELSE 0 END)
               FROM training_baselines"""
        ).fetchone() if "training_baselines" in tables else (0,None,None,0,0)
        report["eligible_baselines"] = {
            "total": int(baseline_row[0] or 0), "oldest_timestamp": baseline_row[1],
            "newest_timestamp": baseline_row[2], "distinct_tokens": int(baseline_row[3] or 0),
            "outcomes_completed": int(baseline_row[4] or 0),
        }
        report["future_observations"] = int(con.execute(
            "SELECT COUNT(*) FROM training_observations"
        ).fetchone()[0]) if "training_observations" in tables else 0
        report["snapshot_eligibility"] = "Baselines are created only after Stage 1 passes; see eligible_baselines."

        if "eligible_training_examples" in tables:
            row = con.execute(
                """SELECT COUNT(*), SUM(CASE WHEN archived_at > ? THEN 1 ELSE 0 END),
                          MIN(captured_at), MAX(captured_at)
                   FROM eligible_training_examples""", (since_timestamp,)
            ).fetchone()
            report["training_examples"] = {
                "total": int(row[0] or 0), "since_last_training": int(row[1] or 0),
                "oldest_timestamp": row[2], "newest_timestamp": row[3],
            }
        else:
            report["training_examples"] = {"total": 0, "since_last_training": 0}

        if "eligible_training_archive_state" in tables:
            state = con.execute(
                "SELECT processed_before, completed_at, horizon_seconds "
                "FROM eligible_training_archive_state WHERE id=1"
            ).fetchone()
            report["archive_state"] = (
                {"processed_before": state[0], "completed_at": state[1],
                 "horizon_hours": float(state[2]) / 3600}
                if state else None
            )
        else:
            report["archive_state"] = None
    finally:
        con.close()
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Memory-bounded ScannerBot XGBoost training")
    ap.add_argument("--db", default=os.getenv("TRAINING_ARCHIVE_PATH", str(Path(os.getenv("MODEL_DIR", "/data/models")) / "training_archive.sqlite")))
    ap.add_argument("--model-dir", default=os.getenv("MODEL_DIR", "/data/models"))
    ap.add_argument("--horizon-hours", type=int, default=2)
    ap.add_argument("--min-rows", type=int, default=100)
    ap.add_argument("--min-tokens", type=int, default=100)
    ap.add_argument("--min-train-positives", type=int, default=2)
    ap.add_argument("--min-train-negatives", type=int, default=2)
    ap.add_argument("--min-test-positives", type=int, default=2)
    ap.add_argument("--min-test-negatives", type=int, default=2)
    ap.add_argument("--since-timestamp", type=float, default=0)
    ap.add_argument("--min-new-samples", type=int, default=0)
    ap.add_argument("--incumbent-dir", default="")
    args = ap.parse_args()

    db_path = Path(args.db).expanduser().resolve()
    diagnostics = training_database_diagnostics(
        db_path, args.horizon_hours, args.since_timestamp,
    )
    print("TRAINING_DATABASE_DIAGNOSTICS=" + json.dumps(diagnostics, sort_keys=True), flush=True)
    if diagnostics.get("error"):
        raise RuntimeError(diagnostics["error"])
    con = sqlite3.connect(str(db_path))
    try:
        new_count = archive_matured_examples(
            con, args.horizon_hours,
        )
        since_count = int(con.execute(
            "SELECT COUNT(*) FROM eligible_training_examples WHERE archived_at > ?",
            (args.since_timestamp,),
        ).fetchone()[0])
        if args.min_new_samples and since_count < args.min_new_samples:
            raise RuntimeError(
                f"Only {since_count} newly matured labelled samples since last completed training attempt; "
                f"need {args.min_new_samples}. Archive added {new_count} this attempt."
            )
        df = load_archive(con)
    finally:
        con.close()

    if df.empty:
        raise RuntimeError("No matured labelled examples in the replay archive yet.")
    if len(df) < args.min_rows:
        raise RuntimeError(f"Only {len(df)} archived training rows found; need at least {args.min_rows}.")
    distinct = df[["chain_id", "token_address"]].drop_duplicates().shape[0]
    if distinct < args.min_tokens:
        raise RuntimeError(f"Only {distinct} distinct archived tokens found; need at least {args.min_tokens}.")

    # Token-disjoint chronological cohorts; each label horizon stays before its cutoff.
    train, calibration, test = chronological_token_splits(df, args.horizon_hours, args.since_timestamp)

    out = Path(args.model_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(df)), "distinct_tokens": int(distinct),
        "newly_archived_this_attempt": int(new_count),
        "new_labelled_samples_since_last_training": int(since_count),
        "historical_archive_retained": True,
        "train_rows": int(len(train)), "calibration_rows": int(len(calibration)),
        "test_rows": int(len(test)), "feature_names": FEATURES, "targets": {},
    }

    for key in TARGETS:
        y = train[f"y_{key}"].astype(int)
        yt = test[f"y_{key}"].astype(int)
        train_positives = int(y.sum())
        train_negatives = int((y == 0).sum())
        if train_positives < args.min_train_positives or train_negatives < args.min_train_negatives:
            raise RuntimeError(
                f"Training data for {key} has {train_positives} positives and {train_negatives} negatives; "
                f"need at least {args.min_train_positives} of each class."
            )
        model = XGBClassifier(
            n_estimators=350, max_depth=4, learning_rate=0.04,
            subsample=0.85, colsample_bytree=0.85, min_child_weight=5,
            reg_lambda=5, objective="binary:logistic", eval_metric="logloss",
            tree_method="hist", scale_pos_weight=max(1.0, (len(y)-int(y.sum()))/max(1,int(y.sum()))),
            n_jobs=1, random_state=42, max_bin=128,
        )
        model.fit(train[FEATURES], y, verbose=False)
        raw_cal = model.predict_proba(calibration[FEATURES])[:, 1]
        cal_y = calibration[f"y_{key}"].astype(int)
        calibrator = None
        if cal_y.nunique() == 2:
            calibrator = LogisticRegression(C=1.0, max_iter=500).fit(logit(raw_cal), cal_y)
        raw_test = model.predict_proba(test[FEATURES])[:, 1]
        p = calibrator.predict_proba(logit(raw_test))[:, 1] if calibrator is not None else raw_test
        pred = (p >= 0.5).astype(int)
        test_positives = int(yt.sum())
        baseline_brier = float(yt.mean() * (1.0 - yt.mean()))
        tn, fp, fn, tp = confusion_matrix(yt, pred, labels=[0,1]).ravel()
        metrics = {
            "train_positives": int(y.sum()), "test_positives": test_positives, "test_negatives": int((yt == 0).sum()),
            "positive_rate": float(yt.mean()), "baseline_brier": baseline_brier,
            "predicted_positive_rate": float(pred.mean()),
            "precision_at_0_5": float(precision_score(yt, pred, zero_division=0)),
            "recall_at_0_5": float(recall_score(yt, pred, zero_division=0)),
            "false_positive_rate_at_0_5": float(fp / max(1, fp + tn)),
            "accuracy_at_0_5": float(accuracy_score(yt, pred)),
            "average_precision": float(average_precision_score(yt, p)),
            "brier_score": float(brier_score_loss(yt, p)),
            "calibrated_with_platt": calibrator is not None,
            "roc_auc": float(roc_auc_score(yt, p)) if yt.nunique() == 2 else None,
        }
        incumbent_dir = Path(args.incumbent_dir) if args.incumbent_dir else None
        if incumbent_dir and (incumbent_dir / f"xgb_{key}.joblib").is_file():
            incumbent_bundle = joblib.load(incumbent_dir / f"xgb_{key}.joblib")
            incumbent_model = incumbent_bundle.get("model") if isinstance(incumbent_bundle, dict) else incumbent_bundle
            incumbent_calibrator = incumbent_bundle.get("calibrator") if isinstance(incumbent_bundle, dict) else None
            incumbent_raw = incumbent_model.predict_proba(test[FEATURES])[:, 1]
            incumbent_p = incumbent_calibrator.predict_proba(logit(incumbent_raw))[:, 1] if incumbent_calibrator is not None else incumbent_raw
            incumbent_pred = (incumbent_p >= 0.5).astype(int)
            itn, ifp, ifn, itp = confusion_matrix(yt, incumbent_pred, labels=[0,1]).ravel()
            incumbent_metrics = {
                "average_precision": float(average_precision_score(yt, incumbent_p)),
                "brier_score": float(brier_score_loss(yt, incumbent_p)),
                "false_positive_rate_at_0_5": float(ifp / max(1, ifp + itn)),
                "precision_at_0_5": float(precision_score(yt, incumbent_pred, zero_division=0)),
                "recall_at_0_5": float(recall_score(yt, incumbent_pred, zero_division=0)),
                "accuracy_at_0_5": float(accuracy_score(yt, incumbent_pred)),
            }
            metrics["incumbent"] = incumbent_metrics
            metrics["challenger_beats_incumbent"] = (
                metrics["average_precision"] > incumbent_metrics["average_precision"]
                and metrics["brier_score"] < incumbent_metrics["brier_score"]
                and metrics["false_positive_rate_at_0_5"] <= incumbent_metrics["false_positive_rate_at_0_5"]
            )
        else:
            metrics["incumbent"] = None
            metrics["challenger_beats_incumbent"] = (
                metrics["average_precision"] > float(yt.mean())
                and metrics["brier_score"] < baseline_brier
            )
        if test_positives < args.min_test_positives:
            metrics["validation_warning"] = f"Only {test_positives} positive test examples; need {args.min_test_positives}."
        report["targets"][key] = metrics
        joblib.dump({"model": model, "calibrator": calibrator}, out / f"xgb_{key}.joblib")

    report["version"] = "scannerbot-v2"
    (out / "metadata.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("NOTE: Labels use Stage 1-eligible baselines and observed future price highs; they are not executable-return proof.")


if __name__ == "__main__":
    main()
