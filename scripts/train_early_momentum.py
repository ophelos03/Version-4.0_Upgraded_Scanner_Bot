from __future__ import annotations

"""Research-only 5m/15m/30m models. These never replace production 2h models."""
import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, precision_score, recall_score
from xgboost import XGBClassifier

from scannerbot.prediction.predictor import FEATURES


def load_samples(db_path: Path, horizon: int) -> pd.DataFrame:
    con = sqlite3.connect(f"file:{db_path.resolve().as_posix()}?mode=ro", uri=True)
    try:
        rows = con.execute(
            f"""SELECT chain_id,token_address,captured_at,features_json,baseline_price,
                       max_price_{horizon}m,min_price_{horizon}m,
                       baseline_liquidity,min_liquidity_{horizon}m
                FROM early_momentum_samples
                WHERE mature_{horizon}m=1 AND baseline_price>0
                  AND max_price_{horizon}m IS NOT NULL AND min_price_{horizon}m IS NOT NULL
                ORDER BY captured_at"""
        ).fetchall()
    finally:
        con.close()
    records = []
    for chain, token, captured, raw, baseline, maximum, minimum, baseline_liquidity, minimum_liquidity in rows:
        try:
            features = json.loads(raw)
        except (TypeError, ValueError):
            continue
        base = float(baseline)
        if base <= 0:
            continue
        record = {key: float(features.get(key, 0) or 0) for key in FEATURES}
        record.update({
            "chain_id": chain, "token_address": token, "captured_at": float(captured),
            "max_multiple": float(maximum) / base,
            "max_drawdown_pct": (float(minimum) / base - 1.0) * 100.0,
            "liquidity_drawdown_pct": (
                (float(minimum_liquidity) / float(baseline_liquidity) - 1.0) * 100.0
                if baseline_liquidity and minimum_liquidity is not None else 0.0
            ),
        })
        record["y_2x"] = int(record["max_multiple"] >= 2.0)
        record["y_5x"] = int(record["max_multiple"] >= 5.0)
        record["y_10x"] = int(record["max_multiple"] >= 10.0)
        records.append(record)
    return pd.DataFrame.from_records(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="/data/models/training_archive.sqlite")
    parser.add_argument("--output", default="/data/models/early_research")
    parser.add_argument("--min-rows", type=int, default=100)
    parser.add_argument("--min-tokens", type=int, default=30)
    args = parser.parse_args()
    db = Path(args.db)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report = {"research_only": True, "production_promotion": False, "horizons": {}}
    for horizon in (5, 15, 30):
        frame = load_samples(db, horizon)
        distinct = frame[["chain_id", "token_address"]].drop_duplicates().shape[0] if not frame.empty else 0
        result = {"rows": len(frame), "distinct_tokens": distinct, "targets": {}}
        if len(frame) < args.min_rows or distinct < args.min_tokens:
            result["status"] = "postponed_insufficient_data"
            report["horizons"][f"{horizon}m"] = result
            continue
        launch_order = frame.groupby(["chain_id", "token_address"])["captured_at"].min().sort_values()
        keys = list(launch_order.index)
        split = max(1, int(len(keys) * 0.8))
        train_keys, test_keys = set(keys[:split]), set(keys[split:])
        if not test_keys:
            result["status"] = "postponed_no_disjoint_test_cohort"
            report["horizons"][f"{horizon}m"] = result
            continue
        train_mask = [(c, t) in train_keys for c, t in zip(frame.chain_id, frame.token_address)]
        test_mask = [(c, t) in test_keys for c, t in zip(frame.chain_id, frame.token_address)]
        train, test = frame[train_mask], frame[test_mask]
        result.update({"train_rows": len(train), "test_rows": len(test), "status": "evaluated"})
        for target in ("2x", "5x", "10x"):
            y_train, y_test = train[f"y_{target}"].astype(int), test[f"y_{target}"].astype(int)
            if y_train.nunique() < 2 or y_test.nunique() < 2:
                result["targets"][target] = {"status": "postponed_single_class"}
                continue
            model = XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.04,
                subsample=0.85, colsample_bytree=0.85, min_child_weight=5,
                reg_lambda=5, objective="binary:logistic", eval_metric="logloss",
                tree_method="hist", n_jobs=1, random_state=42,
            )
            model.fit(train[FEATURES], y_train, verbose=False)
            probability = model.predict_proba(test[FEATURES])[:, 1]
            predicted = (probability >= 0.5).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, predicted, labels=[0, 1]).ravel()
            metrics = {
                "status": "evaluated", "test_positives": int(y_test.sum()),
                "precision_at_0_5": float(precision_score(y_test, predicted, zero_division=0)),
                "recall_at_0_5": float(recall_score(y_test, predicted, zero_division=0)),
                "false_positive_rate_at_0_5": float(fp / max(1, fp + tn)),
                "average_precision": float(average_precision_score(y_test, probability)),
                "brier_score": float(brier_score_loss(y_test, probability)),
                "mean_max_drawdown_pct": float(test["max_drawdown_pct"].mean()),
                "mean_liquidity_drawdown_pct": float(test["liquidity_drawdown_pct"].mean()),
            }
            result["targets"][target] = metrics
            joblib.dump(model, output / f"early_{horizon}m_{target}.joblib")
        report["horizons"][f"{horizon}m"] = result
    (output / "research_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
