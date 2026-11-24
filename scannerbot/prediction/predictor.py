from __future__ import annotations
import json, logging
from pathlib import Path
import numpy as np

log=logging.getLogger(__name__)
FEATURES=["pair_age_minutes","price_usd","liquidity_usd","volume_m5_usd","volume_h1_usd","volume_h24_usd",
"buys_m5","sells_m5","transactions_m5","buys_h1","sells_h1","transactions_h1","buy_share_m5","buy_share_h1",
"price_change_m5_pct","price_change_h1_pct","price_change_h6_pct","price_change_h24_pct","market_cap_usd","fdv_usd",
"liquidity_to_fdv","volume_to_liquidity_5m","volume_to_liquidity_1h","buy_sell_count_ratio_5m","buy_sell_count_ratio_1h",
"price_return_since_snapshot_pct","liquidity_change_since_snapshot_pct","volume_change_since_snapshot_pct",
"transaction_change_since_snapshot_pct","buy_share_change_since_snapshot","volume_acceleration_5m_vs_hour",
"transaction_acceleration_5m_vs_hour","buy_pressure_change_m5","snapshot_interval_seconds","price_return_1m_pct"]

class UpsidePredictor:
    def __init__(self, model_dir: str):
        self.model_dir=Path(model_dir); self.models={}; self.version="untrained"
        self.reload()

    def reload(self, model_dir: str | None = None) -> bool:
        """Load a complete model set first, then swap it in atomically in memory."""
        import joblib
        root=Path(model_dir) if model_dir else self.model_dir
        directory=root
        pointer=root/"CURRENT"
        if pointer.exists():
            release_name=pointer.read_text().strip()
            if release_name and Path(release_name).name == release_name:
                directory=root/"releases"/release_name
        loaded={}
        try:
            meta = directory / "metadata.json"
            if not meta.exists():
                log.info("Model metadata missing in %s; refusing unvalidated model set", directory)
                return False
            metadata = json.loads(meta.read_text())
            if metadata.get("validation_passed") is not True:
                log.info("Model set in %s has not passed promotion validation", directory)
                return False
            targets = metadata.get("targets") or {}
            if not all(key in targets for key in ("2x", "5x", "10x")):
                log.info("Model metadata is missing one or more targets in %s", directory)
                return False
            for key in ("2x","5x","10x"):
                path=directory/f"xgb_{key}.joblib"
                if not path.exists():
                    log.info("Model set incomplete in %s; keeping current model", directory)
                    return False
                loaded[key]=joblib.load(path)
            version=metadata.get("version","unknown")
        except Exception as exc:
            log.warning("Could not load complete XGBoost model set from %s: %s",directory,exc)
            return False
        self.models=loaded
        self.version=version
        self.model_dir=root
        log.info("Loaded model set %s",version)
        return True

    @property
    def ready(self): return len(self.models)==3

    def predict(self, features: dict) -> dict[str,float] | None:
        if not self.ready: return None
        import pandas as pd
        output={}
        for key,bundle in self.models.items():
            model=bundle.get("model") if isinstance(bundle,dict) else bundle
            try:
                booster = model.get_booster()
                expected = list(booster.feature_names or [])
                feature_count = int(booster.num_features())
            except Exception:
                expected, feature_count = [], len(FEATURES)
            columns = expected or FEATURES[:feature_count]
            row={k:float(features.get(k,0) or 0) for k in columns}
            X=pd.DataFrame([row],columns=columns)
            calibrator=bundle.get("calibrator") if isinstance(bundle,dict) else None
            raw=float(model.predict_proba(X)[0,1])
            if calibrator is not None:
                # Training fits Platt scaling on the logit of the raw XGBoost probability.
                eps = 1e-6
                clipped = float(np.clip(raw, eps, 1.0 - eps))
                logit = np.log(clipped / (1.0 - clipped))
                raw = float(calibrator.predict_proba(np.array([[logit]], dtype=float))[0, 1])
            output[key]=max(0.0,min(1.0,raw))
        return output
