from __future__ import annotations
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

def _int(name: str, default: int) -> int:
    try: return int(os.getenv(name, str(default)))
    except (TypeError, ValueError): return default

def _float(name: str, default: float) -> float:
    try: return float(os.getenv(name, str(default)))
    except (TypeError, ValueError): return default

def _bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}

@dataclass(frozen=True)
class Settings:
    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    database_path: str = os.getenv("DATABASE_PATH", "scannerbot.db")
    log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()
    # Solana launch discovery
    enable_shrine: bool = _bool("ENABLE_SHRINE", True)
    enable_pumpportal: bool = _bool("ENABLE_PUMPPORTAL", True)
    pumpportal_api_key: str = os.getenv("PUMPPORTAL_API_KEY", "").strip()
    shrine_api_key: str = os.getenv("SHRINE_API_KEY", "").strip()
    discovery_protocols: str = os.getenv("DISCOVERY_PROTOCOLS", "PUMPFUN,PUMPFUN_MAYHEM,PUMPSWAP,METEORA_DBC,METEORA,METEORA_DAMM_V1,METEORA_DLMM,RAYDIUM,RAYDIUM_CLMM,ORCA,BONK,STONKFUN")
    # DEX Screener enrichment
    dexscreener_discovery_enabled: bool = _bool("DEXSCREENER_DISCOVERY_ENABLED", True)
    dexscreener_discovery_interval_seconds: int = _int("DEXSCREENER_DISCOVERY_INTERVAL_SECONDS", 15)
    scan_interval_seconds: int = _int("SCAN_INTERVAL_SECONDS", 15)
    pair_lookup_interval_seconds: int = _int("PAIR_LOOKUP_INTERVAL_SECONDS", 60)
    pair_lookup_batch_size: int = _int("PAIR_LOOKUP_BATCH_SIZE", 30)
    pair_lookup_concurrency: int = _int("PAIR_LOOKUP_CONCURRENCY", 3)
    # DexPaprika is a rate-limited secondary resolver; the API key is optional.
    dexpaprika_enabled: bool = _bool("DEXPAPRIKA_ENABLED", True)
    dexpaprika_api_key: str = os.getenv("DEXPAPRIKA_API_KEY", "").strip()
    dexpaprika_fallback_max_per_cycle: int = _int("DEXPAPRIKA_FALLBACK_MAX_PER_CYCLE", 1)
    max_active_tokens: int = _int("MAX_ACTIVE_TOKENS", 6000)
    max_pair_age_minutes: int = _int("MAX_PAIR_AGE_MINUTES", 180)
    tracking_retention_hours: int = _int("TRACKING_RETENTION_HOURS", 30)
    min_prediction_age_minutes: int = _int("MIN_PREDICTION_AGE_MINUTES", 0)
    snapshot_interval_seconds: int = _int("SNAPSHOT_INTERVAL_SECONDS", 60)
    min_liquidity_usd: float = _float("MIN_LIQUIDITY_USD", 8000)
    min_volume_5m_usd: float = _float("MIN_VOLUME_5M_USD", 500)
    min_transactions_5m: int = _int("MIN_TRANSACTIONS_5M", 8)
    alert_cooldown_minutes: int = _int("ALERT_COOLDOWN_MINUTES", 60)
    # Security and prediction gates
    rugcheck_enabled: bool = _bool("RUGCHECK_ENABLED", True)
    rugcheck_max_score: float = _float("RUGCHECK_MAX_SCORE", 50)
    min_probability_2x: float = _float("MIN_PROBABILITY_2X", 0.50)
    min_probability_5x: float = _float("MIN_PROBABILITY_5X", 0.15)
    min_probability_10x: float = _float("MIN_PROBABILITY_10X", 0.03)
    model_dir: str = os.getenv("MODEL_DIR", "/data/models")
    # Optional on-chain creator evidence. Public Solana RPC remains the default.
    helius_api_key: str = os.getenv("HELIUS_API_KEY", "").strip()
    solana_rpc_url: str = os.getenv("SOLANA_RPC_URL", "").strip()
    verified_wallet_storage_max_mb: int = _int("VERIFIED_WALLET_STORAGE_MAX_MB", 10)
    prediction_horizon_hours: int = _int("PREDICTION_HORIZON_HOURS", 2)
    # Historical label and evaluation settings
    label_min_age_minutes: int = _int("LABEL_MIN_AGE_MINUTES", 30)
    label_max_age_minutes: int = _int("LABEL_MAX_AGE_MINUTES", 180)
    min_training_rows: int = _int("MIN_TRAINING_ROWS", 100)
    # Automatic training retries until enough new matured labelled samples exist.
    auto_train_enabled: bool = _bool("AUTO_TRAIN_ENABLED", True)
    auto_train_initial_delay_hours: int = _int("AUTO_TRAIN_INITIAL_DELAY_HOURS", 4)
    auto_train_interval_hours: int = _int("AUTO_TRAIN_INTERVAL_HOURS", 4)
    auto_train_retry_hours: int = _int("AUTO_TRAIN_RETRY_HOURS", 2)
    auto_train_min_tokens: int = _int("AUTO_TRAIN_MIN_TOKENS", 100)
    auto_train_min_train_positives: int = _int("AUTO_TRAIN_MIN_TRAIN_POSITIVES", 2)
    auto_train_min_train_negatives: int = _int("AUTO_TRAIN_MIN_TRAIN_NEGATIVES", 2)
    auto_train_min_test_positives: int = _int("AUTO_TRAIN_MIN_TEST_POSITIVES", 2)
    auto_train_min_test_negatives: int = _int("AUTO_TRAIN_MIN_TEST_NEGATIVES", 2)
    auto_train_min_new_samples: int = _int("AUTO_TRAIN_MIN_NEW_SAMPLES", 100)
    auto_train_require_baseline_beat: bool = _bool("AUTO_TRAIN_REQUIRE_BASELINE_BEAT", True)
    min_premodel_score: float = _float("MIN_PREMODEL_SCORE", 60)
    max_market_cap_usd: float = _float("MAX_MARKET_CAP_USD", 80000)
    wash_volume_per_tx_usd: float = _float("WASH_VOLUME_PER_TX_USD", 2500)
    wash_volume_to_liquidity: float = _float("WASH_VOLUME_TO_LIQUIDITY", 3.0)
    wash_one_sided_buy_share: float = _float("WASH_ONE_SIDED_BUY_SHARE", 0.92)

    def validate(self) -> None:
        if not self.telegram_bot_token or not self.telegram_chat_id:
            raise RuntimeError("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env before running.")
        if self.scan_interval_seconds < 5: raise ValueError("SCAN_INTERVAL_SECONDS must be >= 5.")
        if not 1 <= self.pair_lookup_batch_size <= 30: raise ValueError("PAIR_LOOKUP_BATCH_SIZE must be between 1 and 30.")
        if self.pair_lookup_concurrency < 1: raise ValueError("PAIR_LOOKUP_CONCURRENCY must be >= 1.")
        if self.dexpaprika_fallback_max_per_cycle < 0: raise ValueError("DEXPAPRIKA_FALLBACK_MAX_PER_CYCLE must be >= 0.")
        if min(self.auto_train_min_train_positives, self.auto_train_min_train_negatives, self.auto_train_min_test_positives, self.auto_train_min_test_negatives) < 1: raise ValueError("Automatic training class minimums must be >= 1.")
        if self.auto_train_min_new_samples < 1: raise ValueError("AUTO_TRAIN_MIN_NEW_SAMPLES must be >= 1.")
        if self.verified_wallet_storage_max_mb < 1: raise ValueError("VERIFIED_WALLET_STORAGE_MAX_MB must be >= 1.")
        if self.max_pair_age_minutes <= self.min_prediction_age_minutes: raise ValueError("Maximum pair age must exceed minimum prediction age.")
        if not (0 <= self.rugcheck_max_score <= 100): raise ValueError("RUGCHECK_MAX_SCORE must be 0..100.")
        minimum_retention = self.prediction_horizon_hours + self.label_max_age_minutes / 60 + 2
        if self.tracking_retention_hours < minimum_retention:
            raise ValueError(f"TRACKING_RETENTION_HOURS must be at least {minimum_retention:.1f}h so training outcomes can mature.")

settings = Settings()
