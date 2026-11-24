# ScannerBot 2.0 — Solana launch scanner and adaptive research system

ScannerBot 2.0 is a **read-only** Solana memecoin discovery, market-analysis, security-screening and machine-learning research bot. It sends Telegram alerts. It does not contain wallet keys, execute swaps, buy tokens or manage positions.

## What happens when the bot starts

The scanner automatically selects one of two modes based on the model files actually loaded:

### PRE-MODEL / WATCHLIST

This is the default until a complete, validated model release is available.

- Shrine and PumpPortal discover new Solana tokens/pools.
- DEX Screener enriches discovered mints with pair, price, liquidity, volume and transaction data.
- Market features and the existing 0–100 heuristic score continue running.
- Eligible watchlist alerts are sent using the agreed WATCHLIST format.
- RugCheck is checked before an alert. An unavailable/unknown report or rejected result blocks the qualified alert when RugCheck is enabled.
- No ML probability is displayed or implied.

### MODEL-ACTIVE

The bot switches automatically only when all three model files (2x, 5x and 10x) load from the same release **and** that release's metadata contains `validation_passed: true`.

- Discovery, snapshots, market analysis and Telegram alerts continue.
- The same alert layout is retained, with 2x/5x/10x probabilities and model version appended.
- RugCheck remains active.
- The market score is not a probability. Model probabilities are experimental estimates, not guaranteed returns.

The two modes are not separate scanners. The model-ready check changes only whether ML predictions are added and used as an additional alert gate.

## Telegram alert structure

Alerts include token symbol and name, market score, price, liquidity, 1h/24h volume, 1h buys/sells and buy share, 1h price change, market cap, pool age, positive signals, cautions, RugCheck status, DEX, mint address, and separate DexScreener and DexPaprika pool links.

PRE-MODEL alerts are labelled `PRE-MODEL / WATCHLIST`. MODEL-ACTIVE alerts add the three target probabilities and model release version. Very small prices are formatted as decimals rather than scientific notation.

## Discovery and market data

- **Shrine:** new-token Socket.IO feed.
- **PumpPortal:** new-token and migration WebSocket subscriptions. No trade subscription is used.
- **DEX Screener:** primary batched token-pair enrichment, up to 30 mints per request.
- **DexPaprika:** rate-limited secondary Solana pool lookup for a bounded number of DEX Screener misses. Set `DEXPAPRIKA_API_KEY` to a free key for 100K monthly credits and 30 requests/minute; keyless lookup remains available but is throttled to one request every five minutes to stay within its smaller 10K monthly allowance.
- The discovery feeds are complementary and do not guarantee detection of every token or pool.
- The project is Solana-first because the real-time launch adapters in this version are Solana-specific.

## Adaptive training and retained learning

Training is not performed on only the latest batch. Each attempt:

1. Reads raw market snapshots one token at a time, rather than loading the entire SQLite snapshot table into Pandas.
2. Labels matured observations using the configured forward horizon and liquidity rule.
3. Stores compact labelled examples in `training_examples`, keyed by the source snapshot ID.
4. Retains a bounded replay archive (`MAX_TRAINING_EXAMPLES`, default 12,000) so future releases can learn from a mixture of earlier and newer market conditions.
5. Trains three separate XGBoost classifiers (2x, 5x and 10x) from that archive.
6. Evaluates on chronological, token-disjoint train/calibration/test cohorts.
7. Promotes the candidate only if all configured validation checks pass.
8. Loads the complete release and switches `models/CURRENT` atomically. The previous release is retained for rollback.

New labelled examples trigger retraining after `AUTO_TRAIN_MIN_NEW_SAMPLES` have accumulated since the last successful promotion. The scheduler checks again after a failed/insufficient attempt using `AUTO_TRAIN_RETRY_HOURS`. After a successful promotion it waits `AUTO_TRAIN_INTERVAL_HOURS` before the next training attempt.

Raw snapshots are retained long enough for the forward outcome horizon to mature (`TRACKING_RETENTION_HOURS`, default 30h). After the trainer has created a durable replay archive, old raw snapshots and token telemetry can be pruned. The labelled replay archive is separate and is not removed by raw snapshot cleanup. Raw market snapshots are pruned only after a complete successful archive sweep records a watermark covering the relevant retention cutoff and the configured label horizon still matches. A partial/failed sweep cannot authorize pruning. The archive is capped to control RAM and storage use; it is not a complete copy of every market observation.

**Important limitation:** the target labels are research proxies based on the maximum observed DEX Screener price during the configured future horizon while reported liquidity meets the minimum. This is not an executable-return simulation and does not fully model price impact, fees, MEV, failed sells or real route depth. Use walk-forward paper-trading evaluation before relying on probabilities.

## Model validation and release safety

A release must include all three model bundles and metadata with `validation_passed: true`. The automatic trainer sets this flag only after checking:
- minimum matured row count;
- minimum distinct token count;
- minimum positive examples for each target in the unseen test cohort;
- average precision above the test prevalence baseline and Brier score below the baseline, when baseline beating is enabled.

If training, validation or loading fails, the currently active model remains selected. A failed candidate is not used for live predictions. Model probabilities are not shown in PRE-MODEL mode.

The trainer uses Platt scaling on the logit of the raw XGBoost probability. Live prediction applies the same transformation used during calibration.

## Railway and low-resource deployment

The defaults are deliberately bounded:
- XGBoost uses one CPU worker and a capped replay archive.
- Training reads raw snapshots per token rather than loading the complete table.
- DEX batch concurrency is limited.
- Old raw telemetry is pruned after the labelled archive exists.
- Only the current and one previous model release are retained.

Railway storage is persistent only when a persistent volume is mounted at the path containing `scannerbot.db` and `models/`. Confirm the volume mount path and available disk before deploying. SQLite `DELETE` frees pages for reuse but does not necessarily shrink the database file on disk; a large existing database may need a separately planned backup-and-compaction/migration operation. Do not run `VACUUM` against a nearly full volume without a verified backup and sufficient free space.

For a continuously running Railway service, deploy as a worker (not a short-lived request process). Keep the SQLite database and model directory on the mounted persistent volume. Do not commit `.env`, Telegram credentials, the database, or model files to GitHub.

## Installation (Windows PowerShell)

Python 3.11+ is recommended.

```powershell
cd path\to\ScannerBot
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in `.env`, then run:

```powershell
python main.py
```

Run tests:

```powershell
pytest -q
```

## Important environment variables

| Variable | Default | Purpose |
|---|---:|---|
| `ENABLE_SHRINE` | `true` | Enable Shrine launch discovery |
| `ENABLE_PUMPPORTAL` | `true` | Enable PumpPortal discovery |
| `DATABASE_PATH` | `scannerbot.db` | SQLite database path |
| `SCAN_INTERVAL_SECONDS` | `15` | Main scan loop interval |
| `DEXSCREENER_DISCOVERY_INTERVAL_SECONDS` | `15` | DexScreener profile/boost discovery polling interval |\n| `PAIR_LOOKUP_INTERVAL_SECONDS` | `60` | Minimum interval between pair lookups per token |\n| `PAIR_LOOKUP_BATCH_SIZE` | `30` | Tokens per DexScreener batch (maximum 30) |\n| `PAIR_LOOKUP_CONCURRENCY` | `3` | Concurrent token lookup batches |
| `MAX_ACTIVE_TOKENS` | `6000` | Maximum active candidate rows queried |
| `TRACKING_RETENTION_HOURS` | `30` | Raw snapshot retention; must cover horizon + max label age + 2h |
| `MIN_PREDICTION_AGE_MINUTES` | `0` | Earliest alert pool age; tokens can qualify immediately after passing all other checks |
| `MAX_PAIR_AGE_MINUTES` | `180` | Latest alert pool age |
| `MIN_LIQUIDITY_USD` | `8000` | Initial Stage 1 eligibility filter; not reapplied to future outcome observations |
| `MIN_VOLUME_5M_USD` | `500` | Market activity filter |
| `MIN_TRANSACTIONS_5M` | `8` | Market activity filter |
| `MIN_PREMODEL_SCORE` | `60` | Minimum heuristic score for PRE-MODEL watchlist alerts |
| `RUGCHECK_ENABLED` | `true` | Apply RugCheck in both modes |
| `RUGCHECK_MAX_SCORE` | `50` | Maximum normalized RugCheck risk score |
| `MODEL_DIR` | `/data/models` | Persistent model release directory |
| `MIN_PROBABILITY_2X` | `0.50` | ML 2x probability gate |
| `MIN_PROBABILITY_5X` | `0.15` | ML 5x probability gate |
| `MIN_PROBABILITY_10X` | `0.03` | ML 10x probability gate |
| `PREDICTION_HORIZON_HOURS` | `2` | Forward outcome maturity window |
| `MIN_TRAINING_ROWS` | `100` | Minimum matured eligible baselines |
| `AUTO_TRAIN_ENABLED` | `true` | Enable automatic training |
| `AUTO_TRAIN_INITIAL_DELAY_HOURS` | `4` | First attempt after deployment startup |
| `AUTO_TRAIN_INTERVAL_HOURS` | `4` | Regular interval after a completed training run |
| `AUTO_TRAIN_RETRY_HOURS` | `2` | Retry after insufficient matured data or validation class coverage |
| `AUTO_TRAIN_MIN_TOKENS` | `100` | Minimum distinct tokens in archive |
| `AUTO_TRAIN_MIN_NEW_SAMPLES` | `100` | Newly matured eligible baselines required per subsequent cycle |
| `AUTO_TRAIN_MIN_TEST_POSITIVES` | `2` | Minimum positives per target in test cohort |
| `AUTO_TRAIN_MIN_TEST_NEGATIVES` | `2` | Minimum negatives per target in test cohort |

| `ALERT_COOLDOWN_MINUTES` | `60` | Per-token Telegram alert cooldown |

## Discovery dashboard and metrics

The read-only dashboard exposes only two sections: unique token discovery by Shrine, PumpPortal and DexScreener, and unique Stage 1 evaluated/passed/failed counts. It listens on the Railway `PORT` (default 8080). The same five-minute summary is emitted to Railway application logs as `[SCANNER METRICS]`.

## Storage layout

`scannerbot.db` stays on Railway ephemeral application storage. The accumulating eligible training archive is stored separately at `/data/models/training_archive.sqlite` on the existing model volume, so it survives redeploys without moving `scannerbot.db` to persistent storage. Promoted model releases and discovery checkpoints also live under `/data/models`.

## Training data and promotion

Only tokens that pass all existing Stage 1 filters receive a training baseline. One eligible token contributes one training example. The baseline feature vector is fixed at eligibility time; valid future prices are observed over the full two-hour window without reapplying the initial liquidity floor. A window is labelled only when observations cover the start and end with no gap greater than five minutes; incomplete windows remain unknown rather than becoming negatives.

Each target (2x, 5x, 10x) is trained and evaluated independently. The newest matured batch is held out from challenger fitting and compared with the incumbent on the same unseen cases. A challenger is promoted only if it beats the incumbent for every target and each test cohort has enough positive and negative examples. Historical eligible examples accumulate; only promoted releases are retained under `MODEL_DIR`.

The Shrine hourly event archive backfill is active without a key and recovers supported launch/migration events after downtime. Its checkpoints are stored under `/data/models` and survive deployments. The broader live Advanced Data Stream and immediate slot-based resume require a plan-enabled `SHRINE_API_KEY`; without it, the free feed and hourly archive backfill remain active. PumpPortal does not document a historical replay cursor, so its missed launch events are recovered where they overlap with Shrine archive coverage.

## Security and risk limitations

RugCheck is one screening source, not a complete audit. A PASS does not prove a token is safe. This project does not independently decode every token extension, fully cluster insiders/holders, verify all LP-lock semantics, or simulate a sell route. If RugCheck is enabled, UNKNOWN and REJECT statuses block qualified alerts. Set `RUGCHECK_ENABLED=false` only if you explicitly accept alerts without that check.

## Provider references

- Shrine new-token feed: https://account.shrine.trade/pump/data-api/subscribe-new-tokens/
- PumpPortal real-time feed: https://pumpportal.fun/data-api/real-time/
- DEX Screener API: https://docs.dexscreener.com/api/reference
- DexPaprika API and free key: https://docs.dexpaprika.com/ and https://console.dexpaprika.com/
- RugCheck: https://rugcheck.xyz/

## Version 2.0 scope

This is a signal-only research system. It does not promise a specific win rate, accuracy, number of calls, or return multiple. Market coverage depends on discovery-provider availability, DEX indexing, API limits, liquidity, security-report availability and the quality/quantity of matured labels.
