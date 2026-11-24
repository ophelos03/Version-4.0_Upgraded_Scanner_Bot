# Upgrade notes — ScannerBot 2.0

## Important: preserve your current Railway data

This ZIP contains application source code. It does **not** contain your live SQLite database, `.env`, Telegram credentials or trained model files.

Before replacing the deployed source:

1. Back up `scannerbot.db` from the persistent Railway volume.
2. Back up the complete `models/` directory if it contains releases.
3. Confirm the exact persistent volume mount path in Railway.
4. Keep `DATABASE_PATH` and `MODEL_DIR` on that persistent volume.
5. Deploy the new source without deleting or recreating the database.

Do not run `rm scannerbot.db`, do not reset the volume, and do not replace your existing database with a blank database. Version 2.0 creates its `training_examples` table and adds the `archived_at` column if it is missing; it does not require a database reset.

## New environment variables

Add these in Railway Variables:

```dotenv
AUTO_TRAIN_MIN_NEW_SAMPLES=250
MAX_TRAINING_EXAMPLES=12000
MIN_PREMODEL_SCORE=60
```

Keep the existing Telegram, database, discovery, liquidity, prediction, RugCheck and training variables. Ensure `TRACKING_RETENTION_HOURS` is at least the prediction horizon plus maximum label age plus two hours. The defaults (24h horizon, 180m maximum label age, 30h retention) satisfy this.

## What to expect after deployment

- If there is no complete validated 2x/5x/10x release, the bot starts in `PRE-MODEL / WATCHLIST`.
- Market analysis and watchlist alerts can run without trained models.
- RugCheck remains enabled by default. UNKNOWN or rejected security results block qualified alerts.
- The first training attempt is scheduled after the configured initial delay. It may retry hourly while new labelled data accumulates or validation gates are not met.
- A model set becomes active only after all three files load and metadata says `validation_passed: true`.
- Labelled replay examples persist in `training_examples`. Raw market snapshots are pruned only after a complete successful archive sweep records a watermark for the configured prediction horizon; a non-empty archive alone is not sufficient.

## Existing database size warning

SQLite deletes generally make pages reusable but do not automatically reduce the physical database file size. If the existing Railway database is already larger than the mounted volume quota, this code change alone will not shrink it. Do not attempt `VACUUM` on a nearly full volume. First make a verified backup and plan a separate compaction or migration with enough temporary storage.
