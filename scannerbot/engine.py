from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from pathlib import Path

from .config import Settings
from .auto_training import AutoTrainingManager
from .database import Database
from .dexscreener import DexScreenerClient
from .dexpaprika import DexPaprikaClient
from .discovery import DiscoveryCheckpointStore, ShrineDiscovery, ShrineAdvancedDiscovery, ShrineReplayBackfill, PumpPortalDiscovery
from .features import build_features, wash_trading_flags
from .early_momentum import (EarlyMomentumArchive, SolanaEvidenceClient, classify_early_momentum,
                             evaluate_early_entry,
                             early_lookup_priority_signal_count)
from .models import Candidate, canonical_token_key, is_valid_solana_mint, solana_mint_validation_reason
from .prediction.predictor import UpsidePredictor
from .scoring import analyse_pair
from .security.rugcheck import RugCheckClient
from .telegram import Telegram
from .dashboard import DiscoveryDashboard

log = logging.getLogger(__name__)


class ScannerEngine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.dex = DexScreenerClient()
        self.dexpaprika = DexPaprikaClient(settings.dexpaprika_api_key) if settings.dexpaprika_enabled else None
        archive_path = str(Path(settings.model_dir) / "training_archive.sqlite")
        self.db = Database(settings.database_path, settings.prediction_horizon_hours, archive_path)
        self.early_archive = EarlyMomentumArchive(archive_path, settings.verified_wallet_storage_max_mb)
        self.solana_evidence = SolanaEvidenceClient(settings.helius_api_key, settings.solana_rpc_url)
        self._creator_checked: dict[str, float] = {}
        self._onchain_evidence_cache: dict[str, dict] = {}
        self.telegram = Telegram(settings.telegram_bot_token, settings.telegram_chat_id)
        self.rugcheck = RugCheckClient(settings.rugcheck_enabled)
        self.predictor = UpsidePredictor(settings.model_dir)
        self.auto_trainer = AutoTrainingManager(settings, self.predictor, self.telegram)
        protocols = [x.strip().upper() for x in settings.discovery_protocols.split(",") if x.strip()]
        self.discovery = []
        self.discovery_checkpoints = DiscoveryCheckpointStore(str(Path(settings.model_dir) / "discovery_checkpoints.json"))
        if settings.enable_shrine:
            self.discovery.append(ShrineDiscovery(self.on_candidate, protocols))
            self.discovery.append(ShrineReplayBackfill(self.on_candidate, self.discovery_checkpoints))
            if not settings.shrine_api_key:
                log.warning("SHRINE_API_KEY is not configured: the advanced live stream and slot resume are disabled; the free feed and hourly archive backfill remain active.")
            if settings.shrine_api_key:
                self.discovery.append(ShrineAdvancedDiscovery(
                    self.on_candidate, self.discovery_checkpoints, settings.shrine_api_key, protocols,
                ))
        if settings.enable_pumpportal:
            self.discovery.append(PumpPortalDiscovery(self.on_candidate, settings.pumpportal_api_key))
        self._last_lookup: dict[tuple[str, str], float] = {}
        self._early_lookup_priority: dict[tuple[str, str], float] = {}
        self._last_dex_discovery = 0.0
        self._security_cache: dict[str, tuple[float, bool, str, float | None, list[str]]] = {}
        self._last_prune = 0.0
        self._gate_counts: dict[str, int] = {}
        self._invalid_mint_counts: dict[str, int] = {}
        self._last_gate_summary = time.time()
        self._discovery_period_unique: set[tuple[str,str]] = set()
        self._discovery_period_sources: dict[str,set[tuple[str,str]]] = {}
        self._stage1_period_evaluated: set[tuple[str,str]] = set()
        self._stage1_period_passed: set[tuple[str,str]] = set()
        self._dashboard_metrics = {"period_seconds": 0, "updated_at": time.time(), "discovery": {}, "unique_discovered": 0, "stage1": {"evaluated": 0, "passed": 0, "failed": 0}}
        self.dashboard = DiscoveryDashboard(lambda: dict(self._dashboard_metrics))

    def _record_dev_call_after_alert(self, wallet: str | None, chain: str, token: str,
                                     price: float, captured_at: float,
                                     evidence: dict | None = None) -> None:
        if wallet:
            self.early_archive.record_dev_call(
                wallet, chain, token, price,
                evidence or {"source": "solana-rpc", "confidence": "candidate_only"},
                captured_at=captured_at,
            )

    def _mark_early_lookup_priority(self, key: tuple[str, str], now: float) -> None:
        """Refresh promising low-cap candidates twice as often, with a bounded queue."""
        self._early_lookup_priority[key] = now + 600
        while len(self._early_lookup_priority) > 600:
            oldest = min(self._early_lookup_priority, key=self._early_lookup_priority.get)
            self._early_lookup_priority.pop(oldest, None)

    def _reject(self, reason: str, token: str, detail: str = "") -> None:
        self._gate_counts[reason] = self._gate_counts.get(reason, 0) + 1
        suffix = f" ({detail})" if detail else ""
        log.debug("Candidate rejected at %s gate: %s%s", reason, token, suffix)

    def _maybe_log_gate_summary(self) -> None:
        now = time.time()
        if now - self._last_gate_summary < 300:
            return
        counts = dict(sorted(self._gate_counts.items()))
        invalid = dict(sorted(self._invalid_mint_counts.items()))
        period = now - self._last_gate_summary
        sources = {name: len(tokens) for name,tokens in sorted(self._discovery_period_sources.items())}
        evaluated = len(self._stage1_period_evaluated)
        passed = len(self._stage1_period_passed)
        self._dashboard_metrics = {
            "period_seconds": round(period),
            "updated_at": now,
            "discovery": sources,
            "unique_discovered": len(self._discovery_period_unique),
            "stage1": {"evaluated": evaluated, "passed": passed, "failed": max(0,evaluated-passed)},
        }
        log.info(
            "[SCANNER METRICS] period_seconds=%.0f discovery_unique_by_source=%s "
            "discovery_unique_total=%d stage1_evaluated_unique=%d stage1_passed_unique=%d stage1_failed_unique=%d "
            "gate_counts=%s invalid_mints=%s",
            period, sources, len(self._discovery_period_unique), evaluated, passed,
            max(0,evaluated-passed), counts or {}, invalid or {},
        )
        self._gate_counts.clear()
        self._invalid_mint_counts.clear()
        self._discovery_period_unique.clear()
        self._discovery_period_sources.clear()
        self._stage1_period_evaluated.clear()
        self._stage1_period_passed.clear()
        self._last_gate_summary = now

    @property
    def mode(self) -> str:
        # The predictor only reports ready when the complete 2x/5x/10x set is loaded.
        return "MODEL-ACTIVE" if self.predictor.ready else "PRE-MODEL"

    async def on_candidate(self, candidate: Candidate):
        invalid_reason = solana_mint_validation_reason(candidate.token_address)
        if invalid_reason:
            field = str(candidate.metadata.get("_scanner_mint_field") or "unknown_field")
            metric = f"{candidate.source}:{field}:{invalid_reason}"
            self._invalid_mint_counts[metric] = self._invalid_mint_counts.get(metric, 0) + 1
            self._reject("invalid_solana_mint", "<redacted>", f"source={candidate.source} field={field} reason={invalid_reason}")
            return
        key = (candidate.chain_id.lower(), canonical_token_key(candidate.chain_id, candidate.token_address))
        is_new_token, is_new_source = self.db.register_candidate(candidate)
        if is_new_token:
            self._discovery_period_unique.add(key)
        source_name = ("DexScreener" if candidate.source in {"profile", "boost", "profile+boost"} else ("Shrine" if candidate.source.startswith("shrine") else "PumpPortal" if candidate.source == "pumpportal" else candidate.source))
        if is_new_source:
            self._discovery_period_sources.setdefault(source_name, set()).add(key)
        log.debug("Candidate %s observed via %s new_token=%s new_source=%s", candidate.token_address, source_name, is_new_token, is_new_source)

    async def close(self):
        for discovery in self.discovery:
            try:
                await discovery.close()
            except Exception:
                pass
        await self.dex.close()
        if self.dexpaprika:
            await self.dexpaprika.close()
        await self.rugcheck.close()
        await self.telegram.close()
        await self.dashboard.close()
        await self.solana_evidence.close()
        self.early_archive.close()
        self.db.close()

    async def _security(self, mint: str):
        if not is_valid_solana_mint(mint):
            return False, "UNKNOWN", None, ["Invalid Solana mint address; RugCheck request skipped"]
        if not self.settings.rugcheck_enabled:
            return True, "DISABLED", None, ["RugCheck disabled by configuration"]
        now = time.time()
        cached = self._security_cache.get(mint)
        if cached and now - cached[0] < 900:
            return cached[1:]
        report = await self.rugcheck.report(mint)
        passed, status, score, reasons = self.rugcheck.assess(report, self.settings.rugcheck_max_score)
        self._security_cache[mint] = (now, passed, status, score, reasons)
        self.db.save_security("solana", mint, "rugcheck", score, status, {"reasons": reasons})
        return passed, status, score, reasons

    async def _send_alert_once(
        self, analysis, features: dict, status: str, risk_score,
        risk_reasons: list[str], probabilities=None, model_version=None,
    ) -> bool:
        # These are last-moment hard gates on the exact market data used in the alert.
        market_cap = analysis.market_cap
        if market_cap is None:
            self._reject("market_cap_unavailable_at_alert", analysis.token_address)
            return False
        if market_cap > self.settings.max_market_cap_usd:
            self._reject("market_cap_above_alert_maximum", analysis.token_address,
                         f"${market_cap:.0f} > ${self.settings.max_market_cap_usd:.0f}")
            return False
        wash_flags = wash_trading_flags(
            features, self.settings.wash_volume_per_tx_usd,
            self.settings.wash_volume_to_liquidity,
            self.settings.wash_one_sided_buy_share,
        )
        if wash_flags:
            self._reject("wash_trading_suspected", analysis.token_address, "; ".join(wash_flags))
            return False
        alert_timestamp = time.time()
        result = await self.telegram.signal_alert(
            analysis, features, status, risk_score, risk_reasons,
            probabilities=probabilities, model_version=model_version,
            alert_timestamp=alert_timestamp, momentum=features,
        )
        message_id = result.get("message_id") if isinstance(result, dict) else None
        self.db.record_alert(
            analysis.chain_id, analysis.token_address, alert_timestamp,
            analysis.price_usd, analysis.market_cap, analysis.liquidity_usd,
            features, message_id,
        )
        return True

    async def _process_pair(self, row: dict, pair: dict):
        chain, token = row["chain_id"], row["token_address"]
        if not is_valid_solana_mint(token):
            self._reject("invalid_solana_mint", token)
            return
        created = pair.get("pairCreatedAt")
        if not created:
            self._reject("missing_pair_created_at", token)
            return
        pair_created = float(created) / 1000 if float(created) > 10_000_000_000 else float(created)
        age = max(0, (time.time() - pair_created) / 60)
        if age > self.settings.tracking_retention_hours * 60:
            self._reject("tracking_retention_expired", token, f"age={age:.1f}m")
            return

        previous_rows = self.db.recent_snapshots(chain, token, 1)
        previous = json.loads(previous_rows[0]["features_json"]) if previous_rows else None
        if previous is not None:
            previous["_interval_seconds"] = max(0.0, time.time() - float(previous_rows[0]["captured_at"]))
        features = build_features(pair, age, previous)
        raw_price = pair.get("priceUsd")
        if raw_price is None or raw_price == "":
            self._reject("invalid_price", token)
            return
        price = float(raw_price)
        liquidity_raw = (pair.get("liquidity") or {}).get("usd")
        liquidity_observed = liquidity_raw is not None
        liquidity = float(liquidity_raw or 0)
        has_baseline = self.db.has_training_baseline(chain, token)
        if not math.isfinite(price) or price < 0 or (price == 0 and not has_baseline):
            self._reject("invalid_price", token)
            return

        now = time.time()
        # Zero is a valid observed crash price for an already eligible token;
        # missing/non-finite prices remain unknown and are not fabricated.
        self.early_archive.observe_dev_call(chain, token, price)
        if has_baseline:
            self.db.record_training_observation(chain, token, now, price)
            if price == 0:
                self.early_archive.record_zero_price_observation(
                    chain, token, now, liquidity if liquidity_observed else None
                )
                return
            self.early_archive.record_outcome_observation(
                chain, token, now, price, liquidity if liquidity_observed else None
            )
            # Raw snapshots continue only for tokens that already passed Stage 1.
            self.db.save_market_snapshot(
                chain, token, str(pair.get("pairAddress") or ""), now, age, price, liquidity,
                float(pair.get("marketCap") or 0) or None,
                float(pair.get("fdv") or 0) or None, features,
            )

        analysis = analyse_pair(
            chain, token, pair, json.loads(row.get("metadata_json") or "{}"),
            self.settings.min_liquidity_usd, self.settings.min_volume_5m_usd,
            self.settings.min_transactions_5m, 3.0,
        )
        token_key = (chain.lower(), canonical_token_key(chain, token))
        self._stage1_period_evaluated.add(token_key)

        wash_flags = wash_trading_flags(
            features, self.settings.wash_volume_per_tx_usd,
            self.settings.wash_volume_to_liquidity,
            self.settings.wash_one_sided_buy_share,
        )
        if wash_flags:
            self._reject("wash_trading_suspected", token, "; ".join(wash_flags))
            return
        if analysis.market_cap is None:
            self._reject("market_cap_unavailable", token)
            return
        if analysis.market_cap > self.settings.max_market_cap_usd:
            self._reject("market_cap_above_maximum", token,
                         f"${analysis.market_cap:.0f} > ${self.settings.max_market_cap_usd:.0f}")
            return

        # Continue collecting snapshots after the alert window so forward outcomes mature.
        if not (self.settings.min_prediction_age_minutes <= age <= self.settings.max_pair_age_minutes):
            self._reject("outside_alert_age_window", token, f"age={age:.1f}m")
            return
        if liquidity < self.settings.min_liquidity_usd:
            self._reject("liquidity_below_minimum", token, f"${liquidity:.0f} < ${self.settings.min_liquidity_usd:.0f}")
            return
        if features["volume_m5_usd"] < self.settings.min_volume_5m_usd:
            self._reject("volume_below_minimum", token, f"${features['volume_m5_usd']:.0f} < ${self.settings.min_volume_5m_usd:.0f}")
            return
        if features["transactions_m5"] < self.settings.min_transactions_5m:
            self._reject("transactions_below_minimum", token, f"{features['transactions_m5']} < {self.settings.min_transactions_5m}")
            return

        priority_signal_count = early_lookup_priority_signal_count(features, analysis.market_cap)
        if priority_signal_count >= 2:
            self._mark_early_lookup_priority(token_key, now)
        elif analysis.market_cap is not None and analysis.market_cap > 25000:
            self._early_lookup_priority.pop(token_key, None)

        wallet_evidence = {}
        creator_wallet = None
        creator_evidence = None
        refresh_onchain = chain.lower() == "solana" and now - self._creator_checked.get(token, 0.0) >= 300
        if chain.lower() == "solana":
            if refresh_onchain:
                self._onchain_evidence_cache[token] = await self.solana_evidence.creator_for_mint(token) or {}
                self._creator_checked[token] = now
            evidence_result = self._onchain_evidence_cache.get(token, {})
            creator = evidence_result.get("creator") or {}
            concentration = evidence_result.get("holder_concentration") or {}
            if creator.get("wallet"):
                creator_wallet = str(creator["wallet"])
                creator_evidence = {
                    "source": creator.get("provider"), "basis": creator.get("evidence"),
                    "confidence": "candidate_only",
                }
                wallet_row = self.early_archive.developer_wallet(creator_wallet) or {}
                wallet_evidence.update({
                    "creator_candidate": True,
                    "developer_status": wallet_row.get("status", "candidate"),
                    "developer_disqualified": wallet_row.get("status") == "disqualified",
                })
                if wallet_row.get("status") == "disqualified":
                    wallet_evidence["suspicious"] = True
                    wallet_evidence.setdefault("reasons", []).append(
                        "Creator candidate is persistently disqualified after two rug Dev Calls"
                    )
            if concentration.get("available"):
                previous_concentration = (
                    self.early_archive.record_holder_concentration(
                        chain, token, now, concentration, str(evidence_result.get("provider") or "unknown")
                    ) if refresh_onchain else None
                )
                top1 = float(concentration.get("top1_owner_share") or 0)
                top5 = float(concentration.get("top5_owner_share") or 0)
                owner_count = int(concentration.get("sampled_owner_count") or 0)
                concentration_rising = bool(
                    previous_concentration
                    and top5 - float(previous_concentration.get("top5_owner_share") or 0) >= 0.20
                )
                wallet_evidence.update({
                    "holder_concentration": concentration,
                    "low_manipulation": owner_count >= 10 and top1 <= 0.20 and top5 <= 0.70,
                    "suspicious": bool(wallet_evidence.get("developer_disqualified")) or top5 >= 0.90 or concentration_rising,
                    "reasons": (
                        (["Creator candidate is persistently disqualified after two rug Dev Calls"]
                         if wallet_evidence.get("developer_disqualified") else [])
                        + ([concentration.get("warning", "Holder concentration is only a partial screen")]
                           if top5 >= 0.90 else [])
                        + (["Top-five holder concentration increased sharply since the previous sample"]
                           if concentration_rising else [])
                    ),
                })
                if not wallet_evidence["low_manipulation"]:
                    wallet_evidence.setdefault("reasons", []).append(
                        "Holder distribution does not yet establish broad independent buyers; bundle/funder and trade-overlap checks remain incomplete"
                    )
            else:
                wallet_evidence.setdefault("reasons", []).append(
                    "Holder concentration unavailable; wallet, bundle, funder and independent-buyer evidence is insufficient"
                )
        momentum_history = self.early_archive.recent_history(chain, token)
        momentum = classify_early_momentum(features, momentum_history, wallet_evidence)
        early_entry = evaluate_early_entry(
            features, analysis.market_cap, momentum_history, wallet_evidence
        )
        features["early_entry_candidate"] = early_entry["qualified"]
        features["early_entry_signal_score"] = early_entry["score"]
        features["early_entry_signals"] = early_entry["signals"]
        features["early_entry_reasons"] = early_entry["reasons"]
        features["early_entry_qualified"] = False
        features["early_entry_model_gate_bypassed"] = False
        features["early_momentum_classification"] = momentum["classification"]
        features["early_momentum_signals"] = momentum["signals"]
        features["early_momentum_cautions"] = momentum["cautions"]
        features["early_momentum_market_cap_tier"] = momentum["market_cap_tier"]
        features["wallet_evidence_quality"] = momentum["evidence_quality"]
        current_market_cap = float(analysis.market_cap or 0)
        progression = set(self.early_archive.market_cap_progression(chain, token))
        progression.update(threshold for threshold in (15000.0, 30000.0, 50000.0, 80000.0)
                           if current_market_cap >= threshold)
        features["market_cap_progression"] = sorted(progression)
        if wallet_evidence.get("holder_concentration", {}).get("available"):
            concentration = wallet_evidence["holder_concentration"]
            features["top1_holder_owner_share"] = concentration.get("top1_owner_share", 0)
            features["top5_holder_owner_share"] = concentration.get("top5_owner_share", 0)
            features["sampled_holder_owner_count"] = concentration.get("sampled_owner_count", 0)
        self.early_archive.record_observation(
            chain, token, now, price, liquidity,
            float(analysis.market_cap or 0), features, momentum,
        )
        # RugCheck REJECT remains a hard block. UNKNOWN is eligible only for a
        # separately labelled, unverified PRE-MODEL watchlist after all market gates.
        passed, status, risk_score, risk_reasons = await self._security(token)
        if status == "REJECT":
            self._reject("rugcheck_reject", token, "; ".join(risk_reasons[:3]))
            return
        if status == "UNKNOWN":
            if analysis.score < self.settings.min_premodel_score:
                self._reject("unverified_score_below_minimum", token, f"{analysis.score:.0f} < {self.settings.min_premodel_score:.0f}")
                return
            if self.db.should_alert(chain, token, self.settings.alert_cooldown_minutes):
                if await self._send_alert_once(analysis, features, status, risk_score, risk_reasons):
                    self._record_dev_call_after_alert(creator_wallet, chain, token, price, now, creator_evidence)
                    self._gate_counts["unverified_watchlist_alert_sent"] = self._gate_counts.get("unverified_watchlist_alert_sent", 0) + 1
                if self.db.alert_event(chain, token):
                    log.warning("UNVERIFIED WATCHLIST ALERT SENT %s %s score=%.0f; RugCheck UNKNOWN", chain, analysis.symbol, analysis.score)
            else:
                self._reject("token_already_alerted", token)
            return

        early_entry_qualified = bool(
            passed
            and early_entry["qualified"]
            and analysis.score >= max(0.0, self.settings.min_premodel_score - 10.0)
        )
        features["early_entry_qualified"] = early_entry_qualified
        if early_entry_qualified:
            log.info(
                "EARLY ENTRY QUALIFIED %s %s market_cap=$%.0f early_score=%d signals=%s",
                chain, analysis.symbol, float(analysis.market_cap or 0),
                int(early_entry["score"]), early_entry["signals"],
            )

        self._stage1_period_passed.add(token_key)
        # Additive early-momentum observations; no Stage 1 filter or threshold is changed.
        self.db.save_snapshot(analysis)
        # First eligible snapshot is the immutable model input. Subsequent prices
        # are recorded separately and are not subjected to Stage 1 filters again.
        baseline_created = self.db.create_training_baseline(
            chain, token, str(pair.get("pairAddress") or ""), now, age, price, features,
        )
        if baseline_created:
            self.db.save_market_snapshot(
                chain, token, str(pair.get("pairAddress") or ""), now, age, price, liquidity,
                float(pair.get("marketCap") or 0) or None,
                float(pair.get("fdv") or 0) or None, features,
            )

        if not self.predictor.ready:
            if analysis.score < self.settings.min_premodel_score and not early_entry_qualified:
                self._reject("premodel_score_below_minimum", token, f"{analysis.score:.0f} < {self.settings.min_premodel_score:.0f} and early-entry trigger not qualified")
                return
            if self.db.should_alert(chain, token, self.settings.alert_cooldown_minutes):
                if await self._send_alert_once(analysis, features, status, risk_score, risk_reasons):
                    self._record_dev_call_after_alert(creator_wallet, chain, token, price, now, creator_evidence)
                    metric = "early_entry_alert_sent" if early_entry_qualified else "premodel_alert_sent"
                    self._gate_counts[metric] = self._gate_counts.get(metric, 0) + 1
                if self.db.alert_event(chain, token):
                    log.info("PRE-MODEL WATCHLIST ALERT SENT %s %s score=%.0f", chain, analysis.symbol, analysis.score)
            else:
                self._reject("alert_cooldown", token)
            log.info(
                "PRE-MODEL %s %s score=%.0f age=%.1fm liquidity=$%.0f",
                chain, analysis.symbol, analysis.score, age, liquidity,
            )
            return

        probabilities = self.predictor.predict(features)
        if not probabilities:
            self._reject("prediction_unavailable", token)
            return
        self.db.save_prediction(
            chain, token, now, age, probabilities["2x"], probabilities["5x"],
            probabilities["10x"], self.predictor.version,
        )
        model_qualifies = (
            probabilities["2x"] >= self.settings.min_probability_2x
            and (
                probabilities["5x"] >= self.settings.min_probability_5x
                or probabilities["10x"] >= self.settings.min_probability_10x
            )
        )
        qualifies = model_qualifies or early_entry_qualified
        features["early_entry_model_gate_bypassed"] = bool(early_entry_qualified and not model_qualifies)
        if qualifies and self.db.should_alert(chain, token, self.settings.alert_cooldown_minutes):
            if await self._send_alert_once(
                analysis, features, status, risk_score, risk_reasons,
                probabilities=probabilities, model_version=self.predictor.version,
            ):
                self._record_dev_call_after_alert(creator_wallet, chain, token, price, now, creator_evidence)
                metric = "early_entry_model_gate_bypassed_alert_sent" if features.get("early_entry_model_gate_bypassed") else "model_alert_sent"
                self._gate_counts[metric] = self._gate_counts.get(metric, 0) + 1
            if self.db.alert_event(chain, token):
                log.info("MODEL-ACTIVE ALERT SENT %s %s", chain, analysis.symbol)
        elif not qualifies:
            self._reject("model_probability_gate", token)
        else:
            self._reject("alert_cooldown", token)
        log.info(
            "MODEL-ACTIVE %s %s age=%.1fm score=%.0f p2=%.2f p5=%.2f p10=%.2f",
            chain, analysis.symbol, age, analysis.score,
            probabilities["2x"], probabilities["5x"], probabilities["10x"],
        )

    async def scan_once(self):
        # Keep the proven DexScreener profile/boost feeds as an additional
        # Solana discovery route; stream feeds remain enabled alongside it.
        now = time.time()
        if (
            self.settings.dexscreener_discovery_enabled
            and now - self._last_dex_discovery >= self.settings.dexscreener_discovery_interval_seconds
        ):
            self._last_dex_discovery = now
            try:
                discovered = await self.dex.discover()
                solana_candidates = [c for c in discovered if c.chain_id.lower() == "solana"]
                for candidate in solana_candidates:
                    await self.on_candidate(candidate)
                self._gate_counts["dexscreener_discovered"] = self._gate_counts.get("dexscreener_discovered", 0) + len(solana_candidates)
                log.info("DexScreener discovery returned %d Solana candidates", len(solana_candidates))
            except Exception as exc:
                log.warning("DexScreener discovery cycle failed: %s", exc)

        rows = self.db.active_candidates(
            self.settings.tracking_retention_hours * 60, self.settings.max_active_tokens
        )
        now = time.time()
        eligible = []
        for row in rows:
            key = (row["chain_id"], row["token_address"])
            last = self._last_lookup.get(key, 0.0)
            if last == 0.0:
                prior = self.db.recent_snapshots(*key, limit=1)
                if prior:
                    last = float(prior[0]["captured_at"])
            priority_until = self._early_lookup_priority.get(key, 0.0)
            if priority_until and priority_until <= now:
                self._early_lookup_priority.pop(key, None)
                priority_until = 0.0
            lookup_interval = (
                min(self.settings.pair_lookup_interval_seconds, 30)
                if priority_until > now else self.settings.pair_lookup_interval_seconds
            )
            if now - last >= lookup_interval:
                self._last_lookup[key] = now
                eligible.append(row)
        if eligible:
            groups: dict[str, list[dict]] = {}
            for row in eligible:
                groups.setdefault(row["chain_id"], []).append(row)
            batches = []
            for chain, items in groups.items():
                for start in range(0, len(items), self.settings.pair_lookup_batch_size):
                    batches.append((chain, items[start:start + self.settings.pair_lookup_batch_size]))
            sem = asyncio.Semaphore(self.settings.pair_lookup_concurrency)
            fallback_budget = [self.settings.dexpaprika_fallback_max_per_cycle]
            fallback_budget_lock = asyncio.Lock()

            async def fetch_batch(chain: str, items: list[dict]):
                async with sem:
                    try:
                        tokens = [r["token_address"] for r in items]
                        pairs = await self.dex.token_pairs_batch(chain, tokens)
                        by_token = {canonical_token_key(chain, t): [] for t in tokens}
                        for pair in pairs:
                            # DexScreener priceUsd and token metadata describe baseToken.
                            # Matching quoteToken here could score the wrong asset as the candidate.
                            base = canonical_token_key(chain, str((pair.get("baseToken") or {}).get("address") or ""))
                            if base in by_token:
                                by_token[base].append(pair)
                        for row in items:
                            matches = by_token.get(canonical_token_key(chain, row["token_address"]), [])
                            if not matches:
                                self._gate_counts["dex_pair_not_found"] = self._gate_counts.get("dex_pair_not_found", 0) + 1
                                fallback_pair = None
                                if chain.lower() == "solana" and self.dexpaprika:
                                    async with fallback_budget_lock:
                                        if fallback_budget[0] > 0:
                                            fallback_budget[0] -= 1
                                            should_try_fallback = True
                                        else:
                                            should_try_fallback = False
                                    if should_try_fallback:
                                        try:
                                            fallback_pair = await self.dexpaprika.token_pair(row["token_address"])
                                        except Exception as fallback_exc:
                                            log.warning(
                                                "DexPaprika fallback failed for %s: exception_type=%s",
                                                row["token_address"], type(fallback_exc).__name__,
                                            )
                                if fallback_pair:
                                    self._gate_counts["dexpaprika_fallback_resolved"] = self._gate_counts.get("dexpaprika_fallback_resolved", 0) + 1
                                    try:
                                        await self._process_pair(row, fallback_pair)
                                    except Exception as fallback_process_exc:
                                        log.warning(
                                            "Could not process DexPaprika pair for %s: exception_type=%s",
                                            row["token_address"], type(fallback_process_exc).__name__,
                                        )
                                elif chain.lower() == "solana" and self.dexpaprika and should_try_fallback:
                                    self._gate_counts["dexpaprika_unresolved_or_rate_deferred"] = self._gate_counts.get("dexpaprika_unresolved_or_rate_deferred", 0) + 1
                                continue
                            pair = max(
                                matches,
                                key=lambda p: float(((p.get("liquidity") or {}).get("usd") or 0)),
                            )
                            try:
                                await self._process_pair(row, pair)
                            except Exception as exc:
                                detail = str(exc).replace(self.telegram.base, "https://api.telegram.org/<redacted>")
                                if self.telegram.token:
                                    detail = detail.replace(self.telegram.token, "<redacted>")
                                log.warning(
                                    "Could not process %s: exception_type=%s message=%s",
                                    row["token_address"], type(exc).__name__, detail[:500] or "<empty>",
                                )
                    except Exception as exc:
                        log.warning("DexScreener batch lookup failed for %s (%d tokens): %s", chain, len(items), type(exc).__name__)
                        # If the primary provider itself is unavailable, allow one
                        # bounded fallback attempt for this batch during this scan cycle.
                        fallback_row = None
                        if chain.lower() == "solana" and self.dexpaprika and items:
                            async with fallback_budget_lock:
                                if fallback_budget[0] > 0:
                                    fallback_budget[0] -= 1
                                    fallback_row = items[0]
                        if fallback_row:
                            try:
                                fallback_pair = await self.dexpaprika.token_pair(fallback_row["token_address"])
                                if fallback_pair:
                                    self._gate_counts["dexpaprika_primary_error_resolved"] = self._gate_counts.get("dexpaprika_primary_error_resolved", 0) + 1
                                    await self._process_pair(fallback_row, fallback_pair)
                            except Exception as fallback_exc:
                                log.warning("DexPaprika fallback after DexScreener error failed for %s: exception_type=%s",
                                            fallback_row["token_address"], type(fallback_exc).__name__)

            await asyncio.gather(*(fetch_batch(chain, items) for chain, items in batches))

        self._maybe_log_gate_summary()

        if now - self._last_prune >= 3600:
            deleted = self.db.prune_old_data(
                self.settings.tracking_retention_hours,
                self.settings.prediction_horizon_hours,
            )
            self._last_prune = now
            if any(deleted.values()):
                log.info("Pruned expired raw data: %s; labelled replay examples retained=%d",
                         deleted, self.db.training_example_count())

    async def run(self):
        try:
            await self.telegram.send(
                "🟢 ScannerBot v2.0 started
"
                "Chain: Solana
"
                "Mode: " + self.mode + "
"
                "Discovery: " + (", ".join(type(d).__name__ for d in self.discovery) or "disabled") + "
"
                "ML models: " + (self.predictor.version if self.predictor.ready else "not ready — market scoring/watchlist mode") + "
"
                "RugCheck: enabled in both modes; UNKNOWN/rejected reports block qualified alerts
"
                "Training: " + ("automatic" if self.settings.auto_train_enabled else "disabled") + "
"
                "Read-only: no wallet keys, swaps or auto-buy."
            )
        except Exception as exc:
            # Telegram availability must not prevent discovery and scanning from starting.
            log.warning(
                "Telegram startup notification failed; scanner will continue: exception_type=%s detail=%s",
                type(exc).__name__, self.telegram._safe_detail(exc),
            )
        tasks = [asyncio.create_task(d.run()) for d in self.discovery]
        tasks.append(asyncio.create_task(self.dashboard.run()))
        if self.settings.auto_train_enabled:
            tasks.append(asyncio.create_task(self.auto_trainer.run()))
        try:
            while True:
                try:
                    await self.scan_once()
                except Exception:
                    log.exception("Scan cycle failed")
                try:
                    await self.telegram.poll_commands()
                except Exception as exc:
                    log.debug("Telegram polling failed: exception_type=%s detail=%s",
                              type(exc).__name__, self.telegram._safe_detail(exc))
                await asyncio.sleep(self.settings.scan_interval_seconds)
        finally:
            self.auto_trainer.stop()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
