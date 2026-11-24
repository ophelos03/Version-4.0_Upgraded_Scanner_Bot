from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)


class AutoTrainingManager:
    """Runs scheduled XGBoost training and atomically promotes validated models."""

    def __init__(self, settings, predictor, telegram):
        self.settings = settings
        self.predictor = predictor
        self.telegram = telegram
        self.project_root = Path(__file__).resolve().parents[1]
        self.model_root = Path(settings.model_dir).resolve()
        self.model_root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.model_root / "autotraining_state.json"
        self._stop = asyncio.Event()
        self._running = False

    def _read_state(self) -> dict:
        try:
            return json.loads(self.state_path.read_text())
        except (OSError, ValueError):
            return {}

    def _write_state(self, state: dict) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2))
        os.replace(tmp, self.state_path)

    async def _notify(self, message: str) -> None:
        try:
            await self.telegram.send(message)
        except Exception:
            log.exception("Could not send automatic-training Telegram notification")

    def _validation_data_ready(self, report: dict) -> tuple[bool, str]:
        targets = report.get("targets", {})
        for target in ("2x", "5x", "10x"):
            metrics = targets.get(target) or {}
            positives = int(metrics.get("test_positives", 0))
            negatives = int(metrics.get("test_negatives", 0))
            if positives < self.settings.auto_train_min_test_positives:
                return False, f"{target}: insufficient positive validation examples"
            if negatives < self.settings.auto_train_min_test_negatives:
                return False, f"{target}: insufficient negative validation examples"
        return True, "validation cohort has both classes for every target"

    def _candidate_passes(self, report: dict) -> tuple[bool, str]:
        if int(report.get("rows", 0)) < self.settings.min_training_rows:
            return False, "not enough matured training rows"
        if int(report.get("distinct_tokens", 0)) < self.settings.auto_train_min_tokens:
            return False, "not enough distinct tokens"
        targets = report.get("targets", {})
        for target in ("2x", "5x", "10x"):
            metrics = targets.get(target)
            if not metrics:
                return False, f"missing {target} evaluation"
            if not bool(metrics.get("challenger_beats_incumbent")):
                return False, f"{target}: challenger did not outperform incumbent/baseline on held-out cases"
        return True, "all model validation gates passed"

    def _current_release_dir(self) -> Path:
        pointer = self.model_root / "CURRENT"
        if not pointer.is_file():
            return self.model_root
        release_name = pointer.read_text().strip()
        if not release_name or Path(release_name).name != release_name:
            return self.model_root
        return self.model_root / "releases" / release_name

    async def train_once(self) -> bool:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        staging = self.model_root / ".staging" / stamp
        release_name = f"xgb-{stamp}"
        releases = self.model_root / "releases"
        staging.parent.mkdir(parents=True, exist_ok=True)
        staging.mkdir(parents=True, exist_ok=True)
        script = self.project_root / "scripts" / "train_xgboost.py"
        cmd = [
            sys.executable, str(script),
            "--db", str(self.model_root / "training_archive.sqlite"),
            "--model-dir", str(staging),
            "--horizon-hours", str(self.settings.prediction_horizon_hours),
            "--min-rows", str(self.settings.min_training_rows),
            "--min-tokens", str(self.settings.auto_train_min_tokens),
            "--min-train-positives", str(self.settings.auto_train_min_train_positives),
            "--min-train-negatives", str(self.settings.auto_train_min_train_negatives),
            "--min-test-positives", str(self.settings.auto_train_min_test_positives),
            "--min-test-negatives", str(self.settings.auto_train_min_test_negatives),
            "--since-timestamp", str(float(self._read_state().get("last_training_at", self._read_state().get("last_success_at", 0)))),
            "--min-new-samples", str(self.settings.auto_train_min_new_samples),
            "--incumbent-dir", str(self._current_release_dir()),
        ]
        # Short-horizon models are research-only and never affect production
        # model selection. Attempt them on the same scheduler cadence, even when
        # the 2h production-label dataset is not yet mature.
        research_script = self.project_root / "scripts" / "train_early_momentum.py"
        research_cmd = [
            sys.executable, str(research_script),
            "--db", str(self.model_root / "training_archive.sqlite"),
            "--output", str(self.model_root / "early_research"),
            "--min-rows", "100",
            "--min-tokens", "30",
        ]
        try:
            research_proc = await asyncio.create_subprocess_exec(
                *research_cmd, cwd=str(self.project_root),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            )
            research_output, _ = await research_proc.communicate()
            research_text = research_output.decode("utf-8", errors="replace")
            log.info("Short-horizon research training output: %s", research_text[-4000:])
            if research_proc.returncode != 0:
                log.warning("Short-horizon research training failed; production training will continue")
        except Exception:
            log.exception("Short-horizon research training could not run; production training will continue")

        log.info("Starting scheduled XGBoost training")
        proc = None
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, cwd=str(self.project_root),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            )
            output, _ = await proc.communicate()
            output_text = output.decode("utf-8", errors="replace")
            log.info("Automatic training subprocess output: %s", output_text[-5000:])
            if proc.returncode != 0:
                log.warning("Automatic training did not complete: %s", output_text[-5000:])
                await self._notify("⚠️ ScannerBot automatic training postponed.\n" + output_text[-2500:])
                return False

            report_path = staging / "metadata.json"
            report = json.loads(report_path.read_text())
            report["version"] = release_name
            report["automatic_training"] = True
            validation_ready, validation_reason = self._validation_data_ready(report)
            if not validation_ready:
                log.warning("Training validation data is not ready: %s", validation_reason)
                await self._notify("⏳ ScannerBot training postponed; " + validation_reason)
                return False

            passed, reason = self._candidate_passes(report)
            if not passed:
                log.warning("Candidate model set not promoted: %s", reason)
                await self._notify(f"ℹ️ ScannerBot completed a training attempt but kept the incumbent model.\\nReason: {reason}\\nRows: {report.get('rows')} | Tokens: {report.get('distinct_tokens')}")
                state = self._read_state()
                state.update({"last_attempt_at": time.time(), "last_training_at": time.time()})
                self._write_state(state)
                return True

            report["validation_passed"] = True
            report_path.write_text(json.dumps(report, indent=2))

            # Store immutable model releases and switch one small pointer file atomically.
            releases.mkdir(parents=True, exist_ok=True)
            release_dir = releases / release_name
            os.replace(staging, release_dir)
            pointer = self.model_root / "CURRENT"
            previous_release = pointer.read_text().strip() if pointer.exists() else None
            pointer_tmp = self.model_root / "CURRENT.tmp"
            pointer_tmp.write_text(release_name)
            os.replace(pointer_tmp, pointer)

            if not self.predictor.reload():
                # Roll the pointer back so a restart cannot select a broken candidate.
                if previous_release:
                    rollback_tmp = self.model_root / "CURRENT.rollback"
                    rollback_tmp.write_text(previous_release)
                    os.replace(rollback_tmp, pointer)
                else:
                    pointer.unlink(missing_ok=True)
                shutil.rmtree(release_dir, ignore_errors=True)
                log.error("New model release %s could not be loaded; pointer rolled back", release_name)
                await self._notify("⚠️ ScannerBot could not load the candidate model set; the previous release remains selected.")
                return False

            summary = []
            for target in ("2x", "5x", "10x"):
                m = report["targets"][target]
                summary.append(f"{target.upper()}: AP {m['average_precision']:.3f}, precision@0.5 {m['precision_at_0_5']:.3f}")
            state = self._read_state()
            state.update({"last_attempt_at": time.time(), "last_training_at": time.time(), "last_success_at": time.time(), "last_release": release_name})
            self._write_state(state)
            await self._notify(
                "🧠 ScannerBot automatically trained and activated a new XGBoost model set.\n"
                f"Version: {release_name}\nRows: {report['rows']} | Tokens: {report['distinct_tokens']}\n"
                f"New labelled samples: {report.get('new_labelled_samples_since_last_training', 0)}\n" + "\n".join(summary)
            )
            log.info("Promoted model release %s", release_name)
            return True
        except asyncio.CancelledError:
            if proc is not None and proc.returncode is None:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=10)
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
            raise
        except Exception:
            log.exception("Automatic model training failed")
            return False
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)

    async def run(self) -> None:
        if not self.settings.auto_train_enabled:
            log.info("Automatic model training is disabled")
            return
        state = self._read_state()
        now = time.time()
        deployment_id = os.getenv("RAILWAY_DEPLOYMENT_ID", "").strip()
        previous_deployment = str(state.get("deployment_id", ""))
        next_due = float(state.get("next_due_at", 0))
        # Reset initial delay on a genuinely new deployment; preserve cadence on restarts.
        if deployment_id and deployment_id != previous_deployment:
            next_due = now + self.settings.auto_train_initial_delay_hours * 3600
            state.update({"deployment_id": deployment_id, "next_due_at": next_due})
            self._write_state(state)
        elif next_due <= 0:
            last_attempt = float(state.get("last_attempt_at", 0))
            next_due = (last_attempt + self.settings.auto_train_retry_hours * 3600
                        if last_attempt else now + self.settings.auto_train_initial_delay_hours * 3600)
            state["next_due_at"] = next_due
            self._write_state(state)
        log.info("Automatic training scheduler enabled; next attempt in %.1f hours", max(0, (next_due-now)/3600))
        while not self._stop.is_set():
            wait_seconds = max(1, next_due - time.time())
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=wait_seconds)
                break
            except asyncio.TimeoutError:
                pass
            completed = await self.train_once()
            now = time.time()
            state = self._read_state()
            # Completed training advances to the regular interval even if not promoted.
            # Insufficient data or a failed training run uses the shorter retry interval.
            delay_hours = self.settings.auto_train_interval_hours if completed else self.settings.auto_train_retry_hours
            next_due = now + delay_hours * 3600
            state["last_attempt_at"] = now
            state["next_due_at"] = next_due
            self._write_state(state)

    def stop(self) -> None:
        self._stop.set()
