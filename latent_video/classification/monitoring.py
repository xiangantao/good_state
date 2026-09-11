"""Rank-zero W&B runs with a flushed, locally readable metric log."""

from __future__ import annotations

import importlib
import json
import logging
import os
from pathlib import Path

from .config import WandbConfig

logger = logging.getLogger(__name__)


class WandbMonitor:
    def __init__(
        self,
        config: WandbConfig,
        output: Path,
        *,
        rank: int,
        metadata: dict,
        job_type: str,
    ):
        self.run = None
        self.stream = None
        if not config.enabled or rank != 0:
            return
        # Apply before importing the SDK so the run config controls the mode.
        os.environ["WANDB_MODE"] = config.mode
        os.environ["WANDB_ERROR_REPORTING"] = "false"
        wandb = importlib.import_module("wandb")
        output.mkdir(parents=True, exist_ok=True)
        try:
            self.run = wandb.init(
                project=config.project,
                entity=config.entity,
                name=config.name or output.name,
                group=config.group or output.name,
                job_type=job_type,
                mode=config.mode,
                dir=str(output),
                config=metadata,
                save_code=False,
                settings=wandb.Settings(disable_git=True, disable_code=True),
            )
            self.stream = (output / "monitoring.jsonl").open("a", buffering=1)
            for pattern in ("batch/*", "optim/*", "timing/*", "memory/*"):
                self.run.define_metric(pattern, step_metric="step")
            for pattern in ("train/*", "val/*", "evaluation/*"):
                self.run.define_metric(pattern, step_metric="epoch")
        except BaseException:
            self.finish(exit_code=1)
            raise

    def update_config(self, values: dict):
        if self.run is not None:
            self.run.config.update(values, allow_val_change=True)

    def log(self, values: dict):
        if self.run is None:
            return
        assert self.stream is not None
        self.stream.write(json.dumps({"run_id": self.run.id, **values}) + "\n")
        self.run.log(values)

    def finish(self, *, exit_code: int):
        try:
            if self.run is not None:
                self.run.finish(exit_code=exit_code)
        except Exception:
            logger.exception("Failed to finalize the W&B run")
        finally:
            self.run = None
            if self.stream is not None:
                self.stream.close()
                self.stream = None
