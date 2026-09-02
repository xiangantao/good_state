"""Model profiles shared by DAVIS extraction and evaluation scripts."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from heft import COGVIDEOX, COSMOS_2, WAN_2_1, ModelConfig, TrackingConfig
from heft.attn_hook import CaptureSpec, FeatureKind
from heft.tracking import FeatureSelection

# DAVIS_PROFILES is built at import time and `_local` reads the environment while
# building it, so the .env file has to be loaded here rather than inside a
# script's main(): by then the profiles are already frozen at their Hub ids.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def _local(model: ModelConfig) -> ModelConfig:
    """Redirect a profile at an on-disk checkpoint when the env asks for one.

    ``HEFT_MODEL_PATH_<NAME>`` (e.g. ``HEFT_MODEL_PATH_WAN2_1``) overrides the
    Hub repo id so ``from_pretrained`` loads locally instead of downloading.
    """
    key = "HEFT_MODEL_PATH_" + model.name.upper().replace(".", "_").replace("-", "_")
    path = os.getenv(key)
    return replace(model, model_id=path) if path else model


@dataclass(frozen=True, slots=True)
class DavisProfile:
    model: ModelConfig
    layer: int
    head: int
    tracking: TrackingConfig

    @property
    def capture(self) -> CaptureSpec:
        return CaptureSpec(
            step=self.model.start_step,
            layers=(self.layer,),
            heads=(self.head,),
            features=(FeatureKind.QUERY, FeatureKind.KEY),
        )

    @property
    def selection(self) -> FeatureSelection:
        return FeatureSelection(layer=self.layer, head=self.head)


DAVIS_PROFILES = {
    "wan": DavisProfile(
        model=_local(WAN_2_1),
        layer=15,
        head=2,
        tracking=TrackingConfig(
            query_feature=FeatureKind.KEY,
            target_feature=FeatureKind.KEY,
            update_feature=FeatureKind.KEY,
            argmax_radius=17.0,
            search_radius=42.0,
            visibility_threshold=18.0,
            feature_ema_alpha=0.05,
            frequency_range=(0.15, 1.0),
        ),
    ),
    "cosmos2": DavisProfile(
        model=_local(COSMOS_2),
        layer=18,
        head=6,
        tracking=TrackingConfig(
            query_feature=FeatureKind.KEY,
            target_feature=FeatureKind.KEY,
            update_feature=FeatureKind.KEY,
            argmax_radius=18.0,
            search_radius=42.0,
            visibility_threshold=26.0,
            feature_ema_alpha=0.05,
            frequency_range=(0.1, 1.0),
        ),
    ),
    "cogvideox": DavisProfile(
        model=_local(COGVIDEOX),
        layer=15,
        head=16,
        tracking=TrackingConfig(
            query_feature=FeatureKind.KEY,
            target_feature=FeatureKind.KEY,
            update_feature=FeatureKind.KEY,
            argmax_radius=13.0,
            search_radius=43.0,
            visibility_threshold=23.0,
            feature_ema_alpha=0.05,
        ),
    ),
}


def select_videos(samples: tuple[Any, ...]) -> tuple[Any, ...]:
    """Restrict a DAVIS sample tuple to ``HEFT_VIDEOS`` when that env var is set.

    ``HEFT_VIDEOS`` is a comma-separated list of video ids, for smoke tests that
    should not spend a full dataset pass to reach the first error. Unset means
    the whole dataset, which is the normal path.
    """
    wanted = os.getenv("HEFT_VIDEOS", "").strip()
    if not wanted:
        return samples
    names = tuple(name.strip() for name in wanted.split(",") if name.strip())
    available = {sample.video_id for sample in samples}
    missing = [name for name in names if name not in available]
    if missing:
        raise SystemExit(f"HEFT_VIDEOS names no such video: {', '.join(missing)}")
    return tuple(sample for sample in samples if sample.video_id in set(names))
