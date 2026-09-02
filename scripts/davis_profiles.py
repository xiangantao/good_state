"""Model profiles shared by DAVIS extraction and evaluation scripts."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace

from heft import COGVIDEOX, COSMOS_2, WAN_2_1, ModelConfig, TrackingConfig
from heft.attn_hook import CaptureSpec, FeatureKind
from heft.tracking import FeatureSelection


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
