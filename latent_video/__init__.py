"""Video latent experiments, built on the local HeFT and Wan implementations."""

from typing import TYPE_CHECKING

from .config import CANDIDATES, ChannelSelection, ClipConfig

if TYPE_CHECKING:
    from .extractor import ClipLatents, WanLatentExtractor

__all__ = [
    "CANDIDATES",
    "ChannelSelection",
    "ClipConfig",
    "ClipLatents",
    "WanLatentExtractor",
]


def __getattr__(name):
    # Data-loader workers need the configuration, but not the Wan runtime.
    if name in {"ClipLatents", "WanLatentExtractor"}:
        from .extractor import ClipLatents, WanLatentExtractor

        globals().update(ClipLatents=ClipLatents, WanLatentExtractor=WanLatentExtractor)
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
