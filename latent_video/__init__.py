"""Video latent experiments, built on the local HeFT and Wan implementations."""

from .config import CANDIDATES, ChannelSelection, ClipConfig
from .extractor import ClipLatents, WanLatentExtractor

__all__ = [
    "CANDIDATES",
    "ChannelSelection",
    "ClipConfig",
    "ClipLatents",
    "WanLatentExtractor",
]
