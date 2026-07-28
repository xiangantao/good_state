"""Multi-GPU feature extraction API."""

from .api import extract_feature, extract_features
from .config import (
    COGVIDEOX,
    COSMOS_2,
    WAN_2_1,
    ChunkExecutionMetadata,
    ChunkJob,
    ChunkResult,
    ExtractionConfig,
    ExtractionResult,
    ExtractionTask,
    ModelConfig,
    TailFramePolicy,
)
from .pool import FeatureExtractionPool, FeatureExtractor

__all__ = [
    "COGVIDEOX",
    "COSMOS_2",
    "WAN_2_1",
    "ChunkExecutionMetadata",
    "ChunkJob",
    "ChunkResult",
    "ExtractionConfig",
    "ExtractionResult",
    "ExtractionTask",
    "FeatureExtractionPool",
    "FeatureExtractor",
    "ModelConfig",
    "TailFramePolicy",
    "extract_feature",
    "extract_features",
]
