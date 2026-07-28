"""Dataset-backed point tracking evaluation."""

from .config import EvaluationConfig
from .evaluator import evaluate_dataset
from .result import EvaluationReport, MetricValues, VideoMetrics

__all__ = [
    "EvaluationConfig",
    "EvaluationReport",
    "MetricValues",
    "VideoMetrics",
    "evaluate_dataset",
]
