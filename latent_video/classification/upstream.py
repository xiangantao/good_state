"""Import the existing V-JEPA2 implementation without installing or downloading."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VJepaRuntime:
    classifier: Any
    video_dataset: Any
    make_transforms: Any
    init_opt: Any
    provenance: dict


def missing_dependencies() -> list[str]:
    return [
        name
        for name in ("timm", "decord", "pandas")
        if importlib.util.find_spec(name) is None
    ]


def load_upstream(root: Path) -> VJepaRuntime:
    root = root.resolve(strict=True)
    if not (root / "evals/video_classification_frozen/eval.py").is_file():
        raise ValueError(f"Not a local V-JEPA2 checkout: {root}")
    missing = missing_dependencies()
    if missing:
        raise RuntimeError(
            f"Missing V-JEPA2 dependencies: {', '.join(missing)}. "
            "No packages are installed or downloaded automatically."
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    # The upstream eval module changes CUDA visibility when SLURM_LOCALID is set.
    visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        head = importlib.import_module("src.models.attentive_pooler")
        data = importlib.import_module("src.datasets.video_dataset")
        augment = importlib.import_module("evals.video_classification_frozen.utils")
        evaluation = importlib.import_module("evals.video_classification_frozen.eval")
    finally:
        if visibility is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = visibility
    sources = (head, data, augment, evaluation)
    for module in sources:
        if not Path(inspect.getfile(module)).resolve().is_relative_to(root):
            raise RuntimeError("V-JEPA2 import resolved to a different checkout")
    paths = {Path(inspect.getfile(module)).resolve() for module in sources}
    paths.update((root / "src/datasets/utils/video").glob("*.py"))
    paths.update(
        root / relative
        for relative in (
            "src/models/utils/modules.py",
            "src/utils/tensors.py",
            "src/datasets/utils/dataloader.py",
            "src/utils/distributed.py",
        )
    )
    return VJepaRuntime(
        classifier=head.AttentiveClassifier,
        video_dataset=data.VideoDataset,
        make_transforms=augment.make_transforms,
        init_opt=evaluation.init_opt,
        provenance={
            "directory": str(root),
            "sources": {
                str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(paths)
            },
        },
    )
