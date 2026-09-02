"""Extract model-selected Q/K features from the complete TAP-Vid DAVIS dataset."""

from __future__ import annotations

import os
from pathlib import Path

from davis_profiles import DAVIS_PROFILES, select_videos
from dotenv import load_dotenv

from heft import (
    ExtractionConfig,
    ExtractionTask,
    FeatureExtractionPool,
)
from heft.data import TapVidDavisDataset


def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    model_name = os.environ["HEFT_MODEL"]
    profile = DAVIS_PROFILES[model_name]
    dataset_root = Path(os.environ["HEFT_DATASET_ROOT"])
    model_cache = Path(os.environ["HEFT_MODEL_CACHE_DIR"])
    output_root = Path(os.environ["HEFT_FEATURE_OUTPUT_ROOT"].format(model=model_name))
    gpu_ids = tuple(map(int, os.environ["HEFT_GPU_IDS"].split(",")))

    dataset = TapVidDavisDataset(dataset_root)
    samples = select_videos(tuple(dataset[index] for index in range(len(dataset))))
    tasks = tuple(
        ExtractionTask(
            name=sample.video_id,
            input_video=sample.video.read(),
            output_dir=output_root / sample.video_id,
        )
        for sample in samples
    )
    config = ExtractionConfig(
        capture=profile.capture,
        chunk_size=25,
        max_pending_features=4,
        overwrite=os.getenv("HEFT_OVERWRITE", "false").lower() == "true",
    )
    with FeatureExtractionPool(
        model=profile.model,
        config=config,
        gpu_ids=gpu_ids,
        cache_dir=model_cache,
    ) as extractor:
        results = extractor.extract(tasks)

    for result in results:
        print(
            result.task_name,
            [(chunk.chunk, chunk.gpu_id) for chunk in result.chunks],
            result.output_dir,
        )


if __name__ == "__main__":
    main()
