"""Track and evaluate extracted video features on TAP-Vid DAVIS."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

from davis_profiles import DAVIS_PROFILES
from dotenv import load_dotenv

from heft import (
    AnnotationQueries,
    FeatureVideo,
    TrackingTask,
    evaluate_dataset,
    track_features,
)
from heft.data import TapVidDavisDataset


def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    model_name = os.environ["HEFT_MODEL"]
    profile = DAVIS_PROFILES[model_name]
    dataset = TapVidDavisDataset(os.environ["HEFT_DATASET_ROOT"])
    feature_root = Path(os.environ["HEFT_FEATURE_OUTPUT_ROOT"].format(model=model_name))
    output_path = Path(os.environ["HEFT_EVALUATION_OUTPUT"].format(model=model_name))
    gpu_ids = tuple(map(int, os.environ["HEFT_GPU_IDS"].split(",")))
    samples = tuple(dataset[index] for index in range(len(dataset)))
    videos = {
        sample.video_id: FeatureVideo.open(feature_root / sample.video_id)
        for sample in samples
    }

    tasks = []
    for sample in samples:
        features = videos[sample.video_id]
        tasks.append(
            TrackingTask(
                features=features,
                query_points=AnnotationQueries(sample.annotations).generate(features),
                selection=profile.selection,
                video_id=sample.video_id,
            )
        )

    results = track_features(
        tasks,
        config=profile.tracking,
        gpu_ids=gpu_ids,
    )
    report = evaluate_dataset(results, dataset)
    payload = asdict(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["aggregate"], indent=2))
    print(output_path)


if __name__ == "__main__":
    main()
