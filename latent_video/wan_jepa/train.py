"""Mainline training/resume/evaluation entry; never fits the frozen operators."""

import argparse
import importlib.util
import json
import logging
from dataclasses import replace
from functools import partial
from pathlib import Path

import torch

from ..classification.train import preflight, run
from .config import RunConfig
from .encoder import CombinedEncoder, CombinedExtractor
from .operators import digest, load_artifacts


def check(config, evaluation=False):
    issues = preflight(config, evaluation=evaluation)
    if importlib.util.find_spec("einops") is None:
        issues.append("Missing JEPA dependency: einops (no automatic installation)")
    try:
        _, _, manifest = load_artifacts(config.path(config.artifact_dir))
        if (
            digest(config.path(config.jepa_model_path))
            != manifest["required_jepa_weights_sha256"]
        ):
            issues.append("JEPA checkpoint differs from the projection fit checkpoint")
        if (
            digest(config.path(config.channel_mask))
            != manifest["required_channel_mask_sha256"]
        ):
            issues.append("Wan channel mask differs from the groupZCA fit mask")
    except (OSError, ValueError, KeyError, TypeError) as error:
        issues.append(str(error))
    return issues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("ssv2.yaml")
    )
    parser.add_argument("--output-dir", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", type=Path)
    mode.add_argument("--evaluate", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True
    )
    torch.set_float32_matmul_precision("highest")
    config = RunConfig.load(args.config)
    if args.output_dir:
        config = replace(config, output_dir=str(args.output_dir.resolve()))
    issues = check(config, args.evaluate is not None)
    if args.check:
        print(
            json.dumps(
                {
                    "ready": not issues,
                    "issues": issues,
                    "config": config.metadata(),
                    "representation": "Wan896 full16->stride8 + native supervisedJEPA256,4096x1152/video",
                },
                indent=2,
            )
        )
        return int(bool(issues))
    if issues:
        raise ValueError("\n".join(issues))
    factory = type(
        "ConfiguredExtractor",
        (),
        {
            "from_local": staticmethod(
                partial(
                    CombinedExtractor.from_local,
                    jepa_model_path=config.path(config.jepa_model_path),
                    vjepa_root=config.path(config.vjepa_root),
                    artifact_dir=config.path(config.artifact_dir),
                )
            )
        },
    )
    run(
        config,
        resume=args.resume,
        evaluate=args.evaluate,
        extractor_class=factory,
        encoder_factory=CombinedEncoder,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
