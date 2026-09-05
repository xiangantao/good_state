"""Merge completed per-layer OVIS reports without retaining feature tensors."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


RANKING_METRICS = {
    "instance_semantic": "instance_lift",
    "category_semantic": "category_lift",
    "position_control": "position_lift",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    return parser


def _rankings(heads: list[dict[str, Any]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    rankings: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for mode in ("qk", "kk"):
        rankings[mode] = {}
        for label, metric in RANKING_METRICS.items():
            rows = []
            for head in heads:
                if head["mode"] != mode or metric not in head["metrics"]:
                    continue
                summary = head["metrics"][metric]
                rows.append(
                    {
                        "layer": head["layer"],
                        "head": head["head"],
                        "mean": summary["mean"],
                        "median": summary["median"],
                        "top20_video_fraction": summary["top20_video_fraction"],
                    }
                )
            rows.sort(
                key=lambda row: (row["mean"], row["top20_video_fraction"]),
                reverse=True,
            )
            rankings[mode][label] = rows
    return rankings


def _write_csv(path: Path, heads: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "layer",
                "head",
                "mode",
                "samples",
                "metric",
                "mean",
                "median",
                "q25",
                "q75",
                "videos",
                "top20_video_fraction",
            ),
        )
        writer.writeheader()
        for head in heads:
            for metric, summary in sorted(head["metrics"].items()):
                writer.writerow(
                    {
                        "layer": head["layer"],
                        "head": head["head"],
                        "mode": head["mode"],
                        "samples": head["samples"],
                        "metric": metric,
                        **summary,
                    }
                )


def main() -> None:
    args = build_parser().parse_args()
    report_paths = sorted(args.input_dir.glob("layer_*.json"))
    if not report_paths:
        raise SystemExit(f"no layer reports found under {args.input_dir}")

    reports = [json.loads(path.read_text(encoding="utf-8")) for path in report_paths]
    heads = sorted(
        (head for report in reports for head in report["heads"]),
        key=lambda row: (row["layer"], row["head"], row["mode"]),
    )
    completed_layers = sorted({int(head["layer"]) for head in heads})
    first = reports[0]
    config = dict(first["config"])
    config.update(
        {
            "feature_root": "temporary per-layer features deleted after evaluation",
            "execution": "sequential layer extraction -> evaluation -> cleanup",
            "completed_layers": completed_layers,
            "layer_reports": [str(path) for path in report_paths],
        }
    )
    report = {
        "schema": first["schema"],
        "schema_version": first["schema_version"],
        "config": config,
        "evaluated_videos": first["evaluated_videos"],
        "heads": heads,
        "rankings": _rankings(heads),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_csv(args.csv, heads)
    print(
        f"merged {len(completed_layers)} layers / {len(heads)} layer-head-mode rows "
        f"-> {args.output}"
    )


if __name__ == "__main__":
    main()
