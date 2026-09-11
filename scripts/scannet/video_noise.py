"""Checkpoint noise or one exact legacy noise point for video feature probes."""

from pathlib import Path

import torch
from diffusers import UniPCMultistepScheduler

from heft.extraction.config import WAN_2_1

from .extract import selected_noise


class SinglePointUniPCScheduler(UniPCMultistepScheduler):
    """Keep pipeline encoding/add_noise unchanged and evaluate one noise point."""

    def set_timesteps(self, num_inference_steps, device=None):
        if num_inference_steps != 1:
            raise ValueError("An exact noise probe requires one active step")
        super().set_timesteps(num_inference_steps, device)
        self.timesteps = torch.tensor([self.probe_timestep], dtype=torch.int64, device=device)
        self.sigmas = torch.tensor([self.probe_sigma, 0.0], dtype=torch.float32)


def video_scheduler(model_path, mode="heft", timestep=49, shift=3.0):
    if mode not in {"heft", "legacy"}:
        raise ValueError(f"Unknown video noise mode: {mode}")
    if mode == "heft":
        if timestep != WAN_2_1.start_step or shift != 3.0:
            raise ValueError("HEFT noise uses capture step 49 and checkpoint shift 3")
        scheduler = UniPCMultistepScheduler.from_pretrained(
            Path(model_path) / "scheduler", local_files_only=True
        )
        if not scheduler.config.use_flow_sigmas or scheduler.config.flow_shift != shift:
            raise ValueError("Checkpoint flow schedule differs from HEFT's recorded settings")
        scheduler.set_timesteps(WAN_2_1.num_inference_steps)
        info = {
            "requested_timestep": timestep, "timestep_kind": "pipeline_step_index",
            "actual_timestep": float(scheduler.timesteps[timestep]),
            "sigma": float(scheduler.sigmas[timestep]),
            "schedule": type(scheduler).__name__, "shift": shift,
            "num_inference_steps": WAN_2_1.num_inference_steps, "capture_step": timestep,
        }
    else:
        source = selected_noise(timestep, shift)
        scheduler = SinglePointUniPCScheduler.from_pretrained(
            Path(model_path) / "scheduler", flow_shift=shift, local_files_only=True
        )
        if not scheduler.config.use_flow_sigmas or scheduler.config.prediction_type != "flow_prediction":
            raise ValueError("The exact noise probe requires a flow-prediction scheduler")
        scheduler.probe_timestep = source["actual_timestep"]
        scheduler.probe_sigma = source["sigma"]
        scheduler.set_timesteps(1)
        info = {
            **source, "timestep_kind": "legacy_requested_timestep",
            "schedule": type(scheduler).__name__, "source_schedule": source["schedule"],
            "source_num_inference_steps": 1000, "num_inference_steps": 1, "capture_step": 0,
        }
    info.update(noise_mode=mode, active_denoising_steps=1, updates_before_capture=0)
    return scheduler, info
