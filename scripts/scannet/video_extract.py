"""Pooled block/QK capture from the same Wan video pipeline used by HEFT."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from heft.attn_hook import AttentionFeatureCapture, CaptureContext, CaptureSpec, FeatureKind
from heft.extraction.config import WAN_2_1, ExtractionTask, TailFramePolicy
from heft.extraction.pool import _plan_chunks
from heft.extraction.worker import _prepare_input_video

from .video_noise import video_scheduler


def video_jobs(images, scene, seed, chunk_size):
    # The local VAE encoder consumes one frame followed by groups of four.
    if chunk_size < 1 or (chunk_size - 1) % 4:
        raise ValueError("chunk_size must be 4n+1 for the local Wan VAE encoder")
    video = torch.from_numpy(np.stack([np.asarray(image) for image in images]))
    task = ExtractionTask(
        name=scene, input_video=video.permute(0, 3, 1, 2),
        output_dir=Path("."), seed=seed,
    )
    return _plan_chunks([task], chunk_size, tail_policy=TailFramePolicy.PAD)


def chunk_records(jobs, frame_ids):
    return [
        {
            "index": job.chunk, "start": job.start_frame, "stop": job.end_frame,
            "input_frames": job.pipeline_num_frames, "padding_frames": job.padding_frames,
            "frame_ids": frame_ids[job.start_frame:job.end_frame],
            "input_frame_ids": frame_ids[job.start_frame:job.end_frame]
            + [frame_ids[job.end_frame - 1]] * job.padding_frames,
            "seed": job.seed,
        }
        for job in jobs
    ]


def noise_protocol(model_path, mode="heft", timestep=49, shift=3.0):
    return video_scheduler(model_path, mode, timestep, shift)[1]


def pool_video_tokens(tensor, frames, spatial_size, grid_size, valid_frames):
    height, width = spatial_size
    if tensor.ndim != 3 or tensor.shape[:2] != (1, frames * height * width):
        raise ValueError(f"Expected one joint video token sequence, got {tensor.shape}")
    if not 0 < valid_frames <= frames:
        raise ValueError("Invalid unpadded video length")
    maps = tensor[0].reshape(frames, height, width, tensor.shape[-1])
    maps = maps[:valid_frames].permute(0, 3, 1, 2)
    return F.adaptive_avg_pool2d(maps.float(), grid_size).cpu()


class WanVideoExtractor:
    def __init__(self, model_path, device, noise=None):
        from diffusers import WanPipeline

        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Video extraction requires a local CUDA device")
        torch.cuda.set_device(self.device)
        self.noise = noise or noise_protocol(model_path)
        self.model = replace(
            WAN_2_1, model_id=str(Path(model_path).resolve()),
            num_inference_steps=self.noise["num_inference_steps"],
            start_step=self.noise["capture_step"],
        )
        self.pipeline = WanPipeline.from_pretrained(
            self.model.model_id, torch_dtype=self.model.dtype, local_files_only=True
        ).to(self.device)
        if self.noise["noise_mode"] != "heft":
            scheduler, actual_noise = video_scheduler(
                model_path, self.noise["noise_mode"], self.noise["requested_timestep"], self.noise["shift"]
            )
            if actual_noise != self.noise:
                raise ValueError("Video noise metadata differs from the configured scheduler")
            self.pipeline.register_modules(scheduler=scheduler)
        self.pipeline.set_progress_bar_config(disable=True)
        with torch.inference_mode():
            self.prompt_embeds, self.negative_prompt_embeds = self.pipeline.encode_prompt(
                prompt="", device=self.device, do_classifier_free_guidance=True
            )
        # Empty prompt embeddings are shared by all chunks, as in HEFT's cached mode.
        self.pipeline.register_modules(text_encoder=None, tokenizer=None)
        torch.cuda.empty_cache()
        self.versions = {"torch": torch.__version__}
        import diffusers

        self.versions["diffusers"] = diffusers.__version__

    @torch.inference_mode()
    def extract(self, job, blocks, heads, grid_size):
        model, pipe = self.model, self.pipeline
        spatial = tuple(a // b for a, b in zip(model.resolution, model.feature_spatial_stride))
        frames = job.pipeline_num_frames
        block_maps, head_maps, calls, observed = {}, {}, {}, []
        handles = []

        def pool(tensor):
            return pool_video_tokens(tensor, frames, spatial, grid_size, job.num_frames)

        def sink(feature):
            key = (feature.layer, feature.kind, feature.head)
            if key in head_maps:
                raise RuntimeError("Repeated conditional Q/K capture")
            head_maps[key] = pool(feature.tensor)

        def block_hook(block):
            def capture(module, inputs, tensor):
                calls[block] = calls.get(block, 0) + 1
                # Wan calls the conditional branch before the unconditional branch.
                if calls[block] == 1:
                    block_maps[block] = pool(tensor)
            return capture

        def observe(module, inputs, kwargs):
            shape = list(kwargs["hidden_states"].shape)
            if shape[0] != 1 or shape[2] != frames:
                raise ValueError(f"Video time dimension was not preserved: {shape}")
            timestep = kwargs["timestep"][0]
            scheduler = pipe.scheduler
            sigma = scheduler.sigmas[scheduler.index_for_timestep(timestep)]
            observed.append({
                "shape": shape, "timestep": float(timestep), "sigma": float(sigma),
                "sigma_in_latent_dtype": float(sigma.to(kwargs["hidden_states"].dtype)),
                "updates_before_forward": scheduler.lower_order_nums,
            })

        capture = AttentionFeatureCapture(
            spec=CaptureSpec(
                step=model.start_step, layers=blocks, heads=heads,
                features=(FeatureKind.QUERY, FeatureKind.KEY),
            ),
            context=CaptureContext(start_step=model.start_step), sink=sink,
        )
        with capture.attach(pipe.transformer.blocks, processor=model.processor) as session:
            try:
                handles.append(pipe.transformer.register_forward_pre_hook(observe, with_kwargs=True))
                for block in blocks:
                    handles.append(pipe.transformer.blocks[block].register_forward_hook(block_hook(block)))
                session.begin_chunk(chunk=job.chunk, start_step=model.start_step)
                pipe(
                    prompt=None, prompt_embeds=self.prompt_embeds,
                    negative_prompt_embeds=self.negative_prompt_embeds,
                    height=model.resolution[0], width=model.resolution[1], num_frames=frames,
                    num_inference_steps=model.num_inference_steps,
                    guidance_scale=model.guidance_scale, generator=torch.manual_seed(job.seed),
                    input_video=_prepare_input_video(job.input_video, model.resolution),
                    start_step=model.start_step, output_type="latent",
                )
            finally:
                for handle in handles:
                    handle.remove()
        expected_calls = 2 * (model.num_inference_steps - model.start_step)
        if len(observed) != expected_calls or any(calls.get(b) != expected_calls for b in blocks):
            raise RuntimeError("Unexpected Wan conditional/unconditional forward count")
        result = {}
        for block in blocks:
            result[block] = {"features": block_maps[block]}
            for kind in (FeatureKind.QUERY, FeatureKind.KEY):
                result[block][kind.value] = torch.stack(
                    [head_maps[block, kind, head] for head in heads], dim=1
                )
            if any(not torch.isfinite(tensor).all() for tensor in result[block].values()):
                raise ValueError("Non-finite video features")
        return result, {"transformer_forwards": observed, "valid_frames": job.num_frames}
