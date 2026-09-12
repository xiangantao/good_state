"""Batched, two-noise Wan extraction with a shared video VAE encoding."""

from __future__ import annotations

import copy
import hashlib
import inspect
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.pipelines.wan.pipeline_wan import WanPipeline
from diffusers.utils.torch_utils import randn_tensor
from torch import Tensor

import diffusers
from heft.attn_hook import (
    AttentionFeatureCapture,
    CaptureContext,
    CaptureSpec,
    FeatureKind,
)
from heft.extraction.config import WAN_2_1
from heft.extraction.worker import _prepare_input_video
from scripts.scannet.extract import selected_noise
from scripts.scannet.scan import model_identity
from scripts.scannet.video_extract import pool_video_tokens
from scripts.scannet.video_noise import video_scheduler

from .config import (
    CANDIDATES,
    HIDDEN_NAME,
    NOISE_SPECS,
    QK_TARGETS,
    ChannelSelection,
    ClipConfig,
)


def load_noise_branches(model_path: str | Path) -> dict:
    """Resolve and verify the two protocols against the existing scheduler code."""
    branches = {}
    for spec in NOISE_SPECS:
        scheduler, info = video_scheduler(
            model_path, spec.mode, spec.requested_timestep, spec.shift
        )
        if (
            info["actual_timestep"] != spec.actual_timestep
            or info["sigma"] != spec.sigma
        ):
            raise ValueError(
                f"Local scheduler does not reproduce the recorded {spec.name} protocol"
            )
        branches[spec.name] = (scheduler, info)
    return branches


def _tensor_hash(tensor: Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _code_identity(pipeline: WanPipeline, branches: dict) -> dict:
    sources = (
        WanLatentExtractor,
        ClipConfig,
        AttentionFeatureCapture,
        _prepare_input_video,
        video_scheduler,
        selected_noise,
        pool_video_tokens,
        model_identity,
        DiagonalGaussianDistribution,
        randn_tensor,
        type(pipeline),
        type(pipeline.vae),
        type(pipeline.transformer),
        type(pipeline.video_processor),
        *(type(scheduler) for scheduler, _ in branches.values()),
    )
    paths = {Path(inspect.getfile(source)).resolve() for source in sources}
    return {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }


def _require_workspace_runtime():
    root = Path(__file__).resolve().parents[1]
    expected = {
        WanPipeline: "diffusers/src/diffusers/pipelines/wan/pipeline_wan.py",
        AttentionFeatureCapture: "src/heft/attn_hook/capture.py",
        _prepare_input_video: "src/heft/extraction/worker.py",
        video_scheduler: "scripts/scannet/video_noise.py",
        pool_video_tokens: "scripts/scannet/video_extract.py",
    }
    for source, relative in expected.items():
        if Path(inspect.getfile(source)).resolve() != root / relative:
            raise RuntimeError(
                "Use this checkout's runtime: set PYTHONPATH to its src and diffusers/src directories"
            )


@dataclass
class ClipLatents:
    """Six BF16 tensors representing five candidates, plus JSON metadata."""

    tensors: dict[str, Tensor]
    metadata: dict

    def candidate(self, name: str) -> Tensor:
        parts = [self.tensors[key] for key in CANDIDATES[name]]
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)


class _CaptureComplete(Exception):
    """End the native Transformer forward once the complete B15 is available."""


class WanLatentExtractor:
    """Use from_local for the checkpoint, or inject prepared components for tests.

    Each call may batch independent clips. Calls on one instance remain sequential
    because it owns the VAE caches and attention capture state.
    """

    def __init__(
        self,
        pipeline: WanPipeline,
        prompt_embeds: Tensor,
        channels: ChannelSelection,
        branches: dict,
        *,
        config: ClipConfig | None = None,
        provenance: dict | None = None,
    ):
        self.config = config or ClipConfig()
        self.pipeline = pipeline
        self.channels = channels
        self.branches = {
            name: (scheduler, copy.deepcopy(info))
            for name, (scheduler, info) in branches.items()
        }
        if set(self.branches) != {spec.name for spec in NOISE_SPECS}:
            raise ValueError("Both recorded noise branches are required")
        for spec in NOISE_SPECS:
            info = self.branches[spec.name][1]
            if (info["actual_timestep"], info["sigma"], info["shift"]) != (
                spec.actual_timestep,
                spec.sigma,
                spec.shift,
            ):
                raise ValueError(f"Changed noise configuration for {spec.name}")
        pipeline.vae.eval().requires_grad_(False)
        pipeline.transformer.eval().requires_grad_(False)
        # Wan keeps modulation tables in FP32; ModelMixin.dtype skips those tables.
        self.device, self.dtype = (
            pipeline.transformer.device,
            pipeline.transformer.dtype,
        )
        if self.dtype not in (torch.float32, torch.bfloat16):
            raise ValueError(
                "Use BF16 production inference or FP32 reference components"
            )
        if pipeline.vae.dtype != self.dtype or pipeline.vae.device != self.device:
            raise ValueError("VAE and Transformer must share a device and dtype")
        if pipeline.vae_scale_factor_temporal != 1 or any(
            pipeline.vae.temperal_downsample
        ):
            raise ValueError(
                "The adapter requires the local VAE with temporal downsampling disabled"
            )
        if pipeline.vae.use_tiling:
            raise ValueError("The recorded VAE protocol uses untiled encoding")
        model = pipeline.transformer.config
        if model.patch_size[0] != 1 or len(pipeline.transformer.blocks) <= 15:
            raise ValueError("Require temporal patch size 1 and blocks through B15")
        if model.num_attention_heads <= 10:
            raise ValueError("The selected features require heads through H10")
        self.hidden_channels = model.num_attention_heads * model.attention_head_dim
        self.head_channels = model.attention_head_dim
        if max(channels.indices) >= self.hidden_channels:
            raise ValueError("The channel table exceeds the block's channel count")
        if model.in_channels != pipeline.vae.config.z_dim:
            raise ValueError("VAE and Transformer latent channel counts differ")
        self.spatial = tuple(
            size // pipeline.vae_scale_factor_spatial // patch
            for size, patch in zip(
                self.config.resolution, model.patch_size[1:], strict=True
            )
        )
        if (
            prompt_embeds.ndim != 3
            or prompt_embeds.shape[0] != 1
            or prompt_embeds.shape[2] != model.text_dim
        ):
            raise ValueError("Expected one precomputed empty-prompt embedding")
        self.prompt_embeds = prompt_embeds.detach().to(self.device, self.dtype)
        self._channel_indices = torch.tensor(channels.indices, device=self.device)
        if not torch.isfinite(self.prompt_embeds).all():
            raise ValueError("Non-finite prompt embedding")
        self.provenance = copy.deepcopy(provenance or {})
        self.provenance.update(
            code=_code_identity(pipeline, self.branches),
            versions={
                "torch": torch.__version__,
                "diffusers": diffusers.__version__,
                "opencv": cv2.__version__,
            },
            prompt_sha256=_tensor_hash(self.prompt_embeds),
        )

    @classmethod
    def from_local(
        cls,
        model_path: str | Path,
        *,
        channel_mask: str | Path,
        held_out: str | None = None,
        mask_key: str = "ablation_iterative_256",
        device: str = "cuda:0",
        config: ClipConfig | None = None,
    ) -> WanLatentExtractor:
        """Load only local Wan2.1-1.3B files; an explicit channel table is required."""
        _require_workspace_runtime()
        model_path = Path(model_path).resolve()
        if not model_path.is_dir():
            raise ValueError(
                "model_path must be an existing local checkpoint directory"
            )
        channels = ChannelSelection.load(channel_mask, held_out=held_out, key=mask_key)
        branches = load_noise_branches(model_path)
        device_obj = torch.device(device)
        if device_obj.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("The local checkpoint loader requires a CUDA device")
        torch.cuda.set_device(device_obj)
        pipeline = WanPipeline.from_pretrained(
            str(model_path), torch_dtype=torch.bfloat16, local_files_only=True
        ).to(device_obj)
        model = pipeline.transformer.config
        if (model.num_attention_heads, model.attention_head_dim, model.num_layers) != (
            12,
            128,
            30,
        ):
            raise ValueError("The recorded feature definitions require Wan2.1-T2V-1.3B")
        with torch.inference_mode():
            prompt_embeds, _ = pipeline.encode_prompt(
                prompt="", device=device_obj, do_classifier_free_guidance=False
            )
        pipeline.register_modules(text_encoder=None, tokenizer=None)
        pipeline.set_progress_bar_config(disable=True)
        torch.cuda.empty_cache()
        return cls(
            pipeline,
            prompt_embeds,
            channels,
            branches,
            config=config,
            provenance={"model": model_identity(model_path)},
        )

    def _prepare_posterior(
        self, frames: Tensor | Sequence[Tensor]
    ) -> tuple[DiagonalGaussianDistribution, int]:
        clips = [frames] if isinstance(frames, Tensor) else frames
        count = self.config.frames
        padding = (1 - count) % 4
        videos = []
        for clip in clips:
            prepared = _prepare_input_video(clip, self.config.resolution)
            assert prepared is not None
            if padding:
                prepared = torch.cat(
                    (prepared, prepared[-1:].expand(padding, -1, -1, -1))
                )
            videos.append(
                self.pipeline.video_processor.preprocess_video(
                    prepared,
                    height=self.config.resolution[0],
                    width=self.config.resolution[1],
                ).to(device=self.device, dtype=self.dtype)
            )
        video = videos[0] if len(videos) == 1 else torch.cat(videos)
        try:
            posterior = self.pipeline.vae.encode(video).latent_dist
            parameters = posterior.parameters
            if parameters.shape[:3] != (
                len(clips),
                2 * self.pipeline.vae.config.z_dim,
                count + padding,
            ):
                raise ValueError(
                    f"VAE did not preserve the video time dimension: {parameters.shape}"
                )
            # Crop the distribution before sampling so padding consumes no random draws.
            cropped = parameters[:, :, :count].contiguous()
            return DiagonalGaussianDistribution(cropped), padding
        finally:
            self.pipeline.vae.clear_cache()

    def replica(self) -> WanLatentExtractor:
        """Copy an idle, uncompiled extractor without reloading the text encoder."""
        return type(self)(
            copy.deepcopy(self.pipeline),
            self.prompt_embeds.clone(),
            self.channels,
            copy.deepcopy(self.branches),
            config=self.config,
            provenance=self.provenance,
        )

    def _clean_latents(
        self, posterior: DiagonalGaussianDistribution, generator: torch.Generator
    ) -> Tensor:
        latents = posterior.sample(generator=generator).to(self.dtype)
        config = self.pipeline.vae.config
        mean = (
            torch.tensor(config.latents_mean)
            .view(1, config.z_dim, 1, 1, 1)
            .to(latents.device, self.dtype)
        )
        std = (
            torch.tensor(config.latents_std)
            .view(1, config.z_dim, 1, 1, 1)
            .to(latents.device, self.dtype)
        )
        return (latents - mean) / std

    def _capture_branch(
        self, noisy: Tensor, timestep: Tensor, name: str, info: dict
    ) -> tuple[dict, dict]:
        batch_size = noisy.shape[0]
        targets = QK_TARGETS[name]
        outputs, observed, handles = {}, [], []

        def pool(tensor):
            height, width = self.spatial
            frames = self.config.frames
            if tensor.shape[:2] != (batch_size, frames * height * width):
                raise ValueError(f"Unexpected batched video tokens: {tensor.shape}")
            channels = tensor.shape[-1]
            maps = tensor.reshape(batch_size * frames, height, width, channels)
            pooled = F.adaptive_avg_pool2d(
                maps.permute(0, 3, 1, 2).float(), self.config.grid
            )
            return pooled.reshape(batch_size, frames, channels, *self.config.grid)

        def sink(feature):
            key = (feature.layer, feature.kind.value, feature.head)
            if key in targets:
                output_name = targets[key]
                if output_name in outputs:
                    raise RuntimeError(f"Duplicate feature capture: {output_name}")
                outputs[output_name] = (
                    pool(feature.tensor).to(torch.bfloat16).contiguous()
                )

        def block_hook(module, inputs, tensor):
            if HIDDEN_NAME in outputs:
                raise RuntimeError("Duplicate block output capture")
            if tensor.shape[-1] != self.hidden_channels:
                raise ValueError("Unexpected block output width")
            outputs[HIDDEN_NAME] = (
                pool(tensor.index_select(-1, self._channel_indices))
                .to(torch.bfloat16)
                .contiguous()
            )

        def observe(module, inputs, kwargs):
            shape = list(kwargs["hidden_states"].shape)
            actual = kwargs["timestep"].detach().cpu().tolist()
            if (
                shape != list(noisy.shape)
                or shape[2] != self.config.frames
                or actual != [info["actual_timestep"]] * batch_size
            ):
                raise ValueError("Unexpected Transformer input shape or timestep")
            observed.append({"shape": shape, "timestep": actual[0]})

        def stop_after_block(module, inputs, tensor):
            # Registered after the hidden hook: the full B15 output is already captured.
            raise _CaptureComplete

        capture = AttentionFeatureCapture(
            spec=CaptureSpec(
                step=info["capture_step"],
                layers={key[0] for key in targets},
                heads={key[2] for key in targets},
                features=(FeatureKind.QUERY, FeatureKind.KEY),
            ),
            context=CaptureContext(start_step=info["capture_step"]),
            sink=sink,
        )
        with capture.attach(
            self.pipeline.transformer.blocks, processor=WAN_2_1.processor
        ) as session:
            try:
                handles.append(
                    self.pipeline.transformer.register_forward_pre_hook(
                        observe, with_kwargs=True
                    )
                )
                if name == "t299":
                    handles.append(
                        self.pipeline.transformer.blocks[15].register_forward_hook(
                            block_hook
                        )
                    )
                if self.config.stop_after_capture:
                    handles.append(
                        self.pipeline.transformer.blocks[15].register_forward_hook(
                            stop_after_block
                        )
                    )
                session.begin_chunk(chunk=0, start_step=info["capture_step"])
                try:
                    self.pipeline.transformer(
                        hidden_states=noisy,
                        timestep=timestep,
                        encoder_hidden_states=self.prompt_embeds.expand(
                            batch_size, -1, -1
                        ),
                        return_dict=False,
                    )
                except _CaptureComplete:
                    if not self.config.stop_after_capture:
                        raise
            finally:
                for handle in handles:
                    handle.remove()
        expected = set(targets.values()) | ({HIDDEN_NAME} if name == "t299" else set())
        if set(outputs) != expected or len(observed) != 1:
            raise RuntimeError(
                "Incomplete conditional feature capture; verify the local Wan runtime"
            )
        observed[0].update(
            stopped_after_block=15 if self.config.stop_after_capture else None,
            executed_blocks=16
            if self.config.stop_after_capture
            else len(self.pipeline.transformer.blocks),
        )
        return outputs, observed[0]

    @torch.inference_mode()
    def extract(
        self,
        frames: Tensor,
        *,
        seed: int = 42,
        frame_ids: list[int] | tuple[int, ...] | None = None,
        clip_id: str | None = None,
        output_device: str | torch.device = "cpu",
    ) -> ClipLatents:
        """Extract one CPU uint8 RGB clip [T,3,H,W]; return CPU BF16 by default."""
        return self.extract_batch(
            [frames],
            seeds=[seed],
            frame_ids=[frame_ids],
            clip_ids=[clip_id],
            output_device=output_device,
        )[0]

    @torch.inference_mode()
    def extract_batch(
        self,
        frames: Sequence[Tensor],
        *,
        seeds: Sequence[int] | None = None,
        frame_ids: Sequence[Sequence[int] | None] | None = None,
        clip_ids: Sequence[str | None] | None = None,
        output_device: str | torch.device = "cpu",
    ) -> list[ClipLatents]:
        """Batch independent clips; each seed consumes the same draws as extract()."""
        frames = list(frames)
        batch_size = len(frames)
        if not batch_size:
            raise ValueError("Require at least one clip")
        seeds = [42] * batch_size if seeds is None else list(seeds)
        frame_ids = [None] * batch_size if frame_ids is None else list(frame_ids)
        clip_ids = [None] * batch_size if clip_ids is None else list(clip_ids)
        if any(len(items) != batch_size for items in (seeds, frame_ids, clip_ids)):
            raise ValueError("seeds, frame_ids, and clip_ids must match the clip count")
        destination = torch.device(output_device)
        if destination.type == self.device.type and destination.index is None:
            destination = self.device
        if destination.type != "cpu" and destination != self.device:
            raise ValueError("output_device must be CPU or the extractor device")
        for clip, seed, indices, identifier in zip(
            frames, seeds, frame_ids, clip_ids, strict=True
        ):
            if (
                not isinstance(clip, Tensor)
                or clip.ndim != 4
                or clip.shape[:2] != (self.config.frames, 3)
                or min(clip.shape[-2:]) < 1
                or clip.device.type != "cpu"
                or clip.dtype != torch.uint8
            ):
                raise ValueError(
                    f"Expected CPU uint8 RGB frames [{self.config.frames},3,H,W]"
                )
            if type(seed) is not int or not 0 <= seed < 2**63:
                raise ValueError("seed must be an integer in [0,2**63)")
            if indices is not None and (
                len(indices) != self.config.frames
                or any(type(i) is not int or i < 0 for i in indices)
            ):
                raise ValueError(
                    "frame_ids must contain one non-negative integer per input frame"
                )
            if identifier is not None and (
                not isinstance(identifier, str) or not identifier
            ):
                raise ValueError("clip_id must be a non-empty string when supplied")
        posterior, padding = self._prepare_posterior(frames)
        clean_parts, noise_parts = [], []
        # Preserve per-clip CPU RNG shape/order, including posterior sampling first.
        for sample, seed in enumerate(seeds):
            generator = torch.Generator(device="cpu").manual_seed(seed)
            distribution = DiagonalGaussianDistribution(
                posterior.parameters[sample : sample + 1]
            )
            latent = self._clean_latents(distribution, generator)
            clean_parts.append(latent)
            noise_parts.append(
                randn_tensor(
                    latent.shape,
                    generator=generator,
                    device=self.device,
                    dtype=self.dtype,
                )
            )
        clean, noise = torch.cat(clean_parts), torch.cat(noise_parts)
        if not torch.isfinite(clean).all():
            raise ValueError("Non-finite VAE latent")
        tensors, branch_records = {}, {}
        for spec in NOISE_SPECS:
            scheduler, info = self.branches[spec.name]
            scheduler.set_timesteps(info["num_inference_steps"], device=self.device)
            index = info["capture_step"]
            timestep = scheduler.timesteps[index : index + 1]
            sigma = scheduler.sigmas[index]
            if (
                float(timestep[0]) != spec.actual_timestep
                or float(sigma) != spec.sigma
                or scheduler.lower_order_nums != 0
                or scheduler.step_index is not None
            ):
                raise ValueError(
                    f"Scheduler did not reset to the recorded {spec.name} noise point"
                )
            noisy = scheduler.add_noise(clean, noise, timestep)
            outputs, forward = self._capture_branch(
                noisy, timestep.expand(batch_size), spec.name, info
            )
            tensors.update(outputs)
            branch_records[spec.name] = {
                **copy.deepcopy(info),
                "sigma_in_latent_dtype": float(sigma.to(self.dtype)),
                "updates_before_capture": 0,
                "conditional_forwards": 1,
                "unconditional_forwards": 0,
                "scheduler_updates_executed": 0,
                "forward": forward,
            }
        for name, tensor in tensors.items():
            width = 256 if name == HIDDEN_NAME else self.head_channels
            if (
                tensor.shape
                != (batch_size, self.config.frames, width, *self.config.grid)
                or not torch.isfinite(tensor).all()
            ):
                raise ValueError(f"Invalid output feature: {name}")
        tensors = {name: value.to(destination) for name, value in tensors.items()}
        metadata = {
            "schema": "heft.latent_video.clip",
            "schema_version": 1,
            "config": asdict(self.config),
            "vae_input_frames": self.config.frames + padding,
            "padding_frames": padding,
            "padding": "repeat final RGB frame for VAE only; crop posterior before sampling",
            "vae_encodes": 1,
            "latent_distribution": "sample",
            "shared_clean_latent": True,
            "shared_gaussian_noise": True,
            "latent_shape": [1, *clean.shape[1:]],
            "extraction_batch_size": batch_size,
            "inference_dtype": str(self.dtype),
            "storage_dtype": "torch.bfloat16",
            "device": str(self.device),
            "output_device": str(destination),
            "qk_capture": "post-normalization, post-RoPE",
            "hidden_capture": "complete block 15 output",
            "extra_position_encoding": False,
            "channel_selection": self.channels.metadata(),
            "noise_branches": branch_records,
            "candidates": {name: list(parts) for name, parts in CANDIDATES.items()},
            "tensor_shapes": {
                name: list(tensor.shape[1:]) for name, tensor in tensors.items()
            },
            "provenance": copy.deepcopy(self.provenance),
        }
        results = []
        for sample, (clip, seed, indices, identifier) in enumerate(
            zip(frames, seeds, frame_ids, clip_ids, strict=True)
        ):
            record = copy.deepcopy(metadata)
            record.update(
                clip_id=identifier,
                seed=seed,
                frame_ids=None if indices is None else list(indices),
                input_shape=list(clip.shape),
                input_sha256=_tensor_hash(clip),
                batch_index=sample,
            )
            results.append(
                ClipLatents(
                    {name: value[sample] for name, value in tensors.items()}, record
                )
            )
        return results
