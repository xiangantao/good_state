"""Local-files-only Wan block extraction, with per-frame VAE encoding."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def offline_environment(workspace):
    workspace = Path(workspace)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    for key, rel in {
        "HF_HOME": "cache/hf",
        "XDG_CACHE_HOME": "cache/xdg",
        "TORCHINDUCTOR_CACHE_DIR": "cache/inductor",
        "TRITON_CACHE_DIR": "cache/triton",
        "CUDA_CACHE_PATH": "cache/nv",
        "TMPDIR": "tmp",
    }.items():
        path = workspace / rel
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)


def selected_noise(timestep, shift=5.0):
    """Reproduce VEGA FlowUniPC's 1000-step schedule and integer nearest lookup.

    The actual sigma is selected BEFORE integer truncation of the timestep.
    This is not the checkpoint's default flow_shift=3 inference schedule.
    """
    if not 0 <= timestep <= 1000 or shift <= 0:
        raise ValueError("Require timestep in [0,1000] and positive shift")
    sigma_max = float(np.float32(0.999))
    sigmas = np.linspace(sigma_max, 0.0, 1001)[:-1]
    sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
    steps = (sigmas * 1000).astype(np.int64)
    index = int(np.argmin(np.abs(steps - timestep)))
    return {
        "requested_timestep": int(timestep),
        "actual_timestep": int(steps[index]),
        "sigma": float(np.float32(sigmas[index])),
        "schedule_index": index,
        "shift": shift,
        "schedule": "VEGA FlowUniPC 1000-step integer lookup",
    }


def frame_seed(seed, scene, frame):
    key = f"{seed}:{scene}:{frame}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little") % (2**63 - 1)


class WanExtractor:
    def __init__(self, model_path, device="cuda:0", dtype="bf16"):
        model_path = Path(model_path).resolve()
        if not model_path.is_dir():
            raise ValueError("Checkpoint must be an existing local directory")
        self.device = torch.device(device)
        self.dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }[dtype]
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Wan extraction requires a working local CUDA runtime")
        from transformers import T5TokenizerFast, UMT5EncoderModel

        from diffusers import AutoencoderKLWan, WanTransformer3DModel

        print("正在加载本地 Wan 和空文本条件。", flush=True)
        tokenizer = T5TokenizerFast.from_pretrained(
            model_path / "tokenizer", local_files_only=True
        )
        text_encoder = (
            UMT5EncoderModel.from_pretrained(
                model_path / "text_encoder",
                torch_dtype=self.dtype,
                local_files_only=True,
            )
            .eval()
            .requires_grad_(False)
            .to(self.device)
        )
        tokens = tokenizer(
            [""],
            padding="max_length",
            max_length=512,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            context = text_encoder(
                tokens.input_ids, tokens.attention_mask
            ).last_hidden_state
            length = int(tokens.attention_mask.sum())
            context[:, length:] = 0
            self.context = context.to(self.dtype).detach()
        del text_encoder, tokenizer, tokens
        torch.cuda.empty_cache()
        # Retain HEFT's intentional temporal-convolution disablement.
        # Original checkpoint temporal weights are consequently unused.
        self.vae = (
            AutoencoderKLWan.from_pretrained(
                model_path / "vae", torch_dtype=torch.float32, local_files_only=True
            )
            .eval()
            .requires_grad_(False)
            .to(self.device)
        )
        self.transformer = (
            WanTransformer3DModel.from_pretrained(
                model_path / "transformer",
                torch_dtype=self.dtype,
                local_files_only=True,
            )
            .eval()
            .requires_grad_(False)
            .to(self.device)
        )
        self.latent_mean = torch.tensor(
            self.vae.config.latents_mean, device=self.device
        ).view(1, -1, 1, 1, 1)
        self.latent_std = torch.tensor(
            self.vae.config.latents_std, device=self.device
        ).view(1, -1, 1, 1, 1)
        self.versions = {"torch": torch.__version__}
        import diffusers

        self.versions["diffusers"] = diffusers.__version__

    @torch.inference_mode()
    def latents(self, images):
        # Single-frame encoding matches the VEGA Wan path; no temporal VAE mixing.
        results = []
        for image in images:
            x = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)
            x = (x.to(self.device, torch.float32) / 127.5 - 1)[None, :, None]
            z = self.vae.encode(x).latent_dist.mode()
            z = (z - self.latent_mean) / self.latent_std
            results.append(z.cpu())
        return torch.cat(results)

    @torch.inference_mode()
    def blocks(
        self,
        latents,
        blocks,
        timestep,
        shift,
        seed,
        scene,
        frame_ids,
        batch_size=1,
        grid_size=(14, 14),
        save_prepool=False,
    ):
        blocks = sorted(set(blocks))
        if not blocks or min(blocks) < 0 or max(blocks) >= len(self.transformer.blocks):
            raise ValueError("Invalid zero-based block indices")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        noise_info = selected_noise(timestep, shift)
        output = {block: [] for block in blocks}
        prepool = {block: [] for block in blocks} if save_prepool else {}
        handles = []
        patch = self.transformer.config.patch_size
        temporal = latents.shape[2] // patch[0]
        gh, gw = latents.shape[-2] // patch[1], latents.shape[-1] // patch[2]
        if temporal != 1:
            raise ValueError("Spatial probe requires one latent frame per observation")

        def hook_for(block):
            def capture(module, inputs, tensor):
                if tensor.shape[1] != gh * gw:
                    raise ValueError("Unexpected block token grid")
                feature = tensor.transpose(1, 2).reshape(-1, tensor.shape[-1], gh, gw)
                output[block].append(
                    F.adaptive_avg_pool2d(feature.float(), grid_size).cpu()
                )
                if save_prepool:
                    prepool[block].append(feature.to(torch.bfloat16).cpu())

            return capture

        try:
            for block in blocks:
                handles.append(
                    self.transformer.blocks[block].register_forward_hook(
                        hook_for(block)
                    )
                )
            for start in range(0, len(latents), batch_size):
                end = min(start + batch_size, len(latents))
                z = latents[start:end].to(self.device, self.dtype)
                # CPU noise is keyed per original frame; batch size does not change it.
                noise = torch.stack(
                    [
                        torch.randn(
                            latents.shape[1:],
                            generator=torch.Generator().manual_seed(
                                frame_seed(seed, scene, frame_ids[i])
                            ),
                            dtype=torch.float32,
                        )
                        for i in range(start, end)
                    ]
                ).to(self.device, self.dtype)
                sigma = noise_info["sigma"]
                noisy = (1 - sigma) * z + sigma * noise
                times = torch.full(
                    (end - start,),
                    noise_info["actual_timestep"],
                    dtype=torch.long,
                    device=self.device,
                )
                self.transformer(
                    hidden_states=noisy,
                    timestep=times,
                    encoder_hidden_states=self.context.expand(end - start, -1, -1),
                    return_dict=False,
                )
                print(
                    f"{scene}：已提取 {end}/{len(latents)} 帧，blocks={blocks}。",
                    flush=True,
                )
        finally:
            for handle in handles:
                handle.remove()
        return (
            {b: torch.cat(values) for b, values in output.items()},
            {b: torch.cat(values) for b, values in prepool.items()},
            noise_info,
        )
