"""Strict local weights and official multiclip aggregation; no downloads."""

import hashlib
import importlib
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path

import torch


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_target_encoder(model, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping) or "target_encoder" not in checkpoint:
        raise ValueError(
            "Expected official vitl.pt with a target_encoder state dictionary"
        )
    raw = checkpoint["target_encoder"]
    if not isinstance(raw, Mapping):
        raise TypeError("target_encoder must be a state dictionary")
    state = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise TypeError("Invalid encoder state entry")
        while key.startswith(("module.", "backbone.")):
            key = key.split(".", 1)[1]
        if key in state:
            raise ValueError(f"Duplicate normalized encoder key: {key}")
        state[key] = value
    expected = model.state_dict()
    missing, unexpected = (
        sorted(expected.keys() - state.keys()),
        sorted(state.keys() - expected.keys()),
    )
    mismatched = [
        k for k in expected.keys() & state.keys() if expected[k].shape != state[k].shape
    ]
    if missing or unexpected or mismatched:
        raise ValueError(
            f"Encoder weights incompatible: missing={missing}, unexpected={unexpected}, "
            f"shape_mismatch={mismatched}"
        )
    model.load_state_dict(state, strict=True)


class FrozenEncoder:
    def __init__(
        self, wrapper, *, device, extract_batch_size, metadata=None, extract_lanes=1
    ):
        if type(extract_lanes) is not int or extract_lanes not in (1, 2):
            raise ValueError("extract_lanes must be 1 or 2")
        self.device = torch.device(device)
        self.wrapper = wrapper.to(self.device).eval().requires_grad_(False)
        self.wrappers = [self.wrapper] + [
            deepcopy(self.wrapper) for _ in range(extract_lanes - 1)
        ]
        self.embed_dim = wrapper.embed_dim
        self.extract_batch_size = extract_batch_size
        self._metadata = metadata or {}
        self.streams = (
            [
                torch.cuda.Stream(device=self.device)
                if self.device.type == "cuda"
                else None
                for _ in range(extract_lanes)
            ]
            if extract_lanes > 1
            else []
        )
        self.workers = (
            [
                ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix=f"vjepa_lane_{lane}"
                )
                for lane in range(extract_lanes)
            ]
            if extract_lanes > 1
            else []
        )
        self.warmed = [set() for _ in self.workers]
        self.verify_parallel = False
        self.measure = False
        self.comparison = None
        self.last_extract_seconds = 0.0
        self.closed = False

    @classmethod
    def from_local(cls, config, device):
        # load_upstream must run first to establish and validate the local import root.
        root = config.path(config.vjepa_root)
        vit = importlib.import_module("src.models.vision_transformer")
        multiclip = importlib.import_module(
            "evals.video_classification_frozen.modelcustom.vit_encoder_multiclip"
        )
        for module in (vit, multiclip):
            if not Path(module.__file__).resolve().is_relative_to(root):
                raise RuntimeError(
                    "Encoder import resolved to a different V-JEPA2 checkout"
                )
        model = vit.vit_large(
            img_size=256,
            num_frames=16,
            patch_size=16,
            tubelet_size=2,
            uniform_power=True,
            use_rope=True,
        )
        weights = config.path(config.model_path)
        load_target_encoder(model, weights)
        wrapper = multiclip.ClipAggregation(
            model, tubelet_size=2, max_frames=128, use_pos_embed=False
        )
        sources = set((root / "src/models").rglob("*.py"))
        sources.update((root / "src/masks").glob("*.py"))
        sources.add(Path(multiclip.__file__).resolve())
        return cls(
            wrapper,
            device=device,
            extract_batch_size=config.extract_batch_size,
            extract_lanes=config.extract_lanes,
            metadata={
                "model": "V-JEPA2 ViT-L/16",
                "checkpoint_key": "target_encoder",
                "weights_sha256": file_sha256(weights),
                "encoder_parameters": sum(p.numel() for p in model.parameters()),
                "encoder_autocast": "bfloat16",
                "weights_dtype": "float32",
                "clip_tokens": [2048, 1024],
                "video_tokens": [4096, 1024],
                "use_rope": True,
                "aggregation_pos_embed": False,
                "sources": {
                    str(p.relative_to(root)): file_sha256(p) for p in sorted(sources)
                },
            },
        )

    def metadata(self):
        return self._metadata

    def _forward_chunk(self, wrapper, selected, start, count):
        inputs = [
            [x[start : start + count].to(self.device, non_blocking=True)]
            for x in selected
        ]
        with torch.autocast(
            self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"
        ):
            tokens = wrapper(inputs)[0]
        expected = (inputs[0][0].shape[0], 4096, self.embed_dim)
        if tokens.shape != expected:
            raise ValueError(
                f"Unexpected V-JEPA2 tokens: {tokens.shape}, expected {expected}"
            )
        return tokens

    def _serial(self, selected, chunk_size):
        return torch.cat(
            [
                self._forward_chunk(self.wrapper, selected, start, chunk_size)
                for start in range(0, len(selected[0]), chunk_size)
            ],
            dim=0,
        )

    def _run_lane(self, lane, selected, start, count):
        stream = self.streams[lane]
        if stream is not None:
            torch.cuda.set_device(self.device)
        with (
            torch.no_grad(),
            torch.cuda.stream(stream) if stream is not None else nullcontext(),
        ):
            tokens = self._forward_chunk(self.wrappers[lane], selected, start, count)
            if stream is not None:
                complete = torch.cuda.Event()
                complete.record(stream)
        if stream is not None:
            complete.synchronize()
        return tokens

    def _parallel(self, selected, chunk_size):
        consumer = (
            torch.cuda.current_stream(self.device)
            if self.device.type == "cuda"
            else None
        )
        if consumer is not None:
            for stream in self.streams:
                stream.wait_stream(consumer)
        assignments = [
            (index % len(self.workers), start)
            for index, start in enumerate(range(0, len(selected[0]), chunk_size))
        ]
        # Initialize library kernels serially on the same threads used for extraction.
        for lane, start in assignments:
            size = min(chunk_size, len(selected[0]) - start)
            if size not in self.warmed[lane]:
                self.workers[lane].submit(
                    self._run_lane, lane, selected, start, chunk_size
                ).result()
                self.warmed[lane].add(size)
        futures = [
            self.workers[lane].submit(self._run_lane, lane, selected, start, chunk_size)
            for lane, start in assignments
        ]
        wait(futures)
        outputs = [future.result() for future in futures]
        if consumer is not None:
            for tensor in outputs:
                tensor.record_stream(consumer)
        return torch.cat(outputs, dim=0)

    def _synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _verify(self, selected, chunk_size):
        reference = self._serial(selected, chunk_size)
        actual = self._parallel(selected, chunk_size)
        exact = torch.equal(reference, actual)
        self.comparison = {"exact": exact, "shape": list(actual.shape)}
        if not exact:
            self.comparison["max_abs"] = (
                (reference.float() - actual.float()).abs().max().item()
            )
            raise RuntimeError(
                f"Parallel extraction differs from serial: {self.comparison}"
            )
        del reference, actual
        for name, forward in (("serial", self._serial), ("parallel", self._parallel)):
            timings = []
            for _ in range(2):
                self._synchronize()
                start = time.perf_counter()
                tokens = forward(selected, chunk_size)
                self._synchronize()
                timings.append(time.perf_counter() - start)
                del tokens
            self.comparison[f"{name}_seconds"] = timings

    def close(self):
        if not self.closed:
            for worker in self.workers:
                worker.shutdown(wait=True, cancel_futures=True)
            for stream in self.streams:
                if stream is not None:
                    stream.synchronize()
            self.closed = True

    @torch.no_grad()
    def encode_view(self, batch, view):
        if self.closed:
            raise RuntimeError("Encoder is closed")
        clips = batch["clips"]
        if len(clips) != 2:
            raise ValueError("Expected two temporal segments")
        selected = [segment[view] for segment in clips]
        batch_size = selected[0].shape[0]
        for tensor in selected:
            if (
                tensor.shape != (batch_size, 3, 16, 256, 256)
                or not tensor.is_floating_point()
            ):
                raise ValueError(
                    "Expected normalized floating-point [B,3,16,256,256] inputs"
                )
        videos_per_chunk = self.extract_batch_size // len(clips)
        if videos_per_chunk < 1:
            raise ValueError("Extraction batch is smaller than the segment count")
        if self.verify_parallel and self.workers and self.comparison is None:
            self._verify(selected, videos_per_chunk)
        if self.measure:
            self._synchronize()
        start = time.perf_counter()
        tokens = (self._parallel if self.workers else self._serial)(
            selected, videos_per_chunk
        )
        if self.measure:
            self._synchronize()
            self.last_extract_seconds = time.perf_counter() - start
        return tokens
