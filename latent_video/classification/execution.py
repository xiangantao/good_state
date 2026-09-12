"""Fixed-thread extraction lanes with independent model and CUDA stream state."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import nullcontext

import torch

COMPILED_BLOCKS = (*range(13), 14)


def compile_uncaptured_blocks(extractor):
    for index in COMPILED_BLOCKS:
        block = extractor.pipeline.transformer.blocks[index]
        block.forward = torch.compile(block.forward, fullgraph=True, dynamic=False)


class ExtractionPool:
    def __init__(self, extractors, *, compile_blocks: bool = False):
        self.extractors = list(extractors)
        if not self.extractors or len({id(e) for e in self.extractors}) != len(
            self.extractors
        ):
            raise ValueError("Each extraction lane needs a distinct extractor")
        self.device = self.extractors[0].device
        if any(e.device != self.device for e in self.extractors):
            raise ValueError("Extraction lanes must share one device")
        if all(hasattr(e, "pipeline") for e in self.extractors):
            for name in ("vae", "transformer"):
                if len({id(getattr(e.pipeline, name)) for e in self.extractors}) != len(
                    self.extractors
                ):
                    raise ValueError(f"Extraction lanes must not share a {name}")
        self.compiled = compile_blocks
        if compile_blocks:
            if self.device.type != "cuda":
                raise ValueError("Compiled extraction requires CUDA")
            for extractor in self.extractors:
                compile_uncaptured_blocks(extractor)
        self.streams = [
            torch.cuda.Stream(device=self.device)
            if self.device.type == "cuda"
            else None
            for _ in self.extractors
        ]
        self.workers = [
            ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"wan_lane_{lane}")
            for lane in range(len(self.extractors))
        ]
        self.warmed = [set() for _ in self.extractors]
        self.closed = False

    def _run(self, lane, request):
        stream = self.streams[lane]
        if stream is not None:
            torch.cuda.set_device(self.device)
        with (
            torch.inference_mode(),
            torch.autocast(self.device.type, enabled=False),
            torch.cuda.stream(stream) if stream is not None else nullcontext(),
        ):
            result = self.extractors[lane].extract_batch(**request)
            if stream is not None:
                complete = torch.cuda.Event()
                complete.record(stream)
        if stream is not None:
            complete.synchronize()
        return result

    def extract_groups(self, requests, *, concurrent: bool = True):
        if self.closed:
            raise RuntimeError("Extraction pool is closed")
        consumer = (
            torch.cuda.current_stream(self.device)
            if self.device.type == "cuda"
            else None
        )
        if consumer is not None:
            for stream in self.streams:
                stream.wait_stream(consumer)
        assignments = [
            (index % len(self.workers), request)
            for index, request in enumerate(requests)
        ]
        # Compile/autotune new sizes serially, on the same thread used for real work.
        for lane, request in assignments:
            size = len(request["frames"])
            if size not in self.warmed[lane]:
                self.workers[lane].submit(self._run, lane, request).result()
                self.warmed[lane].add(size)
        futures = []
        for lane, request in assignments:
            future = self.workers[lane].submit(self._run, lane, request)
            futures.append(future)
            if not concurrent:
                future.result()
        # Drain all lanes before surfacing a failure or releasing their model state.
        wait(futures)
        results = [future.result() for future in futures]
        if consumer is not None:
            for group in results:
                for result in group:
                    for tensor in result.tensors.values():
                        tensor.record_stream(consumer)
        return results

    def metadata(self):
        return {
            "lanes": len(self.extractors),
            "compiled_blocks": list(COMPILED_BLOCKS) if self.compiled else [],
            "compile_fullgraph": self.compiled,
            "compile_dynamic": False,
            "vae_layout": "native",
            "warmup": "serial per lane and batch size on fixed worker threads",
        }

    def close(self):
        if not self.closed:
            for worker in self.workers:
                worker.shutdown(wait=True, cancel_futures=True)
            for stream in self.streams:
                if stream is not None:
                    stream.synchronize()
            self.closed = True
