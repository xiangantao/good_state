"""Asynchronously transfer captured features from model devices to CPU."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from queue import Queue
from typing import Protocol, Self

import torch
from torch import Tensor

from .capture import CapturedFeature, FeatureKind


@dataclass(frozen=True, slots=True)
class RecordedFeature:
    """An owned CPU feature ready for a storage backend."""

    kind: FeatureKind
    step: int
    layer: int
    head: int
    chunk: int
    tensor: Tensor


class RecordedFeatureSink(Protocol):
    """Synchronous downstream consumer, typically a storage backend."""

    def __call__(self, feature: RecordedFeature) -> None: ...


class FeatureSinkRouter:
    """Route processor captures to the recorder for the active chunk."""

    def __init__(self) -> None:
        self._sink: Callable[[CapturedFeature], None] | None = None

    def bind(self, sink: Callable[[CapturedFeature], None]) -> None:
        self._sink = sink

    def unbind(self) -> None:
        self._sink = None

    def __call__(self, feature: CapturedFeature) -> None:
        sink = self._sink
        if sink is None:
            raise RuntimeError("feature sink router is not bound")
        sink(feature)


@dataclass(slots=True)
class _PendingTransfer:
    feature: RecordedFeature
    event: torch.cuda.Event | None = None
    source: Tensor | None = None


class _StopWorker:
    pass


class FeatureRecorder:
    """Own captured tensors and deliver CPU copies on a worker thread."""

    def __init__(self, *, sink: RecordedFeatureSink, max_pending: int = 8) -> None:
        if not isinstance(max_pending, int) or isinstance(max_pending, bool):
            raise TypeError("max_pending must be an integer")
        if max_pending <= 0:
            raise ValueError("max_pending must be positive")

        self._sink = sink
        self._queue: Queue[_PendingTransfer | _StopWorker] = Queue()
        self._available_slots = threading.BoundedSemaphore(max_pending)
        self._state_lock = threading.Lock()
        self._failure: BaseException | None = None
        self._closed = False
        self._feature_shapes: dict[FeatureKind, tuple[int, ...]] = {}
        self._cuda_streams: dict[torch.device, torch.cuda.Stream] = {}
        self._worker = threading.Thread(
            target=self._run_worker,
            name="heft-feature-recorder",
            daemon=True,
        )
        self._worker.start()

    def __call__(self, feature: CapturedFeature) -> None:
        """Take ownership of one captured feature and enqueue it in order."""

        self._ensure_can_submit()
        self._record_shape(feature)
        self._available_slots.acquire()
        try:
            self._ensure_can_submit()
            pending = self._prepare_transfer(feature)
            self._queue.put(pending)
        except BaseException:
            self._available_slots.release()
            raise

    def feature_shape(self, kind: FeatureKind | str) -> tuple[int, ...]:
        """Return the captured per-head tensor shape for one feature kind."""

        normalized_kind = FeatureKind(kind)
        with self._state_lock:
            try:
                return self._feature_shapes[normalized_kind]
            except KeyError as error:
                raise KeyError(
                    f"no {normalized_kind.value} feature was captured"
                ) from error

    def flush(self) -> None:
        """Wait until every submitted record has reached the sink."""

        self._queue.join()
        self._raise_worker_failure()

    def close(self) -> None:
        """Flush pending records and stop the worker; safe to call repeatedly."""

        with self._state_lock:
            was_closed = self._closed
            self._closed = True
        if was_closed:
            self._raise_worker_failure()
            return

        failure: RuntimeError | None = None
        try:
            self.flush()
        except RuntimeError as error:
            failure = error
        finally:
            self._queue.put(_STOP_WORKER)
            self._worker.join()

        if failure is not None:
            raise failure
        self._raise_worker_failure()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        _traceback: object,
    ) -> None:
        if exc_value is None:
            self.close()
            return
        try:
            self.close()
        except RuntimeError as close_error:
            exc_value.add_note(
                f"FeatureRecorder also failed while closing: {close_error!r}"
            )

    def _prepare_transfer(self, captured: CapturedFeature) -> _PendingTransfer:
        source = captured.tensor.detach()
        if source.device.type == "cuda":
            return self._prepare_cuda_transfer(captured, source)

        owned = torch.empty(source.shape, dtype=source.dtype, device="cpu")
        owned.copy_(source)
        return _PendingTransfer(feature=_to_recorded_feature(captured, owned))

    def _prepare_cuda_transfer(
        self, captured: CapturedFeature, source: Tensor
    ) -> _PendingTransfer:
        device = source.device
        transfer_stream = self._cuda_streams.get(device)
        if transfer_stream is None:
            transfer_stream = torch.cuda.Stream(device=device)
            self._cuda_streams[device] = transfer_stream

        owned = torch.empty(
            source.shape, dtype=source.dtype, device="cpu", pin_memory=True
        )
        transfer_stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(transfer_stream):
            owned.copy_(source, non_blocking=True)
            source.record_stream(transfer_stream)
            event = torch.cuda.Event()
            event.record(transfer_stream)

        return _PendingTransfer(
            feature=_to_recorded_feature(captured, owned),
            event=event,
            source=source,
        )

    def _run_worker(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if isinstance(item, _StopWorker):
                    return
                if item.event is not None:
                    item.event.synchronize()
                if not self._has_worker_failure():
                    self._sink(item.feature)
            # The worker must keep draining queued items even for non-standard
            # sink failures, otherwise producers waiting on backpressure deadlock.
            except BaseException as error:  # noqa: BLE001
                self._set_worker_failure(error)
            finally:
                if isinstance(item, _PendingTransfer):
                    item.source = None
                    self._available_slots.release()
                self._queue.task_done()

    def _ensure_can_submit(self) -> None:
        with self._state_lock:
            if self._closed:
                raise RuntimeError("feature recorder is closed")
            failure = self._failure
        if failure is not None:
            raise RuntimeError("feature recorder worker failed") from failure

    def _record_shape(self, feature: CapturedFeature) -> None:
        shape = tuple(feature.tensor.shape)
        with self._state_lock:
            previous = self._feature_shapes.setdefault(feature.kind, shape)
        if previous != shape:
            raise ValueError(
                f"inconsistent {feature.kind.value} feature shape: {previous} and {shape}"
            )

    def _has_worker_failure(self) -> bool:
        with self._state_lock:
            return self._failure is not None

    def _set_worker_failure(self, error: BaseException) -> None:
        with self._state_lock:
            if self._failure is None:
                self._failure = error

    def _raise_worker_failure(self) -> None:
        with self._state_lock:
            failure = self._failure
        if failure is not None:
            raise RuntimeError("feature recorder worker failed") from failure


def _to_recorded_feature(captured: CapturedFeature, tensor: Tensor) -> RecordedFeature:
    return RecordedFeature(
        kind=captured.kind,
        step=captured.step,
        layer=captured.layer,
        head=captured.head,
        chunk=captured.chunk,
        tensor=tensor,
    )


_STOP_WORKER = _StopWorker()
