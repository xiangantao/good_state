"""Tests for moving captured features out of the model execution path."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import pytest
import torch

from heft.attn_hook import (
    CapturedFeature,
    FeatureKind,
    FeatureRecorder,
    FeatureSinkRouter,
    RecordedFeature,
)


@dataclass
class CollectingSink:
    records: list[RecordedFeature] = field(default_factory=list)
    thread_ids: list[int] = field(default_factory=list)

    def __call__(self, feature: RecordedFeature) -> None:
        self.records.append(feature)
        self.thread_ids.append(threading.get_ident())


def _captured(tensor: torch.Tensor, *, head: int = 2) -> CapturedFeature:
    return CapturedFeature(
        kind=FeatureKind.QUERY,
        step=7,
        layer=3,
        head=head,
        chunk=1,
        tensor=tensor,
    )


def test_recorder_preserves_metadata_dtype_value_and_owns_cpu_memory() -> None:
    sink = CollectingSink()
    source = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)

    with FeatureRecorder(sink=sink) as recorder:
        recorder(_captured(source))
        source.add_(10)

    assert len(sink.records) == 1
    recorded = sink.records[0]
    assert recorded.kind is FeatureKind.QUERY
    assert recorded.step == 7
    assert recorded.layer == 3
    assert recorded.head == 2
    assert recorded.chunk == 1
    assert recorded.tensor.device.type == "cpu"
    assert recorded.tensor.dtype is torch.bfloat16
    assert recorded.tensor.data_ptr() != source.data_ptr()
    torch.testing.assert_close(
        recorded.tensor, torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)
    )


def test_recorder_delivers_records_in_submission_order_on_worker_thread() -> None:
    sink = CollectingSink()
    producer_thread = threading.get_ident()

    with FeatureRecorder(sink=sink) as recorder:
        for head in (4, 1, 7):
            recorder(_captured(torch.tensor([head]), head=head))

    assert [feature.head for feature in sink.records] == [4, 1, 7]
    assert sink.thread_ids
    assert all(thread_id != producer_thread for thread_id in sink.thread_ids)


def test_recorder_materializes_a_contiguous_cpu_buffer() -> None:
    sink = CollectingSink()
    source = torch.arange(12).reshape(3, 4).transpose(0, 1)
    assert not source.is_contiguous()

    with FeatureRecorder(sink=sink) as recorder:
        recorder(_captured(source))

    assert sink.records[0].tensor.is_contiguous()
    torch.testing.assert_close(sink.records[0].tensor, source)


def test_recorder_applies_bounded_backpressure() -> None:
    sink_entered = threading.Event()
    release_sink = threading.Event()

    def blocking_sink(feature: RecordedFeature) -> None:
        del feature
        sink_entered.set()
        if not release_sink.wait(timeout=2):
            raise TimeoutError("test did not release the sink")

    recorder = FeatureRecorder(sink=blocking_sink, max_pending=1)
    recorder(_captured(torch.tensor([1])))
    assert sink_entered.wait(timeout=1)

    second_finished = threading.Event()

    def submit_second() -> None:
        recorder(_captured(torch.tensor([2])))
        second_finished.set()

    producer = threading.Thread(target=submit_second)
    producer.start()
    assert not second_finished.wait(timeout=0.05)

    release_sink.set()
    assert second_finished.wait(timeout=1)
    producer.join(timeout=1)
    recorder.close()


def test_recorder_surfaces_worker_failures() -> None:
    def failing_sink(feature: RecordedFeature) -> None:
        del feature
        raise ValueError("write failed")

    recorder = FeatureRecorder(sink=failing_sink)
    recorder(_captured(torch.tensor([1])))

    with pytest.raises(RuntimeError, match="feature recorder worker failed") as error:
        recorder.close()

    assert isinstance(error.value.__cause__, ValueError)


def test_recorder_rejects_submissions_after_close() -> None:
    recorder = FeatureRecorder(sink=CollectingSink())
    recorder.close()

    with pytest.raises(RuntimeError, match="closed"):
        recorder(_captured(torch.tensor([1])))


def test_sink_router_switches_recorders_between_chunks() -> None:
    first_sink = CollectingSink()
    second_sink = CollectingSink()
    router = FeatureSinkRouter()

    with FeatureRecorder(sink=first_sink) as first_recorder:
        router.bind(first_recorder)
        router(_captured(torch.tensor([1])))
        router.unbind()

    with FeatureRecorder(sink=second_sink) as second_recorder:
        router.bind(second_recorder)
        router(_captured(torch.tensor([2])))
        router.unbind()

    torch.testing.assert_close(first_sink.records[0].tensor, torch.tensor([1]))
    torch.testing.assert_close(second_sink.records[0].tensor, torch.tensor([2]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device is unavailable")
def test_cuda_recorder_uses_pinned_cpu_memory() -> None:
    sink = CollectingSink()
    source = (
        torch.arange(8, device="cuda", dtype=torch.float16)
        .reshape(2, 4)
        .transpose(0, 1)
    )
    assert not source.is_contiguous()
    expected = source.cpu()

    with FeatureRecorder(sink=sink) as recorder:
        recorder(_captured(source))

    recorded = sink.records[0].tensor
    assert recorded.device.type == "cpu"
    assert recorded.is_pinned()
    assert recorded.is_contiguous()
    torch.testing.assert_close(recorded, expected)
