"""Fast, grouped safetensors storage for recorded attention features."""

from __future__ import annotations

import os
import tempfile
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor

from .capture import FeatureKind
from .recorder import RecordedFeature


@dataclass(frozen=True, slots=True)
class _GroupKey:
    chunk: int
    step: int
    layer: int
    kind: FeatureKind


class SafetensorsFeatureStorage:
    """Store one safetensors file per chunk, step, layer, and feature kind."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        expected_heads: Iterable[int],
        overwrite: bool = False,
    ) -> None:
        self.root = Path(root)
        self.expected_heads = _normalize_heads(expected_heads)
        self.overwrite = overwrite
        self._expected_head_set = frozenset(self.expected_heads)
        self._pending: dict[_GroupKey, dict[int, Tensor]] = {}
        self._completed: set[_GroupKey] = set()
        self._closed = False
        self._lock = threading.RLock()

    def __call__(self, feature: RecordedFeature) -> None:
        """Buffer one head and atomically write its group when complete."""

        with self._lock:
            self._ensure_open()
            _validate_feature(feature)
            if feature.head not in self._expected_head_set:
                raise ValueError(
                    f"unexpected head {feature.head}; expected one of {list(self.expected_heads)}"
                )

            key = _GroupKey(
                chunk=feature.chunk,
                step=feature.step,
                layer=feature.layer,
                kind=FeatureKind(feature.kind),
            )
            if key in self._completed:
                raise ValueError(
                    f"feature group {self.path_for_key(key)} has already been written"
                )

            tensors = self._pending.setdefault(key, {})
            if feature.head in tensors:
                raise ValueError(
                    f"duplicate head {feature.head} for feature group {self.path_for_key(key)}"
                )
            _validate_group_tensor(tensors, feature.tensor)
            tensors[feature.head] = feature.tensor

            if tensors.keys() >= self._expected_head_set:
                self._commit(key, tensors)

    def path_for(
        self,
        *,
        chunk: int,
        step: int,
        layer: int,
        kind: FeatureKind | str,
    ) -> Path:
        """Return the canonical file path for a feature group."""

        _validate_index("chunk", chunk)
        _validate_index("step", step)
        _validate_index("layer", layer)
        return self.path_for_key(
            _GroupKey(
                chunk=chunk,
                step=step,
                layer=layer,
                kind=FeatureKind(kind),
            )
        )

    def path_for_key(self, key: _GroupKey) -> Path:
        return (
            self.root
            / f"chunk_{key.chunk:03d}"
            / f"step_{key.step:03d}"
            / f"layer_{key.layer:03d}"
            / f"{key.kind.value}.safetensors"
        )

    def load(
        self,
        *,
        chunk: int,
        step: int,
        layer: int,
        kind: FeatureKind | str,
        heads: Iterable[int] | None = None,
    ) -> dict[int, Tensor]:
        """Load only the requested heads from one feature group."""

        normalized_kind = FeatureKind(kind)
        path = self.path_for(chunk=chunk, step=step, layer=layer, kind=normalized_kind)
        requested_heads = None if heads is None else _normalize_heads(heads)

        with self._lock:
            if not path.is_file():
                raise FileNotFoundError(path)
            with safe_open(path, framework="pt", device="cpu") as file:
                metadata = file.metadata() or {}
                _validate_metadata(
                    metadata,
                    chunk=chunk,
                    step=step,
                    layer=layer,
                    kind=normalized_kind,
                    path=path,
                )
                tensor_names = file.keys()
                available = {_head_from_tensor_key(name): name for name in tensor_names}
                selected = (
                    tuple(sorted(available))
                    if requested_heads is None
                    else requested_heads
                )
                missing = [head for head in selected if head not in available]
                if missing:
                    missing_text = ", ".join(f"head {head}" for head in missing)
                    raise KeyError(f"{missing_text} is missing from {path}")
                return {head: file.get_tensor(available[head]) for head in selected}

    def load_head(
        self,
        *,
        chunk: int,
        step: int,
        layer: int,
        kind: FeatureKind | str,
        head: int,
    ) -> Tensor:
        """Load one head without materializing the other tensors in its file."""

        return self.load(
            chunk=chunk,
            step=step,
            layer=layer,
            kind=kind,
            heads=[head],
        )[head]

    def flush(self) -> None:
        """Verify that every received feature group has been committed."""

        with self._lock:
            self._ensure_open()
            for key, tensors in tuple(self._pending.items()):
                if tensors.keys() >= self._expected_head_set:
                    self._commit(key, tensors)
            if self._pending:
                key, tensors = next(iter(self._pending.items()))
                missing = sorted(self._expected_head_set.difference(tensors))
                raise RuntimeError(
                    f"incomplete feature group {self.path_for_key(key)}; missing heads {missing}"
                )

    def close(self) -> None:
        """Commit complete groups and reject incomplete groups; idempotent."""

        with self._lock:
            if self._closed:
                return
            self.flush()
            self._closed = True

    def __enter__(self) -> Self:
        self._ensure_open()
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
        except (OSError, RuntimeError, ValueError) as close_error:
            exc_value.add_note(
                f"SafetensorsFeatureStorage also failed while closing: {close_error!r}"
            )

    def _commit(self, key: _GroupKey, tensors: dict[int, Tensor]) -> None:
        path = self.path_for_key(key)
        if path.exists() and not self.overwrite:
            raise FileExistsError(f"feature file already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

        tensor_map = {f"head_{head:03d}": tensors[head] for head in self.expected_heads}
        metadata = {
            "schema": _SCHEMA,
            "schema_version": _SCHEMA_VERSION,
            "chunk": str(key.chunk),
            "step": str(key.step),
            "layer": str(key.layer),
            "kind": key.kind.value,
            "heads": ",".join(str(head) for head in self.expected_heads),
        }
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        try:
            save_file(tensor_map, temporary_path, metadata=metadata)
            os.replace(temporary_path, path)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise

        del self._pending[key]
        self._completed.add(key)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("feature storage is closed")


def _normalize_heads(heads: Iterable[int]) -> tuple[int, ...]:
    normalized = tuple(sorted(set(heads)))
    if not normalized:
        raise ValueError("expected_heads must not be empty")
    for head in normalized:
        _validate_index("head", head)
    return normalized


def _validate_feature(feature: RecordedFeature) -> None:
    _validate_index("chunk", feature.chunk)
    _validate_index("step", feature.step)
    _validate_index("layer", feature.layer)
    _validate_index("head", feature.head)
    if feature.tensor.device.type != "cpu":
        raise ValueError("storage accepts only CPU tensors from FeatureRecorder")
    if feature.tensor.layout is not torch.strided:
        raise ValueError("storage accepts only strided tensors")
    if not feature.tensor.is_contiguous():
        raise ValueError("storage requires contiguous tensors from FeatureRecorder")


def _validate_group_tensor(tensors: dict[int, Tensor], tensor: Tensor) -> None:
    if not tensors:
        return
    reference = next(iter(tensors.values()))
    if tensor.shape != reference.shape:
        raise ValueError("all heads in a feature group must have the same shape")
    if tensor.dtype != reference.dtype:
        raise ValueError("all heads in a feature group must have the same dtype")


def _validate_index(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _head_from_tensor_key(name: str) -> int:
    prefix = "head_"
    if not name.startswith(prefix) or not name[len(prefix) :].isdigit():
        raise ValueError(f"invalid tensor key in feature file: {name!r}")
    return int(name[len(prefix) :])


def _validate_metadata(
    metadata: dict[str, str],
    *,
    chunk: int,
    step: int,
    layer: int,
    kind: FeatureKind,
    path: Path,
) -> None:
    expected = {
        "schema": _SCHEMA,
        "schema_version": _SCHEMA_VERSION,
        "chunk": str(chunk),
        "step": str(step),
        "layer": str(layer),
        "kind": kind.value,
    }
    mismatched = {
        name: (metadata.get(name), value)
        for name, value in expected.items()
        if metadata.get(name) != value
    }
    if mismatched:
        raise ValueError(f"invalid feature metadata in {path}: {mismatched}")


_SCHEMA = "heft.attention_features"
_SCHEMA_VERSION = "1"
