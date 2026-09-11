"""Read sampled RGB-D frames directly from ScanNet .sens v4 (no downloads)."""

from __future__ import annotations

import io
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


def _read(f, count):
    data = f.read(count)
    if len(data) != count:
        raise ValueError("Truncated .sens file")
    return data


def _matrix(f):
    return np.frombuffer(_read(f, 64), dtype="<f4").reshape(4, 4).copy()


@dataclass
class Frame:
    index: int
    pose: np.ndarray
    color_timestamp: int
    depth_timestamp: int
    color_offset: int
    color_bytes: int
    depth_offset: int
    depth_bytes: int


class SensScene:
    """Frame poses map depth-camera coordinates to the reconstruction world.

    Color registration uses the explicitly named colorToDepthExtrinsics in the
    scene .txt, rather than treating RGB and depth as pre-aligned.
    """

    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.path = self.directory / (self.directory.name + ".sens")
        self.name = self.directory.name
        self.metadata = {}
        metadata_path = self.directory / (self.name + ".txt")
        if metadata_path.exists():
            for line in metadata_path.read_text().splitlines():
                key, sep, value = line.partition("=")
                if sep:
                    self.metadata[key.strip()] = value.strip()
        self.axis_alignment = np.eye(4, dtype=np.float64)
        self.axis_alignment_present = "axisAlignment" in self.metadata
        if self.axis_alignment_present:
            self.axis_alignment = self._metadata_matrix("axisAlignment")
        # Raw .sens header extrinsics are identity in the local files; the .txt
        # contains the actual RGB/depth calibration and must not be ignored.
        if "colorToDepthExtrinsics" not in self.metadata:
            raise ValueError(f"{self.name}: missing colorToDepthExtrinsics")
        self.color_to_depth = self._metadata_matrix("colorToDepthExtrinsics")
        self.depth_to_color = np.linalg.inv(self.color_to_depth)
        self.frames = []
        file_size = self.path.stat().st_size
        with self.path.open("rb") as f:
            version = struct.unpack("<I", _read(f, 4))[0]
            if version != 4:
                raise ValueError(f"Only .sens v4 supported, got {version}")
            length = struct.unpack("<Q", _read(f, 8))[0]
            if length > 1_000_000:
                raise ValueError("Invalid sensor-name length")
            self.sensor_name = _read(f, length).decode(errors="replace")
            self.color_intrinsic = _matrix(f)
            self.color_extrinsic = _matrix(f)
            self.depth_intrinsic = _matrix(f)
            self.depth_extrinsic = _matrix(f)
            (
                self.color_compression,
                self.depth_compression,
                self.color_width,
                self.color_height,
                self.depth_width,
                self.depth_height,
                self.depth_shift,
                count,
            ) = struct.unpack("<iiIIIIfQ", _read(f, 36))
            if self.color_compression not in (1, 2) or self.depth_compression != 1:
                raise ValueError("Require PNG/JPEG color and zlib uint16 depth")
            if self.depth_shift <= 0 or not np.isfinite(self.depth_shift):
                raise ValueError("Invalid depth scale")
            if count > file_size // 96:
                raise ValueError("Impossible frame count")
            for index in range(count):
                pose = _matrix(f)
                ct, dt, nc, nd = struct.unpack("<QQQQ", _read(f, 32))
                co = f.tell()
                do = co + nc
                if do + nd > file_size:
                    raise ValueError(f"{self.name}: incomplete frame {index}")
                self.frames.append(Frame(index, pose, ct, dt, co, nc, do, nd))
                f.seek(do + nd)

    def _metadata_matrix(self, key):
        matrix = np.fromstring(self.metadata[key], sep=" ").reshape(4, 4)
        if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix[:3, :3])) < 1e-8:
            raise ValueError(f"Invalid {key}")
        return matrix

    def sample(self, count):
        if count < 2:
            raise ValueError("At least two frames required")
        valid = [
            f
            for f in self.frames
            if np.isfinite(f.pose).all()
            and np.allclose(f.pose[3], [0, 0, 0, 1], atol=1e-4)
            and abs(np.linalg.det(f.pose[:3, :3]) - 1) < 0.05
        ]
        if len(valid) < 2:
            raise ValueError(f"{self.name}: fewer than two valid poses")
        indices = np.linspace(0, len(valid) - 1, min(count, len(valid)), dtype=int)
        return [valid[i] for i in indices]

    def decode(self, frame):
        with self.path.open("rb") as f:
            f.seek(frame.color_offset)
            rgb = Image.open(io.BytesIO(_read(f, frame.color_bytes))).convert("RGB")
            rgb.load()
            f.seek(frame.depth_offset)
            raw = zlib.decompress(_read(f, frame.depth_bytes))
        expected = self.depth_height * self.depth_width * 2
        if len(raw) != expected:
            raise ValueError(f"{self.name}/{frame.index}: bad depth size")
        depth = (
            np.frombuffer(raw, dtype="<u2")
            .reshape(self.depth_height, self.depth_width)
            .astype(np.float32)
            / self.depth_shift
        )
        if rgb.size != (self.color_width, self.color_height):
            raise ValueError("RGB dimensions disagree with .sens header")
        return rgb, depth

    def signature(self):
        paths = [self.path, self.directory / (self.name + ".txt")]
        return [
            {
                "path": str(p),
                "bytes": p.stat().st_size,
                "mtime_ns": p.stat().st_mtime_ns,
            }
            for p in paths
            if p.exists()
        ]

    def describe(self, selected):
        ids = [f.index for f in selected]
        return {
            "scene": self.name,
            "frames_in_sens": len(self.frames),
            "frames_in_txt": self.metadata.get("numDepthFrames"),
            "sampled_frame_ids": ids,
            "frame_gaps": np.diff(ids).tolist(),
            "color_timestamps_us": [f.color_timestamp for f in selected],
            "depth_timestamps_us": [f.depth_timestamp for f in selected],
            "axis_alignment_present": self.axis_alignment_present,
            "axis_alignment": self.axis_alignment.tolist(),
            "pose_convention": "depth-camera-to-world",
            "depth_to_color": self.depth_to_color.tolist(),
            "source": self.signature(),
        }


def discover_scenes(root, names=None):
    root = Path(root).resolve()
    directories = (
        [root / name for name in names] if names else sorted(root.glob("scene*"))
    )
    if not directories:
        raise ValueError(f"No scene directories in {root}")
    for directory in directories:
        if (
            not directory.is_dir()
            or not (directory / (directory.name + ".sens")).is_file()
        ):
            raise ValueError(f"Missing scene .sens: {directory}")
    return directories
