from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .imu_converter import ImuSample, load_imu_csv


_IMG_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff")


@dataclass
class FrameRecord:
    frame_index: str
    timestamp: float
    rgb_path: Path
    depth_path: Path


@dataclass
class CameraDataset:
    cam_dir: Path
    frames: list[FrameRecord]
    imu_samples: list[ImuSample]


def _collect_images(folder: Path) -> list[Path]:
    files: list[Path] = []
    for ext in _IMG_EXTS:
        files.extend(folder.glob(ext))
    def _sort_key(p: Path):
        try:
            return (0, int(p.stem))
        except ValueError:
            return (1, p.stem)

    files = sorted(files, key=_sort_key)
    return files


def _to_stem_map(paths: Sequence[Path]) -> dict[str, Path]:
    return {p.stem: p for p in paths}


def _first_existing(base: Path, rel_candidates: Sequence[str]) -> Path | None:
    for rel in rel_candidates:
        p = base / rel
        if p.exists():
            return p
    return None


def _frame_time_from_index(stem: str, fallback_step: float, idx: int) -> float:
    try:
        return float(stem) * 1e-6
    except ValueError:
        return idx * fallback_step


def load_camera_dataset(dataset_root: str | Path, cam_id: str, max_frames: int | None = None) -> CameraDataset:
    dataset_root = Path(dataset_root)
    cam_dir = dataset_root / cam_id
    if not cam_dir.exists():
        raise FileNotFoundError(f"Camera directory not found: {cam_dir}")

    # Backward-compatible path probing:
    # - old style: cam_xx/rgb, cam_xx/depth, cam_xx/imu.csv
    # - new style: 01/RGB, 01/Depth, 01/IMU/imu.csv
    rgb_dir = _first_existing(cam_dir, ["rgb", "RGB", "color", "Color"])
    depth_dir = _first_existing(cam_dir, ["depth", "Depth"])
    imu_csv = _first_existing(cam_dir, ["imu.csv", "IMU/imu.csv", "imu/imu.csv", "IMU/IMU.csv", "imu/IMU.csv"])

    if rgb_dir is None or depth_dir is None or imu_csv is None:
        raise FileNotFoundError(
            "Expected camera folder to contain RGB/depth image folders and IMU csv. "
            "Supported examples: 'cam_01/rgb depth imu.csv' or '01/RGB Depth IMU/imu.csv'."
        )

    rgb_paths = _collect_images(rgb_dir)
    depth_paths = _collect_images(depth_dir)
    if not rgb_paths:
        raise ValueError(f"No RGB images found in: {rgb_dir}")
    if not depth_paths:
        raise ValueError(f"No depth images found in: {depth_dir}")

    depth_by_stem = _to_stem_map(depth_paths)
    imu_samples = load_imu_csv(str(imu_csv))

    # Build frame timestamp lookup from frame_index in imu.csv when available.
    frame_ts_lookup: dict[str, float] = {}
    with open(imu_csv, "r", encoding="utf-8-sig") as f:
        header = f.readline().strip().split(",")
        if "frame_index" in header and "ref_ts_us" in header:
            idx_i = header.index("frame_index")
            ts_i = header.index("ref_ts_us")
            for line in f:
                parts = [x.strip() for x in line.split(",")]
                if len(parts) <= max(idx_i, ts_i):
                    continue
                frame_index = parts[idx_i]
                ref_ts = parts[ts_i]
                if frame_index and ref_ts:
                    frame_ts_lookup[frame_index] = float(ref_ts) * 1e-6

    frames: list[FrameRecord] = []
    fallback_step = 1.0 / 30.0

    for i, rgb in enumerate(rgb_paths):
        stem = rgb.stem
        if stem in depth_by_stem:
            depth = depth_by_stem[stem]
        elif i < len(depth_paths):
            depth = depth_paths[i]
        else:
            break

        ts = frame_ts_lookup.get(stem, _frame_time_from_index(stem, fallback_step, i))
        frames.append(FrameRecord(frame_index=stem, timestamp=ts, rgb_path=rgb, depth_path=depth))

    frames.sort(key=lambda x: x.timestamp)

    if max_frames is not None and max_frames > 0:
        frames = frames[:max_frames]

    return CameraDataset(cam_dir=cam_dir, frames=frames, imu_samples=imu_samples)
