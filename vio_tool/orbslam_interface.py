from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .data_loader import FrameRecord
from .imu_converter import ImuSample, save_orb_imu_txt


@dataclass
class OrbRunResult:
    trajectory_path: Path
    assoc_path: Path
    imu_txt_path: Path
    log_path: Path


def write_association_file(frames: list[FrameRecord], out_path: str | Path) -> None:
    out_path = Path(out_path)
    with out_path.open("w", encoding="utf-8") as f:
        for fr in frames:
            f.write(f"{fr.timestamp:.9f} {fr.rgb_path.as_posix()} {fr.depth_path.as_posix()}\n")


def run_offline_orbslam(
    orb_exec: str | Path,
    vocab_path: str | Path,
    settings_path: str | Path,
    frames: list[FrameRecord],
    imu_samples: list[ImuSample],
    out_dir: str | Path,
    depth_scale: float = 1000.0,
    no_viewer: bool = True,
) -> OrbRunResult:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    assoc_path = out_dir / "rgbd_association.txt"
    imu_txt_path = out_dir / "imu_orb.txt"
    traj_path = out_dir / "traj_ws.txt"
    log_path = out_dir / "orbslam_run.log"

    write_association_file(frames, assoc_path)
    save_orb_imu_txt(imu_samples, str(imu_txt_path))

    cmd = [
        str(orb_exec),
        str(vocab_path),
        str(settings_path),
        str(assoc_path),
        str(imu_txt_path),
        str(traj_path),
        "--depth-scale",
        str(depth_scale),
    ]
    if no_viewer:
        cmd.append("--no-viewer")

    env = os.environ.copy()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)

    with log_path.open("w", encoding="utf-8") as f:
        f.write("# CMD\n")
        f.write(" ".join(cmd) + "\n\n")
        f.write("# STDOUT\n")
        f.write(proc.stdout)
        f.write("\n# STDERR\n")
        f.write(proc.stderr)

    if proc.returncode != 0:
        raise RuntimeError(
            "ORB-SLAM3 runner failed. "
            f"returncode={proc.returncode}. See log: {log_path}"
        )

    if not traj_path.exists():
        raise RuntimeError(f"Trajectory file not found: {traj_path}")

    return OrbRunResult(
        trajectory_path=traj_path,
        assoc_path=assoc_path,
        imu_txt_path=imu_txt_path,
        log_path=log_path,
    )
