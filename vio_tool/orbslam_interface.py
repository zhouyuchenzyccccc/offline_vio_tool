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
    sensor_mode: str = "imu_rgbd",
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
        "--sensor-mode",
        str(sensor_mode),
        "--depth-scale",
        str(depth_scale),
    ]
    if no_viewer:
        cmd.append("--no-viewer")

    env = os.environ.copy()

    # For Pangolin/GUI runs, piping stdout/stderr can be unstable on some systems.
    # Stream directly to log file instead of capture_output.
    with log_path.open("w", encoding="utf-8") as f:
        f.write("# CMD\n")
        f.write(" ".join(cmd) + "\n\n")
        f.write("# STDOUT+STDERR\n")
        f.flush()

        proc = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        returncode = proc.wait()

    if returncode != 0:
        # Some ORB-SLAM3 + Viewer combinations may exit with a non-zero code
        # after trajectory has been successfully saved. Keep the result usable.
        if traj_path.exists() and traj_path.stat().st_size > 0:
            with log_path.open("a", encoding="utf-8") as f:
                f.write("\n# WRAPPER_NOTE\n")
                f.write(
                    "Non-zero return code detected, but trajectory file exists and is non-empty. "
                    f"Continuing. returncode={returncode}\n"
                )
        else:
            raise RuntimeError(
                "ORB-SLAM3 runner failed. "
                f"returncode={returncode}. See log: {log_path}"
            )

    if not traj_path.exists():
        raise RuntimeError(f"Trajectory file not found: {traj_path}")

    return OrbRunResult(
        trajectory_path=traj_path,
        assoc_path=assoc_path,
        imu_txt_path=imu_txt_path,
        log_path=log_path,
    )
