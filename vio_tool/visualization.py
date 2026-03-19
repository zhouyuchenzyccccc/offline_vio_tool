from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .pose_math import Pose, pose_deltas, to_euler_xyz_deg


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_trajectory_3d(poses: list[Pose], out_path: str | Path | None = None) -> None:
    if not poses:
        return
    xs = np.array([p.t[0] for p in poses])
    ys = np.array([p.t[1] for p in poses])
    zs = np.array([p.t[2] for p in poses])

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(xs, ys, zs, "b-", linewidth=1.2, label="trajectory")
    step = max(1, len(poses) // 30)
    ax.scatter(xs[::step], ys[::step], zs[::step], c="r", s=10, label="key points")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Camera Trajectory")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    if out_path:
        fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


def plot_pose_curves(poses: list[Pose], out_path: str | Path | None = None) -> None:
    if not poses:
        return

    t = np.array([p.timestamp for p in poses])
    xyz = np.stack([p.t for p in poses], axis=0)
    eul = np.stack([to_euler_xyz_deg(p.q_xyzw) for p in poses], axis=0)

    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axs[0].plot(t, xyz[:, 0], label="x")
    axs[0].plot(t, xyz[:, 1], label="y")
    axs[0].plot(t, xyz[:, 2], label="z")
    axs[0].set_ylabel("translation (m)")
    axs[0].set_title("Translation vs Time")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(t, eul[:, 0], label="roll")
    axs[1].plot(t, eul[:, 1], label="pitch")
    axs[1].plot(t, eul[:, 2], label="yaw")
    axs[1].set_ylabel("angle (deg)")
    axs[1].set_xlabel("time (s)")
    axs[1].set_title("Euler Angles vs Time")
    axs[1].legend()
    axs[1].grid(True)

    fig.tight_layout()
    if out_path:
        fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


def plot_motion_deltas(poses: list[Pose], out_path: str | Path | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ts, dtrans, drot = pose_deltas(poses)
    if len(ts) == 0:
        return ts, dtrans, drot

    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axs[0].plot(ts, dtrans, "g-")
    axs[0].set_ylabel("delta translation (m)")
    axs[0].grid(True)

    axs[1].plot(ts, drot, "m-")
    axs[1].set_ylabel("delta rotation (deg)")
    axs[1].set_xlabel("time (s)")
    axs[1].grid(True)

    fig.suptitle("Frame-to-Frame Motion")
    fig.tight_layout()
    if out_path:
        fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    return ts, dtrans, drot


def save_all_plots(poses: list[Pose], out_dir: str | Path, prefix: str = "traj") -> dict[str, Path]:
    out_dir = _ensure_dir(out_dir)
    p1 = out_dir / f"{prefix}_3d.png"
    p2 = out_dir / f"{prefix}_pose_curves.png"
    p3 = out_dir / f"{prefix}_motion_delta.png"

    plot_trajectory_3d(poses, p1)
    plot_pose_curves(poses, p2)
    plot_motion_deltas(poses, p3)

    return {"traj3d": p1, "curves": p2, "delta": p3}


def detect_jumps(
    poses: list[Pose],
    trans_thresh_m: float = 0.2,
    rot_thresh_deg: float = 10.0,
) -> list[tuple[float, float, float]]:
    ts, dtrans, drot = pose_deltas(poses)
    issues: list[tuple[float, float, float]] = []
    for t, dt, dr in zip(ts, dtrans, drot):
        if dt > trans_thresh_m or dr > rot_thresh_deg:
            issues.append((float(t), float(dt), float(dr)))
    return issues
