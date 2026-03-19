from __future__ import annotations

from pathlib import Path

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2

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


def save_trajectory_video(
    poses: list[Pose],
    out_path: str | Path,
    fps: int = 20,
    tail_length: int = 120,
) -> Path:
    if not poses:
        raise ValueError("No poses to visualize")

    out_path = Path(out_path)
    xs = np.array([p.t[0] for p in poses], dtype=float)
    ys = np.array([p.t[1] for p in poses], dtype=float)
    zs = np.array([p.t[2] for p in poses], dtype=float)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    z_min, z_max = float(zs.min()), float(zs.max())
    pad_x = max(0.1, (x_max - x_min) * 0.1)
    pad_y = max(0.1, (y_max - y_min) * 0.1)
    pad_z = max(0.1, (z_max - z_min) * 0.1)

    ax.set_xlim(x_min - pad_x, x_max + pad_x)
    ax.set_ylim(y_min - pad_y, y_max + pad_y)
    ax.set_zlim(z_min - pad_z, z_max + pad_z)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Camera Trajectory Animation")
    ax.grid(True)

    line, = ax.plot([], [], [], "b-", linewidth=1.6, label="trajectory")
    point, = ax.plot([], [], [], "ro", markersize=5, label="current")
    ax.legend()

    def _update(frame_idx: int):
        start = max(0, frame_idx - tail_length)
        line.set_data(xs[start:frame_idx + 1], ys[start:frame_idx + 1])
        line.set_3d_properties(zs[start:frame_idx + 1])
        point.set_data([xs[frame_idx]], [ys[frame_idx]])
        point.set_3d_properties([zs[frame_idx]])
        return line, point

    ani = FuncAnimation(fig, _update, frames=len(poses), interval=1000 / max(1, fps), blit=False)

    saved_path = out_path
    try:
        # Preferred output: mp4 via ffmpeg
        ani.save(str(out_path), writer="ffmpeg", fps=fps)
    except Exception:
        # Reliable fallback: render each frame and encode an animated GIF.
        saved_path = out_path.with_suffix(".gif")
        frames: list[Image.Image] = []
        for i in range(len(poses)):
            _update(i)
            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape((h, w, 4))
            frames.append(Image.fromarray(buf[:, :, :3].copy(), mode="RGB"))

        if not frames:
            raise RuntimeError("No frames generated for GIF")

        frames[0].save(
            str(saved_path),
            save_all=True,
            append_images=frames[1:],
            duration=max(1, int(round(1000 / max(1, fps)))),
            loop=0,
            optimize=False,
        )

    plt.close(fig)
    return saved_path


def save_overlay_video(
    frames,
    poses: list[Pose],
    out_path: str | Path,
    fps: int = 20,
) -> Path:
    if not frames:
        raise ValueError("No RGB frames to visualize")
    if not poses:
        raise ValueError("No poses to visualize")

    out_path = Path(out_path)

    # Precompute trajectory map extents in X-Z plane for a compact top-view inset.
    tx = np.array([p.t[0] for p in poses], dtype=float)
    tz = np.array([p.t[2] for p in poses], dtype=float)
    x_min, x_max = float(tx.min()), float(tx.max())
    z_min, z_max = float(tz.min()), float(tz.max())
    span_x = max(1e-6, x_max - x_min)
    span_z = max(1e-6, z_max - z_min)

    first = cv2.imread(str(frames[0].rgb_path), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Cannot read frame: {frames[0].rgb_path}")
    h, w = first.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, float(max(1, fps)), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {out_path}")

    pose_ts = np.array([p.timestamp for p in poses], dtype=float)

    def nearest_pose(ts: float) -> Pose:
        idx = int(np.searchsorted(pose_ts, ts))
        if idx <= 0:
            return poses[0]
        if idx >= len(poses):
            return poses[-1]
        if abs(pose_ts[idx] - ts) < abs(ts - pose_ts[idx - 1]):
            return poses[idx]
        return poses[idx - 1]

    for fr in frames:
        img = cv2.imread(str(fr.rgb_path), cv2.IMREAD_COLOR)
        if img is None:
            continue

        p = nearest_pose(float(fr.timestamp))
        ex, ey, ez = to_euler_xyz_deg(p.q_xyzw)

        # Main HUD text
        cv2.putText(img, f"t={fr.timestamp:.3f}s", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (40, 255, 40), 2)
        cv2.putText(img, f"pos[m]: x={p.t[0]:.3f} y={p.t[1]:.3f} z={p.t[2]:.3f}", (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(img, f"rpy[deg]: {ex:.1f}, {ey:.1f}, {ez:.1f}", (20, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        # Top-view inset (X-Z).
        inset_w = max(220, int(w * 0.28))
        inset_h = max(160, int(h * 0.28))
        ox = w - inset_w - 16
        oy = 16
        cv2.rectangle(img, (ox, oy), (ox + inset_w, oy + inset_h), (32, 32, 32), -1)
        cv2.rectangle(img, (ox, oy), (ox + inset_w, oy + inset_h), (220, 220, 220), 1)

        for i in range(len(poses) - 1):
            x0 = int((poses[i].t[0] - x_min) / span_x * (inset_w - 20)) + ox + 10
            z0 = int((poses[i].t[2] - z_min) / span_z * (inset_h - 20)) + oy + 10
            x1 = int((poses[i + 1].t[0] - x_min) / span_x * (inset_w - 20)) + ox + 10
            z1 = int((poses[i + 1].t[2] - z_min) / span_z * (inset_h - 20)) + oy + 10
            cv2.line(img, (x0, z0), (x1, z1), (80, 180, 255), 1, cv2.LINE_AA)

        cx = int((p.t[0] - x_min) / span_x * (inset_w - 20)) + ox + 10
        cz = int((p.t[2] - z_min) / span_z * (inset_h - 20)) + oy + 10
        cv2.circle(img, (cx, cz), 4, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(img, "Top view (X-Z)", (ox + 10, oy + inset_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)

        writer.write(img)

    writer.release()
    return out_path


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
