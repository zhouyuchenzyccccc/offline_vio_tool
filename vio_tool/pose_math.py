from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class Pose:
    timestamp: float
    t: np.ndarray  # (3,)
    q_xyzw: np.ndarray  # (4,) qx qy qz qw

    def as_matrix(self) -> np.ndarray:
        t = self.t.reshape(3)
        r = Rotation.from_quat(self.q_xyzw.reshape(4)).as_matrix()
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = r
        T[:3, 3] = t
        return T

    @staticmethod
    def from_matrix(timestamp: float, T: np.ndarray) -> "Pose":
        q = Rotation.from_matrix(T[:3, :3]).as_quat()
        t = T[:3, 3].copy()
        return Pose(timestamp=timestamp, t=t, q_xyzw=q)


def invert_se3(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def to_euler_xyz_deg(q_xyzw: np.ndarray) -> np.ndarray:
    return Rotation.from_quat(q_xyzw).as_euler("xyz", degrees=True)


def load_traj_tum(path: str) -> list[Pose]:
    poses: list[Pose] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = line.split()
            if len(vals) != 8:
                continue
            ts, tx, ty, tz, qx, qy, qz, qw = map(float, vals)
            poses.append(Pose(ts, np.array([tx, ty, tz], dtype=np.float64), np.array([qx, qy, qz, qw], dtype=np.float64)))
    return poses


def save_traj_tum(path: str, poses: Iterable[Pose]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for p in poses:
            tx, ty, tz = p.t.tolist()
            qx, qy, qz, qw = p.q_xyzw.tolist()
            f.write(f"{p.timestamp:.9f} {tx:.9f} {ty:.9f} {tz:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n")


def compose(T_ab: np.ndarray, T_bc: np.ndarray) -> np.ndarray:
    return T_ab @ T_bc


def pose_deltas(poses: list[Pose]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(poses) < 2:
        return np.array([]), np.array([]), np.array([])

    dt = []
    dtrans = []
    drot = []
    for i in range(1, len(poses)):
        T_prev = poses[i - 1].as_matrix()
        T_cur = poses[i].as_matrix()
        T_rel = invert_se3(T_prev) @ T_cur
        trans = np.linalg.norm(T_rel[:3, 3])
        ang = Rotation.from_matrix(T_rel[:3, :3]).magnitude() * 180.0 / np.pi

        dt.append(poses[i].timestamp)
        dtrans.append(trans)
        drot.append(ang)

    return np.array(dt), np.array(dtrans), np.array(drot)
