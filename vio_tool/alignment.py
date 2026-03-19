from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .pose_math import Pose, compose, invert_se3


def load_transform_file(path: str | Path) -> np.ndarray:
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()

    # Try JSON first.
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            if "T_WmB" in data:
                arr = np.array(data["T_WmB"], dtype=np.float64)
                if arr.shape == (4, 4):
                    return arr
            if "matrix" in data:
                arr = np.array(data["matrix"], dtype=np.float64)
                if arr.shape == (4, 4):
                    return arr
            if {"tx", "ty", "tz", "qx", "qy", "qz", "qw"}.issubset(set(data.keys())):
                tx, ty, tz = data["tx"], data["ty"], data["tz"]
                qx, qy, qz, qw = data["qx"], data["qy"], data["qz"], data["qw"]
                p = Pose(0.0, np.array([tx, ty, tz], dtype=np.float64), np.array([qx, qy, qz, qw], dtype=np.float64))
                return p.as_matrix()
        elif isinstance(data, list):
            arr = np.array(data, dtype=np.float64)
            if arr.shape == (4, 4):
                return arr
    except json.JSONDecodeError:
        pass

    # Fallback plain text 4x4 matrix.
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        vals = [float(v) for v in line.replace(",", " ").split()]
        if len(vals) == 4:
            rows.append(vals)
    arr = np.array(rows, dtype=np.float64)
    if arr.shape != (4, 4):
        raise ValueError(f"Cannot parse 4x4 transform from: {path}")
    return arr


def estimate_T_wm_ws(T_wm_b: np.ndarray, T_ws_b: np.ndarray) -> np.ndarray:
    return compose(T_wm_b, invert_se3(T_ws_b))


def convert_traj_to_wm(poses_ws: list[Pose], T_wm_ws: np.ndarray) -> list[Pose]:
    out: list[Pose] = []
    for p in poses_ws:
        T_ws_c = p.as_matrix()
        T_wm_c = compose(T_wm_ws, T_ws_c)
        out.append(Pose.from_matrix(p.timestamp, T_wm_c))
    return out
