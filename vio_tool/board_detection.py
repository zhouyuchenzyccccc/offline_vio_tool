from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .pose_math import Pose, compose


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    dist: np.ndarray


def _read_node_float(fs: cv2.FileStorage, keys: list[str]) -> float | None:
    for k in keys:
        node = fs.getNode(k)
        if not node.empty():
            return float(node.real())
    return None


def load_intrinsics_from_orb_yaml(path: str | Path) -> CameraIntrinsics:
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise ValueError(f"Cannot open settings file: {path}")

    fx = _read_node_float(fs, ["Camera.fx", "Camera1.fx", "fx"])
    fy = _read_node_float(fs, ["Camera.fy", "Camera1.fy", "fy"])
    cx = _read_node_float(fs, ["Camera.cx", "Camera1.cx", "cx"])
    cy = _read_node_float(fs, ["Camera.cy", "Camera1.cy", "cy"])

    if None in (fx, fy, cx, cy):
        raise ValueError("Cannot parse intrinsics (fx, fy, cx, cy) from ORB settings file")

    k1 = _read_node_float(fs, ["Camera.k1", "Camera1.k1", "k1"]) or 0.0
    k2 = _read_node_float(fs, ["Camera.k2", "Camera1.k2", "k2"]) or 0.0
    p1 = _read_node_float(fs, ["Camera.p1", "Camera1.p1", "p1"]) or 0.0
    p2 = _read_node_float(fs, ["Camera.p2", "Camera1.p2", "p2"]) or 0.0
    k3 = _read_node_float(fs, ["Camera.k3", "Camera1.k3", "k3"]) or 0.0

    fs.release()
    dist = np.array([k1, k2, p1, p2, k3], dtype=np.float64)
    return CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, dist=dist)


def detect_aruco_T_cb(
    image_path: str | Path,
    intr: CameraIntrinsics,
    marker_length_m: float,
    dict_name: str = "DICT_4X4_50",
    marker_id: int | None = None,
) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    dict_id = getattr(cv2.aruco, dict_name, None)
    if dict_id is None:
        raise ValueError(f"Unknown ArUco dictionary: {dict_name}")

    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    corners, ids, _ = detector.detectMarkers(image)
    if ids is None or len(ids) == 0:
        raise RuntimeError("No ArUco marker detected")

    ids_flat = ids.flatten().tolist()
    pick = 0
    if marker_id is not None:
        if marker_id not in ids_flat:
            raise RuntimeError(f"ArUco id {marker_id} not found. Detected: {ids_flat}")
        pick = ids_flat.index(marker_id)

    K = np.array([[intr.fx, 0.0, intr.cx], [0.0, intr.fy, intr.cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length_m, K, intr.dist)
    rvec = rvecs[pick].reshape(3)
    tvec = tvecs[pick].reshape(3)

    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = tvec
    return T


def detect_apriltag_T_cb(
    image_path: str | Path,
    intr: CameraIntrinsics,
    tag_size_m: float,
    tag_family: str = "tag36h11",
    tag_id: int | None = None,
) -> np.ndarray:
    try:
        from pupil_apriltags import Detector
    except Exception as e:
        raise RuntimeError("AprilTag detection requires package pupil_apriltags") from e

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    det = Detector(families=tag_family)
    tags = det.detect(
        image,
        estimate_tag_pose=True,
        camera_params=(intr.fx, intr.fy, intr.cx, intr.cy),
        tag_size=tag_size_m,
    )

    if not tags:
        raise RuntimeError("No AprilTag detected")

    picked = tags[0]
    if tag_id is not None:
        found = [t for t in tags if t.tag_id == tag_id]
        if not found:
            raise RuntimeError(f"AprilTag id {tag_id} not found")
        picked = found[0]

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = picked.pose_R
    T[:3, 3] = picked.pose_t.reshape(3)
    return T


def compute_T_ws_b(T_ws_c: np.ndarray, T_c_b: np.ndarray) -> np.ndarray:
    return compose(T_ws_c, T_c_b)


def find_nearest_pose_by_timestamp(poses: list[Pose], timestamp: float) -> Pose:
    if not poses:
        raise ValueError("Empty trajectory")
    best = min(poses, key=lambda p: abs(p.timestamp - timestamp))
    return best
