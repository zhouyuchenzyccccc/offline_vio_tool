from __future__ import annotations

import argparse
import json
from pathlib import Path

from vio_tool.alignment import convert_traj_to_wm, estimate_T_wm_ws, load_transform_file
from vio_tool.board_detection import (
    compute_T_ws_b,
    detect_apriltag_T_cb,
    detect_aruco_T_cb,
    find_nearest_pose_by_timestamp,
    load_intrinsics_from_orb_yaml,
)
from vio_tool.data_loader import load_camera_dataset
from vio_tool.orbslam_interface import run_offline_orbslam
from vio_tool.pose_math import load_traj_tum, save_traj_tum
from vio_tool.visualization import detect_jumps, save_all_plots, save_overlay_video, save_trajectory_video


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline RGB-D+IMU VIO with ORB-SLAM3 + world alignment")

    p.add_argument("--dataset", required=True, help="Dataset root, e.g. dataset/IMU_test")
    p.add_argument("--cam_id", required=True, help="Camera id folder, e.g. cam_01")
    p.add_argument("--out_dir", default="outputs", help="Output directory")
    p.add_argument("--max_frames", type=int, default=0, help="Only process first N frames (0 means all)")

    p.add_argument("--run_slam", action="store_true", help="Run ORB-SLAM3 offline")
    p.add_argument("--orb_exec", default="", help="Path to rgbd_inertial_offline executable")
    p.add_argument("--vocab", default="", help="Path to ORBvoc.txt")
    p.add_argument("--settings", default="", help="Path to ORB-SLAM3 camera yaml")
    p.add_argument("--depth_scale", type=float, default=1000.0, help="Depth png scale to meter, e.g. 1000")
    p.add_argument("--sensor_mode", choices=["imu_rgbd", "rgbd"], default="imu_rgbd", help="SLAM sensor mode")
    p.add_argument("--use_viewer", action="store_true", help="Enable ORB-SLAM3 viewer")

    p.add_argument("--traj_ws", default="", help="Existing WS trajectory path. Used when --run_slam is not set")

    p.add_argument("--plot", action="store_true", help="Generate plots")
    p.add_argument("--video", action="store_true", help="Generate trajectory animation video (mp4/gif)")
    p.add_argument("--video_mode", choices=["traj", "overlay", "both"], default="traj", help="Video type: trajectory animation, RGB overlay, or both")
    p.add_argument("--video_fps", type=int, default=20, help="Animation FPS")
    p.add_argument("--video_tail", type=int, default=120, help="Visible trajectory tail length in frames")
    p.add_argument("--jump_trans_thresh", type=float, default=0.2, help="Jump detection threshold in meters")
    p.add_argument("--jump_rot_thresh_deg", type=float, default=10.0, help="Jump detection threshold in degrees")

    p.add_argument("--do_align", action="store_true", help="Estimate T_WmWs and convert trajectory to Wm")
    p.add_argument("--t_wmb_file", default="", help="4x4 transform file for T_WmB (json/txt)")
    p.add_argument("--board_type", choices=["aruco", "apriltag"], default="aruco")
    p.add_argument("--board_size_m", type=float, default=0.08, help="Marker/tag size in meters")
    p.add_argument("--board_frame_index", default="", help="frame_index used for board detection")
    p.add_argument("--board_ts", type=float, default=-1.0, help="Timestamp to pick nearest frame")
    p.add_argument("--aruco_dict", default="DICT_4X4_50")
    p.add_argument("--marker_id", type=int, default=-1, help="Marker/tag id, -1 means first detected")
    p.add_argument("--apriltag_family", default="tag36h11")

    return p.parse_args()


def pick_board_frame(dataset_frames, board_frame_index: str, board_ts: float):
    if board_frame_index:
        for fr in dataset_frames:
            if fr.frame_index == board_frame_index:
                return fr
        raise ValueError(f"board_frame_index not found: {board_frame_index}")

    if board_ts > 0:
        return min(dataset_frames, key=lambda fr: abs(fr.timestamp - board_ts))

    return dataset_frames[0]


def save_matrix_txt(path: Path, T):
    with path.open("w", encoding="utf-8") as f:
        for r in range(4):
            f.write(" ".join([f"{T[r, c]:.9f}" for c in range(4)]) + "\n")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    max_frames = args.max_frames if args.max_frames > 0 else None
    cam_data = load_camera_dataset(args.dataset, args.cam_id, max_frames=max_frames)

    if args.run_slam:
        required = [args.orb_exec, args.vocab, args.settings]
        if not all(required):
            raise ValueError("When --run_slam is set, --orb_exec --vocab --settings are required")

        run_res = run_offline_orbslam(
            orb_exec=args.orb_exec,
            vocab_path=args.vocab,
            settings_path=args.settings,
            frames=cam_data.frames,
            imu_samples=cam_data.imu_samples,
            out_dir=out_dir,
            depth_scale=args.depth_scale,
            sensor_mode=args.sensor_mode,
            no_viewer=not args.use_viewer,
        )
        traj_ws_path = run_res.trajectory_path
    else:
        if not args.traj_ws:
            raise ValueError("Provide --traj_ws when not using --run_slam")
        traj_ws_path = Path(args.traj_ws)

    poses_ws = load_traj_tum(str(traj_ws_path))
    if not poses_ws:
        raise RuntimeError(f"No pose parsed from trajectory: {traj_ws_path}")

    result = {
        "traj_ws": str(traj_ws_path),
        "num_poses": len(poses_ws),
    }

    if args.plot:
        plot_paths = save_all_plots(poses_ws, out_dir, prefix="traj_ws")
        jumps = detect_jumps(
            poses_ws,
            trans_thresh_m=args.jump_trans_thresh,
            rot_thresh_deg=args.jump_rot_thresh_deg,
        )
        jump_path = out_dir / "jump_report_ws.txt"
        with jump_path.open("w", encoding="utf-8") as f:
            for t, dt, dr in jumps:
                f.write(f"{t:.6f} dtrans={dt:.6f}m drot={dr:.6f}deg\n")
        result["plots_ws"] = {k: str(v) for k, v in plot_paths.items()}
        result["jump_report_ws"] = str(jump_path)
        result["jump_count_ws"] = len(jumps)

    if args.video:
        if args.video_mode in ("traj", "both"):
            video_ws = save_trajectory_video(
                poses_ws,
                out_dir / "traj_ws_animation.mp4",
                fps=args.video_fps,
                tail_length=args.video_tail,
            )
            result["video_ws"] = str(video_ws)

        if args.video_mode in ("overlay", "both"):
            overlay_ws = save_overlay_video(
                cam_data.frames,
                poses_ws,
                out_dir / "traj_ws_overlay.mp4",
                fps=args.video_fps,
            )
            result["video_ws_overlay"] = str(overlay_ws)

    if args.do_align:
        if not args.settings:
            raise ValueError("--settings is required for board detection")
        if not args.t_wmb_file:
            raise ValueError("--t_wmb_file is required when --do_align is set")

        intr = load_intrinsics_from_orb_yaml(args.settings)
        marker_id = None if args.marker_id < 0 else args.marker_id

        board_frame = pick_board_frame(cam_data.frames, args.board_frame_index, args.board_ts)

        if args.board_type == "aruco":
            T_cb = detect_aruco_T_cb(
                image_path=board_frame.rgb_path,
                intr=intr,
                marker_length_m=args.board_size_m,
                dict_name=args.aruco_dict,
                marker_id=marker_id,
            )
        else:
            T_cb = detect_apriltag_T_cb(
                image_path=board_frame.rgb_path,
                intr=intr,
                tag_size_m=args.board_size_m,
                tag_family=args.apriltag_family,
                tag_id=marker_id,
            )

        pose_ws_c = find_nearest_pose_by_timestamp(poses_ws, board_frame.timestamp)
        T_ws_b = compute_T_ws_b(pose_ws_c.as_matrix(), T_cb)

        T_wm_b = load_transform_file(args.t_wmb_file)
        T_wm_ws = estimate_T_wm_ws(T_wm_b, T_ws_b)

        poses_wm = convert_traj_to_wm(poses_ws, T_wm_ws)
        traj_wm_path = out_dir / "traj_wm.txt"
        save_traj_tum(str(traj_wm_path), poses_wm)

        save_matrix_txt(out_dir / "T_ws_b.txt", T_ws_b)
        save_matrix_txt(out_dir / "T_wm_ws.txt", T_wm_ws)

        result["board_frame"] = str(board_frame.rgb_path)
        result["traj_wm"] = str(traj_wm_path)
        result["T_ws_b"] = str(out_dir / "T_ws_b.txt")
        result["T_wm_ws"] = str(out_dir / "T_wm_ws.txt")

        if args.plot:
            plot_paths_wm = save_all_plots(poses_wm, out_dir, prefix="traj_wm")
            result["plots_wm"] = {k: str(v) for k, v in plot_paths_wm.items()}

        if args.video:
            if args.video_mode in ("traj", "both"):
                video_wm = save_trajectory_video(
                    poses_wm,
                    out_dir / "traj_wm_animation.mp4",
                    fps=args.video_fps,
                    tail_length=args.video_tail,
                )
                result["video_wm"] = str(video_wm)

            if args.video_mode in ("overlay", "both"):
                overlay_wm = save_overlay_video(
                    cam_data.frames,
                    poses_wm,
                    out_dir / "traj_wm_overlay.mp4",
                    fps=args.video_fps,
                )
                result["video_wm_overlay"] = str(overlay_wm)

    summary_path = out_dir / "run_summary.json"
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
