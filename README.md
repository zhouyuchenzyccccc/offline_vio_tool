# Offline RGB-D + IMU VIO Tool (ORB-SLAM3)

This project provides an offline pipeline to:

1) load recorded RGB, depth, and IMU data,
2) run ORB-SLAM3 in IMU_RGBD mode,
3) optionally align trajectory from SLAM world (Ws) to multi-camera world (Wm),
4) export trajectory and debug visualizations.

The pipeline is designed to avoid modifying ORB-SLAM3 core source code.

## 1. What Is Included

- Python entry script: run_vio.py
- Offline ORB executable source: cpp/rgbd_inertial_offline.cc
- Python modules:
  - vio_tool/data_loader.py
  - vio_tool/imu_converter.py
  - vio_tool/orbslam_interface.py
  - vio_tool/board_detection.py
  - vio_tool/alignment.py
  - vio_tool/visualization.py
  - vio_tool/pose_math.py

## 2. End-to-End Data Flow

1) Input dataset
   - dataset/IMU_test/cam_xx/rgb
   - dataset/IMU_test/cam_xx/depth
   - dataset/IMU_test/cam_xx/imu.csv

2) Convert input for ORB-SLAM3
   - rgbd_association.txt
   - imu_orb.txt (timestamp ax ay az gx gy gz)

3) Run ORB-SLAM3 offline
   - output: traj_ws.txt

4) Optional board-based alignment
   - detect board pose T_CB from one RGB frame
   - compute T_WsB = T_WsC * T_CB
   - load known T_WmB
   - compute T_WmWs = T_WmB * inv(T_WsB)
   - convert all poses: T_WmC = T_WmWs * T_WsC
   - output: traj_wm.txt

5) Optional visualization
   - 3D trajectory
   - translation and Euler curves
   - frame-to-frame motion delta and jump report

## 3. Coordinate Convention

T_AB means transforming a point from frame B to frame A.

- Alignment transform:
  - T_WmWs = T_WmB * inv(T_WsB)

- Trajectory conversion:
  - T_WmC = T_WmWs * T_WsC

- Board pose in SLAM world:
  - T_WsB = T_WsC * T_CB

## 4. Prerequisites (Linux)

Tested target:
- Ubuntu 20.04 or 22.04
- GCC/G++ with C++14 support
- Python 3.9+

Install system dependencies:

```bash
sudo apt update
sudo apt install -y \
  build-essential cmake git pkg-config \
  libeigen3-dev libboost-all-dev \
  libopencv-dev python3-dev python3-pip python3-venv \
  libglew-dev libssl-dev libgsl-dev
```

Notes:
- Pangolin is usually required by ORB-SLAM3 default build.
- If your ORB-SLAM3 already compiles successfully in your environment, keep using that setup.

## 5. Build ORB-SLAM3 (if not already built)

From your ORB-SLAM3 root:

```bash
cd /path/to/ORB-SLAM3
chmod +x build.sh
./build.sh
```

Expected artifacts:
- ORB vocabulary file: /path/to/ORB-SLAM3/Vocabulary/ORBvoc.txt
- ORB shared library in lib or build output path

## 6. Build Offline ORB Runner (this project)

Build the standalone executable that feeds RGB-D + IMU to ORB-SLAM3:

```bash
cd /path/to/cam_pose/offline_vio_tool
cmake -S cpp -B build -DORB_SLAM3_ROOT=/path/to/ORB-SLAM3
cmake --build build -j
```

Expected executable on Linux:
- build/rgbd_inertial_offline

If CMake cannot find ORB-SLAM3 library, ensure ORB-SLAM3 has been built and check output paths in:
- /path/to/ORB-SLAM3/lib
- /path/to/ORB-SLAM3/build/lib

## 7. Create Python Environment

```bash
cd /path/to/cam_pose/offline_vio_tool
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Package purpose:
- numpy/scipy: pose math
- matplotlib: plots
- opencv-contrib-python: ArUco detection
- pupil-apriltags: AprilTag detection (optional if not using AprilTag)

## 8. Prepare Dataset and Settings

### 8.1 Dataset folder

Supported formats:

```text
dataset/
  IMU_test/
    cam_01/
      rgb/
      depth/
      imu.csv

    01/
      RGB/
      Depth/
      IMU/
        imu.csv
```

Supported image extensions:
- png, jpg, jpeg, bmp, tiff

### 8.2 IMU CSV required columns

- frame_index
- ref_ts_us
- accel_ts_us, accel_x, accel_y, accel_z
- gyro_ts_us, gyro_x, gyro_y, gyro_z

Timestamp unit in CSV is microseconds.
The tool converts to seconds for ORB-SLAM3.

### 8.3 Camera settings file

Provide ORB-SLAM3-compatible camera YAML, including intrinsics used for board pose estimation.

Minimum required camera parameters in YAML:
- Camera.fx
- Camera.fy
- Camera.cx
- Camera.cy

Depth handling:
- if depth image is uint16 in millimeters, use --depth_scale 1000
- if depth image is already float in meters, adjust accordingly

### 8.4 Multi-camera board pose file

For alignment mode, prepare known T_WmB.

JSON example:

```json
{
  "T_WmB": [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
  ]
}
```

Plain-text 4x4 matrix is also accepted.

## 9. Run Commands

### 9.1 Quick smoke test (first N frames)

```bash
python run_vio.py \
  --dataset /home/ubuntu/WorkSpace/ZYC/dataset/IMU_test \
  --cam_id 07 \
  --sensor_mode rgbd \
  --run_slam \
  --orb_exec /home/ubuntu/WorkSpace/ZYC/cam_pose/offline_vio_tool/build/rgbd_inertial_offline \
  --vocab /path/to/ORB-SLAM3/Vocabulary/ORBvoc.txt \
  --settings /path/to/cam_settings.yaml \
  --out_dir /home/ubuntu/WorkSpace/ZYC/dataset/IMU_test/output_cam_07_smoke \
  --max_frames 100 \
  --plot
```

### 9.2 Full offline run (trajectory in Ws)

```bash
python run_vio.py \
  --dataset /home/ubuntu/WorkSpace/ZYC/dataset/IMU_test \
  --cam_id 07 \
  --sensor_mode rgbd \
  --run_slam \
  --orb_exec /home/ubuntu/WorkSpace/ZYC/cam_pose/offline_vio_tool/build/rgbd_inertial_offline \
  --vocab /home/ubuntu/WorkSpace/ZYC/cam_pose/ORB-SLAM3/Vocabulary/ORBvoc.txt \
  --settings /home/ubuntu/WorkSpace/ZYC/dataset/IMU_test/camera_params.json \
  --out_dir /home/ubuntu/WorkSpace/ZYC/dataset/IMU_test/output_cam_07 \
  --plot
```

### 9.3 Full run with world alignment (output Wm trajectory)

ArUco mode:

```bash
python run_vio.py \
  --dataset /home/ubuntu/WorkSpace/ZYC/dataset/IMU_test \
  --cam_id 07 \
  --run_slam \
  --orb_exec /home/ubuntu/WorkSpace/ZYC/cam_pose/offline_vio_tool/build/rgbd_inertial_offline \
  --vocab /path/to/ORB-SLAM3/Vocabulary/ORBvoc.txt \
  --settings /path/to/cam_settings.yaml \
  --out_dir /home/ubuntu/WorkSpace/ZYC/dataset/IMU_test/output_cam_07_align_aruco \
  --plot \
  --do_align \
  --board_type aruco \
  --board_size_m 0.08 \
  --aruco_dict DICT_4X4_50 \
  --marker_id 0 \
  --board_frame_index 000123 \
  --t_wmb_file /data/calib/T_WmB.json
```

AprilTag mode:

```bash
python run_vio.py \
  --dataset /home/ubuntu/WorkSpace/ZYC/dataset/IMU_test \
  --cam_id 07 \
  --run_slam \
  --orb_exec /home/ubuntu/WorkSpace/ZYC/cam_pose/offline_vio_tool/build/rgbd_inertial_offline \
  --vocab /path/to/ORB-SLAM3/Vocabulary/ORBvoc.txt \
  --settings /path/to/cam_settings.yaml \
  --out_dir /home/ubuntu/WorkSpace/ZYC/dataset/IMU_test/output_cam_07_align_apriltag \
  --plot \
  --do_align \
  --board_type apriltag \
  --apriltag_family tag36h11 \
  --board_size_m 0.08 \
  --marker_id 0 \
  --board_frame_index 000123 \
  --t_wmb_file /data/calib/T_WmB.json
```

### 9.4 Option meanings and command differences

- What is `--settings`:
  - It is the ORB-SLAM3 camera configuration YAML (intrinsics and related camera params).
  - It is used in two places:
    - ORB-SLAM3 tracking itself (RGB-D-Inertial pipeline).
    - Board pose solving (Aruco/AprilTag) to read `fx fy cx cy`.

- What is `--sensor_mode`:
  - `imu_rgbd`: uses RGB + Depth + IMU (default).
  - `rgbd`: uses RGB + Depth only (recommended first validation when IMU extrinsic/noise are not calibrated).

- Does this include camera calibration:
  - The commands do not run a calibration optimizer.
  - They use existing calibration results from your `cam_settings.yaml`.
  - If `--do_align` is enabled, the pipeline additionally estimates board pose in one frame and computes world alignment transform.

- Difference among commands:
  - `9.1 Quick smoke test`:
    - Runs first N frames (`--max_frames`) for quick debugging.
    - Output is mainly for fast validation.
  - `9.2 Full offline run (Ws)`:
    - Runs full sequence.
    - Outputs trajectory in SLAM world: `traj_ws.txt`.
    - No world alignment.
  - `9.3 Full run with world alignment (Wm)`:
    - Runs full sequence + board detection + world transform solve.
    - Outputs `traj_wm.txt` in multi-camera world.
    - `aruco` and `apriltag` modes only differ in board detector backend.

## 10. Output Files

Generated in out_dir:

- traj_ws.txt
- traj_wm.txt (alignment mode only)
- rgbd_association.txt
- imu_orb.txt
- orbslam_run.log
- T_ws_b.txt (alignment mode)
- T_wm_ws.txt (alignment mode)
- traj_ws_3d.png
- traj_ws_pose_curves.png
- traj_ws_motion_delta.png
- jump_report_ws.txt
- run_summary.json

Trajectory format (TUM-like):
- timestamp tx ty tz qx qy qz qw

## 11. One-Line Command (Linux)

Replace each path variable, then run:

```bash
ORB_ROOT=/path/to/ORB-SLAM3 TOOL_ROOT=/home/ubuntu/WorkSpace/ZYC/cam_pose/offline_vio_tool DATASET=/home/ubuntu/WorkSpace/ZYC/dataset/IMU_test CAM_ID=07 SETTINGS=/path/to/cam_settings.yaml T_WMB=/path/to/T_WmB.json OUT=/home/ubuntu/WorkSpace/ZYC/dataset/IMU_test/output_cam_07_oneliner bash -lc 'python3 -m pip install -r "$TOOL_ROOT/requirements.txt" && cmake -S "$TOOL_ROOT/cpp" -B "$TOOL_ROOT/build" -DORB_SLAM3_ROOT="$ORB_ROOT" && cmake --build "$TOOL_ROOT/build" -j && python3 "$TOOL_ROOT/run_vio.py" --dataset "$DATASET" --cam_id "$CAM_ID" --run_slam --orb_exec "$TOOL_ROOT/build/rgbd_inertial_offline" --vocab "$ORB_ROOT/Vocabulary/ORBvoc.txt" --settings "$SETTINGS" --out_dir "$OUT" --plot --do_align --board_type aruco --board_size_m 0.08 --aruco_dict DICT_4X4_50 --marker_id 0 --board_frame_index 000123 --t_wmb_file "$T_WMB"'
```

## 12. Troubleshooting

### 12.1 CMake cannot find ORB_SLAM3

- Rebuild ORB-SLAM3 first.
- Check ORB library path in ORB-SLAM3/lib or ORB-SLAM3/build/lib.
- Re-run CMake with correct -DORB_SLAM3_ROOT.

### 12.2 Trajectory file is empty

- Check orbslam_run.log.
- Verify image paths in rgbd_association.txt are valid.
- Ensure depth scaling is correct.
- Ensure timestamps are increasing.

### 12.3 Board detection fails

- Verify board is clearly visible in selected frame.
- Check marker id, board size, and dictionary/family.
- Check camera intrinsics in settings YAML.

### 12.4 Large alignment error

- Make sure T_WmB and detected board refer to the same physical board frame and same timestamp.
- Avoid motion blur in board frame.
- Verify frame conventions of all transforms.

## 13. Dependency Clarification

This offline processing tool does not require OrbbecSDK at runtime.

- You may use OrbbecSDK for data collection.
- Offline processing here only depends on saved files, ORB-SLAM3, OpenCV, and Python packages.
