#!/usr/bin/env python3
"""
Convert camera calibration to ORB-SLAM3 YAML settings file.

Supports two modes:
  1. Factory calibration (from read_factory_calib.cpp output):
     python generate_settings.py <factory_calib.json> <output.yaml> --factory [--device_sn SN]
  
  2. Camera params JSON (legacy):
     python generate_settings.py <camera_params.json> <cam_id> <output.yaml>

Examples:
  python generate_settings.py factory_camera_imu_calib.json cam07_settings.yaml --factory
  python generate_settings.py factory_camera_imu_calib.json cam07_settings.yaml --factory --device_sn CPCBC53000M1
  python generate_settings.py camera_params.json 07 cam07_settings.yaml
"""

import json
import sys
import numpy as np
from pathlib import Path


def extract_extrinsic_matrix(extr_dict):
    """
    Convert extrinsic dict {'rotation': [...], 'translation' or 'translation_mm': [...]} to 4x4 matrix.
    """
    R = np.array(extr_dict.get('rotation', np.eye(3)))
    
    # Handle both 'translation' (in meters) and 'translation_mm' (in millimeters)
    if 'translation_mm' in extr_dict:
        t = np.array(extr_dict.get('translation_mm', [0, 0, 0])) / 1000.0  # Convert mm to m
    else:
        t = np.array(extr_dict.get('translation', [0, 0, 0]))  # Already in meters
    
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def generate_settings_from_factory(factory_calib_file, output_yaml, device_sn=None):
    """
    Extract camera and IMU parameters from factory calibration JSON.
    
    Args:
        factory_calib_file: Path to factory_camera_imu_calib.json from read_factory_calib
        output_yaml: Output YAML file path
        device_sn: Device serial number. If None, uses first device in JSON.
    """
    
    # Load factory calibration
    with open(factory_calib_file, 'r') as f:
        factory_data = json.load(f)
    
    # Handle two possible JSON structures:
    # 1. Old format: {"device_sn": {...}, "device_sn2": {...}}
    # 2. New format: {"device_count": N, "devices": {"device_sn": {...}}}
    
    if 'devices' in factory_data:
        # New format
        devices_dict = factory_data['devices']
        device_count = factory_data.get('device_count', len(devices_dict))
    else:
        # Old format - devices are at root level
        devices_dict = {k: v for k, v in factory_data.items() if k not in ['device_count']}
    
    if not devices_dict:
        raise ValueError(f"No devices found in {factory_calib_file}")
    
    # Get device data
    if device_sn is None:
        device_sn = list(devices_dict.keys())[0]
        print(f"No device SN specified, using first device: {device_sn}")
        print(f"Available devices: {list(devices_dict.keys())}")
    
    if device_sn not in devices_dict:
        raise ValueError(f"Device {device_sn} not found in {factory_calib_file}. "
                         f"Available devices: {list(devices_dict.keys())}")
    
    dev_data = devices_dict[device_sn]
    
    # Extract camera intrinsics (may be all zeros from SDK bug)
    camera_param = dev_data.get('camera_param', {})
    
    # Try multiple possible locations for intrinsics
    intr = camera_param.get('rgb_intrinsic') or camera_param.get('camera_intrinsics')
    if not intr:
        intr = {}
    
    fx = float(intr.get('fx', 607.0))
    fy = float(intr.get('fy', 607.0))
    cx = float(intr.get('cx', 640.0))
    cy = float(intr.get('cy', 400.0))
    
    # If intrinsics are zero/invalid, use Gemini 336L defaults
    if fx < 100 or fy < 100:
        print("  ⚠ Camera intrinsics from factory are zero, using Gemini 336L defaults:")
        fx, fy, cx, cy = 607.0, 607.0, 640.0, 400.0
        print(f"    fx={fx}, fy={fy}, cx={cx}, cy={cy}")
    
    # Try multiple possible locations for distortion
    dist = camera_param.get('rgb_distortion') or camera_param.get('distortion')
    if not dist:
        dist = {}
    
    k1 = float(dist.get('k1', 0.0))
    k2 = float(dist.get('k2', 0.0))
    k3 = float(dist.get('k3', 0.0))
    p1 = float(dist.get('p1', 0.0))
    p2 = float(dist.get('p2', 0.0))
    
    # Extract IMU extrinsics (color-to-gyro, which is sensor-to-camera IMU frame)
    stream_extr = dev_data.get('stream_extrinsics', {})
    T_bc = np.eye(4)  # Default: identity
    
    if 'color_to_gyro' in stream_extr:
        T_bc = extract_extrinsic_matrix(stream_extr['color_to_gyro'])
        print(f"  ✓ Found color-to-gyro extrinsic (camera to IMU)")
        trans_mm = (T_bc[:3, 3] * 1000).tolist()
        print(f"    Translation (mm): [{trans_mm[0]:.2f}, {trans_mm[1]:.2f}, {trans_mm[2]:.2f}]")
    elif 'color_to_accel' in stream_extr:
        T_bc = extract_extrinsic_matrix(stream_extr['color_to_accel'])
        print(f"  ✓ Found color-to-accel extrinsic (fallback)")
    else:
        print("  ⚠ No camera-to-IMU extrinsic found, using identity")
    
    # Prepare Tbc matrix string for YAML
    tbc_data = []
    for r in range(4):
        for c in range(4):
            tbc_data.append(f"{T_bc[r, c]:.6f}")
    tbc_str = ", ".join(tbc_data)
    
    # Generate YAML content
    yaml_content = f"""%YAML:1.0

# Camera calibration parameters for ORB-SLAM3 RGB-D+IMU
# Generated from factory calibration (read_factory_calib output)
# Device: {device_sn}
# Source: {Path(factory_calib_file).name}

#--------------------------------------------------------------------------------------------
# Camera Parameters
#--------------------------------------------------------------------------------------------

# Camera type: PinHole or KannalaBrandt8
Camera.type: "PinHole"

# Camera resolution
Camera.width: 1280
Camera.height: 800

# Camera intrinsics (OpenCV model) - from factory or Gemini 336L defaults
Camera.fx: {fx:.1f}
Camera.fy: {fy:.1f}
Camera.cx: {cx:.1f}
Camera.cy: {cy:.1f}

# Distortion parameters (k1, k2, p1, p2, k3) - from factory
Camera.k1: {k1:.6f}
Camera.k2: {k2:.6f}
Camera.p1: {p1:.6f}
Camera.p2: {p2:.6f}
Camera.k3: {k3:.6f}

# Distortion model: opencv for 5-param model
Camera.distortionModel: "opencv"

# Camera frames per second
Camera.fps: 30

# Color order of the images (0: BGR, 1: RGB). It is ignored if the input is a video file
Camera.RGB: 0

# For pinhole models, Bf = baseline * fx (in pixels)
# For RGB-D without stereo, set to 0
Camera.bf: 0.0

#--------------------------------------------------------------------------------------------
# Depth Parameters (RGB-D)
#--------------------------------------------------------------------------------------------

# DepthMapFactor: scales depth values
# For Orbbec (millimeters to meters): 1000
DepthMapFactor: 1000.0

# Maximum depth to be considered valid (meters)
ThDepth: 40.0

#--------------------------------------------------------------------------------------------
# IMU Parameters (for RGB-D-Inertial mode)
#--------------------------------------------------------------------------------------------

# Camera-IMU Transformation Matrix (from factory calibration)
# This is T_BC: transforms from camera frame C to IMU/body frame B
Tbc: !!opencv-matrix
    rows: 4
    cols: 4
    dt: f
    data: [{tbc_str}]

# IMU Frequency (Hz)
IMU.Frequency: 200

# IMU noise parameters
# These should be estimated via checkerboard calibration or Allan variance analysis
# Conservative defaults for Orbbec Gemini 336L (TODO: estimate from your data)
IMU.NoiseGyro: 1.7e-4       # rad/s (gyro measurement noise)
IMU.NoiseAcc: 2.0e-3        # m/s^2 (accelerometer measurement noise)
IMU.GyroWalk: 1.9393e-05    # rad/s/sqrt(s) (gyro bias random walk)
IMU.AccWalk: 3.0e-03        # m/s^3/sqrt(s) (accelerometer bias random walk)

#--------------------------------------------------------------------------------------------
# ORB Parameters
#--------------------------------------------------------------------------------------------

# ORB Extractor: Number of features per image
ORBextractor.nFeatures: 1000

# ORB Extractor: Scale factor between levels in the scale pyramid
ORBextractor.scaleFactor: 1.2

# ORB Extractor: Number of levels in the scale pyramid
ORBextractor.nLevels: 8

# ORB Extractor: Fast threshold
# Image is divided in a grid. At each cell FAST is applied. If no corners are found in a cell then
# its threshold is lowered.
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7

#--------------------------------------------------------------------------------------------
# Viewer Parameters
#--------------------------------------------------------------------------------------------

Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1.0
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2.0
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3.0
Viewer.ViewpointX: 0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -1.8
Viewer.ViewpointF: 500
Viewer.fps: 30
"""
    
    # Write YAML file
    output_path = Path(output_yaml)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"✓ Generated settings file: {output_yaml}")
    print(f"\nCamera parameters extracted:")
    print(f"  fx={fx}, fy={fy}")
    print(f"  cx={cx}, cy={cy}")
    print(f"  k1={k1}, k2={k2}, k3={k3}")
    print(f"  p1={p1}, p2={p2}")
    print(f"\n⚠ TODO: Refine IMU noise parameters (NoiseGyro, NoiseAcc, etc.)")
    print(f"  See: imu_joint_calibration.py for time offset estimation")
    print(f"\nNext: Run your command with --settings {output_yaml}")


def generate_settings_yaml(camera_params_file, cam_id, output_yaml):
    """
    Extract camera parameters for a specific cam_id and generate ORB-SLAM3 YAML (legacy mode).
    
    Args:
        camera_params_file: Path to camera_params.json
        cam_id: Camera ID (e.g., "07", "7", "01", etc.)
        output_yaml: Output YAML file path
    """
    
    # Load camera parameters
    with open(camera_params_file, 'r') as f:
        params = json.load(f)
    
    # Find camera entry - support various id formats
    cam_key = None
    cam_id_str = str(cam_id).lstrip('0') or '0'  # "07" -> "7", "01" -> "1"
    
    for key in params.keys():
        key_normalized = str(key).lstrip('0') or '0'
        if key_normalized == cam_id_str or str(key) == str(cam_id):
            cam_key = key
            break
    
    if not cam_key:
        raise ValueError(f"Camera {cam_id} not found in {camera_params_file}. "
                         f"Available cameras: {list(params.keys())}")
    
    cam_data = params[cam_key]
    
    # Extract RGB intrinsics from nested structure: cam_data['RGB']['intrinsic']
    if 'RGB' not in cam_data:
        raise ValueError(f"Camera {cam_id} has no 'RGB' key. Keys: {list(cam_data.keys())}")
    
    rgb = cam_data['RGB']
    if 'intrinsic' not in rgb:
        raise ValueError(f"Camera {cam_id} RGB has no 'intrinsic' key. Keys: {list(rgb.keys())}")
    
    intr = rgb['intrinsic']
    fx = float(intr.get('fx'))
    fy = float(intr.get('fy'))
    cx = float(intr.get('cx'))
    cy = float(intr.get('cy'))
    
    if None in [fx, fy, cx, cy]:
        raise ValueError(f"Missing intrinsics for camera {cam_id}: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
    
    # Extract distortion parameters from RGB.distortion
    dist = rgb.get('distortion', {})
    k1 = float(dist.get('k1', 0.0))
    k2 = float(dist.get('k2', 0.0))
    k3 = float(dist.get('k3', 0.0))
    p1 = float(dist.get('p1', 0.0))
    p2 = float(dist.get('p2', 0.0))
    
    # Generate YAML content
    yaml_content = f"""%YAML:1.0

# Camera calibration parameters for ORB-SLAM3 RGB-D
# Generated from {Path(camera_params_file).name}
# Camera ID: {cam_id}

#--------------------------------------------------------------------------------------------
# Camera Parameters
#--------------------------------------------------------------------------------------------

# Camera type: PinHole or KannalaBrandt8
Camera.type: "PinHole"

# Camera resolution
Camera.width: 1280
Camera.height: 800

# Camera intrinsics (OpenCV model)
Camera.fx: {fx:.1f}
Camera.fy: {fy:.1f}
Camera.cx: {cx:.1f}
Camera.cy: {cy:.1f}

# Distortion parameters (k1, k2, p1, p2, k3)
Camera.k1: {k1:.6f}
Camera.k2: {k2:.6f}
Camera.p1: {p1:.6f}
Camera.p2: {p2:.6f}
Camera.k3: {k3:.6f}

# Distortion model: opencv for 5-param model
Camera.distortionModel: "opencv"

# Camera frames per second
Camera.fps: 30

# Color order of the images (0: BGR, 1: RGB). It is ignored if the input is a video file
Camera.RGB: 0

# For pinhole models, Bf = baseline * fx (in pixels)
# For RGB-D without stereo, set to 0
Camera.bf: 0.0

#--------------------------------------------------------------------------------------------
# Depth Parameters (RGB-D)
#--------------------------------------------------------------------------------------------

# DepthMapFactor: scales depth values
# For Realsense (millimeters): 1000
# For normalized [0,1]: 1
DepthMapFactor: 1000.0

# Maximum depth to be considered valid (meters)
ThDepth: 40.0

#--------------------------------------------------------------------------------------------
# IMU Parameters (not used in RGBD-only mode)
#--------------------------------------------------------------------------------------------

# Camera-IMU Transformation Matrix
# Even in RGB-D mode, some builds still validate this key.
Tbc: !!opencv-matrix
    rows: 4
    cols: 4
    dt: f
    data: [1.0, 0.0, 0.0, 0.0,
                 0.0, 1.0, 0.0, 0.0,
                 0.0, 0.0, 1.0, 0.0,
                 0.0, 0.0, 0.0, 1.0]

# IMU noise parameters (from ORB-SLAM3 style defaults / EuRoC-like values)
# These are required by IMU_RGBD parser. For pure RGBD path they are not used.
IMU.Frequency: 200
IMU.NoiseGyro: 1.7e-4
IMU.NoiseAcc: 2.0e-3
IMU.GyroWalk: 1.9393e-05
IMU.AccWalk: 3.0e-03

#--------------------------------------------------------------------------------------------
# ORB Parameters
#--------------------------------------------------------------------------------------------

# ORB Extractor: Number of features per image
ORBextractor.nFeatures: 1000

# ORB Extractor: Scale factor between levels in the scale pyramid
ORBextractor.scaleFactor: 1.2

# ORB Extractor: Number of levels in the scale pyramid
ORBextractor.nLevels: 8

# ORB Extractor: Fast threshold
# Image is divided in a grid. At each cell FAST is applied. If no corners are found in a cell then
# its threshold is lowered.
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7

#--------------------------------------------------------------------------------------------
# Viewer Parameters
#--------------------------------------------------------------------------------------------

Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1.0
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2.0
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3.0
Viewer.ViewpointX: 0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -1.8
Viewer.ViewpointF: 500
Viewer.fps: 30
"""
    
    # Write YAML file
    output_path = Path(output_yaml)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"✓ Generated settings file: {output_yaml}")
    print(f"\nCamera parameters extracted:")
    print(f"  fx={fx}, fy={fy}")
    print(f"  cx={cx}, cy={cy}")
    print(f"  k1={k1}, k2={k2}, k3={k3}")
    print(f"  p1={p1}, p2={p2}")
    print(f"\nNext: Run your command with --settings {output_yaml}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate ORB-SLAM3 settings from calibration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Factory calibration mode (new - recommended):
    python generate_settings.py factory_camera_imu_calib.json cam07_settings.yaml --factory
    python generate_settings.py factory_camera_imu_calib.json cam07_settings.yaml --factory --device_sn CPCBC53000M1

  Legacy camera_params mode:
    python generate_settings.py camera_params.json cam07_settings.yaml --cam_id 07
        """)
    
    parser.add_argument('input_file', help='Input calibration JSON file')
    parser.add_argument('output_yaml', help='Output YAML settings file')
    parser.add_argument('--factory', action='store_true', help='Use factory calibration mode (from read_factory_calib)')
    parser.add_argument('--device_sn', default=None, help='Device serial number (for factory mode)')
    parser.add_argument('--cam_id', default=None, help='Camera ID (for legacy camera_params mode)')
    
    args = parser.parse_args()
    
    try:
        if args.factory:
            print(f"[Factory Mode] Loading factory calibration from {args.input_file}")
            generate_settings_from_factory(args.input_file, args.output_yaml, args.device_sn)
        elif args.cam_id:
            print(f"[Legacy Mode] Loading camera_params with cam_id={args.cam_id}")
            generate_settings_yaml(args.input_file, args.cam_id, args.output_yaml)
        else:
            # Try auto-detect: attempt factory mode first if it looks like factory JSON
            print("Attempting auto-detect mode...")
            try:
                with open(args.input_file, 'r') as f:
                    data = json.load(f)
                    # Factory format has 'devices' or device SN keys with 'stream_extrinsics'
                    if 'devices' in data:
                        # New format with devices dict
                        devices_dict = data['devices']
                        first_key = list(devices_dict.keys())[0] if devices_dict else None
                    else:
                        # Old format - devices at root level
                        devices_dict = {k: v for k, v in data.items() if k not in ['device_count']}
                        first_key = list(devices_dict.keys())[0] if devices_dict else None
                    
                    if first_key and isinstance(devices_dict.get(first_key), dict) and 'stream_extrinsics' in devices_dict.get(first_key, {}):
                        print("  → Detected factory calibration format")
                        generate_settings_from_factory(args.input_file, args.output_yaml, args.device_sn)
                    else:
                        raise ValueError("Not factory format, provide --cam_id for legacy mode")
            except Exception as e:
                print(f"  ✗ Auto-detect failed: {e}")
                raise ValueError("Please specify --factory or --cam_id flag")
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
