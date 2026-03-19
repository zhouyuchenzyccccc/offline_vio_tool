#!/usr/bin/env python3
"""
Convert camera_params.json to ORB-SLAM3 YAML settings file.
Usage:
  python generate_settings.py <camera_params.json> <cam_id> <output.yaml>
Example:
  python generate_settings.py /path/to/camera_params.json 07 /path/to/cam07_settings.yaml
"""

import json
import sys
from pathlib import Path


def generate_settings_yaml(camera_params_file, cam_id, output_yaml):
    """
    Extract camera parameters for a specific cam_id and generate ORB-SLAM3 YAML.
    
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
    fx = intr.get('fx')
    fy = intr.get('fy')
    cx = intr.get('cx')
    cy = intr.get('cy')
    
    if None in [fx, fy, cx, cy]:
        raise ValueError(f"Missing intrinsics for camera {cam_id}: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
    
    # Extract distortion parameters from RGB.distortion
    dist = rgb.get('distortion', {})
    k1 = dist.get('k1', 0.0)
    k2 = dist.get('k2', 0.0)
    k3 = dist.get('k3', 0.0)
    p1 = dist.get('p1', 0.0)
    p2 = dist.get('p2', 0.0)
    
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
Camera.fx: {fx}
Camera.fy: {fy}
Camera.cx: {cx}
Camera.cy: {cy}

# Distortion parameters (k1, k2, p1, p2, k3)
Camera.k1: {k1}
Camera.k2: {k2}
Camera.p1: {p1}
Camera.p2: {p2}
Camera.k3: {k3}

# Distortion model: opencv for 5-param model
Camera.distortionModel: "opencv"

# Camera frames per second
Camera.fps: 30

# Color order of the images (0: BGR, 1: RGB). It is ignored if the input is a video file
Camera.RGB: 0

# For pinhole models, Bf = baseline * fx (in pixels)
# For RGB-D without stereo, set to 0
Camera.bf: 0

#--------------------------------------------------------------------------------------------
# Depth Parameters (RGB-D)
#--------------------------------------------------------------------------------------------

# DepthMapFactor: scales depth values
# For Realsense (millimeters): 1000
# For normalized [0,1]: 1
Camera.DepthMapFactor: 1000.0

# Maximum depth to be considered valid (meters)
ThDepth: 40.0

#--------------------------------------------------------------------------------------------
# IMU Parameters (not used in RGBD-only mode)
#--------------------------------------------------------------------------------------------

# Camera-IMU Transformation Matrix (if no IMU, use identity or this will be ignored)
# Tbc: SE(3) transform from camera to IMU frame
# Format: 
#   Tbc.tx: translation x
#   Tbc.ty: translation y
#   Tbc.tz: translation z
#   Tbc.qx, Tbc.qy, Tbc.qz, Tbc.qw: rotation quaternion (identity for no extrinsic)
Tbc.tx: 0.0
Tbc.ty: 0.0
Tbc.tz: 0.0
Tbc.qx: 0.0
Tbc.qy: 0.0
Tbc.qz: 0.0
Tbc.qw: 1.0

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
    if len(sys.argv) != 4:
        print("Usage: python generate_settings.py <camera_params.json> <cam_id> <output.yaml>")
        print("Example: python generate_settings.py camera_params.json 07 cam07_settings.yaml")
        sys.exit(1)
    
    camera_params_file = sys.argv[1]
    cam_id = sys.argv[2]
    output_yaml = sys.argv[3]
    
    try:
        generate_settings_yaml(camera_params_file, cam_id, output_yaml)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
