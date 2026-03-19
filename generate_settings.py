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
        cam_id: Camera ID (e.g., "07", "07_right", "7", etc.)
        output_yaml: Output YAML file path
    """
       
    # Load camera parameters
    with open(camera_params_file, 'r') as f:
        params = json.load(f)
    
    # Find camera entry (handle multiple id formats)
    cam_key = None
    for key in params.get('cameras', {}).keys():
        # Match 07 or cam_07 or camera_07 or 7
        if str(cam_id) in str(key) or key.endswith(str(cam_id)):
            cam_key = key
            break
    
    if not cam_key:
        raise ValueError(f"Camera {cam_id} not found in {camera_params_file}. "
                         f"Available cameras: {list(params.get('cameras', {}).keys())}")
    
    cam = params['cameras'][cam_key]
    
    # Extract intrinsics
    # Common keys: fx, fy, cx, cy (or K[0,0], K[1,1], K[0,2], K[1,2])
    if 'intrinsics' in cam:
        intr = cam['intrinsics']
        fx = intr.get('fx', intr.get('K', [[None]*3]*3)[0][0])
        fy = intr.get('fy', intr.get('K', [[None]*3]*3)[1][1])
        cx = intr.get('cx', intr.get('K', [[None]*3]*3)[0][2])
        cy = intr.get('cy', intr.get('K', [[None]*3]*3)[1][2])
    else:
        fx = cam.get('fx')
        fy = cam.get('fy')
        cx = cam.get('cx')
        cy = cam.get('cy')
    
    if None in [fx, fy, cx, cy]:
        raise ValueError(f"Missing intrinsics for camera {cam_id}: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
    
    # Extract distortion parameters
    # Common formats: k1, k2, k3, p1, p2 (OpenCV 5-param model)
    dist = cam.get('distortion', {})
    k1 = dist.get('k1', 0.0)
    k2 = dist.get('k2', 0.0)
    k3 = dist.get('k3', 0.0)
    p1 = dist.get('p1', 0.0)
    p2 = dist.get('p2', 0.0)
    
    # Generate YAML content
    yaml_content = f"""%YAML:1.0

# Camera calibration parameters for ORB-SLAM3
# Generated from {Path(camera_params_file).name}
# Camera ID: {cam_id}

# Camera calibration and distortion parameters (OpenCV model)
Camera.type: "PinHole"

# Camera.fx: focal length in x with value in pixels
Camera.fx: {fx}

# Camera.fy: focal length in y with value in pixels
Camera.fy: {fy}

# Camera.cx: principal point x coordinate in pixels
Camera.cx: {cx}

# Camera.cy: principal point y coordinate in pixels
Camera.cy: {cy}

# Camera.k1: distortion parameter k1
Camera.k1: {k1}

# Camera.k2: distortion parameter k2
Camera.k2: {k2}

# Camera.p1: distortion parameter p1
Camera.p1: {p1}

# Camera.p2: distortion parameter p2
Camera.p2: {p2}

# Camera.k3: distortion parameter k3
Camera.k3: {k3}

# Camera distortion model - use "opencv" for k1,k2,p1,p2,k3
Camera.distortionModel: "opencv"

# DepthMapFactor: "1/DepthMapFactor" is the depth value in meters
# For Realsense in millimeters: DepthMapFactor = 1000
# For normalized depth [0,1]: DepthMapFactor = 1
Camera.DepthMapFactor: 1000.0

# RGB-D camera resolution (example, adjust as needed)
Camera.width: 1280
Camera.height: 800

# Viewer parameters
# These are for GUI visualization, use default values if --no-viewer is used
Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3
Viewer.ViewpointX: 0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -1.8
Viewer.ViewpointF: 500

# Feature detection parameters
ORBextractor.nFeatures: 1000
ORBextractor.scaleFactor: 1.2
ORBextractor.nLevels: 8
ORBextractor.fIniThFAST: 20
ORBextractor.fMinThFAST: 7

# Viewer refresh rate
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
