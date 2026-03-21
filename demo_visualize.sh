#!/bin/bash
# Quick visualization demo script
# Usage: ./demo_visualize.sh

set -e

# Check dependencies
python3 -c "import open3d" 2>/dev/null || {
    echo "Installing Open3D..."
    pip install open3d scipy -q
}

echo "==========================================
3D Visualization Demo for ORB-SLAM3
=========================================="

# Settings
DATASET="${1:-.}"
CAM_ID="${2:-07}"
MAX_FRAMES="${3:-100}"
VIS_MODE="${4:-basic}"  # basic or full

# Auto-detect paths
if [ ! -f "cam${CAM_ID}_settings.yaml" ]; then
    echo "⚠️  Settings file not found: cam${CAM_ID}_settings.yaml"
    echo "   Please run: python generate_settings.py factory_camera_imu_calib.json cam${CAM_ID}_settings.yaml --factory"
    exit 1
fi

if [ ! -f "./build/rgbd_inertial_offline" ]; then
    echo "⚠️  ORB-SLAM3 executable not found: ./build/rgbd_inertial_offline"
    echo "   Please build first: cmake -S cpp -B build && cmake --build build -j"
    exit 1
fi

ORB_EXEC="./build/rgbd_inertial_offline"
VOCAB="./ORB-SLAM3/Vocabulary/ORBvoc.txt"
if [ ! -f "$VOCAB" ]; then
    # Try alternative path
    VOCAB="/home/ubuntu/WorkSpace/ZYC/cam_pose/ORB-SLAM3/Vocabulary/ORBvoc.txt"
fi

if [ ! -f "$VOCAB" ]; then
    echo "⚠️  ORB vocabulary not found. Edit this script to set correct VOCAB path."
    exit 1
fi

OUT_DIR="./output_vis_demo_cam${CAM_ID}"

echo "
Configuration:
  Dataset: $DATASET
  Camera ID: $CAM_ID
  Max frames: $MAX_FRAMES
  Visualization: $VIS_MODE
  Output: $OUT_DIR
  Settings: cam${CAM_ID}_settings.yaml
  ORB-SLAM3: $ORB_EXEC
  Vocabulary: $VOCAB
"

# Build command
CMD="python run_vio.py"
CMD="$CMD --dataset $DATASET"
CMD="$CMD --cam_id $CAM_ID"
CMD="$CMD --sensor_mode rgbd"
CMD="$CMD --run_slam"
CMD="$CMD --orb_exec $ORB_EXEC"
CMD="$CMD --vocab $VOCAB"
CMD="$CMD --settings cam${CAM_ID}_settings.yaml"
CMD="$CMD --out_dir $OUT_DIR"
CMD="$CMD --max_frames $MAX_FRAMES"
CMD="$CMD --visualize"

if [ "$VIS_MODE" = "full" ]; then
    CMD="$CMD --vis_with_pointcloud"
    echo "🔶 Running with point cloud (slower)..."
else
    echo "🔵 Running basic visualization..."
fi

echo ""
echo "Running: $CMD"
echo ""

$CMD

echo ""
echo "✓ Visualization complete!"
echo "Results saved to: $OUT_DIR/"
