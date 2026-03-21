"""
Real-time 3D visualization for RGB-D trajectory with camera poses and point cloud.
Uses Open3D for fast, interactive visualization.
"""

import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
import cv2

try:
    import open3d as o3d
except ImportError:
    raise ImportError("Please install open3d: pip install open3d")


class CameraPoseVisualizer:
    """Visualize camera poses along trajectory with point cloud."""
    
    def __init__(self, width: int = 1920, height: int = 1440, window_name: str = "ORB-SLAM3 Trajectory"):
        """
        Initialize visualizer.
        
        Args:
            width: Visualization window width
            height: Visualization window height
            window_name: Window title
        """
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name=window_name, width=width, height=height, left=0, top=0)
        
        # Visualization elements
        self.trajectory_line = None
        self.camera_models: List[o3d.geometry.TriangleMesh] = []
        self.point_cloud = o3d.geometry.PointCloud()
        self.coordinate_frames: List[o3d.geometry.TriangleMesh] = []
        
        # Settings
        self.render_options = self.vis.get_render_option()
        self.render_options.background_color = np.array([0.8, 0.8, 0.8])
        self.render_options.point_size = 5.0  # Larger points for better visibility
        self.render_options.line_width = 2.0
        
        # Camera control
        self.view_control = self.vis.get_view_control()
        self._setup_view()
    
    def _setup_view(self):
        """Setup default camera view (third-person perspective)."""
        # Set a good third-person view
        camera_params = o3d.io.read_pinhole_camera_intrinsic(
            "pinhole_camera_intrinsic.json"
        ) if Path("pinhole_camera_intrinsic.json").exists() else None
        
        # Use view control to set viewing angle
        self.view_control.set_zoom(0.8)
        self.view_control.rotate(100, 50)  # Third-person view
    
    def add_trajectory(self, poses: List[np.ndarray], color: np.ndarray = None):
        """
        Add trajectory as a line connecting camera positions.
        
        Args:
            poses: List of 4x4 pose matrices (camera to world)
            color: RGB color for trajectory [0, 1]
        """
        if color is None:
            color = np.array([0.0, 1.0, 0.0])  # Green
        
        # Extract translation vectors
        positions = np.array([pose[:3, 3] for pose in poses])
        
        # Create line from positions
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(positions)
        
        # Create connections (line segments)
        if len(positions) > 1:
            lines = np.array([[i, i + 1] for i in range(len(positions) - 1)])
            line_set.lines = o3d.utility.Vector2iVector(lines)
        
        # Set color
        colors = np.tile(color, (len(lines) if len(positions) > 1 else 1, 1))
        line_set.colors = o3d.utility.Vector3dVector(colors)
        
        self.trajectory_line = line_set
        self.vis.add_geometry(self.trajectory_line, reset_bounding_box=False)
    
    def add_camera_model(self, pose: np.ndarray, size: float = 0.1, 
                        focal_length: float = 500, is_keyframe: bool = False):
        """
        Add camera frustum at given pose.
        
        Args:
            pose: 4x4 pose matrix (camera to world)
            size: Size of camera model
            focal_length: Camera focal length for frustum shape
            is_keyframe: If True, use red; otherwise blue
        """
        # Create camera frustum mesh
        mesh = self._create_camera_mesh(size, focal_length)
        
        # Transform to world coordinates
        mesh.transform(pose)
        
        # Color: red for keyframes, blue for others
        color = np.array([1.0, 0.0, 0.0]) if is_keyframe else np.array([0.0, 0.0, 1.0])
        mesh.paint_uniform_color(color)
        
        self.vis.add_geometry(mesh, reset_bounding_box=False)
        self.camera_models.append(mesh)
    
    @staticmethod
    def _create_camera_mesh(size: float = 0.1, focal_length: float = 500) -> o3d.geometry.TriangleMesh:
        """
        Create a simple camera frustum mesh.
        Simple pyramid: apex at origin, base at distance.
        """
        # Camera pyramid vertices (in camera frame)
        # Apex at origin, four corners of frustum base
        vertices = np.array([
            [0, 0, 0],  # Camera center (apex)
            [-size, -size, size * focal_length],  # Top-left
            [size, -size, size * focal_length],   # Top-right
            [size, size, size * focal_length],    # Bottom-right
            [-size, size, size * focal_length],   # Bottom-left
        ])
        
        # Faces (pyramid)
        faces = np.array([
            [0, 1, 2],  # Left face
            [0, 2, 3],  # Right face
            [0, 3, 4],  # Bottom face
            [0, 4, 1],  # Top face
            [1, 2, 3],  # Back face 1
            [1, 3, 4],  # Back face 2
        ])
        
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(vertices),
            o3d.utility.Vector3iVector(faces)
        )
        mesh.compute_vertex_normals()
        return mesh
    
    def add_point_cloud(self, rgb_images: List[np.ndarray], depth_images: List[np.ndarray],
                       poses: List[np.ndarray], intrinsics: np.ndarray,
                       depth_scale: float = 1000.0, sample_rate: int = 5):
        """
        Build and add point cloud from RGB-D images and poses.
        
        Args:
            rgb_images: List of RGB images (H, W, 3)
            depth_images: List of depth images (H, W) in mm
            poses: List of 4x4 pose matrices (camera to world)
            intrinsics: 3x3 camera intrinsic matrix
            depth_scale: Depth image scale (mm to meters)
            sample_rate: Sample every nth pixel to reduce computation
        """
        print(f"Creating point cloud from {len(rgb_images)} images...")
        points = []
        colors = []
        
        # Process every nth image
        for frame_idx in range(0, len(rgb_images), sample_rate):
            if frame_idx >= len(depth_images) or frame_idx >= len(poses):
                continue
                
            rgb = rgb_images[frame_idx]
            depth = depth_images[frame_idx]
            pose = poses[frame_idx]
            
            # Skip if invalid
            if rgb is None or depth is None:
                continue
            
            # Convert depth from mm to meters
            depth_m = depth.astype(np.float32) / float(depth_scale)
            
            # Get valid depth pixels (depth > 0.05m and < 5m)
            valid = (depth_m > 0.05) & (depth_m < 5.0)
            
            if not valid.any():
                continue
            
            # Back-project using intrinsics
            h, w = depth_m.shape
            v, u = np.where(valid)
            
            # Normalize image coordinates using camera intrinsics
            # K = [[fx,  0, cx],
            #      [ 0, fy, cy],
            #      [ 0,  0,  1]]
            fx = intrinsics[0, 0]
            fy = intrinsics[1, 1]
            cx = intrinsics[0, 2]
            cy = intrinsics[1, 2]
            
            x = (u - cx) / fx
            y = (v - cy) / fy
            z = depth_m[valid]
            
            # 3D points in camera frame
            pts_cam = np.stack([x * z, y * z, z], axis=-1)  # (N, 3)
            
            # Transform to world frame: P_world = R @ P_cam + t
            R = pose[:3, :3]
            t = pose[:3, 3]
            pts_world = (R @ pts_cam.T).T + t
            points.append(pts_world)
            
            # Get corresponding RGB values and normalize
            rgb_vals = rgb[valid].astype(np.float32) / 255.0
            colors.append(rgb_vals)
            
            print(f"  Frame {frame_idx}: {len(pts_world)} points")
        
        if not points:
            print("Warning: No valid points in point cloud")
            return
            
        # Combine all points and colors
        all_points = np.vstack(points)
        all_colors = np.vstack(colors)
        
        print(f"Total points: {len(all_points)}, colors shape: {all_colors.shape}")
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_points)
        pcd.colors = o3d.utility.Vector3dVector(all_colors)
        
        # Add to visualizer
        self.vis.add_geometry(pcd, reset_bounding_box=False)
        print("Point cloud added to visualizer")
    
    def add_coordinate_frame(self, pose: np.ndarray, size: float = 0.1):
        """
        Add coordinate frame (X-Red, Y-Green, Z-Blue) at pose.
        
        Args:
            pose: 4x4 pose matrix
            size: Size of coordinate frame
        """
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        coord_frame.transform(pose)
        self.vis.add_geometry(coord_frame, reset_bounding_box=False)
        self.coordinate_frames.append(coord_frame)
    
    def update_camera_position(self, eye: np.ndarray, target: np.ndarray, up: np.ndarray):
        """
        Update camera view parameters.
        
        Args:
            eye: Camera position in world coordinates
            target: Target point to look at
            up: Up vector
        """
        self.view_control.set_lookat(target)
        self.view_control.set_front(eye - target)
        self.view_control.set_up(up)
    
    def render(self):
        """Render one frame."""
        self.vis.poll_events()
        self.vis.update_renderer()
    
    def run(self, title: str = "ORB-SLAM3 Trajectory Visualization"):
        """
        Run interactive visualization loop.
        
        Args:
            title: Window title (note: cannot be changed after creation in Open3D)
        """
        # Note: Window title is set during initialization, cannot be changed dynamically
        # Interactive visualization loop
        while True:
            self.render()
            
            # Check if window is still open
            if not self.vis.poll_events():
                break
    
    def save_screenshot(self, filename: str):
        """Save screenshot of current view."""
        self.vis.capture_screen_float_buffer(filename)
        print(f"Screenshot saved: {filename}")
    
    def destroy(self):
        """Cleanup and close window."""
        self.vis.destroy_window()


def visualize_trajectory(
    poses: List[np.ndarray],
    rgb_images: Optional[List[np.ndarray]] = None,
    depth_images: Optional[List[np.ndarray]] = None,
    intrinsics: Optional[np.ndarray] = None,
    depth_scale: float = 1000.0,
    keyframe_indices: Optional[List[int]] = None,
    interactive: bool = True,
    window_name: str = "ORB-SLAM3 Trajectory",
):
    """
    Visualize trajectory with optional point cloud.
    
    Args:
        poses: List of 4x4 camera pose matrices (camera to world)
        rgb_images: List of RGB images (optional for point cloud)
        depth_images: List of depth images in mm (optional for point cloud)
        intrinsics: 3x3 camera intrinsic matrix (required if using images)
        depth_scale: Depth image scale factor (default 1000 = mm to meters)
        keyframe_indices: Indices of keyframes for highlighting
        interactive: If True, run interactive viewer; else just render
    """
    visualizer = CameraPoseVisualizer(window_name=window_name)
    
    # Add trajectory
    visualizer.add_trajectory(poses)
    
    # Add camera models
    for idx, pose in enumerate(poses):
        is_keyframe = keyframe_indices is not None and idx in keyframe_indices
        visualizer.add_camera_model(pose, size=0.05, is_keyframe=is_keyframe)
    
    # Add coordinate frame at first pose
    if len(poses) > 0:
        visualizer.add_coordinate_frame(poses[0], size=0.2)
    
    # Add point cloud if images provided
    if rgb_images is not None and depth_images is not None and intrinsics is not None:
        print("Building point cloud... this may take a moment")
        # Adaptive sampling rate based on number of frames
        num_frames = len(rgb_images)
        if num_frames <= 10:
            sample_rate = 1  # Use all pixels for few frames
        elif num_frames <= 30:
            sample_rate = 2  # Use half the pixels
        else:
            sample_rate = 5  # Use 1/5 of pixels for many frames
        
        print(f"Using sample_rate={sample_rate} for {num_frames} frames")
        visualizer.add_point_cloud(
            rgb_images, depth_images, poses,
            intrinsics, depth_scale, sample_rate=sample_rate
        )
    
    # Run or render
    if interactive:
        visualizer.run("ORB-SLAM3 Trajectory Visualization")
    else:
        visualizer.render()
    
    visualizer.destroy()


if __name__ == "__main__":
    # Example usage
    print("Realtime visualizer module. Use in run_vio.py with --visualize flag.")
