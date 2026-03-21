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
    
    def __init__(self, width: int = 1920, height: int = 1440):
        """
        Initialize visualizer.
        
        Args:
            width: Visualization window width
            height: Visualization window height
        """
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=width, height=height, left=0, top=0)
        
        # Visualization elements
        self.trajectory_line = None
        self.camera_models: List[o3d.geometry.TriangleMesh] = []
        self.point_cloud = o3d.geometry.PointCloud()
        self.coordinate_frames: List[o3d.geometry.TriangleMesh] = []
        
        # Settings
        self.render_options = self.vis.get_render_option()
        self.render_options.background_color = np.array([0.8, 0.8, 0.8])
        self.render_options.point_size = 2.0
        
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
        points = []
        colors = []
        
        # Process every nth image
        for idx in range(0, len(rgb_images), sample_rate):
            rgb = rgb_images[idx]
            depth = depth_images[idx] / depth_scale  # Convert to meters
            pose = poses[idx]
            
            # Get valid depth pixels
            valid = depth > 0.1  # Skip close artifacts and invalid pixels
            
            if not valid.any():
                continue
            
            # Back-project using intrinsics
            h, w = depth.shape
            v, u = np.where(valid)
            
            # Normalize image coordinates
            x = (u - intrinsics[0, 2]) / intrinsics[0, 0]
            y = (v - intrinsics[1, 2]) / intrinsics[1, 1]
            z = depth[valid]
            
            # 3D points in camera frame
            pts_cam = np.stack([x * z, y * z, z], axis=-1)  # (N, 3)
            
            # Transform to world frame
            pts_world = (pose[:3, :3] @ pts_cam.T + pose[:3, 3:]).T
            points.append(pts_world)
            
            # Get corresponding RGB values
            colors.append(rgb[valid] / 255.0)  # Normalize to [0, 1]
        
        if points:
            points = np.vstack(points)
            colors = np.vstack(colors)
            
            self.point_cloud.points = o3d.utility.Vector3dVector(points)
            self.point_cloud.colors = o3d.utility.Vector3dVector(colors)
            self.vis.add_geometry(self.point_cloud, reset_bounding_box=False)
    
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
            title: Window title
        """
        self.vis.get_window().set_title(title)
        
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
    visualizer = CameraPoseVisualizer()
    
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
        visualizer.add_point_cloud(
            rgb_images, depth_images, poses,
            intrinsics, depth_scale, sample_rate=5
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
