import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Union
import cv2


def visualize_pointcloud_with_trajectories(
    pointcloud: np.ndarray,
    trajectory1: np.ndarray,
    trajectory2: Optional[np.ndarray] = None,
    pointcloud_colors: Optional[np.ndarray] = None,
    # Default colors: trajectory1 uses Metric3D GT color (#FDE725), trajectory2 uses #6DCD59
    trajectory1_color: Tuple[float, float, float] = (253/255.0, 231/255.0, 37/255.0),
    trajectory2_color: Tuple[float, float, float] = (109/255.0, 205/255.0, 89/255.0),
    image_size: Tuple[int, int] = (800, 600),
    background_color: Tuple[float, float, float] = (0.1, 0.1, 0.1),  # Dark gray
    point_size: float = 1.0,
    line_width: float = 5.0,
    camera_distance: float = 3.0,
    trajectory_as_lines: bool = True
) -> List[np.ndarray]:
    """
    Visualize a pointcloud with one or two 3D trajectories from 4 different camera viewpoints.
    
    Args:
        pointcloud: Point cloud data (N, 3) where N is number of points
        trajectory1: First trajectory (M, 3) where M is number of trajectory points
        trajectory2: Optional second trajectory (K, 3), can be None
        pointcloud_colors: Optional colors for pointcloud (N, 3) in range [0, 1]. If None, uses white
        trajectory1_color: RGB color for first trajectory (0-1 range)
        trajectory2_color: RGB color for second trajectory (0-1 range)
        image_size: Output image size (width, height)
        background_color: Background color (R, G, B) in range [0, 1]
        point_size: Size of points in the pointcloud
        line_width: Width of trajectory lines
        camera_distance: Distance of camera from center
        trajectory_as_lines: If True, draw trajectories as connected lines, else as points
    
    Returns:
        List of 5 numpy arrays, each representing a rendered image from different viewpoints
    """
    
    # Validate inputs
    if pointcloud.shape[1] != 3:
        raise ValueError("Pointcloud must have shape (N, 3)")
    if trajectory1.shape[1] != 3:
        raise ValueError("Trajectory1 must have shape (M, 3)")
    if trajectory2 is not None and trajectory2.shape[1] != 3:
        raise ValueError("Trajectory2 must have shape (K, 3)")
    
    # Create Open3D visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=image_size[0], height=image_size[1], visible=False)
    
    # Create pointcloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud)
    
    # Set pointcloud colors
    if pointcloud_colors is not None:
        if pointcloud_colors.shape != pointcloud.shape:
            raise ValueError("Pointcloud colors must have same shape as pointcloud")
        # Ensure colors are in [0, 1] range
        if pointcloud_colors.max() > 1.0:
            pointcloud_colors = pointcloud_colors / 255.0
        pcd.colors = o3d.utility.Vector3dVector(pointcloud_colors)
    else:
        # Default to white
        pcd.colors = o3d.utility.Vector3dVector(np.ones_like(pointcloud) * 0.8)
    
    # Create trajectory line sets
    trajectory_geometries = []
    
    # First trajectory
    if len(trajectory1) > 1:
        if trajectory_as_lines:
            lines = [[i, i + 1] for i in range(len(trajectory1) - 1)]
            line_set1 = o3d.geometry.LineSet()
            line_set1.points = o3d.utility.Vector3dVector(trajectory1)
            line_set1.lines = o3d.utility.Vector2iVector(lines)
            line_set1.colors = o3d.utility.Vector3dVector([trajectory1_color] * len(lines))
            trajectory_geometries.append(line_set1)
        else:
            # Draw as points
            traj_pcd1 = o3d.geometry.PointCloud()
            traj_pcd1.points = o3d.utility.Vector3dVector(trajectory1)
            traj_pcd1.colors = o3d.utility.Vector3dVector([trajectory1_color] * len(trajectory1))
            trajectory_geometries.append(traj_pcd1)
    
    # Second trajectory (if provided)
    if trajectory2 is not None and len(trajectory2) > 1:
        if trajectory_as_lines:
            lines = [[i, i + 1] for i in range(len(trajectory2) - 1)]
            line_set2 = o3d.geometry.LineSet()
            line_set2.points = o3d.utility.Vector3dVector(trajectory2)
            line_set2.lines = o3d.utility.Vector2iVector(lines)
            line_set2.colors = o3d.utility.Vector3dVector([trajectory2_color] * len(lines))
            trajectory_geometries.append(line_set2)
        else:
            # Draw as points
            traj_pcd2 = o3d.geometry.PointCloud()
            traj_pcd2.points = o3d.utility.Vector3dVector(trajectory2)
            traj_pcd2.colors = o3d.utility.Vector3dVector([trajectory2_color] * len(trajectory2))
            trajectory_geometries.append(traj_pcd2)
    
    # Calculate scene bounds for camera positioning
    all_points = [pointcloud, trajectory1]
    if trajectory2 is not None:
        all_points.append(trajectory2)
    all_points = np.vstack(all_points)
    
    center = np.mean(all_points, axis=0)
    bbox_size = np.max(all_points, axis=0) - np.min(all_points, axis=0)
    max_extent = np.max(bbox_size)
    
    # Define camera basis consistent with the point cloud convention from create_pointcloud.py:
    # X: right (+), Y: down (+), Z: forward (+)
    # Therefore the visual "up" direction is -Y.
    base_front = np.array([0.0, 0.0, 1.0], dtype=np.float64)  # look along +Z
    base_up = np.array([0.0, -1.0, 0.0], dtype=np.float64)    # up is -Y
    base_right = np.cross(base_front, base_up)                # should be +X
    base_front /= np.linalg.norm(base_front)
    base_up /= np.linalg.norm(base_up)
    base_right /= np.linalg.norm(base_right)

    # Prepare five slight view variants (yaw, pitch in degrees)
    # Positive yaw rotates to the right around the up axis; positive pitch pitches downward (toward +Z) around the right axis
    yaw_pitch_list = [
        (0.0, 0.0),        # center
        (25.0, -8.0),      # right, up
        (-25.0, -8.0),     # left, up
        (0.0, -18.0),      # more up
        (0.0, 18.0),       # more down
    ]

    # All cameras orbit the scene center
    look_at_point = center
    
    rendered_images = []
    
    for i, (yaw_deg, pitch_deg) in enumerate(yaw_pitch_list):
        # Clear previous geometries
        vis.clear_geometries()
        
        # Add pointcloud and trajectories
        vis.add_geometry(pcd)
        for geom in trajectory_geometries:
            vis.add_geometry(geom)
        
        # Set render options
        render_option = vis.get_render_option()
        render_option.background_color = np.array(background_color)
        render_option.point_size = point_size
        render_option.line_width = line_width
        
        # Set camera parameters
        view_control = vis.get_view_control()
        
        # Build rotation from yaw (around up) and pitch (around right)
        yaw_rad = np.deg2rad(yaw_deg)
        pitch_rad = np.deg2rad(pitch_deg)
        # Rodrigues expects axis-angle; ensure axes are unit vectors
        R_yaw, _ = cv2.Rodrigues(base_up * yaw_rad)
        R_pitch, _ = cv2.Rodrigues(base_right * pitch_rad)
        R = R_pitch @ R_yaw

        rotated_front = (R @ base_front).astype(np.float64)
        rotated_up = (R @ base_up).astype(np.float64)

        view_control.set_lookat(look_at_point)
        # Invert front to avoid ending up on the opposite side of the scene
        view_control.set_front((-rotated_front).tolist())
        view_control.set_up(rotated_up.tolist())
        # Increase zoom for a closer framing
        view_control.set_zoom(0.6)
        
        # Update and render
        vis.poll_events()
        vis.update_renderer()
        
        # Capture image
        image = vis.capture_screen_float_buffer(do_render=True)
        image_np = np.asarray(image)
        
        # Convert from float [0,1] to uint8 [0,255]
        image_np = (image_np * 255).astype(np.uint8)
        
        rendered_images.append(image_np)
    
    # Clean up
    vis.destroy_window()
    
    return rendered_images


def save_visualization_images(
    images: List[np.ndarray], 
    output_prefix: str, 
    format: str = 'png'
) -> List[str]:
    """
    Save rendered images to files.
    
    Args:
        images: List of rendered images from visualize_pointcloud_with_trajectories
        output_prefix: Prefix for output filenames (e.g., 'scene_01')
        format: Image format ('png', 'jpg', etc.)
    
    Returns:
        List of saved file paths
    """
    saved_paths = []
    
    for i, image in enumerate(images):
        filename = f"{output_prefix}_view_{i+1}.{format}"
        
        # OpenCV expects BGR format
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image
            
        cv2.imwrite(filename, image_bgr)
        saved_paths.append(filename)
    
    return saved_paths


def create_combined_view(images: List[np.ndarray]) -> np.ndarray:
    """
    Combine 5 rendered images into a single grid image.
    Layout: [0] [1] [2]
            [3] [4] [ ]
    
    Args:
        images: List of 5 rendered images
    
    Returns:
        Combined image as numpy array
    """
    if len(images) != 5:
        raise ValueError("Expected exactly 5 images")
    
    # Ensure all images have the same size
    h, w = images[0].shape[:2]
    for img in images[1:]:
        if img.shape[:2] != (h, w):
            raise ValueError("All images must have the same dimensions")
    
    # Create 2x3 grid with last spot empty
    # Top row: images 0, 1, 2
    # Bottom row: images 3, 4, empty
    
    # Create empty image for the last spot
    if len(images[0].shape) == 3:
        empty_img = np.zeros_like(images[0])
    else:
        empty_img = np.zeros((h, w), dtype=images[0].dtype)
    
    top_row = np.hstack([images[0], images[1], images[2]])
    bottom_row = np.hstack([images[3], images[4], empty_img])
    combined = np.vstack([top_row, bottom_row])
    
    return combined
