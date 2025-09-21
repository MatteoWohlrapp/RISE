#!/usr/bin/env python3
"""
Test script for visualization functions using matplotlib backend (fallback)
This can be used when Open3D is not available.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from typing import Optional, Tuple, List
import os


def visualize_pointcloud_with_trajectories_matplotlib(
    pointcloud: np.ndarray,
    trajectory1: np.ndarray,
    trajectory2: Optional[np.ndarray] = None,
    pointcloud_colors: Optional[np.ndarray] = None,
    trajectory1_color: Tuple[float, float, float] = (1.0, 0.0, 0.0),  # Red
    trajectory2_color: Tuple[float, float, float] = (0.0, 1.0, 0.0),  # Green
    image_size: Tuple[int, int] = (800, 600),
    background_color: Tuple[float, float, float] = (0.1, 0.1, 0.1),  # Dark gray
    point_size: float = 1.0,
    camera_distance: float = 3.0
) -> List[np.ndarray]:
    """
    Matplotlib-based fallback visualization function for testing purposes.
    """
    
    # Validate inputs
    if pointcloud.shape[1] != 3:
        raise ValueError("Pointcloud must have shape (N, 3)")
    if trajectory1.shape[1] != 3:
        raise ValueError("Trajectory1 must have shape (M, 3)")
    if trajectory2 is not None and trajectory2.shape[1] != 3:
        raise ValueError("Trajectory2 must have shape (K, 3)")
    
    # Calculate scene bounds for camera positioning
    all_points = [pointcloud, trajectory1]
    if trajectory2 is not None:
        all_points.append(trajectory2)
    all_points = np.vstack(all_points)
    
    center = np.mean(all_points, axis=0)
    bbox_size = np.max(all_points, axis=0) - np.min(all_points, axis=0)
    max_extent = np.max(bbox_size)
    
    # Define 4 different viewing angles
    elevation_angles = [20, 20, -10, 45]  # Different elevation angles
    azimuth_angles = [45, -45, 135, 0]    # Different azimuth angles
    
    rendered_images = []
    
    # Set up matplotlib for rendering without display
    plt.ioff()  # Turn off interactive mode
    
    for i, (elev, azim) in enumerate(zip(elevation_angles, azimuth_angles)):
        fig = plt.figure(figsize=(image_size[0]/100, image_size[1]/100), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        
        # Set background color
        fig.patch.set_facecolor(background_color)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor(background_color)
        ax.yaxis.pane.set_edgecolor(background_color)
        ax.zaxis.pane.set_edgecolor(background_color)
        
        # Plot point cloud
        if pointcloud_colors is not None:
            if pointcloud_colors.max() > 1.0:
                colors = pointcloud_colors / 255.0
            else:
                colors = pointcloud_colors
            ax.scatter(pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2], 
                      c=colors, s=point_size, alpha=0.6)
        else:
            ax.scatter(pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2], 
                      c='lightgray', s=point_size, alpha=0.6)
        
        # Plot first trajectory
        ax.plot(trajectory1[:, 0], trajectory1[:, 1], trajectory1[:, 2], 
                color=trajectory1_color, linewidth=3, label='Trajectory 1')
        
        # Plot second trajectory if provided
        if trajectory2 is not None:
            ax.plot(trajectory2[:, 0], trajectory2[:, 1], trajectory2[:, 2], 
                    color=trajectory2_color, linewidth=3, label='Trajectory 2')
        
        # Set viewing angle
        ax.view_init(elev=elev, azim=azim)
        
        # Set axis limits based on scene bounds
        margin = max_extent * 0.1
        ax.set_xlim(center[0] - max_extent/2 - margin, center[0] + max_extent/2 + margin)
        ax.set_ylim(center[1] - max_extent/2 - margin, center[1] + max_extent/2 + margin)
        ax.set_zlim(center[2] - max_extent/2 - margin, center[2] + max_extent/2 + margin)
        
        # Remove axis labels and ticks for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        
        # Render to numpy array
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        rendered_images.append(buf)
        
        plt.close(fig)
    
    plt.ion()  # Turn interactive mode back on
    
    return rendered_images


def test_visualization():
    """Test the visualization with sample data"""
    print("Testing visualization functions...")
    
    # Create sample pointcloud (random points in a box)
    np.random.seed(42)
    pointcloud = np.random.randn(500, 3) * 2
    
    # Create sample trajectory (spiral)
    t = np.linspace(0, 4*np.pi, 50)
    trajectory = np.column_stack([
        np.cos(t) * 2,
        np.sin(t) * 2,
        t * 0.1
    ])
    
    # Generate random colors for pointcloud
    colors = np.random.rand(500, 3)
    
    print("Generating visualization with matplotlib backend...")
    
    # Test matplotlib backend
    images = visualize_pointcloud_with_trajectories_matplotlib(
        pointcloud=pointcloud,
        trajectory1=trajectory,
        pointcloud_colors=colors,
        image_size=(400, 300)
    )
    
    print(f"Generated {len(images)} images")
    
    # Save test images
    os.makedirs("test_output", exist_ok=True)
    for i, image in enumerate(images):
        filename = f"test_output/test_view_{i+1}.png"
        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, image_bgr)
        print(f"Saved: {filename}")
    
    # Create combined view
    if len(images) == 4:
        # Create 2x2 grid
        h, w = images[0].shape[:2]
        top_row = np.hstack([images[0], images[1]])
        bottom_row = np.hstack([images[2], images[3]])
        combined = np.vstack([top_row, bottom_row])
        
        combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
        cv2.imwrite("test_output/test_combined.png", combined_bgr)
        print("Saved: test_output/test_combined.png")
    
    print("Test completed successfully!")


if __name__ == "__main__":
    test_visualization()
