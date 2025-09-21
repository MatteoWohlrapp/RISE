#!/usr/bin/env python3
"""
Batch Visualization Script for Point Clouds and Trajectories

This script processes folders created by create_pointcloud.py and generates
visualizations for each scene containing:
- Point cloud from pointcloud.ply
- 3D trajectory from tmp_metadata.json (points_3d field)
- 4 different camera viewpoints for each scene

Usage:
    python batch_visualize.py <input_folder>

Input folder structure (created by create_pointcloud.py):
    input_folder/
    ├── scene_001/
    │   ├── pointcloud.ply
    │   ├── tmp_metadata.json  (contains points_3d trajectory)
    │   └── ...
    ├── scene_002/
    │   ├── pointcloud.ply
    │   ├── tmp_metadata.json
    │   └── ...
    └── ...

Output structure:
    input_folder/
    ├── scene_001/
    │   ├── pointcloud.ply
    │   ├── tmp_metadata.json
    │   └── vis/
    │       ├── view_1.png     # Camera perspective
    │       ├── view_2.png     # Slight right-back elevated
    │       ├── view_3.png     # Slight left-back elevated
    │       ├── view_4.png     # Right-front lower
    │       ├── view_5.png     # Left-front lower
    │       └── combined.png   # 2x3 grid of all views
    ├── scene_002/
    │   ├── pointcloud.ply
    │   ├── tmp_metadata.json
    │   └── vis/
    │       ├── view_1.png     # Camera perspective
    │       ├── view_2.png     # Slight right-back elevated
    │       ├── view_3.png     # Slight left-back elevated
    │       ├── view_4.png     # Right-front lower
    │       ├── view_5.png     # Left-front lower
    │       └── combined.png   # 2x3 grid of all views
    └── ...
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from plyfile import PlyData
import logging
from typing import Optional, Tuple, List

# Import the visualization functions
from utils.visualize import (
    visualize_pointcloud_with_trajectories,
    save_visualization_images,
    create_combined_view
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_pointcloud_from_ply(ply_path: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load point cloud data from PLY file.
    
    Args:
        ply_path: Path to PLY file
    
    Returns:
        tuple: (points, colors) where points is (N, 3) and colors is (N, 3) or None
    """
    try:
        ply_data = PlyData.read(str(ply_path))
        vertices = ply_data['vertex'].data
        
        # Extract coordinates
        points = np.column_stack([
            vertices['x'],
            vertices['y'], 
            vertices['z']
        ])
        
        # Extract colors if available
        colors = None
        field_names = vertices.dtype.names
        if field_names and 'red' in field_names and 'green' in field_names and 'blue' in field_names:
            colors = np.column_stack([
                vertices['red'],
                vertices['green'],
                vertices['blue']
            ]) / 255.0  # Normalize to [0, 1]
        
        logger.info(f"Loaded point cloud: {len(points)} points from {ply_path}")
        if colors is not None:
            logger.info(f"Loaded colors for {len(colors)} points")
        return points, colors
        
    except Exception as e:
        logger.error(f"Failed to load point cloud from {ply_path}: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def load_trajectory_from_metadata(metadata_path: Path) -> Optional[np.ndarray]:
    """
    Load 3D trajectory from tmp_metadata.json.
    
    Args:
        metadata_path: Path to tmp_metadata.json file
    
    Returns:
        3D trajectory as numpy array (N, 3) or None if not found/invalid
    """
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        if 'points_3d' not in metadata:
            logger.warning(f"No 'points_3d' field found in {metadata_path}")
            return None
        
        points_3d = np.array(metadata['points_3d'])
        
        # Validate shape
        if len(points_3d.shape) != 2 or points_3d.shape[1] != 3:
            logger.warning(f"Invalid points_3d shape in {metadata_path}: {points_3d.shape}")
            return None
        
        logger.info(f"Loaded trajectory: {len(points_3d)} points from {metadata_path}")
        return points_3d
        
    except Exception as e:
        logger.error(f"Failed to load trajectory from {metadata_path}: {e}")
        return None


def process_scene(scene_path: Path, image_size: Tuple[int, int] = (800, 600)) -> bool:
    """
    Process a single scene folder and generate visualizations.
    
    Args:
        scene_path: Path to scene folder
        image_size: Output image size (width, height)
    
    Returns:
        bool: Success status
    """
    scene_name = scene_path.name
    logger.info(f"Processing scene: {scene_name}")
    
    # Create vis directory inside scene folder
    vis_dir = scene_path / "vis"
    vis_dir.mkdir(exist_ok=True)
    
    # Check for required files
    pointcloud_path = scene_path / "pointcloud.ply"
    metadata_path = scene_path / "tmp_metadata.json"
    
    if not pointcloud_path.exists():
        logger.warning(f"No pointcloud.ply found in {scene_path}")
        return False
    
    if not metadata_path.exists():
        logger.warning(f"No tmp_metadata.json found in {scene_path}")
        return False
    
    try:
        # Load point cloud
        pointcloud, pointcloud_colors = load_pointcloud_from_ply(pointcloud_path)
        if pointcloud is None:
            logger.error(f"Failed to load point cloud for {scene_name}")
            return False
        
        # Load trajectory
        trajectory = load_trajectory_from_metadata(metadata_path)
        if trajectory is None:
            logger.error(f"Failed to load trajectory for {scene_name}")
            return False
        
        # Generate visualizations
        logger.info(f"Generating visualizations for {scene_name}...")
        images = visualize_pointcloud_with_trajectories(
            pointcloud=pointcloud,
            trajectory1=trajectory,
            trajectory2=None,  # Only one trajectory from metadata
            pointcloud_colors=pointcloud_colors,
            # Use Metric3D GT yellow (#FDE725)
            trajectory1_color=(253/255.0, 231/255.0, 37/255.0),
            image_size=image_size,
            background_color=(0.05, 0.05, 0.05),  # Dark background
            point_size=2.0,
            line_width=12.0,
            trajectory_as_lines=True
        )
        
        # Save individual view images
        output_prefix = str(vis_dir / "view")
        saved_paths = save_visualization_images(images, output_prefix, format='png')
        logger.info(f"Saved individual views: {saved_paths}")
        
        # Create and save combined view
        combined_image = create_combined_view(images)
        combined_path = vis_dir / "combined.png"
        
        # Save combined image (OpenCV format)
        import cv2
        combined_bgr = cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(combined_path), combined_bgr)
        logger.info(f"Saved combined view: {combined_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing scene {scene_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Batch visualization of point clouds and trajectories")
    parser.add_argument("--input_folder", type=str, help="Input folder containing scene subfolders")
    parser.add_argument("--image-size", nargs=2, type=int, default=[512, 512], 
                       help="Output image size (width height), default: 800 600")
    
    args = parser.parse_args()
    
    input_folder = Path(args.input_folder)
    
    if not input_folder.exists():
        logger.error(f"Input folder does not exist: {input_folder}")
        sys.exit(1)
    
    if not input_folder.is_dir():
        logger.error(f"Input path is not a directory: {input_folder}")
        sys.exit(1)
    
    logger.info(f"Processing scenes in: {input_folder}")
    
    # Find all scene subfolders with required files
    scene_folders = []
    for item in input_folder.iterdir():
        if item.is_dir():
            pointcloud_path = item / "pointcloud.ply"
            metadata_path = item / "tmp_metadata.json"
            if pointcloud_path.exists() and metadata_path.exists():
                scene_folders.append(item)
            else:
                logger.debug(f"Skipping {item.name} - missing required files")
    
    if not scene_folders:
        logger.warning(f"No valid scene folders found in {input_folder}")
        logger.info("Looking for folders containing both 'pointcloud.ply' and 'tmp_metadata.json'")
        sys.exit(1)
    
    logger.info(f"Found {len(scene_folders)} valid scene folders to process")
    
    # Process each scene
    success_count = 0
    image_size = tuple(args.image_size)
    
    for scene_folder in sorted(scene_folders):
        if process_scene(scene_folder, image_size):
            success_count += 1
    
    logger.info(f"Successfully processed {success_count}/{len(scene_folders)} scenes")
    logger.info(f"Visualizations saved inside each scene's 'vis/' directory")
    
    # Generate summary
    if success_count > 0:
        total_images = success_count * 6  # 5 views + 1 combined per scene
        logger.info(f"Generated {total_images} visualization images total")
        logger.info(f"Each scene has 5 individual views and 1 combined 2x3 grid in its vis/ folder")


if __name__ == "__main__":
    main()
