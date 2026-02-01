"""
Depth Estimation from 2D Images using MiDaS
Converts 2D knuckle images to 3D point clouds
"""

import torch
import cv2
import numpy as np
from PIL import Image


class DepthEstimator:
    """
    Monocular Depth Estimation using MiDaS
    Converts 2D images to depth maps, then to 3D point clouds
    """
    
    def __init__(self, model_type="DPT_Large"):
        """
        Initialize MiDaS depth estimator
        
        Args:
            model_type: "DPT_Large", "DPT_Hybrid", or "MiDaS_small"
        """
        print(f"[DepthEstimator] Loading MiDaS model: {model_type}...")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            # Load MiDaS model
            self.midas = torch.hub.load("intel-isl/MiDaS", model_type)
            self.midas.to(self.device)
            self.midas.eval()
            
            # Load transforms
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            
            if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
                self.transform = midas_transforms.dpt_transform
            else:
                self.transform = midas_transforms.small_transform
            
            print(f"[DepthEstimator] ✓ MiDaS loaded on {self.device}")
            
        except Exception as e:
            print(f"[DepthEstimator] Error loading MiDaS: {e}")
            print("[DepthEstimator] Using fallback depth estimation")
            self.midas = None
    
    
    def estimate_depth(self, image_path):
        """
        Estimate depth map from 2D image
        
        Args:
            image_path: Path to input image
        
        Returns:
            depth_map: (H, W) numpy array with depth values
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.midas is not None:
            # Use MiDaS
            input_batch = self.transform(img_rgb).to(self.device)
            
            with torch.no_grad():
                prediction = self.midas(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img_rgb.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            
            depth_map = prediction.cpu().numpy()
        else:
            # Fallback: Use gradient-based depth estimation
            depth_map = self._fallback_depth_estimation(img_rgb)
        
        # Normalize to [0, 1]
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
        
        return depth_map
    
    
    def _fallback_depth_estimation(self, img_rgb):
        """
        Simple gradient-based depth estimation (fallback)
        """
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        
        # Compute gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Magnitude
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Invert (darker = closer)
        depth = 255 - gray.astype(float)
        depth = depth + magnitude * 0.3
        
        return depth
    
    
    def depth_to_pointcloud(self, image_path, depth_map=None, num_points=1024):
        """
        Convert depth map to 3D point cloud
        
        Args:
            image_path: Path to image
            depth_map: Pre-computed depth map (optional)
            num_points: Target number of points
        
        Returns:
            points: (num_points, 3) numpy array - 3D coordinates
        """
        if depth_map is None:
            depth_map = self.estimate_depth(image_path)
        
        # Load image for texture info
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        h, w = depth_map.shape
        
        # Create meshgrid
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        
        # Normalize coordinates to [-1, 1]
        xx = (xx / w - 0.5) * 2
        yy = (yy / h - 0.5) * 2
        
        # Create point cloud
        points = np.stack([
            xx.flatten(),
            yy.flatten(),
            depth_map.flatten()
        ], axis=1)
        
        # Remove invalid points
        valid_mask = ~np.isnan(points).any(axis=1)
        points = points[valid_mask]
        
        # Subsample to target number of points
        if len(points) > num_points:
            # Farthest Point Sampling
            indices = self._farthest_point_sampling(points, num_points)
            points = points[indices]
        elif len(points) < num_points:
            # Pad with duplicates
            pad_size = num_points - len(points)
            pad_indices = np.random.choice(len(points), pad_size)
            points = np.vstack([points, points[pad_indices]])
        
        return points
    
    
    def _farthest_point_sampling(self, points, num_samples):
        """
        Farthest Point Sampling for point cloud downsampling
        """
        N = points.shape[0]
        centroids = np.zeros(num_samples, dtype=np.int32)
        distance = np.ones(N) * 1e10
        farthest = np.random.randint(0, N)
        
        for i in range(num_samples):
            centroids[i] = farthest
            centroid = points[farthest, :]
            dist = np.sum((points - centroid) ** 2, axis=1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = np.argmax(distance)
        
        return centroids
    
    
    def process_multiple_images(self, image_paths, num_points=1024):
        """
        Process multiple images and merge into single point cloud
        
        Args:
            image_paths: List of image paths
            num_points: Total points in merged cloud
        
        Returns:
            merged_cloud: (num_points, 3) numpy array
        """
        all_points = []
        
        for img_path in image_paths:
            points = self.depth_to_pointcloud(img_path, num_points=num_points//len(image_paths))
            all_points.append(points)
        
        merged = np.vstack(all_points)
        
        # Resample to exact num_points
        if len(merged) > num_points:
            indices = self._farthest_point_sampling(merged, num_points)
            merged = merged[indices]
        
        return merged


if __name__ == "__main__":
    print("="*60)
    print("Testing Depth Estimator")
    print("="*60)
    
    estimator = DepthEstimator(model_type="MiDaS_small")
    
    print("\n✓ Depth estimator ready!")
    print("\nTo use:")
    print("  depth_map = estimator.estimate_depth('image.jpg')")
    print("  points_3d = estimator.depth_to_pointcloud('image.jpg')")
    print("="*60)
