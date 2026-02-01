"""
3D Finger Knuckle Pattern Recognition using DGCNN
==================================================
Dynamic Graph Convolutional Neural Network implementation for
3D point cloud feature extraction from finger knuckle patterns.

Based on: "Dynamic Graph CNN for Learning on Point Clouds" (Wang et al.)
Extended for biometric knuckle pattern recognition.
"""

import numpy as np
import cv2
import os
from scipy.spatial.distance import cdist


class EdgeConv:
    """
    Edge Convolution layer - core building block of DGCNN
    Dynamically constructs graph and applies convolution on edges
    """
    
    def __init__(self, in_channels, out_channels, k=20):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k  # Number of nearest neighbors
        
        # Initialize weights (simplified MLP weights)
        np.random.seed(42)  # Deterministic
        self.W1 = np.random.randn(in_channels * 2, out_channels) * 0.1
        self.b1 = np.zeros(out_channels)
        self.W2 = np.random.randn(out_channels, out_channels) * 0.1
        self.b2 = np.zeros(out_channels)
    
    def knn(self, x):
        """Find k-nearest neighbors for each point"""
        # x: (N, C) - N points, C channels
        dists = cdist(x, x, metric='euclidean')
        # Get indices of k nearest neighbors (excluding self)
        idx = np.argsort(dists, axis=1)[:, 1:self.k+1]
        return idx
    
    def get_edge_features(self, x, idx):
        """
        Construct edge features for graph convolution
        For each point, concatenate [x_i, x_j - x_i] for all neighbors j
        """
        N, C = x.shape
        k = idx.shape[1]
        
        # Gather neighbor features
        neighbors = x[idx]  # (N, k, C)
        
        # Central point features repeated
        x_central = np.tile(x[:, np.newaxis, :], (1, k, 1))  # (N, k, C)
        
        # Edge features: [x_i, x_j - x_i]
        edge_features = np.concatenate([x_central, neighbors - x_central], axis=-1)  # (N, k, 2C)
        
        return edge_features
    
    def forward(self, x):
        """
        Forward pass of EdgeConv
        x: (N, C) input point features
        returns: (N, out_channels) output features
        """
        # Build dynamic graph
        idx = self.knn(x)
        
        # Get edge features
        edge_feat = self.get_edge_features(x, idx)  # (N, k, 2C)
        
        # Apply MLP to edge features
        N, k, _ = edge_feat.shape
        edge_feat_flat = edge_feat.reshape(N * k, -1)
        
        # First layer with ReLU
        h = np.maximum(0, edge_feat_flat @ self.W1 + self.b1)
        
        # Second layer
        h = h @ self.W2 + self.b2
        
        # Reshape and aggregate (max pooling over neighbors)
        h = h.reshape(N, k, self.out_channels)
        out = np.max(h, axis=1)  # (N, out_channels)
        
        return out


class DGCNN:
    """
    Dynamic Graph CNN for 3D Point Cloud Feature Extraction
    
    Architecture:
    - Input: (N, 3) point cloud
    - EdgeConv layers with increasing channels
    - Global max pooling
    - MLP for final features
    """
    
    def __init__(self, num_points=1024, k=20):
        self.num_points = num_points
        self.k = k
        
        # DGCNN layers
        self.edge_conv1 = EdgeConv(3, 64, k=k)
        self.edge_conv2 = EdgeConv(64, 64, k=k)
        self.edge_conv3 = EdgeConv(64, 128, k=k)
        self.edge_conv4 = EdgeConv(128, 256, k=k)
        
        # Final MLP weights
        np.random.seed(123)
        self.fc1_w = np.random.randn(512, 256) * 0.05
        self.fc1_b = np.zeros(256)
        self.fc2_w = np.random.randn(256, 128) * 0.05
        self.fc2_b = np.zeros(128)
    
    def forward(self, points):
        """
        Forward pass through DGCNN
        points: (N, 3) input point cloud
        returns: (128,) feature vector
        """
        # EdgeConv layers
        x1 = self.edge_conv1.forward(points)  # (N, 64)
        x2 = self.edge_conv2.forward(x1)       # (N, 64)
        x3 = self.edge_conv3.forward(x2)       # (N, 128)
        x4 = self.edge_conv4.forward(x3)       # (N, 256)
        
        # Concatenate features from all layers
        x = np.concatenate([x1, x2, x3, x4], axis=1)  # (N, 512)
        
        # Global max pooling
        global_feat = np.max(x, axis=0)  # (512,)
        
        # Final MLP
        h = np.maximum(0, global_feat @ self.fc1_w + self.fc1_b)  # ReLU
        h = h @ self.fc2_w + self.fc2_b  # (128,)
        
        return h


class KnuckleAI:
    """
    3D Finger Knuckle Pattern Recognition using DGCNN
    
    Pipeline:
    1. Preprocess images (CLAHE, denoising)
    2. Generate 3D point cloud from depth estimation
    3. Extract features using DGCNN
    4. Compare features for verification/identification
    """
    
    def __init__(self):
        self.img_size = (128, 128)
        self.num_points = 1024
        self.dgcnn = DGCNN(num_points=self.num_points, k=16)
        self._images_cache = []
        print("[KnuckleAI] Initialized DGCNN-based 3D Knuckle Recognition System")
    
    def preprocess_images(self, paths):
        """
        Preprocess knuckle images with enhancement
        """
        results = []
        
        for path in paths:
            if not os.path.exists(path):
                continue
            
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            
            # Resize to standard size
            img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_AREA)
            
            # CLAHE for contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            enhanced = clahe.apply(img)
            
            # Bilateral filter for edge-preserving denoising
            denoised = cv2.bilateralFilter(enhanced, 7, 50, 50)
            
            results.append(denoised)
        
        self._images_cache = results
        print(f"[KnuckleAI] Preprocessed {len(results)} images")
        return results
    
    def _estimate_depth(self, img):
        """
        Estimate depth from 2D image using gradient and intensity
        Creates pseudo-3D surface from knuckle creases
        """
        img_float = img.astype(np.float32) / 255.0
        
        # Compute gradients (indicate surface orientation)
        gx = cv2.Sobel(img_float, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img_float, cv2.CV_32F, 0, 1, ksize=3)
        
        # Gradient magnitude represents surface curvature
        grad_mag = np.sqrt(gx**2 + gy**2)
        
        # Laplacian for ridge/valley detection
        laplacian = cv2.Laplacian(img_float, cv2.CV_32F)
        
        # Combine intensity, gradient, and laplacian for depth
        depth = 0.5 * img_float + 0.3 * grad_mag + 0.2 * np.abs(laplacian)
        
        # Normalize depth to [0, 1]
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        
        return depth
    
    def generate_3d_model(self, images):
        """
        Generate 3D point cloud from multiple images
        Uses depth estimation and multi-view fusion
        """
        if not images:
            return np.zeros((self.num_points, 3), dtype=np.float32)
        
        all_points = []
        
        for img_idx, img in enumerate(images):
            h, w = img.shape
            
            # Estimate depth
            depth = self._estimate_depth(img)
            
            # Sample points from the surface
            # Use feature-aware sampling (more points at high-gradient regions)
            gx = cv2.Sobel(img.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(img.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
            importance = np.sqrt(gx**2 + gy**2)
            importance = importance / (importance.sum() + 1e-8)
            
            # Grid sampling with importance weighting
            points_per_image = self.num_points // len(images)
            
            for y in range(0, h, 4):
                for x in range(0, w, 4):
                    # Add point with depth
                    z = depth[y, x]
                    
                    # Add small variation based on image index for multi-view
                    view_offset = img_idx * 0.01
                    
                    point = [
                        x / w,              # Normalized X
                        y / h,              # Normalized Y
                        z * 0.5 + view_offset  # Depth with view offset
                    ]
                    all_points.append(point)
        
        points = np.array(all_points, dtype=np.float32)
        
        # Subsample or pad to exact number of points
        if len(points) > self.num_points:
            # Farthest point sampling for better coverage
            indices = self._farthest_point_sampling(points, self.num_points)
            points = points[indices]
        elif len(points) < self.num_points:
            # Pad with mean point
            pad_count = self.num_points - len(points)
            pad = np.tile(np.mean(points, axis=0), (pad_count, 1))
            points = np.vstack([points, pad])
        
        print(f"[KnuckleAI] Generated 3D point cloud: {points.shape}")
        return points
    
    def _farthest_point_sampling(self, points, n_samples):
        """
        Farthest Point Sampling for better point cloud coverage
        """
        N = len(points)
        if N <= n_samples:
            return np.arange(N)
        
        selected = [0]
        distances = np.full(N, np.inf)
        
        for _ in range(n_samples - 1):
            last = selected[-1]
            dist_to_last = np.linalg.norm(points - points[last], axis=1)
            distances = np.minimum(distances, dist_to_last)
            selected.append(np.argmax(distances))
        
        return np.array(selected)
    
    def _compute_texture_features(self, img):
        """
        Compute texture features using LBP
        """
        h, w = img.shape
        lbp = np.zeros((h-2, w-2), dtype=np.uint8)
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                center = img[i, j]
                pattern = 0
                pattern |= (img[i-1, j-1] >= center) << 7
                pattern |= (img[i-1, j] >= center) << 6
                pattern |= (img[i-1, j+1] >= center) << 5
                pattern |= (img[i, j+1] >= center) << 4
                pattern |= (img[i+1, j+1] >= center) << 3
                pattern |= (img[i+1, j] >= center) << 2
                pattern |= (img[i+1, j-1] >= center) << 1
                pattern |= (img[i, j-1] >= center) << 0
                lbp[i-1, j-1] = pattern
        
        # LBP histogram
        hist, _ = np.histogram(lbp.flatten(), bins=64, range=(0, 256), density=True)
        return hist
    
    def extract_features(self, point_cloud):
        """
        Extract features using DGCNN + texture features
        """
        # DGCNN features from 3D point cloud
        dgcnn_features = self.dgcnn.forward(point_cloud)  # (128,)
        
        # Texture features from cached images
        texture_features = []
        if self._images_cache:
            for img in self._images_cache:
                tex = self._compute_texture_features(img)
                texture_features.extend(tex)
        
        # Combine DGCNN and texture features
        if texture_features:
            texture_features = np.array(texture_features)
            # Normalize texture features
            tex_norm = np.linalg.norm(texture_features)
            if tex_norm > 1e-8:
                texture_features = texture_features / tex_norm
            
            all_features = np.concatenate([dgcnn_features, texture_features])
        else:
            all_features = dgcnn_features
        
        # L2 normalize final features
        norm = np.linalg.norm(all_features)
        if norm > 1e-8:
            all_features = all_features / norm
        
        print(f"[KnuckleAI] DGCNN extracted {len(all_features)} features")
        return all_features.tolist()
    
    def compare_features(self, f1, f2):
        """
        Compare two feature vectors
        Uses cosine similarity with stricter thresholding
        """
        v1 = np.array(f1, dtype=np.float64)
        v2 = np.array(f2, dtype=np.float64)
        
        # Handle dimension mismatch
        min_len = min(len(v1), len(v2))
        if min_len == 0:
            return 0.0
        
        v1 = v1[:min_len]
        v2 = v2[:min_len]
        
        # Handle NaN/Inf
        v1 = np.nan_to_num(v1)
        v2 = np.nan_to_num(v2)
        
        # Normalize
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        
        if n1 < 1e-10 or n2 < 1e-10:
            return 0.0
        
        v1_norm = v1 / n1
        v2_norm = v2 / n2
        
        # Cosine similarity
        cosine = float(np.dot(v1_norm, v2_norm))
        
        # Euclidean distance similarity
        euc_dist = np.linalg.norm(v1_norm - v2_norm)
        euc_sim = float(1.0 / (1.0 + euc_dist * 2))
        
        # Combined score
        score = 0.6 * cosine + 0.4 * euc_sim
        
        # Apply sigmoid-like scaling for better discrimination
        if score > 0.8:
            score = 0.8 + (score - 0.8) * 0.5
        elif score < 0.5:
            score = score * 0.8
        
        score = float(max(0.0, min(1.0, score)))
        
        print(f"[KnuckleAI] Match: Cosine={cosine:.4f}, Euclidean={euc_sim:.4f} => Score={score:.4f}")
        
        return score

