"""
Optical Flow Analysis Module.

Computes dense optical flow using Farneback method to provide:
1. Independent motion vectors for near-miss velocity validation
2. Scene-level motion patterns for anomaly detection
3. Flow-based speed estimation for objects without stable tracks
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from .config import PipelineConfig
from .utils import FrameResult, TrackedObject, Timer


class OpticalFlowAnalyzer:
    """Dense optical flow computation and analysis.
    
    Uses Farneback method for dense flow estimation. While more
    computationally expensive than sparse (Lucas-Kanade), it provides
    per-pixel motion vectors that can be sampled at arbitrary locations
    (e.g., within bounding boxes of tracked objects).
    
    Flow is computed at intervals (default: every 2 frames) to balance
    accuracy and CPU performance.
    """
    
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.flow_cfg = cfg.optical_flow
        
        self.flow_maps: Dict[int, np.ndarray] = {}      # frame_idx -> flow
        self.scene_flow_stats: List[dict] = []           # Per-frame scene stats
        self.object_flow_speeds: Dict[int, List[float]] = defaultdict(list)
    
    def compute(self, video_path: str, 
                frame_results: List[FrameResult]) -> Dict[int, np.ndarray]:
        """Compute dense optical flow for the entire video.
        
        Args:
            video_path: Path to input video
            frame_results: Detection results for object-level flow extraction
            
        Returns:
            Dictionary mapping frame indices to flow fields (H, W, 2)
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f'  Computing optical flow (interval={self.flow_cfg.compute_interval})...')
        
        ret, prev_frame = cap.read()
        if not ret:
            raise IOError("Cannot read first frame")
        
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        frame_idx = 0
        computed_count = 0
        
        with Timer("Optical Flow") as timer:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                
                # Compute flow at intervals
                if frame_idx % self.flow_cfg.compute_interval != 0:
                    continue
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Farneback dense optical flow
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray,
                    None,
                    pyr_scale=self.flow_cfg.pyr_scale,
                    levels=self.flow_cfg.levels,
                    winsize=self.flow_cfg.winsize,
                    iterations=self.flow_cfg.iterations,
                    poly_n=self.flow_cfg.poly_n,
                    poly_sigma=self.flow_cfg.poly_sigma,
                    flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
                )
                
                self.flow_maps[frame_idx] = flow
                
                # Compute scene-level statistics
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                scene_stats = {
                    'frame_idx': frame_idx,
                    'mean_magnitude': float(np.mean(mag)),
                    'max_magnitude': float(np.max(mag)),
                    'std_magnitude': float(np.std(mag)),
                    'dominant_direction': float(np.mean(ang)),
                    'motion_area_ratio': float(
                        np.sum(mag > self.flow_cfg.motion_threshold) / mag.size
                    )
                }
                self.scene_flow_stats.append(scene_stats)
                
                # Extract object-level flow
                if frame_idx < len(frame_results):
                    self._extract_object_flow(
                        flow, frame_results[frame_idx]
                    )
                
                prev_gray = gray
                computed_count += 1
                
                if computed_count % 50 == 0:
                    print(f'    Computed flow for {computed_count} frames...')
        
        cap.release()
        print(f'  Optical flow computed for {computed_count} frames')
        
        return self.flow_maps
    
    def _extract_object_flow(self, flow: np.ndarray, 
                              frame_result: FrameResult) -> None:
        """Extract average flow within each object's bounding box.
        
        This provides an independent velocity estimate that can be
        compared with the Kalman-filtered tracking velocity.
        """
        h, w = flow.shape[:2]
        
        for obj in frame_result.objects:
            if obj.track_id < 0:
                continue
            
            # Clamp bbox to frame boundaries
            x1 = max(0, int(obj.bbox[0]))
            y1 = max(0, int(obj.bbox[1]))
            x2 = min(w, int(obj.bbox[2]))
            y2 = min(h, int(obj.bbox[3]))
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Extract flow within bbox
            roi_flow = flow[y1:y2, x1:x2]
            
            # Compute magnitude
            mag = np.sqrt(roi_flow[..., 0]**2 + roi_flow[..., 1]**2)
            
            # Use median (robust to outliers) as object flow speed
            flow_speed = float(np.median(mag))
            self.object_flow_speeds[obj.track_id].append(flow_speed)
    
    def get_object_flow_velocity(self, track_id: int,
                                  frame_idx: int) -> Optional[np.ndarray]:
        """Get optical flow velocity vector for a specific object.
        
        Uses the nearest available flow field and samples within
        the object's last known bounding box.
        """
        # Find nearest flow frame
        nearest = min(self.flow_maps.keys(), 
                      key=lambda k: abs(k - frame_idx),
                      default=None)
        
        if nearest is None or abs(nearest - frame_idx) > 5:
            return None
        
        return self.flow_maps.get(nearest)
    
    def detect_flow_anomalies(self, threshold_multiplier: float = 2.5) -> List[dict]:
        """Detect frames with anomalous motion patterns.
        
        Anomalies in optical flow often correlate with sudden braking,
        swerving, or other evasive actions — indicators of near-miss events.
        
        Uses z-score based detection: frames where motion magnitude
        exceeds mean + threshold_multiplier * std are flagged.
        """
        if not self.scene_flow_stats:
            return []
        
        magnitudes = [s['mean_magnitude'] for s in self.scene_flow_stats]
        mean_mag = np.mean(magnitudes)
        std_mag = np.std(magnitudes)
        
        threshold = mean_mag + threshold_multiplier * std_mag
        
        anomalies = []
        for stats in self.scene_flow_stats:
            if stats['mean_magnitude'] > threshold:
                anomalies.append({
                    'frame_idx': stats['frame_idx'],
                    'magnitude': stats['mean_magnitude'],
                    'z_score': (stats['mean_magnitude'] - mean_mag) / (std_mag + 1e-8),
                    'motion_area': stats['motion_area_ratio']
                })
        
        if anomalies:
            print(f'  ⚠ Found {len(anomalies)} flow anomaly frames '
                  f'(threshold={threshold:.2f})')
        
        return anomalies
    
    def get_flow_consistency_score(self, track_id: int) -> float:
        """Compare optical flow speed with tracking-based speed.
        
        Returns a consistency score (0-1) indicating how well the
        flow-based and tracking-based velocities agree. Low scores
        suggest tracking errors or complex motion.
        """
        flow_speeds = self.object_flow_speeds.get(track_id, [])
        if not flow_speeds:
            return 1.0  # No data, assume consistent
        
        return float(np.mean(flow_speeds))
    
    def visualize_flow(self, flow: np.ndarray) -> np.ndarray:
        """Convert flow field to HSV visualization.
        
        Hue = direction, Value = magnitude.
        Returns a BGR image for display.
        """
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
        hsv[..., 0] = ang * 180 / np.pi / 2      # Hue: direction
        hsv[..., 1] = 255                          # Saturation: full
        hsv[..., 2] = cv2.normalize(               # Value: magnitude
            mag, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
        
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
