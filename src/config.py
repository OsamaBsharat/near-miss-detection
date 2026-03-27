"""
Configuration module for Near-Miss Incident Detection System.

Centralizes all hyperparameters, thresholds, and settings.
All values are documented with rationale for selection.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np


@dataclass
class DetectionConfig:
    """Object detection settings."""
    
    # YOLOv8 Nano: 3.2M params, optimal CPU speed/accuracy tradeoff
    # Benchmarked: ~30ms/frame on CPU vs ~60ms for YOLOv8s
    model_name: str = 'model/yolov8n.pt'
    
    # Confidence threshold: 0.35 balances precision/recall
    # Lower = more detections but more false positives
    # Higher = fewer false positives but missed objects
    confidence: float = 0.35
    
    # NMS IoU threshold: controls duplicate suppression
    iou_threshold: float = 0.45
    
    # COCO class IDs relevant to traffic scenes
    target_classes: list = field(default_factory=lambda: [0, 1, 2, 3, 5, 7])
    
    # Human-readable class names
    class_names: Dict[int, str] = field(default_factory=lambda: {
        0: 'Pedestrian', 1: 'Cyclist', 2: 'Car',
        3: 'Motorcycle', 5: 'Bus', 7: 'Truck'
    })
    
    # Vulnerability classification for risk weighting
    class_categories: Dict[int, str] = field(default_factory=lambda: {
        0: 'vulnerable', 1: 'vulnerable', 2: 'vehicle',
        3: 'vehicle', 5: 'vehicle', 7: 'vehicle'
    })
    
    # Minimum bounding box area (filters noise/artifacts)
    min_object_area: int = 400


@dataclass
class TrackingConfig:
    """Multi-object tracking settings."""
    
    # ByteTrack: IoU-based tracker, no ReID model needed
    tracker_type: str = 'bytetrack.yaml'
    
    # Frames to retain in track history for velocity estimation
    history_length: int = 30
    
    # EMA smoothing factor for velocity (0.3 = moderate smoothing)
    velocity_smoothing: float = 0.3
    
    # Maximum frame gap to still compute velocity
    max_frame_gap: int = 5


@dataclass
class KalmanConfig:
    """Kalman filter settings for trajectory prediction."""
    
    # State: [x, y, vx, vy, ax, ay] — position, velocity, acceleration
    state_dim: int = 6
    measurement_dim: int = 2  # Observe only position
    
    # Process noise — higher = more responsive to changes
    process_noise: float = 1.0
    
    # Measurement noise — higher = smoother but less responsive
    measurement_noise: float = 5.0
    
    # Initial covariance
    initial_covariance: float = 100.0
    
    # Prediction horizon (frames) for TTC estimation
    prediction_horizon: int = 30


@dataclass
class NearMissConfig:
    """Near-miss detection thresholds and parameters."""
    
    # Distance thresholds (pixels) — calibrated for 720p traffic footage
    # These approximate: HIGH ≈ <2m, MEDIUM ≈ <4m, LOW ≈ <6m at typical distances
    distance_high: float = 60.0
    distance_medium: float = 120.0
    distance_low: float = 200.0
    
    # Time-to-Collision thresholds (seconds)
    # Based on traffic safety literature:
    # < 0.8s = critical (driver has <1s to react)
    # < 1.5s = dangerous (evasive action needed)
    # < 3.0s = warning (potential hazard developing)
    ttc_high: float = 0.8
    ttc_medium: float = 1.5
    ttc_low: float = 3.0
    
    # Minimum closing speed to consider (pixels/frame)
    # Filters out stationary or co-moving objects
    min_closing_speed: float = 2.0
    
    # Vulnerability multiplier: 30% higher risk for pedestrians/cyclists
    vulnerability_weight: float = 1.3
    
    # Risk score weights: TTC is weighted higher as it's more predictive
    distance_weight: float = 0.4
    ttc_weight: float = 0.6


@dataclass
class FilterConfig:
    """False positive reduction parameters."""
    
    # Minimum consecutive frames for a valid incident
    # Eliminates single-frame detection noise
    min_incident_frames: int = 3
    
    # Merge events within this frame gap into one incident
    merge_gap_frames: int = 10
    
    # Minimum IoU overlap between consecutive detections
    min_track_consistency: float = 0.3


@dataclass
class OpticalFlowConfig:
    """Optical flow analysis settings."""
    
    # Farneback dense optical flow parameters
    pyr_scale: float = 0.5     # Pyramid scale
    levels: int = 3             # Pyramid levels
    winsize: int = 15           # Averaging window size
    iterations: int = 3         # Iterations at each level
    poly_n: int = 5             # Polynomial expansion neighborhood
    poly_sigma: float = 1.2    # Gaussian std for polynomial expansion
    
    # Flow magnitude threshold for motion detection
    motion_threshold: float = 2.0
    
    # Compute flow every N frames (performance optimization)
    compute_interval: int = 2


@dataclass
class VisualizationConfig:
    """Visualization and annotation settings."""
    
    # Risk level colors (BGR for OpenCV)
    risk_colors: Dict[str, Tuple[int, int, int]] = field(default_factory=lambda: {
        'HIGH': (0, 0, 255),       # Red
        'MEDIUM': (0, 165, 255),   # Orange
        'LOW': (0, 255, 255),      # Yellow
        'NONE': (0, 255, 0)        # Green
    })
    
    # Object class colors (BGR)
    class_colors: Dict[str, Tuple[int, int, int]] = field(default_factory=lambda: {
        'Pedestrian': (255, 100, 100),
        'Cyclist': (255, 200, 0),
        'Car': (0, 200, 0),
        'Motorcycle': (0, 255, 255),
        'Bus': (200, 100, 0),
        'Truck': (100, 50, 200),
    })
    
    # Trail visualization
    max_trail_points: int = 20
    trail_thickness_range: Tuple[int, int] = (1, 3)
    
    # HUD settings
    hud_height: int = 55
    hud_opacity: float = 0.7
    font_scale: float = 0.5


@dataclass
class PipelineConfig:
    """Master configuration aggregating all sub-configs."""
    
    # Video settings
    video_url: str = 'https://www.youtube.com/watch?v=r86kxxU-LUY'
    video_path: str = 'inputs/traffic_video.mp4'
    output_video: str = 'outputs/annotated_near_miss.mp4'
    
    # Sub-configurations
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    kalman: KalmanConfig = field(default_factory=KalmanConfig)
    near_miss: NearMissConfig = field(default_factory=NearMissConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    optical_flow: OpticalFlowConfig = field(default_factory=OpticalFlowConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    
    def summary(self) -> str:
        """Print configuration summary."""
        lines = [
            "=" * 55,
            "  Pipeline Configuration Summary",
            "=" * 55,
            f"  Detector:    {self.detection.model_name} (conf={self.detection.confidence})",
            f"  Tracker:     ByteTrack (history={self.tracking.history_length})",
            f"  Kalman:      {self.kalman.state_dim}D state, σ_process={self.kalman.process_noise}",
            f"  Near-Miss:   D_high={self.near_miss.distance_high}px, TTC_high={self.near_miss.ttc_high}s",
            f"  Filtering:   min_frames={self.filter.min_incident_frames}, merge_gap={self.filter.merge_gap_frames}",
            f"  Opt. Flow:   interval={self.optical_flow.compute_interval}, threshold={self.optical_flow.motion_threshold}",
            f"  Classes:     {list(self.detection.class_names.values())}",
            "=" * 55,
        ]
        return "\n".join(lines)
