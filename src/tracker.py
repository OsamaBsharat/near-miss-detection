"""
Tracking & Trajectory Module — Kalman-filtered trajectory estimation.

Builds track histories from detection results, applies Kalman filtering
for smooth velocity estimation and trajectory prediction.
"""

import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple
from filterpy.kalman import KalmanFilter

from .config import PipelineConfig
from .utils import TrackedObject, FrameResult


class KalmanTracker:
    """6-state Kalman filter for single object trajectory estimation.
    
    State vector: [x, y, vx, vy, ax, ay]
    - Position (x, y): object centroid in pixel space
    - Velocity (vx, vy): centroid displacement rate (pixels/frame)
    - Acceleration (ax, ay): velocity change rate
    
    Only position is measured (from detection); velocity and acceleration
    are estimated by the filter. This provides:
    1. Smooth velocity estimates even with noisy detections
    2. Forward trajectory prediction for TTC estimation
    3. Handling of temporary occlusions via prediction
    """
    
    def __init__(self, initial_pos: np.ndarray, cfg):
        self.kf = KalmanFilter(dim_x=cfg.state_dim, dim_z=cfg.measurement_dim)
        
        dt = 1.0  # 1 frame timestep
        
        # State transition matrix: constant acceleration model
        # x' = x + vx*dt + 0.5*ax*dt²
        # vx' = vx + ax*dt
        # ax' = ax
        self.kf.F = np.array([
            [1, 0, dt, 0,  0.5*dt**2, 0        ],
            [0, 1, 0,  dt, 0,         0.5*dt**2 ],
            [0, 0, 1,  0,  dt,        0         ],
            [0, 0, 0,  1,  0,         dt        ],
            [0, 0, 0,  0,  1,         0         ],
            [0, 0, 0,  0,  0,         1         ],
        ])
        
        # Measurement matrix: observe only position
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
        ])
        
        # Process noise (tuned for traffic dynamics)
        q = cfg.process_noise
        self.kf.Q = np.diag([
            q * 0.25, q * 0.25,   # position noise (low)
            q * 1.0,  q * 1.0,    # velocity noise (medium)
            q * 2.0,  q * 2.0,    # acceleration noise (higher)
        ])
        
        # Measurement noise
        r = cfg.measurement_noise
        self.kf.R = np.diag([r, r])
        
        # Initial state
        self.kf.x = np.array([
            initial_pos[0], initial_pos[1],
            0, 0,  # initial velocity unknown
            0, 0,  # initial acceleration unknown
        ]).reshape(-1, 1)
        
        # Initial covariance (high uncertainty)
        self.kf.P *= cfg.initial_covariance
        
        self.prediction_horizon = cfg.prediction_horizon
        self.initialized = True
    
    def update(self, measurement: np.ndarray) -> None:
        """Update filter with new position measurement."""
        self.kf.predict()
        self.kf.update(measurement.reshape(-1, 1))
    
    def predict_only(self) -> np.ndarray:
        """Predict next state without measurement (occlusion handling)."""
        self.kf.predict()
        return self.kf.x.flatten()[:2]
    
    @property
    def position(self) -> np.ndarray:
        """Current filtered position [x, y]."""
        return self.kf.x.flatten()[:2]
    
    @property
    def velocity(self) -> np.ndarray:
        """Current filtered velocity [vx, vy] in pixels/frame."""
        return self.kf.x.flatten()[2:4]
    
    @property
    def acceleration(self) -> np.ndarray:
        """Current filtered acceleration [ax, ay]."""
        return self.kf.x.flatten()[4:6]
    
    @property
    def speed(self) -> float:
        """Current speed magnitude in pixels/frame."""
        return float(np.linalg.norm(self.velocity))
    
    def predict_trajectory(self, n_steps: int = None) -> np.ndarray:
        """Predict future trajectory for n_steps frames.
        
        Uses current state to propagate forward, considering velocity
        and acceleration. Returns array of shape (n_steps, 2) with
        predicted (x, y) positions.
        """
        if n_steps is None:
            n_steps = self.prediction_horizon
        
        trajectory = np.zeros((n_steps, 2))
        state = self.kf.x.flatten().copy()
        F = self.kf.F.copy()
        
        for i in range(n_steps):
            state = F @ state
            trajectory[i] = state[:2]
        
        return trajectory


class TrackHistory:
    """Maintains complete history for a single tracked object.
    
    Combines raw position history with Kalman-filtered state estimation.
    """
    
    def __init__(self, max_length: int, kalman_cfg):
        self.positions = deque(maxlen=max_length)
        self.raw_velocities = deque(maxlen=max_length)
        self.class_name = None
        self.class_id = None
        self.last_seen_frame = -1
        self.total_frames = 0
        
        # Kalman filter (initialized on first update)
        self._kalman: Optional[KalmanTracker] = None
        self._kalman_cfg = kalman_cfg
    
    def update(self, frame_idx: int, centroid: np.ndarray,
               class_name: str, class_id: int) -> None:
        """Add new observation and update Kalman filter."""
        self.class_name = class_name
        self.class_id = class_id
        self.last_seen_frame = frame_idx
        self.total_frames += 1
        
        # Initialize Kalman on first observation
        if self._kalman is None:
            self._kalman = KalmanTracker(centroid, self._kalman_cfg)
        else:
            self._kalman.update(centroid)
        
        # Store raw position
        self.positions.append((frame_idx, centroid[0], centroid[1]))
        
        # Compute raw velocity for comparison
        if len(self.positions) >= 2:
            prev = self.positions[-2]
            dt = frame_idx - prev[0]
            if 0 < dt < 5:
                vx = (centroid[0] - prev[1]) / dt
                vy = (centroid[1] - prev[2]) / dt
                self.raw_velocities.append(np.array([vx, vy]))
    
    @property
    def kalman_velocity(self) -> np.ndarray:
        """Kalman-filtered velocity estimate."""
        if self._kalman is not None:
            return self._kalman.velocity
        return np.array([0.0, 0.0])
    
    @property
    def kalman_position(self) -> np.ndarray:
        """Kalman-filtered position estimate."""
        if self._kalman is not None:
            return self._kalman.position
        if len(self.positions) > 0:
            return np.array([self.positions[-1][1], self.positions[-1][2]])
        return np.array([0.0, 0.0])
    
    @property
    def speed(self) -> float:
        """Current Kalman-estimated speed (pixels/frame)."""
        if self._kalman is not None:
            return self._kalman.speed
        return 0.0
    
    @property
    def last_position(self) -> Optional[np.ndarray]:
        """Most recent raw position."""
        if len(self.positions) > 0:
            return np.array([self.positions[-1][1], self.positions[-1][2]])
        return None
    
    def predict_trajectory(self, n_steps: int = 30) -> Optional[np.ndarray]:
        """Predict future trajectory using Kalman filter."""
        if self._kalman is not None:
            return self._kalman.predict_trajectory(n_steps)
        return None


class TrajectoryAnalyzer:
    """Builds and manages track histories for all objects.
    
    Provides:
    - Kalman-filtered velocity and position for each track
    - Closing speed computation between object pairs
    - Trajectory prediction for TTC estimation
    """
    
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.histories: Dict[int, TrackHistory] = {}
    
    def build(self, frame_results: List[FrameResult]) -> Dict[int, TrackHistory]:
        """Build track histories from detection results."""
        self.histories = {}
        
        for fr in frame_results:
            for obj in fr.objects:
                if obj.track_id < 0:
                    continue
                
                if obj.track_id not in self.histories:
                    self.histories[obj.track_id] = TrackHistory(
                        max_length=self.cfg.tracking.history_length,
                        kalman_cfg=self.cfg.kalman
                    )
                
                self.histories[obj.track_id].update(
                    fr.frame_idx, obj.centroid,
                    obj.class_name, obj.class_id
                )
        
        print(f'  Built {len(self.histories)} track histories with Kalman filtering')
        return self.histories
    
    def compute_closing_speed(self, track_id_1: int, 
                               track_id_2: int) -> Tuple[float, Optional[np.ndarray]]:
        """Compute closing speed between two tracked objects.
        
        Uses Kalman-filtered velocities for robust estimation.
        
        Returns:
            Tuple of (closing_speed, predicted_collision_point)
            closing_speed > 0 means objects are approaching
        """
        hist1 = self.histories.get(track_id_1)
        hist2 = self.histories.get(track_id_2)
        
        if hist1 is None or hist2 is None:
            return 0.0, None
        
        pos1 = hist1.kalman_position
        pos2 = hist2.kalman_position
        
        # Direction vector from obj1 to obj2
        direction = pos2 - pos1
        dist = np.linalg.norm(direction)
        if dist < 1e-6:
            return 0.0, None
        direction_unit = direction / dist
        
        # Relative velocity using Kalman estimates
        v1 = hist1.kalman_velocity
        v2 = hist2.kalman_velocity
        relative_vel = v1 - v2
        
        # Closing speed = projection onto direction vector
        closing = float(np.dot(relative_vel, direction_unit))
        
        # Predict collision point if approaching
        collision_point = None
        if closing > 0 and dist > 0:
            t_collision = dist / closing  # frames to collision
            collision_point = pos1 + v1 * t_collision
        
        return closing, collision_point
