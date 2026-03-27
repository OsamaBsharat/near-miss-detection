"""
Data models and utility functions for the Near-Miss Detection System.

Contains all dataclasses, video I/O helpers, and shared utilities.
"""

import cv2
import os
import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from datetime import timedelta
from collections import defaultdict


# ============================================================
# DATA MODELS
# ============================================================

@dataclass
class TrackedObject:
    """A single detected and tracked object in one frame."""
    track_id: int
    class_id: int
    class_name: str
    bbox: np.ndarray         # [x1, y1, x2, y2]
    centroid: np.ndarray     # [cx, cy]
    confidence: float
    area: float
    velocity: Optional[np.ndarray] = None  # [vx, vy] pixels/frame
    speed: float = 0.0                     # magnitude pixels/frame


@dataclass
class FrameResult:
    """All detection + tracking results for a single frame."""
    frame_idx: int
    timestamp: float
    objects: List[TrackedObject] = field(default_factory=list)
    optical_flow_magnitude: Optional[float] = None  # Avg flow magnitude


@dataclass
class NearMissEvent:
    """A single near-miss interaction between two objects at one frame."""
    frame_idx: int
    timestamp: float
    track_id_1: int
    track_id_2: int
    class_1: str
    class_2: str
    distance: float
    ttc: float                # seconds, inf if diverging
    closing_speed: float      # pixels/frame
    risk_level: str           # HIGH / MEDIUM / LOW
    risk_score: float         # 0.0 - 1.0
    bbox_1: np.ndarray = field(repr=False)
    bbox_2: np.ndarray = field(repr=False)
    predicted_collision_point: Optional[np.ndarray] = None  # Kalman predicted


@dataclass
class Incident:
    """A grouped near-miss incident spanning multiple frames."""
    incident_id: int
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    duration_frames: int
    duration_sec: float
    max_risk_level: str
    max_risk_score: float
    avg_risk_score: float
    min_distance: float
    min_ttc: float
    involved_tracks: set
    involved_classes: set
    events: List[NearMissEvent] = field(repr=False)
    peak_frame: int = 0      # Frame with highest risk


# ============================================================
# VIDEO UTILITIES
# ============================================================

def download_video(url: str, output_path: str) -> str:
    """Download video from YouTube using yt-dlp.
    
    Selects 720p or lower to keep file size manageable
    and ensure CPU processing stays within time limits.
    """
    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f'  Video already exists: {output_path} ({size_mb:.1f} MB)')
        return output_path
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    cmd = (
        f'yt-dlp -f "bestvideo[height<=720][ext=mp4]'
        f'+bestaudio[ext=m4a]/best[height<=720]" '
        f'--merge-output-format mp4 -o "{output_path}" "{url}"'
    )
    
    print(f'  Downloading video from: {url}')
    os.system(cmd)
    
    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f'  Downloaded successfully: {size_mb:.1f} MB')
    else:
        raise FileNotFoundError('Download failed! Check URL and network connection.')
    
    return output_path


def get_video_info(video_path: str) -> dict:
    """Extract video metadata for pipeline calibration."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f'Cannot open video: {video_path}')
    
    info = {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    info['duration_sec'] = info['total_frames'] / info['fps']
    info['duration_str'] = str(timedelta(seconds=int(info['duration_sec'])))
    cap.release()
    return info


def extract_sample_frames(video_path: str, n_frames: int = 6) -> List[Tuple[int, np.ndarray]]:
    """Extract evenly-spaced frames for visual inspection."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append((idx, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


# ============================================================
# MATH UTILITIES
# ============================================================

def compute_bbox_distance(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """Compute minimum edge-to-edge distance between two bounding boxes.
    
    Uses the signed separation along each axis. Returns 0 if boxes overlap.
    More accurate than centroid distance for large/differently-sized objects.
    """
    dx = max(0, max(bbox1[0], bbox2[0]) - min(bbox1[2], bbox2[2]))
    dy = max(0, max(bbox1[1], bbox2[1]) - min(bbox1[3], bbox2[3]))
    return np.sqrt(dx**2 + dy**2)


def compute_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """Compute Intersection over Union between two bounding boxes."""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


# ============================================================
# STATISTICS UTILITIES
# ============================================================

def analyze_tracking_quality(frame_results: List[FrameResult]) -> dict:
    """Compute comprehensive detection and tracking statistics."""
    total_detections = 0
    class_counts = defaultdict(int)
    track_ids = set()
    objects_per_frame = []
    track_lengths = defaultdict(int)
    confidences = []
    
    for fr in frame_results:
        objects_per_frame.append(len(fr.objects))
        for obj in fr.objects:
            total_detections += 1
            class_counts[obj.class_name] += 1
            confidences.append(obj.confidence)
            if obj.track_id >= 0:
                track_ids.add(obj.track_id)
                track_lengths[obj.track_id] += 1
    
    track_len_values = list(track_lengths.values()) if track_lengths else [0]
    
    return {
        'total_frames': len(frame_results),
        'total_detections': total_detections,
        'unique_tracks': len(track_ids),
        'class_counts': dict(class_counts),
        'avg_objects_per_frame': float(np.mean(objects_per_frame)),
        'max_objects_per_frame': max(objects_per_frame) if objects_per_frame else 0,
        'avg_track_length': float(np.mean(track_len_values)),
        'median_track_length': float(np.median(track_len_values)),
        'longest_track': max(track_len_values),
        'avg_confidence': float(np.mean(confidences)) if confidences else 0,
        'objects_per_frame_std': float(np.std(objects_per_frame)),
    }


class Timer:
    """Simple context-manager timer for profiling pipeline stages."""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.elapsed = 0.0
    
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
        if self.name:
            print(f'  {self.name}: {self.elapsed:.1f}s')
