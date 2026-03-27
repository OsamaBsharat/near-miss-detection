"""
Near-Miss Incident Detection System
=====================================

A modular computer vision pipeline for detecting and analyzing
near-miss traffic incidents from video footage.

Modules:
    config      — Centralized configuration and hyperparameters
    utils       — Data models, video I/O, and math utilities
    detector    — YOLOv8-based multi-class object detection
    tracker     — Kalman-filtered trajectory estimation and analysis
    near_miss   — Near-miss detection logic and incident grouping
    optical_flow— Dense optical flow analysis (Farneback)
    visualizer  — Video annotation and dashboard generation
    report      — HTML report and JSON export
"""

from .config import PipelineConfig
from .utils import (
    TrackedObject, FrameResult, NearMissEvent, Incident,
    download_video, get_video_info, extract_sample_frames,
    analyze_tracking_quality, Timer
)
from .detector import ObjectDetector
from .tracker import TrajectoryAnalyzer
from .near_miss import NearMissDetector
from .optical_flow import OpticalFlowAnalyzer
from .visualizer import VideoAnnotator, DashboardGenerator
from .report import generate_html_report, export_json_results

__version__ = '1.0.0'
__all__ = [
    'PipelineConfig', 'ObjectDetector', 'TrajectoryAnalyzer',
    'NearMissDetector', 'OpticalFlowAnalyzer', 'VideoAnnotator',
    'DashboardGenerator', 'generate_html_report', 'export_json_results',
]
