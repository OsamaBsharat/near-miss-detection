"""
Object Detection Module — YOLOv8-based multi-class detector.

Handles model initialization, inference, and result parsing
with built-in ByteTrack multi-object tracking.
"""

import time
import numpy as np
from typing import List
from ultralytics import YOLO

from .config import PipelineConfig
from .utils import TrackedObject, FrameResult, Timer


class ObjectDetector:
    """YOLOv8 Nano detector with integrated ByteTrack tracking.
    
    Why YOLOv8n?
    ┌────────────┬──────────┬─────────┬────────────┐
    │ Model      │ Params   │ mAP@50  │ CPU Speed  │
    ├────────────┼──────────┼─────────┼────────────┤
    │ YOLOv8n    │ 3.2M     │ 37.3    │ ~30ms      │ ← Selected
    │ YOLOv8s    │ 11.2M    │ 44.9    │ ~60ms      │
    │ YOLOv8m    │ 25.9M    │ 50.2    │ ~150ms     │
    │ Faster-RCNN│ 41.8M    │ 42.0    │ ~200ms     │
    └────────────┴──────────┴─────────┴────────────┘
    
    Selected for optimal CPU speed/accuracy balance, meeting the
    <5 min total inference requirement for 2-3 min video.
    """
    
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.det_cfg = cfg.detection
        self.trk_cfg = cfg.tracking
        self.model = None
    
    def initialize(self) -> None:
        """Load model and run warmup inference."""
        print(f'Loading {self.det_cfg.model_name}...')
        self.model = YOLO(self.det_cfg.model_name)
        
        # Warmup inference to initialize model buffers
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        _ = self.model(dummy, verbose=False)
        
        param_count = sum(p.numel() for p in self.model.model.parameters())
        print(f'Model loaded: {param_count:,} parameters')
    
    def run(self, video_path: str, fps: float) -> List[FrameResult]:
        """Run detection + tracking on full video.
        
        Uses ultralytics built-in ByteTrack for multi-object tracking.
        Results are streamed frame-by-frame to minimize memory usage.
        
        Args:
            video_path: Path to input video
            fps: Video FPS for timestamp calculation
            
        Returns:
            List of FrameResult objects with tracked objects per frame
        """
        if self.model is None:
            self.initialize()
        
        results_list = []
        
        print(f'  Running detection + tracking...')
        with Timer("Detection + Tracking") as timer:
            tracking_results = self.model.track(
                source=video_path,
                conf=self.det_cfg.confidence,
                iou=self.det_cfg.iou_threshold,
                classes=self.det_cfg.target_classes,
                tracker=self.trk_cfg.tracker_type,
                stream=True,
                verbose=False,
                persist=True
            )
            
            for frame_idx, result in enumerate(tracking_results):
                objects = self._parse_detections(result, frame_idx)
                
                frame_result = FrameResult(
                    frame_idx=frame_idx,
                    timestamp=frame_idx / fps,
                    objects=objects
                )
                results_list.append(frame_result)
                
                if (frame_idx + 1) % 100 == 0:
                    elapsed = time.time() - timer.start
                    speed = (frame_idx + 1) / elapsed
                    print(f'    Frame {frame_idx+1} | '
                          f'{speed:.1f} FPS | '
                          f'{len(objects)} objects')
        
        total = len(results_list)
        avg_fps = total / timer.elapsed if timer.elapsed > 0 else 0
        print(f'Processed {total} frames at {avg_fps:.1f} FPS')
        
        return results_list
    
    def _parse_detections(self, result, frame_idx: int) -> List[TrackedObject]:
        """Parse YOLO result into TrackedObject instances.
        
        Filters by minimum area and extracts track IDs from ByteTrack.
        """
        objects = []
        
        if result.boxes is None or len(result.boxes) == 0:
            return objects
        
        boxes = result.boxes
        for i in range(len(boxes)):
            bbox = boxes.xyxy[i].cpu().numpy()
            cls_id = int(boxes.cls[i].cpu())
            conf = float(boxes.conf[i].cpu())
            
            # Track ID from ByteTrack (may be None if tracking lost)
            track_id = -1
            if boxes.id is not None:
                track_id = int(boxes.id[i].cpu())
            
            # Compute centroid and area
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            
            # Filter small/noisy detections
            if area < self.det_cfg.min_object_area:
                continue
            
            obj = TrackedObject(
                track_id=track_id,
                class_id=cls_id,
                class_name=self.det_cfg.class_names.get(cls_id, 'Unknown'),
                bbox=bbox,
                centroid=np.array([cx, cy]),
                confidence=conf,
                area=area
            )
            objects.append(obj)
        
        return objects
