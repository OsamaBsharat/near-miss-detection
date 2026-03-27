"""
Near-Miss Detection Module.

Implements the core near-miss detection algorithm with:
- Multi-criteria risk assessment (distance + TTC + velocity)
- Kalman-predicted trajectory-based TTC
- Composite risk scoring with vulnerability weighting
- Temporal filtering and incident grouping for false positive reduction
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from .config import PipelineConfig
from .utils import (
    FrameResult, NearMissEvent, Incident,
    compute_bbox_distance
)
from .tracker import TrajectoryAnalyzer


class NearMissDetector:
    """Multi-criteria near-miss incident detector.
    
    Detection Pipeline:
    ┌─────────────────┐
    │  For each frame  │
    │  For each pair   │──► Distance check ──► TTC estimation ──► Risk scoring
    └─────────────────┘         │                    │                  │
                                ▼                    ▼                  ▼
                         Proximity filter    Closing speed       Composite score
                                │            + Kalman TTC             │
                                └────────────────┬───────────────────┘
                                                 ▼
                                          Risk Level Assignment
                                          (HIGH/MEDIUM/LOW/NONE)
    """
    
    def __init__(self, cfg: PipelineConfig, 
                 trajectory_analyzer: TrajectoryAnalyzer):
        self.cfg = cfg
        self.nm_cfg = cfg.near_miss
        self.filter_cfg = cfg.filter
        self.traj = trajectory_analyzer
        
        # Risk level priority for comparison
        self._risk_priority = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1, 'NONE': 0}
    
    def detect(self, frame_results: List[FrameResult],
               fps: float) -> List[NearMissEvent]:
        """Run near-miss detection on all frames.
        
        For each frame, evaluates all pairs of tracked objects
        against distance, TTC, and velocity criteria.
        
        Args:
            frame_results: Detection/tracking results per frame
            fps: Video FPS for time conversion
            
        Returns:
            List of raw NearMissEvent detections
        """
        events = []
        
        for fr in frame_results:
            objects = [o for o in fr.objects if o.track_id >= 0]
            n = len(objects)
            
            for i in range(n):
                for j in range(i + 1, n):
                    event = self._evaluate_pair(
                        objects[i], objects[j],
                        fr.frame_idx, fr.timestamp, fps
                    )
                    if event is not None:
                        events.append(event)
        
        print(f'  Raw near-miss events: {len(events)}')
        return events
    
    def _evaluate_pair(self, obj1, obj2,
                       frame_idx: int, timestamp: float,
                       fps: float) -> Optional[NearMissEvent]:
        """Evaluate a single object pair for near-miss conditions.
        
        Multi-criteria assessment:
        1. Edge-to-edge bounding box distance
        2. Closing speed from Kalman-filtered velocities
        3. Time-to-Collision estimation
        4. Composite risk scoring with vulnerability weighting
        """
        # 1. Distance check (fast rejection)
        distance = compute_bbox_distance(obj1.bbox, obj2.bbox)
        if distance > self.nm_cfg.distance_low * 1.5:
            return None  # Too far, skip
        
        # 2. Closing speed & TTC via Kalman
        closing_speed, collision_point = self.traj.compute_closing_speed(
            obj1.track_id, obj2.track_id
        )
        
        ttc = float('inf')
        if closing_speed > self.nm_cfg.min_closing_speed:
            # TTC = distance / (closing_speed_in_pixels_per_frame * fps)
            ttc = distance / (closing_speed * fps)
        
        # 3. Risk scoring
        cat1 = self.cfg.detection.class_categories.get(obj1.class_id, 'vehicle')
        cat2 = self.cfg.detection.class_categories.get(obj2.class_id, 'vehicle')
        risk_level, risk_score = self._compute_risk(
            distance, ttc, cat1, cat2
        )
        
        if risk_level == 'NONE':
            return None
        
        return NearMissEvent(
            frame_idx=frame_idx,
            timestamp=timestamp,
            track_id_1=obj1.track_id,
            track_id_2=obj2.track_id,
            class_1=obj1.class_name,
            class_2=obj2.class_name,
            distance=distance,
            ttc=ttc,
            closing_speed=closing_speed,
            risk_level=risk_level,
            risk_score=risk_score,
            bbox_1=obj1.bbox.copy(),
            bbox_2=obj2.bbox.copy(),
            predicted_collision_point=collision_point
        )
    
    def _compute_risk(self, distance: float, ttc: float,
                      cat1: str, cat2: str) -> Tuple[str, float]:
        """Compute composite risk level and score.
        
        Risk Score Formula:
            score = (w_d * dist_score + w_t * ttc_score) * vuln_mult
        
        where:
            dist_score = max(0, 1 - distance / D_low)
            ttc_score  = max(0, 1 - ttc / TTC_low)
            vuln_mult  = 1.3 if pedestrian/cyclist involved, else 1.0
        """
        nm = self.nm_cfg
        
        # Vulnerability multiplier
        vuln_mult = 1.0
        if cat1 == 'vulnerable' or cat2 == 'vulnerable':
            vuln_mult = nm.vulnerability_weight
        
        # Distance-based score (0-1, higher = more dangerous)
        dist_score = max(0.0, 1.0 - distance / nm.distance_low)
        
        # TTC-based score (0-1)
        if ttc == float('inf') or ttc < 0:
            ttc_score = 0.0
        else:
            ttc_score = max(0.0, 1.0 - ttc / nm.ttc_low)
        
        # Weighted composite
        raw_score = (nm.distance_weight * dist_score + 
                     nm.ttc_weight * ttc_score) * vuln_mult
        risk_score = min(1.0, raw_score)
        
        # Risk level classification
        if (distance < nm.distance_high or 
            (0 < ttc < nm.ttc_high)):
            risk_level = 'HIGH'
        elif (distance < nm.distance_medium or 
              (0 < ttc < nm.ttc_medium)):
            risk_level = 'MEDIUM'
        elif (distance < nm.distance_low or 
              (0 < ttc < nm.ttc_low)):
            risk_level = 'LOW'
        else:
            risk_level = 'NONE'
        
        return risk_level, risk_score
    
    def group_incidents(self, events: List[NearMissEvent]) -> List[Incident]:
        """Group raw events into discrete incidents.
        
        Applies three-layer false positive reduction:
        1. Group events by object pair
        2. Cluster temporally close events (merge_gap)
        3. Filter clusters shorter than min_incident_frames
        
        This typically reduces raw event count by 60-80%.
        """
        # Group by object pair
        pair_events = defaultdict(list)
        for evt in events:
            pair_key = tuple(sorted([evt.track_id_1, evt.track_id_2]))
            pair_events[pair_key].append(evt)
        
        incidents = []
        
        for pair_key, pevents in pair_events.items():
            pevents.sort(key=lambda e: e.frame_idx)
            
            # Temporal clustering
            clusters = self._temporal_cluster(pevents)
            
            for cluster in clusters:
                # Apply minimum duration filter
                if len(cluster) < self.filter_cfg.min_incident_frames:
                    continue
                
                incident = self._build_incident(cluster, len(incidents))
                incidents.append(incident)
        
        # Sort chronologically and re-index
        incidents.sort(key=lambda inc: inc.start_frame)
        for i, inc in enumerate(incidents):
            inc.incident_id = i
        
        # Statistics
        raw_count = len(events)
        filtered_pct = (1 - len(incidents) / max(1, raw_count)) * 100
        print(f'  {len(incidents)} incidents (from {raw_count} raw events, '
              f'{filtered_pct:.0f}% filtered)')
        
        return incidents
    
    def _temporal_cluster(self, events: List[NearMissEvent]) -> List[List[NearMissEvent]]:
        """Split event sequence into temporal clusters based on gap threshold."""
        if not events:
            return []
        
        clusters = []
        current = [events[0]]
        
        for evt in events[1:]:
            if evt.frame_idx - current[-1].frame_idx <= self.filter_cfg.merge_gap_frames:
                current.append(evt)
            else:
                clusters.append(current)
                current = [evt]
        clusters.append(current)
        
        return clusters
    
    def _build_incident(self, cluster: List[NearMissEvent], 
                        incident_id: int) -> Incident:
        """Build an Incident object from a cluster of events."""
        risk_levels = [e.risk_level for e in cluster]
        max_risk = max(risk_levels, 
                       key=lambda r: self._risk_priority.get(r, 0))
        
        risk_scores = [e.risk_score for e in cluster]
        distances = [e.distance for e in cluster]
        ttcs = [e.ttc for e in cluster if e.ttc != float('inf')]
        
        peak_evt = max(cluster, key=lambda e: e.risk_score)
        
        tracks = set()
        classes = set()
        for e in cluster:
            tracks.update([e.track_id_1, e.track_id_2])
            classes.update([e.class_1, e.class_2])
        
        return Incident(
            incident_id=incident_id,
            start_frame=cluster[0].frame_idx,
            end_frame=cluster[-1].frame_idx,
            start_time=cluster[0].timestamp,
            end_time=cluster[-1].timestamp,
            duration_frames=cluster[-1].frame_idx - cluster[0].frame_idx + 1,
            duration_sec=cluster[-1].timestamp - cluster[0].timestamp,
            max_risk_level=max_risk,
            max_risk_score=float(max(risk_scores)),
            avg_risk_score=float(np.mean(risk_scores)),
            min_distance=float(min(distances)),
            min_ttc=float(min(ttcs)) if ttcs else float('inf'),
            involved_tracks=tracks,
            involved_classes=classes,
            events=cluster,
            peak_frame=peak_evt.frame_idx
        )
