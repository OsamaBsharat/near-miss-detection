"""
Visualization Module — Video annotation and dashboard generation.

Handles all visual output:
- Video annotation with bounding boxes, trails, risk indicators, HUD
- Statistical dashboard with 6 analysis charts
- Key frame extraction at peak risk moments
"""

import cv2
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from typing import List, Dict, Tuple
from collections import defaultdict

from .config import PipelineConfig
from .utils import FrameResult, NearMissEvent, Incident, Timer
from .tracker import TrajectoryAnalyzer


class VideoAnnotator:
    
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.vis_cfg = cfg.visualization
    
    def annotate(self, video_path: str, output_path: str,
                 frame_results: List[FrameResult],
                 frame_event_map: Dict[int, List[NearMissEvent]],
                 track_histories: dict,
                 incidents: List[Incident],
                 flow_analyzer=None) -> str:
        """Generate fully annotated output video.
        
        Overlays include:
        - Color-coded bounding boxes per object class
        - Track ID labels
        - Trajectory trails (last 20 positions)
        - Near-miss risk indicators (connecting lines + risk badges)
        - Real-time HUD with timestamp, incident counter, risk level
        - Optional: optical flow visualization overlay
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        incident_set = set()
        
        print(f'  Generating annotated video...')
        with Timer("Video Annotation") as timer:
            for fidx in range(total):
                ret, frame = cap.read()
                if not ret:
                    break
                
                fr = frame_results[fidx] if fidx < len(frame_results) else None
                active_events = frame_event_map.get(fidx, [])
                
                # Track incident count
                for inc in incidents:
                    if inc.start_frame <= fidx:
                        incident_set.add(inc.incident_id)
                
                # Draw detections
                if fr:
                    self._draw_detections(frame, fr, track_histories)
                
                # Draw near-miss indicators
                self._draw_near_miss(frame, active_events)
                
                # Draw optical flow overlay (small corner)
                if flow_analyzer and fidx in flow_analyzer.flow_maps:
                    self._draw_flow_overlay(
                        frame, flow_analyzer.flow_maps[fidx]
                    )
                
                # Draw HUD
                self._draw_hud(frame, fidx, fidx / fps,
                              active_events, len(incident_set),
                              len(incidents), width)
                
                out.write(frame)
                
                if (fidx + 1) % 200 == 0:
                    print(f'    Annotated {fidx+1}/{total} frames...')
        
        cap.release()
        out.release()
        
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f'  Saved: {output_path} ({size_mb:.1f} MB)')
        return output_path
    
    def _draw_detections(self, frame, fr: FrameResult, 
                         track_histories: dict) -> None:
        """Draw bounding boxes, labels, and trajectory trails."""
        for obj in fr.objects:
            color = self.vis_cfg.class_colors.get(
                obj.class_name, (200, 200, 200)
            )
            x1, y1, x2, y2 = obj.bbox.astype(int)
            
            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            
            # Label with background
            label = f'{obj.class_name} #{obj.track_id}'
            sz = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            cv2.rectangle(frame, (x1, y1 - sz[1] - 6),
                         (x1 + sz[0] + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255),
                       1, cv2.LINE_AA)
            
            # Trajectory trail
            if obj.track_id in track_histories:
                hist = track_histories[obj.track_id]
                self._draw_trail(frame, hist.positions, color)
    
    def _draw_trail(self, frame, positions, color,
                    max_points: int = 20) -> None:
        """Draw fading trajectory trail."""
        pts = list(positions)[-max_points:]
        for k in range(1, len(pts)):
            p1 = (int(pts[k-1][1]), int(pts[k-1][2]))
            p2 = (int(pts[k][1]), int(pts[k][2]))
            thickness = max(1, int(np.ceil(k / len(pts) * 3)))
            cv2.line(frame, p1, p2, color, thickness, cv2.LINE_AA)
    
    def _draw_near_miss(self, frame, 
                        events: List[NearMissEvent]) -> None:
        """Draw near-miss indicators between involved objects."""
        for evt in events:
            risk_color = self.vis_cfg.risk_colors[evt.risk_level]
            
            # Centroids
            c1 = ((evt.bbox_1[:2] + evt.bbox_1[2:]) / 2).astype(int)
            c2 = ((evt.bbox_2[:2] + evt.bbox_2[2:]) / 2).astype(int)
            
            # Connecting line
            thickness = 3 if evt.risk_level == 'HIGH' else 2
            cv2.line(frame, tuple(c1), tuple(c2), 
                    risk_color, thickness, cv2.LINE_AA)
            
            # Risk label at midpoint
            mid = ((c1 + c2) / 2).astype(int)
            ttc_str = f'TTC:{evt.ttc:.1f}s' if evt.ttc != float('inf') else ''
            label = f'{evt.risk_level} D:{evt.distance:.0f}px {ttc_str}'
            cv2.putText(frame, label, tuple(mid),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, risk_color, 
                       1, cv2.LINE_AA)
            
            # Highlight bbox in risk color
            for bbox in [evt.bbox_1, evt.bbox_2]:
                b = bbox.astype(int)
                cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]),
                             risk_color, 3)
            
            # Draw predicted collision point
            if evt.predicted_collision_point is not None:
                pt = evt.predicted_collision_point.astype(int)
                cv2.drawMarker(frame, tuple(pt), risk_color,
                              cv2.MARKER_CROSS, 15, 2)
    
    def _draw_flow_overlay(self, frame, flow: np.ndarray) -> None:
        """Draw optical flow visualization in corner."""
        h, w = frame.shape[:2]
        
        # Create flow visualization
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, 
                                     cv2.NORM_MINMAX).astype(np.uint8)
        flow_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Resize to small overlay
        overlay_h, overlay_w = h // 5, w // 5
        flow_small = cv2.resize(flow_vis, (overlay_w, overlay_h))
        
        # Place in bottom-right corner with border
        x_off = w - overlay_w - 10
        y_off = h - overlay_h - 10
        cv2.rectangle(frame, (x_off - 2, y_off - 18),
                     (x_off + overlay_w + 2, y_off + overlay_h + 2),
                     (40, 40, 40), -1)
        cv2.putText(frame, 'Optical Flow', (x_off, y_off - 4),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200),
                   1, cv2.LINE_AA)
        frame[y_off:y_off + overlay_h, 
              x_off:x_off + overlay_w] = flow_small
    
    def _draw_hud(self, frame, frame_idx: int, timestamp: float,
                  active_events: list, incident_count: int,
                  total_incidents: int, width: int) -> None:
        """Draw heads-up display overlay."""
        # Semi-transparent header
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, self.vis_cfg.hud_height),
                     (30, 30, 30), -1)
        cv2.addWeighted(overlay, self.vis_cfg.hud_opacity, frame, 
                       1 - self.vis_cfg.hud_opacity, 0, frame)
        
        y_text = 35
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = self.vis_cfg.font_scale
        
        # Timestamp
        cv2.putText(frame, f't={timestamp:.1f}s | Frame {frame_idx}',
                   (10, y_text), font, fs, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Incident counter
        cv2.putText(frame, f'Incidents: {incident_count}/{total_incidents}',
                   (width - 220, y_text), font, fs, (255, 255, 255), 
                   1, cv2.LINE_AA)
        
        # Active risk badge
        if active_events:
            max_evt = max(active_events, key=lambda e: e.risk_score)
            rc = self.vis_cfg.risk_colors[max_evt.risk_level]
            badge_x = width // 2 - 70
            cv2.rectangle(frame, (badge_x, 5), 
                         (badge_x + 140, 45), rc, -1)
            cv2.putText(frame, f'{max_evt.risk_level} RISK',
                       (badge_x + 10, 33), font, 0.55, 
                       (255, 255, 255), 2, cv2.LINE_AA)


import os  # needed for file size check


class DashboardGenerator:
    """Generates comprehensive statistical analysis dashboard."""
    
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.risk_colors_plt = {
            'HIGH': '#FF3333', 'MEDIUM': '#FF9933', 'LOW': '#FFCC33'
        }
    
    def create(self, incidents: List[Incident],
               raw_events: List[NearMissEvent],
               tracking_stats: dict,
               video_info: dict,
               flow_stats: list = None,
               save_path: str = 'outputs/dashboard.png') -> None:
        """Generate 6-panel analysis dashboard.
        
        Panels:
        1. Risk distribution (pie)
        2. Incident timeline (horizontal bar)
        3. Proximity distribution (histogram)
        4. TTC distribution (histogram)
        5. Object class involvement (bar)
        6. Risk score over time (scatter + rolling avg)
        """
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
        fig.suptitle('Near-Miss Incident Detection — Analysis Dashboard',
                     fontsize=18, fontweight='bold', y=0.98)
        
        self._plot_risk_distribution(fig, gs[0, 0], incidents)
        self._plot_timeline(fig, gs[0, 1:], incidents, video_info)
        self._plot_distance_hist(fig, gs[1, 0], raw_events)
        self._plot_ttc_hist(fig, gs[1, 1], raw_events)
        self._plot_class_involvement(fig, gs[1, 2], incidents)
        self._plot_risk_timeline(fig, gs[2, :], raw_events, video_info)
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f'  Dashboard saved: {save_path}')
    
    def create_flow_dashboard(self, flow_stats: list,
                              flow_anomalies: list,
                              video_info: dict,
                              save_path: str = 'outputs/flow_dashboard.png') -> None:
        """Generate optical flow analysis dashboard."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Optical Flow Analysis', fontsize=16, fontweight='bold')
        
        if flow_stats:
            # 1. Flow magnitude over time
            ax = axes[0, 0]
            frames = [s['frame_idx'] for s in flow_stats]
            times = [f / video_info['fps'] for f in frames]
            mags = [s['mean_magnitude'] for s in flow_stats]
            ax.plot(times, mags, 'b-', alpha=0.7, linewidth=1)
            ax.fill_between(times, mags, alpha=0.2)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Mean Flow Magnitude')
            ax.set_title('Scene Motion Over Time', fontweight='bold')
            
            # Mark anomalies
            if flow_anomalies:
                anom_times = [a['frame_idx'] / video_info['fps'] for a in flow_anomalies]
                anom_mags = [a['magnitude'] for a in flow_anomalies]
                ax.scatter(anom_times, anom_mags, c='red', s=50, 
                          zorder=5, label='Anomalies')
                ax.legend()
            
            # 2. Motion area ratio
            ax = axes[0, 1]
            areas = [s['motion_area_ratio'] * 100 for s in flow_stats]
            ax.plot(times, areas, 'g-', alpha=0.7)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Motion Area (%)')
            ax.set_title('Active Motion Area', fontweight='bold')
            
            # 3. Flow magnitude distribution
            ax = axes[1, 0]
            ax.hist(mags, bins=30, color='#4488CC', edgecolor='black', alpha=0.8)
            ax.axvline(np.mean(mags), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(mags):.2f}')
            ax.set_xlabel('Flow Magnitude')
            ax.set_ylabel('Count')
            ax.set_title('Flow Magnitude Distribution', fontweight='bold')
            ax.legend()
            
            # 4. Direction histogram
            ax = axes[1, 1]
            dirs = [s['dominant_direction'] for s in flow_stats]
            ax.hist(dirs, bins=36, color='#CC8844', edgecolor='black', alpha=0.8)
            ax.set_xlabel('Dominant Direction (radians)')
            ax.set_ylabel('Count')
            ax.set_title('Flow Direction Distribution', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f'  Flow dashboard saved: {save_path}')
    
    # === Private plot methods ===
    
    def _plot_risk_distribution(self, fig, gs_pos, incidents):
        ax = fig.add_subplot(gs_pos)
        risk_counts = defaultdict(int)
        for inc in incidents:
            risk_counts[inc.max_risk_level] += 1
        
        if risk_counts:
            labels = list(risk_counts.keys())
            sizes = list(risk_counts.values())
            colors = [self.risk_colors_plt.get(l, '#999') for l in labels]
            wedges, texts, autotexts = ax.pie(
                sizes, labels=labels, colors=colors, autopct='%1.0f%%',
                startangle=90, textprops={'fontsize': 11}
            )
            for at in autotexts:
                at.set_fontweight('bold')
        ax.set_title('Risk Distribution', fontsize=13, fontweight='bold')
    
    def _plot_timeline(self, fig, gs_pos, incidents, video_info):
        ax = fig.add_subplot(gs_pos)
        for inc in incidents:
            color = self.risk_colors_plt.get(inc.max_risk_level, '#999')
            ax.barh(0.5, max(0.1, inc.duration_sec), left=inc.start_time,
                   height=0.6, color=color, alpha=0.8, 
                   edgecolor='black', linewidth=0.5)
            ax.text(inc.start_time + inc.duration_sec / 2, 0.5,
                   f'#{inc.incident_id}', ha='center', va='center', fontsize=7)
        
        ax.set_xlabel('Time (seconds)')
        ax.set_title('Incident Timeline', fontsize=13, fontweight='bold')
        ax.set_yticks([])
        ax.set_xlim(0, video_info['duration_sec'])
        legend_patches = [mpatches.Patch(color=c, label=l) 
                         for l, c in self.risk_colors_plt.items()]
        ax.legend(handles=legend_patches, loc='upper right', fontsize=9)
    
    def _plot_distance_hist(self, fig, gs_pos, raw_events):
        ax = fig.add_subplot(gs_pos)
        if raw_events:
            distances = [e.distance for e in raw_events]
            ax.hist(distances, bins=30, color='#4488CC', alpha=0.8, 
                   edgecolor='black')
            nm = self.cfg.near_miss
            ax.axvline(nm.distance_high, color='red', linestyle='--',
                      label=f'High ({nm.distance_high}px)')
            ax.axvline(nm.distance_medium, color='orange', linestyle='--',
                      label=f'Medium ({nm.distance_medium}px)')
            ax.legend(fontsize=8)
        ax.set_xlabel('Distance (pixels)')
        ax.set_ylabel('Count')
        ax.set_title('Proximity Distribution', fontsize=13, fontweight='bold')
    
    def _plot_ttc_hist(self, fig, gs_pos, raw_events):
        ax = fig.add_subplot(gs_pos)
        if raw_events:
            ttcs = [e.ttc for e in raw_events if 0 < e.ttc < 10]
            if ttcs:
                ax.hist(ttcs, bins=25, color='#CC4444', alpha=0.8, 
                       edgecolor='black')
                nm = self.cfg.near_miss
                ax.axvline(nm.ttc_high, color='red', linestyle='--',
                          label=f'High ({nm.ttc_high}s)')
                ax.axvline(nm.ttc_medium, color='orange', linestyle='--',
                          label=f'Medium ({nm.ttc_medium}s)')
                ax.legend(fontsize=8)
        ax.set_xlabel('Time-to-Collision (seconds)')
        ax.set_ylabel('Count')
        ax.set_title('TTC Distribution', fontsize=13, fontweight='bold')
    
    def _plot_class_involvement(self, fig, gs_pos, incidents):
        ax = fig.add_subplot(gs_pos)
        if incidents:
            class_inv = defaultdict(int)
            for inc in incidents:
                for cls in inc.involved_classes:
                    class_inv[cls] += 1
            classes = list(class_inv.keys())
            counts = list(class_inv.values())
            bars = ax.barh(classes, counts, color='#44AA88', edgecolor='black')
            for bar, count in zip(bars, counts):
                ax.text(bar.get_width() + 0.2,
                       bar.get_y() + bar.get_height() / 2,
                       str(count), va='center', fontsize=10)
        ax.set_xlabel('Incidents Involved')
        ax.set_title('Class Involvement', fontsize=13, fontweight='bold')
    
    def _plot_risk_timeline(self, fig, gs_pos, raw_events, video_info):
        ax = fig.add_subplot(gs_pos)
        if raw_events:
            timestamps = [e.timestamp for e in raw_events]
            scores = [e.risk_score for e in raw_events]
            colors = [self.risk_colors_plt.get(e.risk_level, '#999') 
                     for e in raw_events]
            ax.scatter(timestamps, scores, c=colors, alpha=0.5, 
                      s=15, edgecolor='none')
            
            if len(timestamps) > 20:
                idx = np.argsort(timestamps)
                ts = np.array(timestamps)[idx]
                sc = np.array(scores)[idx]
                w = min(50, len(ts) // 5)
                if w > 1:
                    avg = np.convolve(sc, np.ones(w) / w, mode='valid')
                    ax.plot(ts[:len(avg)], avg, color='black', linewidth=2,
                           label='Rolling Average')
                    ax.legend(fontsize=9)
        
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Risk Score')
        ax.set_title('Risk Score Timeline', fontsize=13, fontweight='bold')
        ax.set_xlim(0, video_info['duration_sec'])
        ax.set_ylim(0, 1.05)
    
    def extract_peak_frames(self, video_path: str, 
                            incidents: List[Incident],
                            n_frames: int = 4,
                            save_path: str = 'outputs/peak_moments.png') -> None:
        """Extract and display frames at peak risk moments."""
        cap = cv2.VideoCapture(video_path)
        
        sorted_inc = sorted(incidents, key=lambda i: -i.max_risk_score)[:n_frames]
        if not sorted_inc:
            print('  No incidents to display.')
            return
        
        fig, axes = plt.subplots(1, len(sorted_inc), 
                                 figsize=(6 * len(sorted_inc), 5))
        if len(sorted_inc) == 1:
            axes = [axes]
        
        fig.suptitle('Peak Near-Miss Moments', fontsize=14, fontweight='bold')
        
        for ax, inc in zip(axes, sorted_inc):
            cap.set(cv2.CAP_PROP_POS_FRAMES, inc.peak_frame)
            ret, frame = cap.read()
            if ret:
                ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                ax.set_title(
                    f'Incident #{inc.incident_id} | {inc.max_risk_level} Risk\n'
                    f't={inc.start_time:.1f}s | TTC={inc.min_ttc:.2f}s'
                    if inc.min_ttc != float('inf') else
                    f'Incident #{inc.incident_id} | {inc.max_risk_level} Risk\n'
                    f't={inc.start_time:.1f}s',
                    fontsize=10
                )
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        cap.release()
        print(f'  Peak moments saved: {save_path}')
