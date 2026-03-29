"""
Report Generation Module | Professional HTML summary report.

Generates a comprehensive, self-contained HTML report with:
- Embedded images (base64)
- Interactive-style tables
- Responsive design
- Complete methodology documentation
"""

import base64
import os
import json
import numpy as np
from typing import List
from collections import defaultdict

from ultralytics import cfg

from .config import PipelineConfig
from .utils import NearMissEvent, Incident


def _img_b64(path: str) -> str:
    """Read image file and return base64 encoded string."""
    if not os.path.exists(path):
        return ''
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def generate_html_report(
    incidents: List[Incident],
    raw_events: List[NearMissEvent],
    tracking_stats: dict,
    video_info: dict,
    cfg: PipelineConfig,
    flow_anomaly_count: int = 0,
    output_path: str = 'outputs/near_miss_report.html'
) -> str:
    """Generate comprehensive HTML analysis report.
    
    The report is self-contained with all images embedded as base64.
    Designed to be professional enough to present to stakeholders.
    """
    
    # Risk counts
    risk_counts = defaultdict(int)
    for inc in incidents:
        risk_counts[inc.max_risk_level] += 1
    
    dashboard_b64 = _img_b64(os.path.join(cfg.output_dir, 'dashboard.png'))
    peaks_b64 = _img_b64(os.path.join(cfg.output_dir, 'peak_moments.png'))
    samples_b64 = _img_b64(os.path.join(cfg.output_dir, 'sample_frames.png'))
    flow_b64 = _img_b64(os.path.join(cfg.output_dir, 'flow_dashboard.png'))
    class_b64 = _img_b64(os.path.join(cfg.output_dir, 'class_pair_analysis.png'))
    
    # Incident table rows
    incident_rows = ''
    for inc in incidents:
        ttc = f'{inc.min_ttc:.2f}s' if inc.min_ttc != float('inf') else 'N/A'
        rl = inc.max_risk_level.lower()
        incident_rows += f'''
        <tr>
            <td>{inc.incident_id}</td>
            <td>{inc.start_time:.1f}s – {inc.end_time:.1f}s</td>
            <td>{inc.duration_sec:.2f}s</td>
            <td><span class="badge {rl}">{inc.max_risk_level}</span></td>
            <td>{inc.max_risk_score:.2f}</td>
            <td>{inc.min_distance:.0f}px</td>
            <td>{ttc}</td>
            <td>{", ".join(sorted(inc.involved_classes))}</td>
        </tr>'''
    
    # Flow section (conditional)
    flow_section = ''
    if flow_b64:
        flow_section = f'''
        <section class="card">
            <h2><span class="icon">🌊</span> Optical Flow Analysis</h2>
            <p>Dense optical flow (Farneback method) provides scene-level motion analysis 
               independent of object tracking. Anomalous motion patterns often correlate 
               with sudden braking or evasive maneuvers.</p>
            <p><strong>Flow anomaly frames detected:</strong> {flow_anomaly_count}</p>
            <div class="img-wrap">
                <img src="data:image/png;base64,{flow_b64}" alt="Flow Dashboard">
            </div>
        </section>'''
    
    # Class pair section
    class_section = ''
    if class_b64:
        class_section = f'''
        <section class="card">
            <h2><span class="icon">🔀</span> Multi-Class Interaction Analysis</h2>
            <div class="img-wrap">
                <img src="data:image/png;base64,{class_b64}" alt="Class Analysis">
            </div>
        </section>'''
    
    # Build HTML
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Near-Miss Incident Detection | Analysis Report</title>
<style>
:root {{
    --primary: #0a1628;
    --primary-light: #162a4a;
    --accent: #3b82f6;
    --accent-glow: rgba(59, 130, 246, 0.15);
    --danger: #ef4444;
    --warning: #f59e0b;
    --caution: #eab308;
    --success: #22c55e;
    --surface: #ffffff;
    --surface-alt: #f8fafc;
    --border: #e2e8f0;
    --text: #1e293b;
    --text-muted: #64748b;
    --radius: 12px;
    --shadow: 0 1px 3px rgba(0,0,0,0.08), 0 4px 12px rgba(0,0,0,0.04);
    --shadow-lg: 0 4px 20px rgba(0,0,0,0.1);
}}

* {{ margin: 0; padding: 0; box-sizing: border-box; }}

body {{
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
    background: var(--surface-alt);
    color: var(--text);
    line-height: 1.65;
    -webkit-font-smoothing: antialiased;
}}

.container {{ max-width: 1100px; margin: 0 auto; padding: 24px 20px; }}

/* ── Header ── */
.hero {{
    background: linear-gradient(135deg, var(--primary) 0%, var(--primary-light) 60%, #1e3a5f 100%);
    color: white;
    padding: 48px 40px;
    border-radius: var(--radius);
    margin-bottom: 28px;
    text-align: center;
    position: relative;
    overflow: hidden;
    box-shadow: var(--shadow-lg);
}}
.hero::before {{
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(59,130,246,0.12) 0%, transparent 70%);
    border-radius: 50%;
}}
.hero h1 {{
    font-size: 2.1em;
    font-weight: 700;
    margin-bottom: 8px;
    letter-spacing: -0.02em;
    position: relative;
}}
.hero .subtitle {{
    opacity: 0.8;
    font-size: 1.1em;
    font-weight: 400;
    position: relative;
}}
.hero .meta {{
    margin-top: 16px;
    font-size: 0.85em;
    opacity: 0.6;
    position: relative;
}}

/* ── Stat Cards ── */
.stats {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 16px;
    margin-bottom: 28px;
}}
.stat {{
    background: var(--surface);
    border-radius: var(--radius);
    padding: 22px 18px;
    text-align: center;
    box-shadow: var(--shadow);
    border: 1px solid var(--border);
    transition: transform 0.15s;
}}
.stat:hover {{ transform: translateY(-2px); box-shadow: var(--shadow-lg); }}
.stat .num {{
    font-size: 2.2em;
    font-weight: 700;
    color: var(--primary);
    line-height: 1.1;
}}
.stat .num.danger {{ color: var(--danger); }}
.stat .num.warn {{ color: var(--warning); }}
.stat .num.ok {{ color: var(--success); }}
.stat .lbl {{
    font-size: 0.82em;
    color: var(--text-muted);
    margin-top: 6px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 600;
}}

/* ── Cards ── */
.card {{
    background: var(--surface);
    border-radius: var(--radius);
    padding: 28px;
    margin-bottom: 20px;
    box-shadow: var(--shadow);
    border: 1px solid var(--border);
}}
.card h2 {{
    color: var(--primary);
    font-size: 1.3em;
    margin-bottom: 16px;
    padding-bottom: 10px;
    border-bottom: 2px solid var(--border);
    display: flex;
    align-items: center;
    gap: 8px;
}}
.card h2 .icon {{ font-size: 1.2em; }}

/* ── Callout ── */
.callout {{
    background: var(--accent-glow);
    border-left: 4px solid var(--accent);
    padding: 16px 20px;
    border-radius: 0 8px 8px 0;
    margin: 16px 0;
    font-size: 0.95em;
}}
.callout p {{ margin-bottom: 6px; }}
.callout p:last-child {{ margin-bottom: 0; }}
.callout strong {{ color: var(--primary); }}

/* ── Table ── */
table {{
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-size: 0.88em;
    overflow: hidden;
    border-radius: 8px;
    border: 1px solid var(--border);
}}
thead th {{
    background: var(--primary);
    color: white;
    padding: 11px 12px;
    text-align: left;
    font-weight: 600;
    font-size: 0.9em;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}}
tbody td {{
    padding: 9px 12px;
    border-bottom: 1px solid var(--border);
}}
tbody tr:last-child td {{ border-bottom: none; }}
tbody tr:hover {{ background: var(--surface-alt); }}

/* ── Badges ── */
.badge {{
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 0.78em;
    font-weight: 700;
    letter-spacing: 0.04em;
    color: white;
}}
.badge.high {{ background: var(--danger); }}
.badge.medium {{ background: var(--warning); }}
.badge.low {{ background: var(--caution); color: #333; }}

/* ── Images ── */
.img-wrap {{
    text-align: center;
    margin: 18px 0;
}}
.img-wrap img {{
    max-width: 100%;
    border-radius: 8px;
    box-shadow: var(--shadow);
}}

/* ── Config Table ── */
.config-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 16px;
    margin-top: 12px;
}}
.config-item {{
    display: flex;
    justify-content: space-between;
    padding: 8px 0;
    border-bottom: 1px solid var(--border);
}}
.config-item .key {{ font-weight: 600; color: var(--text-muted); font-size: 0.9em; }}
.config-item .val {{ font-weight: 500; font-family: 'SF Mono', Consolas, monospace; font-size: 0.9em; }}

/* ── Footer ── */
.footer {{
    text-align: center;
    padding: 24px;
    color: var(--text-muted);
    font-size: 0.85em;
}}

/* ── Responsive ── */
@media (max-width: 700px) {{
    .stats {{ grid-template-columns: repeat(2, 1fr); }}
    .hero {{ padding: 32px 24px; }}
    .hero h1 {{ font-size: 1.5em; }}
    .card {{ padding: 20px 16px; }}
}}
</style>
</head>
<body>
<div class="container">

    <!-- HERO -->
    <div class="hero">
        <h1>🚦 Near-Miss Incident Detection Report</h1>
        <p class="subtitle">Automated Traffic Safety Analysis Using Computer Vision</p>
        <p class="meta">YOLOv8 Nano • ByteTrack • Kalman Filter • Optical Flow</p>
    </div>

    <!-- STATS GRID -->
    <div class="stats">
        <div class="stat">
            <div class="num">{len(incidents)}</div>
            <div class="lbl">Total Incidents</div>
        </div>
        <div class="stat">
            <div class="num danger">{risk_counts.get("HIGH", 0)}</div>
            <div class="lbl">High Risk</div>
        </div>
        <div class="stat">
            <div class="num warn">{risk_counts.get("MEDIUM", 0)}</div>
            <div class="lbl">Medium Risk</div>
        </div>
        <div class="stat">
            <div class="num ok">{risk_counts.get("LOW", 0)}</div>
            <div class="lbl">Low Risk</div>
        </div>
        <div class="stat">
            <div class="num">{tracking_stats["unique_tracks"]}</div>
            <div class="lbl">Objects Tracked</div>
        </div>
        <div class="stat">
            <div class="num">{video_info["duration_str"]}</div>
            <div class="lbl">Video Duration</div>
        </div>
    </div>

    <!-- METHODOLOGY -->
    <section class="card">
        <h2><span class="icon">🔬</span> Methodology</h2>
        <div class="callout">
            <p><strong>Detection:</strong> YOLOv8 Nano (3.2M parameters) , CPU-optimized with mAP@50 of 37.3, processing at ~30ms/frame</p>
            <p><strong>Tracking:</strong> ByteTrack , two-stage IoU-based association with low-confidence second pass for occlusion handling</p>
            <p><strong>Trajectory:</strong> 6-state Kalman Filter [x, y, vx, vy, ax, ay] , constant acceleration model with EMA-smoothed velocity estimation</p>
            <p><strong>Optical Flow:</strong> Farneback dense flow , independent motion validation and scene anomaly detection</p>
            <p><strong>Risk Assessment:</strong> Composite scoring: score = (0.4 × dist_score + 0.6 × ttc_score) × vuln_multiplier</p>
            <p><strong>False Positive Reduction:</strong> 3-layer filtering , temporal persistence (≥{cfg.filter.min_incident_frames} frames), area threshold ({cfg.detection.min_object_area}px²), incident merging (gap ≤{cfg.filter.merge_gap_frames} frames)</p>
        </div>
    </section>

    <!-- DETECTION OVERVIEW -->
    <section class="card">
        <h2><span class="icon">📊</span> Detection Overview</h2>
        <div class="config-grid">
            <div>
                <div class="config-item">
                    <span class="key">Total Detections</span>
                    <span class="val">{tracking_stats["total_detections"]:,}</span>
                </div>
                <div class="config-item">
                    <span class="key">Frames Processed</span>
                    <span class="val">{tracking_stats["total_frames"]:,}</span>
                </div>
                <div class="config-item">
                    <span class="key">Unique Tracks</span>
                    <span class="val">{tracking_stats["unique_tracks"]}</span>
                </div>
            </div>
            <div>
                <div class="config-item">
                    <span class="key">Avg Objects/Frame</span>
                    <span class="val">{tracking_stats["avg_objects_per_frame"]:.1f}</span>
                </div>
                <div class="config-item">
                    <span class="key">Avg Track Length</span>
                    <span class="val">{tracking_stats["avg_track_length"]:.0f} frames</span>
                </div>
                <div class="config-item">
                    <span class="key">Avg Confidence</span>
                    <span class="val">{tracking_stats.get("avg_confidence", 0):.2f}</span>
                </div>
            </div>
        </div>
        <div class="img-wrap">
            <img src="data:image/png;base64,{samples_b64}" alt="Sample Frames">
        </div>
    </section>

    <!-- DASHBOARD -->
    <section class="card">
        <h2><span class="icon">📈</span> Analysis Dashboard</h2>
        <div class="img-wrap">
            <img src="data:image/png;base64,{dashboard_b64}" alt="Dashboard">
        </div>
    </section>

    <!-- PEAK MOMENTS -->
    <section class="card">
        <h2><span class="icon">⚠️</span> Peak Near-Miss Moments</h2>
        <p>Frames captured at the highest risk score within each top incident:</p>
        <div class="img-wrap">
            <img src="data:image/png;base64,{peaks_b64}" alt="Peak Moments">
        </div>
    </section>

    {flow_section}

    {class_section}

    <!-- INCIDENT LOG -->
    <section class="card">
        <h2><span class="icon">📋</span> Incident Log</h2>
        <table>
            <thead>
                <tr>
                    <th>ID</th><th>Time Range</th><th>Duration</th>
                    <th>Risk</th><th>Score</th><th>Min Dist</th>
                    <th>Min TTC</th><th>Objects</th>
                </tr>
            </thead>
            <tbody>
                {incident_rows}
            </tbody>
        </table>
    </section>

    <!-- CONFIGURATION -->
    <section class="card">
        <h2><span class="icon">⚙️</span> Technical Configuration</h2>
        <div class="config-grid">
            <div>
                <div class="config-item">
                    <span class="key">Model</span>
                    <span class="val">{cfg.detection.model_name}</span>
                </div>
                <div class="config-item">
                    <span class="key">Confidence</span>
                    <span class="val">{cfg.detection.confidence}</span>
                </div>
                <div class="config-item">
                    <span class="key">Tracker</span>
                    <span class="val">ByteTrack</span>
                </div>
                <div class="config-item">
                    <span class="key">Kalman State</span>
                    <span class="val">{cfg.kalman.state_dim}D (pos+vel+acc)</span>
                </div>
            </div>
            <div>
                <div class="config-item">
                    <span class="key">Distance Thresholds</span>
                    <span class="val">H:{cfg.near_miss.distance_high} / M:{cfg.near_miss.distance_medium} / L:{cfg.near_miss.distance_low}px</span>
                </div>
                <div class="config-item">
                    <span class="key">TTC Thresholds</span>
                    <span class="val">H:{cfg.near_miss.ttc_high} / M:{cfg.near_miss.ttc_medium} / L:{cfg.near_miss.ttc_low}s</span>
                </div>
                <div class="config-item">
                    <span class="key">Resolution</span>
                    <span class="val">{video_info["width"]}×{video_info["height"]} @ {video_info["fps"]:.0f}fps</span>
                </div>
                <div class="config-item">
                    <span class="key">Optical Flow</span>
                    <span class="val">Farneback (interval={cfg.optical_flow.compute_interval})</span>
                </div>
            </div>
        </div>
    </section>

    <!-- FOOTER -->
    <div class="footer">
        <p>Generated by Eng. Osama Bsharat | +970594145999</p>
        <p>AI/ML Technical Assessment | Computer Vision Challenge</p>
    </div>

</div>
</body>
</html>'''
    
    with open(output_path, 'w', encoding='utf-8') as f:
         f.write(html)
    
    print(f'  Report saved: {output_path}')
    return output_path


def export_json_results(incidents: List[Incident],
                        tracking_stats: dict,
                        video_info: dict,
                        output_path: str = 'outputs/analysis_results.json') -> str:
    """Export machine-readable results as JSON."""
    results = {
        'video_info': video_info,
        'tracking_stats': {
            k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
            for k, v in tracking_stats.items()
        },
        'summary': {
            'total_incidents': len(incidents),
            'high_risk': sum(1 for i in incidents if i.max_risk_level == 'HIGH'),
            'medium_risk': sum(1 for i in incidents if i.max_risk_level == 'MEDIUM'),
            'low_risk': sum(1 for i in incidents if i.max_risk_level == 'LOW'),
        },
        'incidents': [
            {
                'id': inc.incident_id,
                'start_time': inc.start_time,
                'end_time': inc.end_time,
                'duration_sec': inc.duration_sec,
                'risk_level': inc.max_risk_level,
                'risk_score': float(inc.max_risk_score),
                'min_distance_px': float(inc.min_distance),
                'min_ttc_sec': float(inc.min_ttc) if inc.min_ttc != float('inf') else None,
                'involved_classes': sorted(list(inc.involved_classes)),
                'peak_frame': inc.peak_frame,
            }
            for inc in incidents
        ]
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f'  JSON results saved: {output_path}')
    return output_path
