# 🚦 Near-Miss Incident Detection System

## AI/ML Technical Assessment | Computer Vision Challenge

**Author:** Osama Bsharat  
**Date:** 2026  
**Environment:** Google Colab (Free Tier) / CPU

---

## Overview

An automated system that detects and analyzes near-miss traffic incidents from video footage using state-of-the-art computer vision. The system identifies situations where accidents were narrowly avoided and provides detailed risk analysis with professional reporting.

### System Architecture

```
Video Input ──► YOLOv8n Detection ──► ByteTrack Tracking ──► Kalman Filter
                                                                  │
                              ┌───────────────────────────────────┘
                              ▼
Optical Flow ──► Flow Anomalies ──► Near-Miss Detection ──► Risk Scoring
                                           │
                              ┌────────────┴────────────┐
                              ▼                         ▼
                    False Positive Filter      Incident Grouping
                              │                         │
                              ▼                         ▼
                    Annotated Video          Dashboard + Report
```

## Project Structure

```
near-miss-detection/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
│
├── src/                                # Modular source code
│   ├── __init__.py                     # Package exports
│   ├── config.py                       # Centralized configuration
│   ├── utils.py                        # Data models + utilities
│   ├── detector.py                     # YOLOv8 detection module
│   ├── tracker.py                      # Kalman-filtered tracking
│   ├── near_miss.py                    # Near-miss detection logic
│   ├── optical_flow.py                 # Dense optical flow analysis
│   ├── visualizer.py                   # Video annotation + dashboards
│   └── report.py                       # HTML report + JSON export
│
├── notebooks/
│   ├── Near_Miss_Detection.ipynb       # Main pipeline notebook
│   └── Evaluation.ipynb                # Evaluation & analysis notebook
│
├── outputs/                            # Generated at runtime
│   ├── annotated_near_miss.mp4
│   ├── near_miss_report.html
│   ├── dashboard.png
│   ├── flow_dashboard.png
│   ├── peak_moments.png
│   ├── class_pair_analysis.png
│   └── analysis_results.json
│
└── demo_video.mp4                      # Demo video (record separately)
```
