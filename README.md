# MaritimeMAS — Gulf of Mexico Multi-Agent System (Production Upgrade)

A real-time, production-grade maritime operations dashboard combining three coordinated AI agents over the US Gulf Coast (lat 27–30°N, lon −95–−88°W).

---

## 🚀 Quick Start

```powershell
# Navigate to project folder
cd "C:\Users\LAKSHYA SINHA\.gemini\antigravity\scratch\maritime-mas"

# Run automated smoke test
python smoke_test.py

# Launch the live dashboard
python -m streamlit run dashboard.py
```

Open **`http://localhost:8501`** in your browser. Click **▶️ Auto-Play** in the sidebar to start the simulation.

---

## 🌟 Upgraded Features & Capabilities

### 🛰️ Real Satellite Imagery Base Map
- Built on **Esri World Imagery** satellite tile services (`https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer`).
- Displays true satellite coastline, offshore shelves, and land boundaries across the Gulf of Mexico without needing Mapbox tokens.

### ⚓ Real-Time Animated Vessel & Wake Layer
- Vessels display vector headings and directional indicators matching true AIS course.
- **Fading Wake Trails**: Tracks each vessel's historical positions over time for smooth visual movement.

### 💡 AI Explainability Panel ("Why was this flagged?")
- Collapsible detailed explainability cards for every dark vessel flagged by `sklearn.ensemble.IsolationForest`.
- Surfacing exact metrics: transponder gap duration (min), dead-reckoning displacement error (km), speed delta (kts), and heading shift (°), paired with plain-language explanation text.

### 🌀 NOAA Hurricane Ida Demo Mode (Cat 4 Storm Avoidance)
- Replays real **NOAA HURDAT2** historical storm track data for **Hurricane Ida (Aug 2021)** as it traversed the Gulf of Mexico.
- **Dynamic Cross-Agent Rerouting**: When enabled, the storm eye and gale force radius add dynamic cost penalties to the grid graph, triggering Agent 1 to dynamically route around the Cat 4 eye in real time.

### 🐟 Global Fishing Watch (GFW) Spatial Layer
- Overlays real commercial fishing effort zones across the Mississippi Delta Shrimping Fleet, Texas Menhaden Fishery, and Pelagic Reef Fishing areas.
- Allows operators to contextualize dark vessel flags against known fishing grounds.

### ⏱️ Interactive Timeline Scrubber
- Full state snapshotting engine allowing operators to scrub backwards and forwards to replay past simulation ticks.

---

## 📊 Data Authenticity Breakdown

| Dataset | Data Origin | Status | Description & Date Range |
|---|---|---|---|
| **Weather** | Open-Meteo Marine API | **REAL** | Live hourly wave height, wind speed/direction, and ocean current velocity sampled via HTTP API. |
| **Storm Track** | NOAA HURDAT2 Database | **REAL HISTORICAL** | Actual track points, wind speeds (kts), and central pressure (mb) for Hurricane Ida (Aug 27–29, 2021). |
| **Fishing Zones** | Global Fishing Watch (GFW) | **REAL SPATIAL** | EEZ boundaries, shrimping fleet coordinates, and pelagic fishing effort zones in the Gulf shelf. |
| **AIS Tracks** | NOAA Cadastre Schema | **SYNTHETIC (HIGH DENSITY)** | 50 vessels generated along canonical Gulf shipping lanes over a 24-hour density window. |
| **Debris Sightings**| NOAA Marine Debris Program | **SYNTHETIC** | 25 coastal seed points near Louisiana & Texas shelf matching NOAA marine debris survey schema. |

---

## 🤖 Multi-Agent Architecture

```
maritime-mas/
├── data/
│   └── loader.py          # Real Open-Meteo API, NOAA HURDAT2 storm track, GFW zones, AIS, Debris
├── agents/
│   ├── route.py           # Agent 1: A* grid solver with weather & storm avoidance cost multipliers
│   ├── dark_vessel.py     # Agent 2: AIS transponder gap injection & IsolationForest anomaly detection
│   └── debris.py          # Agent 3: Drift spawner, DBSCAN clustering & collector vessel assignment
├── orchestrator.py        # Central state, tick history snapshotter, cross-agent reroute engine
├── dashboard.py           # Streamlit + Pydeck Esri Satellite UI app
├── smoke_test.py          # End-to-end verification script
├── requirements.txt       # Dependencies
└── README.md
```

### Agent 1 — Route Optimization (`agents/route.py`)
- Grid graph solver over 0.25° grid resolution (~27 km).
- Edge weights = Distance × (1 + Wave Penalty + Headwind Penalty + Obstacle Penalty).
- Computes baseline straight-line costs vs optimized A* route to output net fuel savings %.

### Agent 2 — Dark Vessel Detection (`agents/dark_vessel.py`)
- Injects realistic AIS transponder cut-outs.
- Extracts per-vessel feature vectors: `max_gap_minutes`, `mean_gap_minutes`, `displacement_error_km`, `speed_change_after_gap`, `heading_change_after_gap`, `ping_count`.
- Trains `IsolationForest` (contamination=0.25) to flag suspicious non-transponding vessels with precision & recall metrics.

### Agent 3 — Marine Debris Coordination (`agents/debris.py`)
- Simulates coastal drift by spawning new sightings near existing hotspots.
- Runs **DBSCAN** clustering (`eps` ≈ 55 km, `min_samples` = 2).
- Assigns collector vessels based on transit distance and computes ETA for cleanup operations.

---

## ⚡ Verification & Testing

To run the complete automated test suite:
```powershell
python smoke_test.py
```
Outputs `[PASS]` for all 22+ validation checks across data loading, graph pathfinding, anomaly detection, collector assignment, storm mode toggle, and snapshot replay.
