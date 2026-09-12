# Maritime Autonomous Multi-Agent System (GulfMAS)

A 24-hour hackathon MVP of a coordinated multi-agent maritime system for the **Gulf of Mexico / US Gulf Coast** region.

Three AI agents share a common state and coordinate through a lightweight in-process orchestrator, surfaced on an interactive Streamlit + pydeck command dashboard.

---

## 🚀 Quick Start

### 1. Installation

Ensure Python 3.11+ is installed, then install the dependencies:

```bash
pip install -r requirements.txt
```

### 2. Launch the Dashboard

```bash
streamlit run dashboard.py
```

Open your browser to `http://localhost:8501`.

---

## 🏗️ Architecture & Agents

### Agent 1 — Route Optimization (`agents/route.py`)
- **Technology:** `networkx` weighted grid graph over bounding box (Lat 27–30°N, Lon -95–-88°W).
- **Edge Cost:** Haversine distance penalized by weather factors (wave height and wind speed/direction).
- **Algorithm:** $A^*$ heuristic search (`astar_path`).
- **Baseline Comparison:** Straight-line path through identical weather fields to compute realistic fuel/cost savings %.
- **Dynamic Hazard Avoidance:** Dynamically inflates edge weights near any flagged dark vessels (`avoid_points`) to reroute vessels around security or collision hazards.

### Agent 2 — Dark Vessel Detection (`agents/dark_vessel.py`)
- **Technology:** `scikit-learn` `IsolationForest` anomaly detection.
- **Workflow:** Simulates AIS transponder tampering by dropping ping windows.
- **Feature Engineering:**
  1. Maximum time-gap length (minutes).
  2. Speed delta across the gap.
  3. Heading change across the gap.
  4. Spatial position deviation between dead-reckoning (predicted straight line) and actual reappearance.
- Evaluates against ground-truth labels and outputs confidence-weighted anomaly flags.

### Agent 3 — Debris Coordination (`agents/debris.py`)
- **Technology:** `scikit-learn` `DBSCAN` clustering.
- **Workflow:** Takes coastal debris sightings and clusters active sightings into high-density pollution hotspots.
- **Collector Dispatch:** Simulates a fleet of 4 autonomous cleanup vessels (`COLLECTOR_1` through `COLLECTOR_4`) and assigns each hotspot to the nearest collector vessel via Euclidean/Haversine distance minimization.

### In-Process Orchestrator (`orchestrator.py`)
- Maintains a single thread-safe shared state dictionary.
- Coordinates cross-agent interaction: **If a flagged dark vessel's position falls within 35 nautical miles of the active route, Agent 1 is invoked to recompute the path around the hazard.**
- Logs all state changes and operational interventions in timestamped plain English.

---

## 📊 Data Sources (Real vs. Synthetic)

| Dataset | Status | Details |
|---|---|---|
| **AIS Vessel Tracks** | **Synthetic** | NOAA Marine Cadastre files are multi-gigabyte archives. Generated ~30 vessels navigating realistic Gulf shipping lanes (Galveston, Houston, Mississippi Delta, Mobile) with kinematic heading and speed jitter. |
| **Marine Weather** | **Synthetic** | Generated grid with wave heights and wind speeds, featuring an intentional storm cell near $28.5^\circ\text{N}, -91.5^\circ\text{W}$ to demonstrate routing variance. *(Open-Meteo live API integration code is structured in the data module for production expansion)*. |
| **Marine Debris** | **Synthetic** | NOAA Marine Debris Program seed locations placed along the Texas/Louisiana/Alabama coastline with dynamic multi-step drift and new sighting spawn loops. |

---

## 🖥️ Dashboard Features

- **PyDeck Visual Map:**
  - 🟢 **Green Path:** Weather-optimized $A^*$ route.
  - 🔵 **Cyan Points:** Commercial AIS traffic.
  - 🔴 **Red Nodes:** Flagged dark/anomalous vessels.
  - 🟡 **Yellow Circles:** Debris hotspots sized by cluster density.
  - 🟣 **Purple Nodes & Vector Lines:** Autonomous cleanup vessels and their dispatch targets.
- **Side Panel Metrics:** Live fuel savings %, count of intercepted dark vessels, and active debris hotspots covered.
- **Multi-Agent Event Feed:** Chronological real-time plain-English audit log of agent operations and rerouting triggers.
- **Interactive Controls:** Step simulation button (tick) and continuous auto-play toggle.

---

## ⚠️ Known Shortcuts & Hackathon Limitations

1. **Single-Process Model:** Designed for speed and simplicity during a 24-hour sprint; does not use Celery or Kafka.
2. **Nearest-Neighbor Dispatch:** Debris collector assignment uses greedy nearest distance rather than a full Hungarian algorithm or auction bidding.
3. **Grid Discretization:** The routing graph uses a $0.25^\circ$ (~15 nm) grid resolution for sub-second $A^*$ response.
