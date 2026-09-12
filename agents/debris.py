"""
agents/debris.py
================
AGENT 3 — MARINE DEBRIS COORDINATION

Pipeline:
1. Start from debris seed points (lat/lon from data/loader.py).
2. On each simulation tick, randomly spawn a few new sightings near existing
   hotspots (debris drifts / new sightings reported).
3. Run DBSCAN on all active sightings to cluster them into hotspots.
4. Maintain 3-4 "collector" vessels with starting positions.
5. Assign each hotspot to the nearest collector (Euclidean on lat/lon).
6. Return full state for orchestrator and dashboard.

Public API:
    init_collectors() -> list[dict]
    get_debris_state(sightings_df, collectors) -> list[dict]
    spawn_new_sightings(sightings_df, n=3) -> pd.DataFrame
"""

import logging
import math
import random
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

logger = logging.getLogger(__name__)

BBOX = {"lat_min": -40.0, "lat_max": 60.0, "lon_min": -180.0, "lon_max": 180.0}
DEBRIS_TYPES = ["Plastic", "Derelict Gear", "Foam", "Metal", "Rope", "Mixed"]

# DBSCAN parameters (degrees ≈ km at these latitudes)
# eps=1.2° ≈ 130 km — clusters real regional monitoring stations into oceanic hotspots
DBSCAN_EPS = 1.2
DBSCAN_MIN_SAMPLES = 2


# --------------------------------------------------------------------------- #
#  Geometry helper                                                             #
# --------------------------------------------------------------------------- #
def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


# --------------------------------------------------------------------------- #
#  Collector vessels                                                           #
# --------------------------------------------------------------------------- #
COLLECTOR_START_POSITIONS = [
    (29.5, -94.8, "CC-01"),
    (29.9, -90.1, "CC-02"),
    (51.0, 1.5, "CC-03"),
    (36.1, -5.4, "CC-04"),
    (30.0, 32.5, "CC-05"),
    (1.3, 103.8, "CC-06"),
    (35.5, 139.8, "CC-07"),
    (8.9, -79.5, "CC-08"),
]


def init_collectors() -> list[dict[str, Any]]:
    """Initialise collector vessels at their home positions."""
    return [
        {
            "collector_id": cid,
            "lat": lat,
            "lon": lon,
            "status": "IDLE",
            "assigned_hotspot": None,
        }
        for lat, lon, cid in COLLECTOR_START_POSITIONS
    ]


# --------------------------------------------------------------------------- #
#  Sighting spawning (drift simulation)                                       #
# --------------------------------------------------------------------------- #
_spawn_counter = [0]  # global counter for unique IDs


def spawn_new_sightings(sightings_df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """
    Spawn n new debris sightings near existing ones (simulating drift / new reports).
    Returns the updated sightings DataFrame.
    """
    if sightings_df.empty:
        return sightings_df
    new_rows = []
    for _ in range(n):
        seed_row = sightings_df.sample(1).iloc[0]
        new_lat = seed_row["lat"] + random.gauss(0, 0.15)
        new_lon = seed_row["lon"] + random.gauss(0, 0.2)
        # Clamp to bounding box
        new_lat = max(BBOX["lat_min"], min(BBOX["lat_max"], new_lat))
        new_lon = max(BBOX["lon_min"], min(BBOX["lon_max"], new_lon))
        _spawn_counter[0] += 1
        new_rows.append(
            {
                "debris_id": f"DS{_spawn_counter[0]:04d}",
                "lat": round(new_lat, 4),
                "lon": round(new_lon, 4),
                "debris_type": random.choice(DEBRIS_TYPES),
                "timestamp": pd.Timestamp.now(),
                "severity": random.choice(["Low", "Medium", "High"]),
            }
        )
    return pd.concat([sightings_df, pd.DataFrame(new_rows)], ignore_index=True)


# --------------------------------------------------------------------------- #
#  DBSCAN clustering                                                          #
# --------------------------------------------------------------------------- #
def _cluster_sightings(sightings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run DBSCAN on sightings lat/lon.
    Returns sightings_df with an added 'cluster' column (-1 = noise).
    """
    coords = sightings_df[["lat", "lon"]].values
    if len(coords) < DBSCAN_MIN_SAMPLES:
        sightings_df = sightings_df.copy()
        sightings_df["cluster"] = -1
        return sightings_df

    # Convert degrees to radians for haversine metric
    coords_rad = np.radians(coords)
    # eps in radians: 0.5° in radians = ~0.00873
    db = DBSCAN(
        eps=np.radians(DBSCAN_EPS),
        min_samples=DBSCAN_MIN_SAMPLES,
        algorithm="ball_tree",
        metric="haversine",
    )
    labels = db.fit_predict(coords_rad)
    sightings_df = sightings_df.copy()
    sightings_df["cluster"] = labels
    return sightings_df


def _build_hotspots(sightings_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Build hotspot summaries from clustered sightings."""
    hotspots = []
    clustered = sightings_df[sightings_df["cluster"] >= 0]
    for cluster_id, grp in clustered.groupby("cluster"):
        centre_lat = float(grp["lat"].mean())
        centre_lon = float(grp["lon"].mean())
        hotspots.append(
            {
                "hotspot_id": f"HS{int(cluster_id):03d}",
                "center": [round(centre_lat, 4), round(centre_lon, 4)],
                "sighting_count": len(grp),
                "debris_types": grp["debris_type"].value_counts().to_dict(),
                "severity": grp["severity"].mode()[0] if not grp["severity"].empty else "Low",
                "collector_assigned": None,
                "status": "UNASSIGNED",
            }
        )
    return hotspots


# --------------------------------------------------------------------------- #
#  Nearest-collector assignment                                               #
# --------------------------------------------------------------------------- #
def _assign_collectors(
    hotspots: list[dict],
    collectors: list[dict],
    priority_hotspot_ids: set[str] | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    Assign each hotspot to the nearest available collector.
    Hotspots listed in priority_hotspot_ids are assigned first (cross-agent
    bump from nearby flagged dark vessels).
    """
    if not hotspots or not collectors:
        return hotspots, collectors

    priority = set(priority_hotspot_ids or [])

    for c in collectors:
        c["assigned_hotspot"] = None
        c["status"] = "IDLE"

    hotspots_sorted = sorted(
        hotspots,
        key=lambda h: (0 if h["hotspot_id"] in priority else 1, -h["sighting_count"]),
    )

    available_collectors = list(collectors)  # shallow copy for tracking

    for hotspot in hotspots_sorted:
        if not available_collectors:
            hotspot["collector_assigned"] = None
            hotspot["status"] = "UNASSIGNED"
            continue
        hlat, hlon = hotspot["center"]
        best = min(
            available_collectors,
            key=lambda c: _haversine(c["lat"], c["lon"], hlat, hlon),
        )
        dist_km = _haversine(best["lat"], best["lon"], hlat, hlon)
        eta_hours = dist_km / (12.0 * 1.852)  # 12 knots in km/h
        eta_str = f"{int(eta_hours * 60)} min" if eta_hours < 1.0 else f"{eta_hours:.1f} hrs"

        hotspot["collector_assigned"] = best["collector_id"]
        hotspot["status"] = "ASSIGNED"
        hotspot["distance_km"] = round(dist_km, 1)
        hotspot["eta"] = eta_str

        best["assigned_hotspot"] = hotspot["hotspot_id"]
        best["status"] = "EN_ROUTE"
        best["eta"] = eta_str
        best["distance_km"] = round(dist_km, 1)

        # Move collector slightly toward hotspot (visual effect on map)
        best["lat"] = round(best["lat"] + (hlat - best["lat"]) * 0.12, 5)
        best["lon"] = round(best["lon"] + (hlon - best["lon"]) * 0.12, 5)
        available_collectors.remove(best)

    return hotspots, collectors


# --------------------------------------------------------------------------- #
#  Public API                                                                  #
# --------------------------------------------------------------------------- #
def get_debris_state(
    sightings_df: pd.DataFrame,
    collectors: list[dict[str, Any]],
    spawn_new: bool = True,
    n_new: int = 2,
    priority_hotspot_ids: set[str] | None = None,
) -> tuple[list[dict], list[dict], pd.DataFrame]:
    """
    Full pipeline for one tick.

    Parameters
    ----------
    sightings_df  : current active debris sightings DataFrame
    collectors    : list of collector vessel dicts (from init_collectors or previous tick)
    spawn_new     : whether to spawn new sightings this tick
    n_new         : how many new sightings to spawn

    Returns
    -------
    (hotspots, collectors, updated_sightings_df)
    """
    if spawn_new:
        sightings_df = spawn_new_sightings(sightings_df, n=n_new)

    sightings_df = _cluster_sightings(sightings_df)
    hotspots = _build_hotspots(sightings_df)
    hotspots, collectors = _assign_collectors(
        hotspots, collectors, priority_hotspot_ids=priority_hotspot_ids
    )

    n_hs = len(hotspots)
    n_assigned = sum(1 for h in hotspots if h["collector_assigned"])
    logger.info("Debris: %d sightings, %d hotspots, %d assigned", len(sightings_df), n_hs, n_assigned)

    return hotspots, collectors, sightings_df
