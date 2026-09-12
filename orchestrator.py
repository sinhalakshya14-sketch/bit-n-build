"""
orchestrator.py
===============
Central simulation engine for MaritimeMAS (Gulf of Mexico).

Responsibilities:
  - Load all datasets once via data.loader.load_all()
  - Coordinate the three agents:
      Agent 1 (agents/route.py)       — A* weather-optimised routing
      Agent 2 (agents/dark_vessel.py) — IsolationForest dark-vessel detection
      Agent 3 (agents/debris.py)      — DBSCAN debris clustering & collectors
  - Advance the simulation clock tick-by-tick (tick())
  - Cross-agent rerouting: Agent 1's route is recomputed with cost penalties
    around flagged dark vessels and the Hurricane Ida storm eye whenever the
    storm-replay demo mode is active.
  - Maintain wake trails, the live event log, KPI metrics and a snapshot
    history that powers the dashboard timeline scrubber.

Public API:
    initialise()
    get_state() -> dict
    tick()
    toggle_storm_mode(force: bool | None = None)
    get_tick_history() -> list[dict]
    ROUTE_ORIGIN, ROUTE_DESTINATION

Run (via the dashboard):
    python -m streamlit run dashboard.py
"""

import time
import logging
from typing import Any

from data.loader import load_all, DATA_STATUS
from agents.route import get_route, _snap_to_grid
from agents.dark_vessel import detect_dark_vessels
from agents import debris as debris_agent

logger = logging.getLogger(__name__)

# Route under management: Galveston Entrance → Mississippi River South Pass
ROUTE_ORIGIN = (29.30, -94.80)
ROUTE_DESTINATION = (29.90, -90.10)

HISTORY_LIMIT = 120      # snapshots kept for the timeline scrubber
EVENT_LOG_LIMIT = 200    # event feed entries kept
WAKE_LENGTH = 15         # wake-trail points per vessel
FLAGGED_RADIUS_KM = 65.0 # dark-vessel avoidance radius around route waypoints

_state: dict[str, Any] = {}


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _log(msg: str) -> None:
    """Append a timestamped line to the live event feed."""
    _state["event_log"].append(f"[{time.strftime('%H:%M:%S')}] {msg}")


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    import math
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _build_tracks(ais_df) -> dict[str, list[tuple]]:
    """Per-vessel cyclic playback track: (lat, lon, speed, heading, type)."""
    tracks: dict[str, list[tuple]] = {}
    for vid, grp in ais_df.sort_values("timestamp").groupby("vessel_id"):
        tracks[vid] = [
            (round(float(r.lat), 5), round(float(r.lon), 5), round(float(r.speed), 1),
             round(float(r.heading), 1), str(r.vessel_type))
            for r in grp.itertuples()
        ]
    return tracks


def _vessel_state_at_tick(t: int) -> list[dict[str, Any]]:
    """Current position/status of every vessel, merged with detection results."""
    out = []
    for vid, det in _state["detection_by_id"].items():
        track = _state["tracks"][vid]
        lat, lon, speed, heading, vtype = track[t % len(track)]
        out.append(
            {
                "vessel_id": vid,
                "vessel_type": vtype,
                "lat": lat,
                "lon": lon,
                "speed": speed,
                "heading": heading,
                "flagged": det["flagged"],
                "confidence": det["confidence"],
                "max_gap_minutes": det["max_gap_minutes"],
                "displacement_error_km": det["displacement_error_km"],
                "explanation": det["explanation"],
            }
        )
    return out


def _run_dark_vessel_scan() -> None:
    """Run Agent 2 and refresh flagged-vessel metrics."""
    results = detect_dark_vessels(_state["ais_df"])
    _state["detections"] = results
    _state["detection_by_id"] = {r["vessel_id"]: r for r in results}
    flagged = [r for r in results if r["flagged"]]
    _state["flagged_vessels"] = flagged

    tp = sum(1 for r in flagged if r["is_ground_truth_dark"])
    fp = len(flagged) - tp
    fn = sum(1 for r in results if not r["flagged"] and r["is_ground_truth_dark"])
    m = _state["metrics"]
    m["vessels_flagged"] = len(flagged)
    m["precision"] = tp / (tp + fp) if (tp + fp) else 0.0
    m["recall"] = tp / (tp + fn) if (tp + fn) else 0.0
    _log(f"🚨 Dark vessel scan: {len(flagged)} flagged (P {m['precision']:.0%} / R {m['recall']:.0%})")


def _flagged_nodes_near_route(waypoints: list) -> list[tuple[float, float]]:
    """Grid nodes near flagged dark vessels that lie close to the route."""
    nodes: set[tuple[float, float]] = set()
    for det in _state["detections"]:
        if not det["flagged"]:
            continue
        lat, lon = det["last_known_position"]
        node = _snap_to_grid(lat, lon)
        for wp in waypoints:
            if _haversine(node[0], node[1], wp[0], wp[1]) <= FLAGGED_RADIUS_KM:
                nodes.add(node)
                break
    return sorted(nodes)


def _storm_cost_nodes() -> list[tuple[float, float]]:
    """Grid nodes around the active storm eye (≈1° penalty footprint)."""
    eye = _state.get("storm_eye_pos")
    if not eye:
        return []
    return [
        _snap_to_grid(eye[0] + i * 0.25, eye[1] + j * 0.25)
        for i in range(-4, 5)
        for j in range(-4, 5)
    ]


def _update_route(reason: str) -> None:
    """Cross-agent reroute: recompute Agent 1's path with hazard cost nodes."""
    extra = _flagged_nodes_near_route(_state["current_route"]["waypoints"]) + _storm_cost_nodes()
    if not extra:
        return
    new_route = get_route(
        ROUTE_ORIGIN, ROUTE_DESTINATION, _state["weather_df"], extra_cost_nodes=extra
    )
    if new_route["waypoints"] != _state["current_route"]["waypoints"]:
        _state["current_route"] = new_route
        if new_route.get("naive_waypoints"):
            _state["baseline_route"] = {"waypoints": new_route["naive_waypoints"]}
        _state["metrics"]["fuel_savings_pct"] = new_route["savings_pct"]
        _state["metrics"]["reroute_count"] += 1
        _log(
            f"⚠️ Rerouted ({reason}) — {len(extra)} hazard nodes penalised, "
            f"new fuel savings {new_route['savings_pct']}%"
        )


def _snapshot() -> dict[str, Any]:
    """Point-in-time state snapshot for the timeline scrubber."""
    return {
        "tick": _state["tick"],
        "vessel_positions": [dict(v) for v in _state["vessel_positions"]],
        "debris_hotspots": [dict(h) for h in _state["debris_hotspots"]],
        "current_route": _state["current_route"],
        "storm_mode_active": _state["storm_mode_active"],
        "storm_eye_pos": _state["storm_eye_pos"],
    }


# --------------------------------------------------------------------------- #
#  Public API                                                                  #
# --------------------------------------------------------------------------- #
def _gulf_storm_rows(storm_df):
    """Storm track rows inside the Gulf bounding box, for eye replay."""
    return storm_df[(storm_df["lat"] >= 25.0) & (storm_df["lat"] <= 31.0)].reset_index(drop=True)


def initialise() -> None:
    """Load data, run all three agents once, seed state and history."""
    ais_df, weather_df, debris_df, storm_df, fishing_zones = load_all()

    _state.clear()
    _state.update(
        tick=0,
        event_log=[],
        history=[],
        vessel_wakes={},
        vessel_positions=[],
        ais_df=ais_df,
        weather_df=weather_df,
        storm_df=storm_df,
        fishing_zones=fishing_zones,
        data_status=dict(DATA_STATUS),
        storm_mode_active=False,
        storm_eye_pos=None,
        sightings_df=debris_df,
        gulf_storm_track=_gulf_storm_rows(storm_df),
        collectors=debris_agent.init_collectors(),
        debris_hotspots=[],
        current_route=None,
        metrics={
            "fuel_savings_pct": 0.0,
            "vessels_flagged": 0,
            "precision": 0.0,
            "recall": 0.0,
            "hotspots_covered": 0,
            "reroute_count": 0,
        },
    )
    _log("🌊 MaritimeMAS engine initialised — Open-Meteo & NOAA data loaded")
    _log(
        f"📡 Datasets: {len(ais_df)} AIS pings · {len(weather_df)} weather points · "
        f"{len(storm_df)} NOAA storm points · {len(fishing_zones)} GFW zones"
    )

    # Agent 2 — dark vessel detection
    _run_dark_vessel_scan()

    # Vessel playback tracks + seed wakes
    _state["tracks"] = _build_tracks(ais_df)
    _state["vessel_positions"] = _vessel_state_at_tick(0)
    for v in _state["vessel_positions"]:
        track = _state["tracks"][v["vessel_id"]]
        idx = 0 % len(track)
        _state["vessel_wakes"][v["vessel_id"]] = [
            [p[0], p[1]] for p in track[max(0, idx - 3) : idx + 1]
        ]

    # Agent 3 — debris clustering & collector assignment (no spawning on init)
    hotspots, collectors, sightings = debris_agent.get_debris_state(
        _state["sightings_df"], _state["collectors"], spawn_new=False
    )
    _state["debris_hotspots"] = hotspots
    _state["collectors"] = collectors
    _state["sightings_df"] = sightings
    _state["metrics"]["hotspots_covered"] = sum(1 for h in hotspots if h["collector_assigned"])
    _log(f"🗑️ {len(hotspots)} debris hotspots clustered, collectors assigned")

    # Agent 1 — initial optimised route
    _state["current_route"] = get_route(ROUTE_ORIGIN, ROUTE_DESTINATION, weather_df)
    naive_wps = _state["current_route"].get("naive_waypoints")
    _state["baseline_route"] = {
        "waypoints": naive_wps if naive_wps else [list(ROUTE_ORIGIN), list(ROUTE_DESTINATION)]
    }
    _state["metrics"]["fuel_savings_pct"] = _state["current_route"]["savings_pct"]
    _log(
        f"🧭 Route optimised: {len(_state['current_route']['waypoints'])} waypoints, "
        f"fuel savings {_state['current_route']['savings_pct']}%"
    )

    # Cross-agent reroute around flagged vessels near the corridor
    _update_route("initial dark-vessel avoidance")

    _state["history"].append(_snapshot())
    logger.info("Orchestrator initialised at tick 0")


def tick() -> None:
    """Advance the simulation by one tick."""
    if not _state:
        initialise()

    _state["tick"] += 1
    t = _state["tick"]

    # Storm replay progression (NOAA HURDAT2 points inside the Gulf box)
    if _state["storm_mode_active"]:
        track = _state["gulf_storm_track"]
        row = track.iloc[t % len(track)]
        _state["storm_eye_pos"] = [float(row["lat"]), float(row["lon"])]
        if t % 3 == 0:
            _log(
                f"🌀 Hurricane Ida eye at {row['lat']:.1f}°N {abs(row['lon']):.1f}°W — "
                f"{row['max_wind_kts']} kts ({row['cat']})"
            )
        _update_route("storm eye moved")

    # Vessels advance along their AIS tracks; wake trails updated
    vpos = _vessel_state_at_tick(t)
    _state["vessel_positions"] = vpos
    for v in vpos:
        wake = _state["vessel_wakes"].setdefault(v["vessel_id"], [])
        wake.append([v["lat"], v["lon"]])
        del wake[:-WAKE_LENGTH]

    # Agent 3 — debris drift, clustering & collector reassignment
    hotspots, collectors, sightings = debris_agent.get_debris_state(
        _state["sightings_df"], _state["collectors"], spawn_new=True, n_new=2
    )
    _state["debris_hotspots"] = hotspots
    _state["collectors"] = collectors
    _state["sightings_df"] = sightings
    _state["metrics"]["hotspots_covered"] = sum(1 for h in hotspots if h["collector_assigned"])
    if t % 5 == 0:
        assigned = sum(1 for h in hotspots if h["collector_assigned"])
        _log(f"🗑️ {len(hotspots)} hotspots tracked · {assigned} collectors assigned")

    # Periodic Agent 2 rescan + reroute check
    if t % 10 == 0:
        _run_dark_vessel_scan()
        _update_route("dark vessel rescan")

    _state["event_log"] = _state["event_log"][-EVENT_LOG_LIMIT:]
    _state["history"].append(_snapshot())
    del _state["history"][:-HISTORY_LIMIT]


def toggle_storm_mode(force: bool | None = None) -> None:
    """
    Toggle the Hurricane Ida (NOAA HURDAT2) storm-replay demo mode.

    force=True/False sets the mode explicitly; with no argument the mode
    simply flips. Activating triggers an immediate storm-avoidance reroute.
    """
    if not _state:
        initialise()

    new_active = (not _state["storm_mode_active"]) if force is None else bool(force)
    _state["storm_mode_active"] = new_active

    if new_active:
        track = _state["gulf_storm_track"]
        first = track.iloc[0]
        _state["storm_eye_pos"] = [float(first["lat"]), float(first["lon"])]
        _log("🌀 Hurricane Ida replay STARTED — Agent 1 avoiding Cat 4 eye")
        _update_route("storm avoidance")
    else:
        _state["storm_eye_pos"] = None
        _log("🌩️ Hurricane Ida replay stopped — storm penalties cleared")
        _update_route("storm cleared")


def get_state() -> dict[str, Any]:
    """Full dashboard-facing state snapshot."""
    if not _state:
        initialise()
    return {
        "tick": _state["tick"],
        "metrics": dict(_state["metrics"]),
        "event_log": list(_state["event_log"]),
        "vessel_positions": _state["vessel_positions"],
        "vessel_wakes": _state["vessel_wakes"],
        "fishing_zones": _state["fishing_zones"],
        "debris_hotspots": _state["debris_hotspots"],
        "collectors": _state["collectors"],
        "current_route": _state["current_route"],
        "baseline_route": _state["baseline_route"],
        "flagged_vessels": _state["flagged_vessels"],
        "storm_mode_active": _state["storm_mode_active"],
        "storm_eye_pos": _state["storm_eye_pos"],
        "storm_df": _state["storm_df"],
        "weather_df": _state["weather_df"],
        "data_status": _state["data_status"],
    }


def get_tick_history() -> list[dict[str, Any]]:
    """List of state snapshots, oldest first (for the timeline scrubber)."""
    if not _state:
        initialise()
    return _state["history"]
