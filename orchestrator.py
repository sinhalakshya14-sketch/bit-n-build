"""
orchestrator.py
===============
Central orchestrator for the Maritime Multi-Agent System (MaritimeMAS).

Maintains a single shared state dict, coordinates all three agents,
stores simulation tick history for timeline replay, tracks vessel wake trails,
and implements key cross-agent interactions:
  1. Dark vessel rerouting: elevate edge costs near flagged dark vessels.
  2. Storm mode (Hurricane Ida): elevate edge costs inside hurricane gale radius.
"""

import logging
import math
import random
import copy
from datetime import datetime
from typing import Any

import pandas as pd

from data.loader import load_all, DATA_STATUS
from agents import route as route_agent
from agents import dark_vessel as dark_agent
from agents import debris as debris_agent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("orchestrator")

# --------------------------------------------------------------------------- #
#  Constants                                                                   #
# --------------------------------------------------------------------------- #
REROUTE_THRESHOLD_KM = 150.0  # flag vessel within this km of route → reroute
EXTRA_COST_FACTOR = 4.0       # edge cost multiplier near flagged vessels

# Canonical route: Galveston Entrance Channel → Louisiana Mississippi South Pass Approach (Open Marine Waters)
ROUTE_ORIGIN = (29.30, -94.75)
ROUTE_DESTINATION = (29.10, -89.50)

# --------------------------------------------------------------------------- #
#  Shared state                                                               #
# --------------------------------------------------------------------------- #
_state: dict[str, Any] = {
    "tick": 0,
    "vessel_positions": [],
    "vessel_wakes": {},         # vessel_id -> list of [lat, lon] historical trail
    "dark_vessel_results": [],
    "flagged_vessels": [],
    "current_route": {},
    "baseline_route": {},       # naive straight line route comparison
    "debris_hotspots": [],
    "collectors": [],
    "sightings_df": None,
    "storm_df": None,
    "fishing_zones": [],
    "storm_mode_active": False,
    "storm_eye_pos": None,
    "event_log": [],
    "data_status": {},
    "metrics": {
        "fuel_savings_pct": 0.0,
        "vessels_flagged": 0,
        "hotspots_covered": 0,
        "reroute_count": 0,
        "precision": 0.0,
        "recall": 0.0,
    },
}

# Tick history storage for the timeline scrubber replay
_tick_history: list[dict[str, Any]] = []

_ais_df: pd.DataFrame | None = None
_weather_df: pd.DataFrame | None = None
_storm_df: pd.DataFrame | None = None
_fishing_zones: list[dict] = []
_initialised = False

# Dedupe tracking so the event feed only reports *new* reroute triggers
# instead of re-logging the same trigger on every tick it remains active.
_last_triggering_vessel_ids: set = set()
_last_storm_grid_cell: tuple | None = None


# --------------------------------------------------------------------------- #
#  Helpers                                                                    #
# --------------------------------------------------------------------------- #
def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _log_event(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    _state["event_log"].insert(0, f"[{ts}] {msg}")
    _state["event_log"] = _state["event_log"][:50]
    logger.info("EVENT: %s", msg)


def _point_to_segment_dist(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> float:
    ab_len2 = (bx - ax) ** 2 + (by - ay) ** 2
    if ab_len2 == 0:
        return math.sqrt((px - ax) ** 2 + (py - ay) ** 2)
    t = max(0, min(1, ((px - ax) * (bx - ax) + (py - ay) * (by - ay)) / ab_len2))
    cx, cy = ax + t * (bx - ax), ay + t * (by - ay)
    return math.sqrt((px - cx) ** 2 + (py - cy) ** 2)


def _min_dist_to_route(vessel_lat: float, vessel_lon: float, waypoints: list) -> float:
    if not waypoints or len(waypoints) < 2:
        return float("inf")
    min_d = float("inf")
    for i in range(len(waypoints) - 1):
        seg_d_deg = _point_to_segment_dist(
            vessel_lat, vessel_lon,
            waypoints[i][0], waypoints[i][1],
            waypoints[i + 1][0], waypoints[i + 1][1],
        )
        min_d = min(min_d, seg_d_deg * 111.0)
    return min_d


def _snap_to_grid(lat: float, lon: float, step: float = 0.25) -> tuple[float, float]:
    return round(round(lat / step) * step, 4), round(round(lon / step) * step, 4)


def _snapshot_tick() -> None:
    """Save current state snapshot to tick history for playback timeline scrubber."""
    snap = {
        "tick": _state["tick"],
        "vessel_positions": copy.deepcopy(_state["vessel_positions"]),
        "current_route": copy.deepcopy(_state["current_route"]),
        "debris_hotspots": copy.deepcopy(_state["debris_hotspots"]),
        "collectors": copy.deepcopy(_state["collectors"]),
        "flagged_vessels": copy.deepcopy(_state["flagged_vessels"]),
        "metrics": copy.deepcopy(_state["metrics"]),
        "storm_eye_pos": _state["storm_eye_pos"],
    }
    _tick_history.append(snap)
    if len(_tick_history) > 100:
        _tick_history.pop(0)


# --------------------------------------------------------------------------- #
#  Initialise                                                                 #
# --------------------------------------------------------------------------- #
def initialise() -> None:
    global _ais_df, _weather_df, _storm_df, _fishing_zones, _initialised, _tick_history

    _log_event("System initialising — loading datasets...")
    ais_df, weather_df, debris_df, storm_df, fishing_zones = load_all()
    _ais_df = ais_df
    _weather_df = weather_df
    _storm_df = storm_df
    _fishing_zones = fishing_zones

    _state["data_status"] = dict(DATA_STATUS)
    _state["storm_df"] = storm_df
    _state["fishing_zones"] = fishing_zones
    _tick_history = []

    # --- Agent 3: Debris initialisation ---
    _state["collectors"] = debris_agent.init_collectors()
    hotspots, collectors, sightings = debris_agent.get_debris_state(
        debris_df, _state["collectors"], spawn_new=False
    )
    _state["debris_hotspots"] = hotspots
    _state["collectors"] = collectors
    _state["sightings_df"] = sightings
    _log_event(f"Debris Agent: {len(hotspots)} initial hotspots detected, {len(collectors)} collectors assigned.")

    # --- Agent 2: Dark vessel detection ---
    dv_results = dark_agent.detect_dark_vessels(ais_df)
    _state["dark_vessel_results"] = dv_results
    flagged = [r for r in dv_results if r["flagged"]]
    _state["flagged_vessels"] = flagged
    _state["metrics"]["vessels_flagged"] = len(flagged)

    from sklearn.metrics import precision_score, recall_score
    y_true = [int(r["is_ground_truth_dark"]) for r in dv_results]
    y_pred = [int(r["flagged"]) for r in dv_results]
    _state["metrics"]["precision"] = round(precision_score(y_true, y_pred, zero_division=0), 2)
    _state["metrics"]["recall"] = round(recall_score(y_true, y_pred, zero_division=0), 2)
    _log_event(
        f"Dark Vessel Agent: {len(flagged)} vessels flagged "
        f"(Precision={_state['metrics']['precision']:.2f}, Recall={_state['metrics']['recall']:.2f})."
    )

    # --- Agent 1: Initial route & Naive baseline comparison ---
    route = route_agent.get_route(ROUTE_ORIGIN, ROUTE_DESTINATION, weather_df)
    _state["current_route"] = route
    _state["baseline_route"] = {
        "waypoints": [list(ROUTE_ORIGIN), list(ROUTE_DESTINATION)],
        "cost": route.get("baseline_cost", 0),
    }
    _state["metrics"]["fuel_savings_pct"] = route["savings_pct"]
    _log_event(
        f"Route Agent: Optimized path computed ({len(route['waypoints'])} waypoints, "
        f"{route['savings_pct']:.1f}% fuel savings vs. straight-line)."
    )

    # --- Cross-agent: check reroute for dark vessels ---
    _check_and_reroute_for_dark_vessels()

    # --- Initial vessel positions & wake history ---
    _update_vessel_positions(ais_df)

    _state["metrics"]["hotspots_covered"] = sum(
        1 for h in _state["debris_hotspots"] if h["collector_assigned"]
    )

    _initialised = True
    _snapshot_tick()
    _log_event("System ready — live multi-agent simulation running.")


# --------------------------------------------------------------------------- #
#  Cross-Agent Logic: Rerouting for Dark Vessels & Hurricanes               #
# --------------------------------------------------------------------------- #
def _check_and_reroute_for_dark_vessels() -> None:
    global _last_triggering_vessel_ids, _last_storm_grid_cell

    waypoints = _state["current_route"].get("waypoints", [])
    flagged = _state["flagged_vessels"]

    if not waypoints:
        return

    extra_cost_nodes = []
    triggering_vessels = []

    # 1. Dark vessel cost penalties
    if flagged:
        for vessel in flagged:
            pos = vessel.get("last_known_position", [0, 0])
            if not pos or pos == [0, 0]:
                continue
            vlat, vlon = pos
            dist_km = _min_dist_to_route(vlat, vlon, waypoints)
            if dist_km < REROUTE_THRESHOLD_KM:
                snapped = _snap_to_grid(vlat, vlon)
                extra_cost_nodes.append(snapped)
                triggering_vessels.append((vessel["vessel_id"], vlat, vlon, dist_km))

    # 2. Hurricane Ida Storm Mode cost penalties
    storm_grid_cell = None
    if _state["storm_mode_active"] and _storm_df is not None:
        idx = min(_state["tick"], len(_storm_df) - 1)
        storm_row = _storm_df.iloc[idx]
        storm_lat, storm_lon = float(storm_row["lat"]), float(storm_row["lon"])
        _state["storm_eye_pos"] = [storm_lat, storm_lon]
        storm_grid_cell = _snap_to_grid(storm_lat, storm_lon, step=0.5)

        # Add 3x3 grid obstacle around hurricane eye
        for dlat in [-0.5, -0.25, 0.0, 0.25, 0.5]:
            for dlon in [-0.5, -0.25, 0.0, 0.25, 0.5]:
                extra_cost_nodes.append(_snap_to_grid(storm_lat + dlat, storm_lon + dlon))

    if not extra_cost_nodes:
        # Nothing forcing a detour right now — reset dedupe trackers so a
        # future trigger is treated as new again.
        _last_triggering_vessel_ids = set()
        _last_storm_grid_cell = None
        return

    new_route = route_agent.get_route(
        ROUTE_ORIGIN,
        ROUTE_DESTINATION,
        _weather_df,
        extra_cost_nodes=extra_cost_nodes,
        extra_cost_factor=EXTRA_COST_FACTOR,
    )
    _state["current_route"] = new_route
    _state["metrics"]["fuel_savings_pct"] = new_route["savings_pct"]

    # Only treat vessels/storm positions that are *new* since the last check
    # as an actual reroute event — otherwise the feed would repeat the same
    # line every tick the vessel simply stays near the route.
    current_ids = {vid for vid, *_ in triggering_vessels}
    new_ids = current_ids - _last_triggering_vessel_ids
    new_vessels = [t for t in triggering_vessels if t[0] in new_ids]

    storm_is_new = (
        storm_grid_cell is not None and storm_grid_cell != _last_storm_grid_cell
    )

    if new_vessels or storm_is_new:
        _state["metrics"]["reroute_count"] += 1

        for vid, vlat, vlon, dist_km in new_vessels:
            loc_str = f"({vlat:.2f}°N, {abs(vlon):.2f}°W)"
            _log_event(
                f"⚠️  Rerouted around flagged dark vessel {vid} near {loc_str} "
                f"({dist_km:.0f} km from route). New savings: {new_route['savings_pct']:.1f}%."
            )

        if storm_is_new:
            _log_event(
                f"🌀 Hurricane Ida Reroute: Avoided Cat 4 storm eye at "
                f"({_state['storm_eye_pos'][0]:.1f}°N, {abs(_state['storm_eye_pos'][1]):.1f}°W). "
                f"Dynamic path cost: {new_route['cost']:.1f}."
            )

    _last_triggering_vessel_ids = current_ids
    _last_storm_grid_cell = storm_grid_cell


# --------------------------------------------------------------------------- #
#  Vessel Position & Wake Trail Tracking                                      #
# --------------------------------------------------------------------------- #
def _update_vessel_positions(ais_df: pd.DataFrame) -> None:
    # Select ping slice based on tick index
    tick_step = _state["tick"] % max(1, (len(ais_df) // 50))
    latest = (
        ais_df.sort_values("timestamp")
        .groupby("vessel_id")
        .nth(tick_step)
        .reset_index()
        [["vessel_id", "lat", "lon", "speed", "heading", "vessel_type"]]
    )

    dv_map = {r["vessel_id"]: r for r in _state["dark_vessel_results"]}
    positions = []

    for _, row in latest.iterrows():
        vid = row["vessel_id"]
        vlat, vlon = float(row["lat"]), float(row["lon"])
        dv = dv_map.get(vid, {})

        # Append to wake trail history (max 6 points)
        wake = _state["vessel_wakes"].get(vid, [])
        wake.append([vlat, vlon])
        if len(wake) > 6:
            wake.pop(0)
        _state["vessel_wakes"][vid] = wake

        positions.append(
            {
                "vessel_id": vid,
                "lat": vlat,
                "lon": vlon,
                "speed": float(row["speed"]),
                "heading": float(row["heading"]),
                "vessel_type": row.get("vessel_type", "CARGO"),
                "flagged": dv.get("flagged", False),
                "confidence": dv.get("confidence", 0.0),
                "explanation": dv.get("explanation", ""),
                "max_gap_minutes": dv.get("max_gap_minutes", 0),
                "displacement_error_km": dv.get("displacement_error_km", 0),
                "speed_change_after_gap": dv.get("speed_change_after_gap", 0),
                "heading_change_after_gap": dv.get("heading_change_after_gap", 0),
            }
        )

    _state["vessel_positions"] = positions


# --------------------------------------------------------------------------- #
#  Simulation Tick Engine                                                     #
# --------------------------------------------------------------------------- #
def tick() -> dict[str, Any]:
    global _ais_df

    if not _initialised:
        initialise()
        return _state

    _state["tick"] += 1
    t = _state["tick"]

    if _ais_df is not None:
        _update_vessel_positions(_ais_df)

    # Debris tick
    hotspots, collectors, sightings = debris_agent.get_debris_state(
        _state["sightings_df"],
        _state["collectors"],
        spawn_new=(t % 2 == 0),
        n_new=random.randint(1, 2),
    )
    _state["debris_hotspots"] = hotspots
    _state["collectors"] = collectors
    _state["sightings_df"] = sightings
    _state["metrics"]["hotspots_covered"] = sum(
        1 for h in hotspots if h["collector_assigned"]
    )

    # Dark vessel scan refresh (every 3 ticks)
    if t % 3 == 0 and _ais_df is not None:
        dv_results = dark_agent.detect_dark_vessels(_ais_df)
        _state["dark_vessel_results"] = dv_results
        flagged = [r for r in dv_results if r["flagged"]]
        _state["flagged_vessels"] = flagged
        _state["metrics"]["vessels_flagged"] = len(flagged)

        from sklearn.metrics import precision_score, recall_score
        y_true = [int(r["is_ground_truth_dark"]) for r in dv_results]
        y_pred = [int(r["flagged"]) for r in dv_results]
        _state["metrics"]["precision"] = round(precision_score(y_true, y_pred, zero_division=0), 2)
        _state["metrics"]["recall"] = round(recall_score(y_true, y_pred, zero_division=0), 2)

    # Check reroutes
    _check_and_reroute_for_dark_vessels()

    # Route refresh (every 5 ticks)
    if t % 5 == 0 and _weather_df is not None:
        route = route_agent.get_route(ROUTE_ORIGIN, ROUTE_DESTINATION, _weather_df)
        _state["current_route"] = route
        _state["metrics"]["fuel_savings_pct"] = route["savings_pct"]

    _snapshot_tick()
    return _state


def toggle_storm_mode(active: bool | None = None) -> bool:
    """Toggle Hurricane Ida storm avoidance mode."""
    if active is None:
        _state["storm_mode_active"] = not _state["storm_mode_active"]
    else:
        _state["storm_mode_active"] = active

    if _state["storm_mode_active"]:
        _log_event("🌀 Hurricane Ida Mode ENABLED — dynamic storm track avoidance active.")
    else:
        _state["storm_eye_pos"] = None
        _log_event("🌀 Hurricane Ida Mode DISABLED — normal navigation restored.")

    _check_and_reroute_for_dark_vessels()
    return _state["storm_mode_active"]


# --------------------------------------------------------------------------- #
#  Public API                                                                 #
# --------------------------------------------------------------------------- #
def get_state() -> dict[str, Any]:
    return _state

def get_tick_history() -> list[dict[str, Any]]:
    return _tick_history
