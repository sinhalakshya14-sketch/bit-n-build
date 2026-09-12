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

from data.loader import load_all, DATA_STATUS, load_debris_zones
from agents.route import get_route, get_candidate_routes, _snap_to_grid, _haversine
from agents.dark_vessel import detect_dark_vessels
from agents import debris as debris_agent
from agents.investigator import investigate_vessel
from storage import log_zone_flag, get_zone_rolling_stats, log_route_decision, get_recent_override_count

logger = logging.getLogger(__name__)

# Route under management: Galveston Entrance → Mississippi River South Pass
ROUTE_ORIGIN = (29.30, -94.75)
ROUTE_DESTINATION = (29.10, -89.50)

DEFAULT_RISK_AVOIDANCE_THRESHOLD_PCT = 5.0

# Hardcoded catalog of real ports covering active vessel simulation lanes
PORT_CATALOG = [
    {"name": "Houston / Galveston (US)", "lat": 29.30, "lon": -94.75, "region": "Gulf of Mexico"},
    {"name": "New Orleans / South Pass (US)", "lat": 29.10, "lon": -89.50, "region": "Gulf of Mexico"},
    {"name": "Corpus Christi (US)", "lat": 27.80, "lon": -97.20, "region": "Gulf of Mexico"},
    {"name": "Mobile (US)", "lat": 30.65, "lon": -88.05, "region": "Gulf of Mexico"},
    {"name": "Tampa (US)", "lat": 27.95, "lon": -82.55, "region": "Gulf of Mexico"},
    {"name": "Panama Canal (PA)", "lat": 8.90, "lon": -79.50, "region": "Central America"},
    {"name": "Gibraltar (GI)", "lat": 36.14, "lon": -5.35, "region": "Mediterranean"},
    {"name": "Port Said / Suez (EG)", "lat": 31.26, "lon": 32.30, "region": "Mediterranean / Red Sea"},
    {"name": "Singapore (SG)", "lat": 1.29, "lon": 103.85, "region": "Southeast Asia"},
    {"name": "Rotterdam (NL)", "lat": 51.92, "lon": 4.48, "region": "North Sea"},
    {"name": "Cape Town (ZA)", "lat": -33.92, "lon": 18.42, "region": "South Africa"},
    {"name": "Honolulu (US)", "lat": 21.30, "lon": -157.85, "region": "Pacific (Remote / No Fleet)"},
]


def find_nearest_active_vessel(lat: float, lon: float, max_dist_km: float = 300.0) -> dict | None:
    """
    Find the closest active (non-flagged) vessel within max_dist_km of (lat, lon).
    Returns vessel dict or None if no active vessel is within threshold.
    """
    vessels = _state.get("vessel_positions", [])
    best = None
    min_dist = float("inf")
    for v in vessels:
        if v.get("flagged", False):
            continue
        vlat, vlon = v.get("lat"), v.get("lon")
        if vlat is None or vlon is None:
            continue
        d = _haversine(lat, lon, vlat, vlon)
        if d < min_dist:
            min_dist = d
            best = {
                "vessel_id": v.get("vessel_id", "Unknown"),
                "vessel_type": v.get("vessel_type", "CARGO"),
                "speed": v.get("speed", 0.0),
                "heading": v.get("heading", 0.0),
                "lat": vlat,
                "lon": vlon,
                "dist_km": round(min_dist, 1),
            }
    if best and best["dist_km"] <= max_dist_km:
        return best
    return None


def set_active_route_ports(origin_name: str, dest_name: str) -> None:
    """Set active commercial corridor ports and re-run route tradeoff."""
    port_map = {p["name"]: p for p in PORT_CATALOG}
    orig_p = port_map.get(origin_name, PORT_CATALOG[0])
    dest_p = port_map.get(dest_name, PORT_CATALOG[1])
    _state["active_origin_name"] = orig_p["name"]
    _state["active_dest_name"] = dest_p["name"]
    _state["active_route_origin"] = (orig_p["lat"], orig_p["lon"])
    _state["active_route_destination"] = (dest_p["lat"], dest_p["lon"])
    _run_route_tradeoff(f"port change: {orig_p['name']} -> {dest_p['name']}")


GULF_ZONES = [
    {"id": "Zone 1", "name": "Texas Inshore / Galveston Shelf", "lat_min": 28.5, "lat_max": 30.5, "lon_min": -96.5, "lon_max": -94.0},
    {"id": "Zone 2", "name": "Louisiana Coastal Shelf", "lat_min": 28.5, "lat_max": 30.5, "lon_min": -94.0, "lon_max": -91.5},
    {"id": "Zone 3", "name": "Mississippi Delta / South Pass", "lat_min": 28.5, "lat_max": 30.5, "lon_min": -91.5, "lon_max": -89.0},
    {"id": "Zone 4", "name": "Mobile Bay & Florida Panhandle", "lat_min": 28.5, "lat_max": 30.5, "lon_min": -89.0, "lon_max": -86.5},
    {"id": "Zone 5", "name": "South Texas Deepwater", "lat_min": 26.5, "lat_max": 28.5, "lon_min": -96.5, "lon_max": -94.0},
    {"id": "Zone 6", "name": "Central Gulf Deepwater", "lat_min": 26.5, "lat_max": 28.5, "lon_min": -94.0, "lon_max": -91.5},
    {"id": "Zone 7", "name": "Mississippi Canyon", "lat_min": 26.5, "lat_max": 28.5, "lon_min": -91.5, "lon_max": -89.0},
    {"id": "Zone 8", "name": "DeSoto Canyon / East Gulf", "lat_min": 26.5, "lat_max": 28.5, "lon_min": -89.0, "lon_max": -86.5},
]

def get_zone_for_point(lat: float, lon: float) -> dict[str, Any]:
    for z in GULF_ZONES:
        if z["lat_min"] <= lat <= z["lat_max"] and z["lon_min"] <= lon <= z["lon_max"]:
            return z
    best = GULF_ZONES[0]
    best_dist = float("inf")
    for z in GULF_ZONES:
        clat = (z["lat_min"] + z["lat_max"]) / 2.0
        clon = (z["lon_min"] + z["lon_max"]) / 2.0
        d = (lat - clat) ** 2 + (lon - clon) ** 2
        if d < best_dist:
            best_dist = d
            best = z
    return best

HISTORY_LIMIT = 120      # snapshots kept for the timeline scrubber
EVENT_LOG_LIMIT = 200    # event feed entries kept
WAKE_LENGTH = 15         # wake-trail points per vessel
FLAGGED_RADIUS_KM = 65.0 # dark-vessel avoidance radius around route waypoints
DARK_HOTSPOT_RADIUS_KM = 150.0  # Dark Vessel Agent → Debris Agent coupling

_state: dict[str, Any] = {}


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _log(msg: str) -> None:
    """Append a timestamped line to the live event feed."""
    _state["event_log"].append(f"[{time.strftime('%H:%M:%S')}] {msg}")


def append_event(msg: str) -> None:
    """Public event-feed append (watchlist re-alerts, etc.)."""
    if not _state:
        initialise()
    _log(msg)


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
                "scan_confidence": det.get("scan_confidence", det["confidence"]),
                "flag_count": det.get("flag_count", 1 if det["flagged"] else 0),
                "flag_ticks_window": det.get("flag_ticks_window", []),
                "memory_note": det.get("memory_note", ""),
                "max_gap_minutes": det["max_gap_minutes"],
                "displacement_error_km": det["displacement_error_km"],
                "explanation": det["explanation"],
            }
        )
    return out


def _run_dark_vessel_scan() -> None:
    """Run Agent 2 and refresh flagged-vessel metrics."""
    th = _state.get("detection_thresholds") or {}
    results = detect_dark_vessels(
        _state["ais_df"],
        gap_threshold_min=th.get("gap_threshold_min", 45.0),
        dr_threshold_km=th.get("dr_threshold_km", 8.0),
    )
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
    _log(f"[Surveillance Agent] -> [Orchestrator]: Flagged {len(flagged)} dark vessels in latest scan (P {m['precision']:.0%} / R {m['recall']:.0%})")
    _update_vessel_memory(_state.get("tick", 0))
    _maybe_investigate_flagged()


MEMORY_WINDOW = 6  # ticks counted in the explainability "last N ticks" line


def _update_vessel_memory(tick: int) -> None:
    """
    Persistent per-vessel suspicion.

    vessel_memory[vessel_id] = {
        "flag_ticks": list[int],            # ticks this vessel was flagged
        "flag_count": int,                  # lifetime scan/tick flags
        "cumulative_suspicion": float,      # 0–1, grows with repeats
        "last_scan_confidence": float,      # this scan's IF/rule score
        "last_flagged_tick": int | None,
    }
    Displayed confidence is cumulative_suspicion, not the raw scan score.
    """
    mem: dict[str, dict[str, Any]] = _state.setdefault("vessel_memory", {})
    detections = _state.get("detections") or []
    flagged_ids = {r["vessel_id"] for r in detections if r.get("flagged")}
    for det in detections:
        vid = det["vessel_id"]
        rec = mem.setdefault(
            vid,
            {
                "flag_ticks": [],
                "flag_count": 0,
                "cumulative_suspicion": 0.0,
                "last_scan_confidence": 0.0,
                "last_flagged_tick": None,
            },
        )
        rec["last_scan_confidence"] = float(det.get("confidence") or 0.0)
        if vid in flagged_ids:
            if rec["last_flagged_tick"] != tick:
                rec["flag_ticks"].append(int(tick))
                rec["last_flagged_tick"] = int(tick)
            rec["flag_ticks"] = [t for t in rec["flag_ticks"] if t >= tick - 40]
            rec["flag_count"] = len(rec["flag_ticks"])
            n_window = sum(1 for t in rec["flag_ticks"] if t >= tick - MEMORY_WINDOW + 1)
            extra = max(0, n_window - 1)
            rec["cumulative_suspicion"] = round(
                min(0.99, rec["last_scan_confidence"] + 0.08 * extra + 0.02 * max(0, rec["flag_count"] - 1)),
                3,
            )
            rec["memory_note"] = (
                f"Flagged {n_window} time(s) in the last {MEMORY_WINDOW} ticks "
                f"({rec['flag_count']} lifetime flag events). "
                f"Scan score {rec['last_scan_confidence']:.0%}; "
                f"repeat-flag lift applied."
            )
        else:
            rec["cumulative_suspicion"] = round(rec["last_scan_confidence"] * 0.9, 3)
            rec["memory_note"] = "No active flag this tick; suspicion decaying toward the last scan score."
        window = [t for t in rec["flag_ticks"] if t >= tick - MEMORY_WINDOW + 1]
        det["scan_confidence"] = rec["last_scan_confidence"]
        det["confidence"] = rec["cumulative_suspicion"]
        det["flag_count"] = rec["flag_count"]
        det["flag_ticks_window"] = window
        det["memory_note"] = rec["memory_note"]
    _state["flagged_vessels"] = [r for r in detections if r.get("flagged")]
    if _state.get("detection_by_id"):
        for r in detections:
            _state["detection_by_id"][r["vessel_id"]] = r


def _maybe_investigate_flagged() -> None:
    """Cache at most 3 investigative briefs per scan (feature-flagged)."""
    if not _state.get("llm_enabled"):
        return
    briefs = _state.setdefault("llm_briefs", {})
    tick = int(_state.get("tick") or 0)
    ranked = sorted(
        _state.get("flagged_vessels") or [],
        key=lambda r: r.get("confidence", 0),
        reverse=True,
    )[:3]
    for det in ranked:
        key = f"{det['vessel_id']}@t{tick}"
        if key in briefs:
            det["investigation"] = briefs[key]
            continue
        briefs[key] = investigate_vessel(det, use_llm=True)
        det["investigation"] = briefs[key]
        _log(f"[Investigator Agent] -> [Orchestrator]: Completed background brief for V{det['vessel_id']} ({briefs[key].get('model')})")
    if len(briefs) > 40:
        for old in list(briefs.keys())[:-30]:
            briefs.pop(old, None)


def set_llm_enabled(enabled: bool) -> None:
    if not _state:
        initialise()
    _state["llm_enabled"] = bool(enabled)


def refresh_investigations() -> None:
    if not _state:
        initialise()
    _state["llm_enabled"] = True
    _maybe_investigate_flagged()


def get_vessel_origin_dest(vid: str) -> dict[str, Any]:
    if not _state or "tracks" not in _state:
        return {}
    
    # vid might be an int or string depending on dataframe parsing, so string cast for robustness
    track = _state["tracks"].get(str(vid)) or _state["tracks"].get(int(vid) if str(vid).isdigit() else vid)
    if not track:
        return {}
    
    first = track[0]
    last = track[-1]
    
    from agents.investigator import nearest_port
    origin = nearest_port(first[0], first[1])
    dest = nearest_port(last[0], last[1])
    return {"origin": origin, "dest": dest}


def generate_single_brief(vid: str) -> None:
    if not _state:
        initialise()
    
    # We will let single brief generate regardless of global llm_enabled toggle.
    # The investigator tool handles API key absence internally.
        
    det = _state.get("detection_by_id", {}).get(vid)
    # If not in detection_by_id, it might be a normal vessel. Let's create a minimal det object.
    if not det:
        track = _state["tracks"].get(str(vid)) or _state["tracks"].get(int(vid) if str(vid).isdigit() else vid)
        if not track:
            return
        last_pos = track[-1]
        det = {
            "vessel_id": vid,
            "lat": last_pos[0],
            "lon": last_pos[1],
            "max_gap_minutes": 0,
            "displacement_error_km": 0,
            "confidence": 0,
        }
        
    briefs = _state.setdefault("llm_briefs", {})
    tick = int(_state.get("tick") or 0)
    key = f"{det['vessel_id']}@t{tick}"
    
    if key not in briefs:
        briefs[key] = investigate_vessel(det, use_llm=True)
        det["investigation"] = briefs[key]
        _log(f"[Investigator Agent] -> [Orchestrator]: Completed on-demand brief for V{det['vessel_id']} ({briefs[key].get('model')})")


def set_detection_thresholds(gap_threshold_min: float, dr_threshold_km: float) -> None:
    """Update IMO-style rule cutoffs and rescan if they actually changed."""
    if not _state:
        initialise()
    new = {
        "gap_threshold_min": float(gap_threshold_min),
        "dr_threshold_km": float(dr_threshold_km),
    }
    old = _state.get("detection_thresholds") or {}
    if (
        old.get("gap_threshold_min") == new["gap_threshold_min"]
        and old.get("dr_threshold_km") == new["dr_threshold_km"]
    ):
        return
    _state["detection_thresholds"] = new
    _run_dark_vessel_scan()
    _update_route("detection-threshold change")


def _flagged_near_route(waypoints: list) -> list[dict[str, Any]]:
    """Flagged detections whose last position is near the managed route."""
    hits = []
    for det in _state.get("detections") or []:
        if not det.get("flagged"):
            continue
        lat, lon = det["last_known_position"]
        node = _snap_to_grid(lat, lon)
        for wp in waypoints:
            dist = _haversine(node[0], node[1], wp[0], wp[1])
            if dist <= FLAGGED_RADIUS_KM:
                hits.append({"det": det, "node": node, "dist_km": dist})
                break
    return hits


def _flagged_nodes_near_route(waypoints: list) -> list[tuple[float, float]]:
    return sorted({h["node"] for h in _flagged_near_route(waypoints)})


def _dark_near_hotspots(
    hotspots: list[dict],
    detections: list[dict] | None = None,
) -> list[dict[str, Any]]:
    """Pairs of flagged vessels within DARK_HOTSPOT_RADIUS_KM of a debris hotspot."""
    dets = detections if detections is not None else (_state.get("detections") or [])
    pairs = []
    for det in dets:
        if not det.get("flagged"):
            continue
        lat, lon = det["last_known_position"]
        best = None
        best_d = DARK_HOTSPOT_RADIUS_KM
        for hs in hotspots:
            c = hs.get("center") or [0, 0]
            d = _haversine(lat, lon, c[0], c[1])
            if d <= best_d:
                best_d = d
                best = hs
        if best is not None:
            pairs.append({"det": det, "hotspot": best, "dist_km": best_d})
    return pairs


def _run_route_tradeoff(
    reason: str = "scheduled",
    origin: tuple[float, float] | None = None,
    destination: tuple[float, float] | None = None,
) -> bool:
    """
    Multi-Agent Tradeoff Engine (Phases 1 & 2):
    1. Surveillance Agent: Maps flagged dark vessels to maritime zones, logs to SQLite, computes rolling risk.
    2. Route Planner Agent: Generates Candidate A (fuel-optimal), Candidate B (zone-avoiding), and Candidate C (balanced).
    3. Debris Agent: Evaluates opportunistic proximity to hotspots with assigned collectors.
    4. Feedback Loop: Dynamically adjusts risk-avoidance threshold from dispatcher overrides in SQLite.
    5. Orchestrator: Synthesizes N candidate options into exactly one recommended decision with live reasoning trace.
    """
    if _state.get("weather_df") is None:
        return False

    t = int(_state.get("tick") or 0)

    origin = origin or _state.get("active_route_origin", ROUTE_ORIGIN)
    destination = destination or _state.get("active_route_destination", ROUTE_DESTINATION)
    orig_name = _state.get("active_origin_name", "Houston / Galveston (US)")
    dest_name = _state.get("active_dest_name", "New Orleans / South Pass (US)")

    # ── 1. Surveillance Agent: Aggregate dark vessel events by Maritime Zone ──
    flagged = _state.get("flagged_vessels") or []
    zone_flag_counts: dict[str, int] = {}
    for fv in flagged:
        lat, lon = fv["last_known_position"]
        z = get_zone_for_point(lat, lon)
        zid = z["id"]
        zone_flag_counts[zid] = zone_flag_counts.get(zid, 0) + 1

    for zid, count in zone_flag_counts.items():
        zname = next((z["name"] for z in GULF_ZONES if z["id"] == zid), zid)
        try:
            log_zone_flag(zid, zname, t, count)
        except Exception as e:
            logger.debug(f"Failed to log zone flag: {e}")

    try:
        zone_stats = get_zone_rolling_stats(window_ticks=30, current_tick=t)
    except Exception:
        zone_stats = {}

    # ── 2. Hazard footprint near corridor ──
    curr_wps = (_state.get("current_route") or {}).get("waypoints", [])
    near = _flagged_near_route(curr_wps) if curr_wps else []
    PORT_APPROACH_KM = 35.0
    mid_route_hits = [
        h for h in near
        if _haversine(h["node"][0], h["node"][1], origin[0], origin[1]) > PORT_APPROACH_KM
        and _haversine(h["node"][0], h["node"][1], destination[0], destination[1]) > PORT_APPROACH_KM
    ]
    mid_route_hits.sort(key=lambda h: h["det"].get("confidence", 0), reverse=True)
    extra_nodes = sorted({h["node"] for h in mid_route_hits[:3]}) + _storm_cost_nodes()

    # ── 3. Route Planner Agent: 3 Candidate Routes ──
    candidates = get_candidate_routes(
        origin,
        destination,
        _state["weather_df"],
        extra_cost_nodes=extra_nodes,
        extra_cost_factor=2.5,
    )
    cand_a = candidates["candidate_a"]
    cand_b = candidates["candidate_b"]
    cand_c = candidates["candidate_c"]
    fuel_diff_b = candidates.get("fuel_diff_pct", 0.0)
    fuel_diff_c = candidates.get("fuel_diff_c_pct", 0.0)

    # Check which zones Candidate A intersects
    crossed_zones_a = []
    seen_zones_a = set()
    for wp in cand_a["waypoints"]:
        z = get_zone_for_point(wp[0], wp[1])
        if z["id"] not in seen_zones_a:
            seen_zones_a.add(z["id"])
            flags_here = zone_flag_counts.get(z["id"], 0)
            hist_count = zone_stats.get(z["id"], {}).get("count", 0)
            if flags_here > 0 or hist_count > 0:
                z_data = dict(z)
                z_data["flag_count"] = flags_here or hist_count
                z_data["risk_label"] = zone_stats.get(z["id"], {}).get("risk_label", "moderate")
                crossed_zones_a.append(z_data)

    a_hits = _flagged_near_route(cand_a["waypoints"])
    has_active_corridor_risk = bool(a_hits or crossed_zones_a or extra_nodes)

    # ── 4. Adaptive Feedback Loop: Check dispatcher overrides (Phase 2C) ──
    recent_overrides = 0
    try:
        recent_overrides = get_recent_override_count(limit=10)
    except Exception:
        recent_overrides = 0

    base_threshold = DEFAULT_RISK_AVOIDANCE_THRESHOLD_PCT
    feedback_note = None
    if recent_overrides >= 3:
        threshold = 7.0
        feedback_note = f"Risk-avoidance threshold adjusted from 5.0% to 7.0% based on {recent_overrides} recent dispatcher overrides."
    else:
        threshold = base_threshold

    _state["risk_threshold"] = threshold

    # ── 5. Orchestrator Selection Rule (Generalized to N Paths) ──
    if has_active_corridor_risk and extra_nodes and (cand_b["waypoints"] != cand_a["waypoints"]):
        if fuel_diff_b <= threshold:
            chosen = cand_b
            chosen_id = "B"
            decision_reason = (
                f"risk avoidance justifies +{fuel_diff_b:.1f}% fuel cost (threshold: {threshold:.1f}%)"
            )
        elif fuel_diff_c <= threshold and (cand_c["waypoints"] != cand_a["waypoints"]):
            chosen = cand_c
            chosen_id = "C"
            decision_reason = (
                f"full avoidance (+{fuel_diff_b:.1f}%) exceeds threshold ({threshold:.1f}%); "
                f"selected Candidate C (Balanced) providing partial safety buffer at +{fuel_diff_c:.1f}% within budget"
            )
        else:
            chosen = cand_a
            chosen_id = "A"
            decision_reason = (
                f"risk avoidance fuel penalty (+{fuel_diff_b:.1f}%) exceeds threshold ({threshold:.1f}%); "
                f"selecting fuel-efficient Candidate A with surveillance advisory"
            )
    else:
        chosen = cand_a
        chosen_id = "A"
        decision_reason = (
            f"no critical risk on primary corridor; selected Candidate A for optimal fuel efficiency ({cand_a['savings_pct']:.1f}% savings)"
        )

    # ── 6. Debris Agent: Opportunistic proximity to active collector hotspots ──
    hotspots = _state.get("debris_hotspots") or []
    debris_bonus = None
    for hs in hotspots:
        if not hs.get("collector_assigned"):
            continue
        c = hs.get("center") or [0, 0]
        min_d = min((_haversine(wp[0], wp[1], c[0], c[1]) for wp in chosen["waypoints"]), default=999.0)
        if min_d <= 45.0:
            debris_bonus = {
                "hotspot_id": hs["hotspot_id"],
                "collector": hs["collector_assigned"],
                "dist_km": round(min_d, 1),
            }
            break

    if debris_bonus:
        decision_reason += f", plus opportunistic debris proximity ({debris_bonus['dist_km']}km from {debris_bonus['hotspot_id']})"

    # Look up nearest active vessel to origin port (Part 1D)
    assigned_vessel = find_nearest_active_vessel(origin[0], origin[1], max_dist_km=300.0)

    # ── 7. Live Condensed Reasoning Trace (Phase 1C & Part 2) ──
    trace = []
    # Surveillance Agent trace line
    if crossed_zones_a:
        for z in crossed_zones_a[:2]:
            trace.append({
                "agent": "Surveillance Agent",
                "icon": "🛰️",
                "text": f"{z['id']} ({z['name']}) flagged ({z['flag_count']} dark-vessel events, risk: {z['risk_label']})",
                "type": "warning" if z["risk_label"] == "elevated" else "info",
            })
    elif flagged:
        top_z = get_zone_for_point(flagged[0]["last_known_position"][0], flagged[0]["last_known_position"][1])
        cnt = zone_flag_counts.get(top_z["id"], 1)
        rlabel = zone_stats.get(top_z["id"], {}).get("risk_label", "moderate")
        trace.append({
            "agent": "Surveillance Agent",
            "icon": "🛰️",
            "text": f"{top_z['id']} ({top_z['name']}) flagged ({cnt} dark-vessel events, risk: {rlabel})",
            "type": "warning" if rlabel == "elevated" else "info",
        })
    else:
        trace.append({
            "agent": "Surveillance Agent",
            "icon": "🛰️",
            "text": "All maritime patrol sectors clear; 0 dark vessel incursions detected on primary corridor.",
            "type": "info",
        })

    # Route Planner trace lines
    trace.append({
        "agent": "Route Planner",
        "icon": "🧭",
        "text": f"Generated 3 candidate routes: Candidate A ({cand_a['savings_pct']:.1f}% fuel saved), Candidate C (+{fuel_diff_c:.1f}% fuel, balanced), Candidate B (+{fuel_diff_b:.1f}% fuel, zone-avoidance).",
        "type": "info",
    })

    # Debris Agent trace line
    if debris_bonus:
        trace.append({
            "agent": "Debris Agent",
            "icon": "🌊",
            "text": f"Candidate {chosen_id} passes within {debris_bonus['dist_km']}km of Hotspot {debris_bonus['hotspot_id']} (collector {debris_bonus['collector']} assigned)",
            "type": "success",
        })
    else:
        trace.append({
            "agent": "Debris Agent",
            "icon": "🌊",
            "text": "Debris Agent: No active collector hotspots intersecting transit corridor.",
            "type": "info",
        })

    # Orchestrator Decision trace line
    trace.append({
        "agent": "Orchestrator",
        "icon": "🤖",
        "text": f"Selecting Candidate {chosen_id} — {decision_reason}",
        "type": "decision",
    })

    # Feedback Loop trace line if threshold was modified
    if feedback_note:
        trace.append({
            "agent": "Feedback Loop",
            "icon": "🔄",
            "text": feedback_note,
            "type": "feedback",
        })

    decision_payload = {
        "chosen_candidate": chosen_id,
        "chosen_name": chosen["name"],
        "cost": chosen["cost"],
        "savings_pct": chosen["savings_pct"],
        "fuel_diff_pct": fuel_diff_b,
        "fuel_diff_c_pct": fuel_diff_c,
        "threshold_used": threshold,
        "has_risk_avoidance": (chosen_id in ["B", "C"] and has_active_corridor_risk),
        "debris_bonus": debris_bonus,
        "reason": decision_reason,
        "feedback_applied": bool(feedback_note),
        "assigned_vessel": assigned_vessel,
        "origin_name": orig_name,
        "dest_name": dest_name,
        "origin_pos": list(origin),
        "dest_pos": list(destination),
        "tick": t,
    }

    candidate_routes_dict = {
        "candidate_a": cand_a,
        "candidate_b": cand_b,
        "candidate_c": cand_c,
        "baseline_cost": candidates.get("baseline_cost", chosen.get("baseline_cost", 0)),
        "fuel_diff_pct": fuel_diff_b,
        "fuel_diff_c_pct": fuel_diff_c,
        "chosen_id": chosen_id,
    }

    _state["candidate_routes"] = candidate_routes_dict
    _state["tradeoff_decision"] = decision_payload
    _state["reasoning_trace"] = trace

    current_route_payload = {
        "waypoints": chosen["waypoints"],
        "naive_waypoints": candidates.get("naive_waypoints", []),
        "cost": chosen["cost"],
        "baseline_cost": candidates.get("baseline_cost", chosen["cost"]),
        "savings_pct": chosen["savings_pct"],
        "chosen_candidate": chosen_id,
        "origin": list(origin),
        "destination": list(destination),
    }

    prev_wps = (_state.get("current_route") or {}).get("waypoints")
    changed = (prev_wps != current_route_payload["waypoints"])
    _state["current_route"] = current_route_payload
    _state["baseline_route"] = {"waypoints": candidates.get("naive_waypoints", [])}
    _state["metrics"]["fuel_savings_pct"] = chosen["savings_pct"]
    if changed:
        _state["metrics"]["reroute_count"] = _state["metrics"].get("reroute_count", 0) + 1
        _log(f"[Orchestrator] -> [Fleet Dispatch]: Tradeoff selection updated to Candidate {chosen_id} ({decision_reason})")

    return changed


def _update_route(reason: str) -> bool:
    """Trigger multi-agent tradeoff engine."""
    return _run_route_tradeoff(reason)


def _run_debris_tick(*, spawn_new: bool, n_new: int = 2) -> None:
    """Agent 3, with Dark Vessel Agent flags raising hotspot assignment priority."""
    prev_assign = {
        c["collector_id"]: c.get("assigned_hotspot")
        for c in _state.get("collectors") or []
    }
    pairs = _dark_near_hotspots(_state.get("debris_hotspots") or [])
    # First cluster, then pair against new hotspots: cluster without priority, then re-assign.
    hotspots, collectors, sightings = debris_agent.get_debris_state(
        _state["sightings_df"],
        _state["collectors"],
        spawn_new=spawn_new,
        n_new=n_new,
        priority_hotspot_ids=None,
    )
    pairs = _dark_near_hotspots(hotspots)
    priority_ids = {p["hotspot"]["hotspot_id"] for p in pairs}
    if priority_ids:
        hotspots, collectors = debris_agent._assign_collectors(
            hotspots, collectors, priority_hotspot_ids=priority_ids
        )
    _state["debris_hotspots"] = hotspots
    _state["collectors"] = collectors
    _state["sightings_df"] = sightings
    _state["metrics"]["hotspots_covered"] = sum(1 for h in hotspots if h["collector_assigned"])

    logged = 0
    for p in pairs:
        hs = next((h for h in hotspots if h["hotspot_id"] == p["hotspot"]["hotspot_id"]), None)
        if not hs or not hs.get("collector_assigned"):
            continue
        cid = hs["collector_assigned"]
        prev = prev_assign.get(cid)
        if prev == hs["hotspot_id"]:
            continue
        _log(f"[Surveillance Agent] -> [Orchestrator]: Flagged vessel V{p['det']['vessel_id']} near debris hotspot {hs['hotspot_id']}")
        _log(f"[Orchestrator] -> [Debris Agent]: Priority assignment requested for hotspot {hs['hotspot_id']}")
        _log(f"[Debris Agent] -> [Orchestrator]: Assigned collector {cid} to priority hotspot {hs['hotspot_id']}")
        logged += 1
        if logged >= 3:
            break


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
        debris_zones=load_debris_zones(),
        data_status=dict(DATA_STATUS),
        vessel_memory={},
        detection_thresholds={
            "gap_threshold_min": 45.0,
            "dr_threshold_km": 8.0,
        },
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
        llm_enabled=False,
        llm_briefs={},
    )
    _log("[Orchestrator] Initialising MaritimeMAS engine...")
    _log(
        f"[Data Layer] Open-Meteo & NOAA data loaded — {len(ais_df)} AIS pings · {len(weather_df)} weather points · "
        f"{len(storm_df)} NOAA storm points · {len(fishing_zones)} GFW zones"
    )

    # Agent 2 — dark vessel detection
    _run_dark_vessel_scan()

    # Vessel playback tracks + seed wakes
    _state["tracks"] = _build_tracks(ais_df)
    _state["vessel_positions"] = _vessel_state_at_tick(0)
    _state["vessel_origins"] = {}
    for v in _state["vessel_positions"]:
        _state["vessel_origins"][v["vessel_id"]] = (v["lat"], v["lon"])
        track = _state["tracks"][v["vessel_id"]]
        idx = 0 % len(track)
        _state["vessel_wakes"][v["vessel_id"]] = [
            [p[0], p[1]] for p in track[max(0, idx - 3) : idx + 1]
        ]

    # Agent 3 — debris clustering & collector assignment (priority from Dark Vessel Agent)
    _run_debris_tick(spawn_new=False)
    _log(
        f"🗑️ {len(_state['debris_hotspots'])} debris hotspots clustered, collectors assigned"
    )

    # Agent 1 — initial commercial corridor and ports
    _state["active_origin_name"] = "Houston / Galveston (US)"
    _state["active_dest_name"] = "New Orleans / South Pass (US)"
    _state["active_route_origin"] = ROUTE_ORIGIN
    _state["active_route_destination"] = ROUTE_DESTINATION

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
    if "vessel_origins" not in _state:
        _state["vessel_origins"] = {}
    for v in vpos:
        vid = v["vessel_id"]
        if vid not in _state["vessel_origins"]:
            _state["vessel_origins"][vid] = (v["lat"], v["lon"])
        wake = _state["vessel_wakes"].setdefault(vid, [])
        wake.append([v["lat"], v["lon"]])
        del wake[:-WAKE_LENGTH]

    # Agent 3 — debris drift, clustering & collector reassignment (Dark Vessel priority)
    _run_debris_tick(spawn_new=True, n_new=2)
    if t % 5 == 0:
        assigned = sum(1 for h in _state["debris_hotspots"] if h["collector_assigned"])
        _log(
            f"🗑️ {len(_state['debris_hotspots'])} hotspots tracked · {assigned} collectors assigned"
        )

    # Periodic Agent 2 rescan + reroute check
    if t % 10 == 0:
        _run_dark_vessel_scan()
        _update_route("dark vessel rescan")
    else:
        _update_vessel_memory(t)
        _run_route_tradeoff(f"tick {t}")

    _state["vessel_positions"] = _vessel_state_at_tick(t)

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
        "candidate_routes": _state.get("candidate_routes"),
        "tradeoff_decision": _state.get("tradeoff_decision"),
        "reasoning_trace": _state.get("reasoning_trace") or [],
        "risk_threshold": _state.get("risk_threshold", DEFAULT_RISK_AVOIDANCE_THRESHOLD_PCT),
        "gulf_zones": GULF_ZONES,
        "flagged_vessels": _state["flagged_vessels"],
        "storm_mode_active": _state["storm_mode_active"],
        "storm_eye_pos": _state["storm_eye_pos"],
        "storm_df": _state["storm_df"],
        "weather_df": _state["weather_df"],
        "data_status": _state["data_status"],
        "vessel_memory": dict(_state.get("vessel_memory") or {}),
        "llm_enabled": bool(_state.get("llm_enabled")),
        "llm_briefs": dict(_state.get("llm_briefs") or {}),
        "detection_thresholds": dict(_state.get("detection_thresholds") or {}),
        "vessel_origins": dict(_state.get("vessel_origins") or {}),
        "active_origin_name": _state.get("active_origin_name", "Houston / Galveston (US)"),
        "active_dest_name": _state.get("active_dest_name", "New Orleans / South Pass (US)"),
        "active_route_origin": _state.get("active_route_origin", ROUTE_ORIGIN),
        "active_route_destination": _state.get("active_route_destination", ROUTE_DESTINATION),
    }


def get_tick_history() -> list[dict[str, Any]]:
    """List of state snapshots, oldest first (for the timeline scrubber)."""
    if not _state:
        initialise()
    return _state["history"]


def _demo_conditions() -> dict[str, bool]:
    flagged = bool(_state.get("flagged_vessels"))
    assigned = any(h.get("collector_assigned") for h in (_state.get("debris_hotspots") or []))
    causal = any("in response to" in line for line in (_state.get("event_log") or []))
    rerouted = int((_state.get("metrics") or {}).get("reroute_count") or 0) > 0 or causal
    return {"flagged_vessel": flagged, "debris_assignment": assigned, "reroute": rerouted}


def guided_demo(max_ticks: int = 24, min_ticks: int = 3) -> dict[str, Any]:
    """
    Drive the live simulation until a flag, a collector assignment, and a
    reroute have all occurred. Does not inject fake events.
    """
    if not _state:
        initialise()
    _log("🎬 Guided Demo started — advancing the live engine (no scripted data)")
    ticks_run = 0
    saw = _demo_conditions()
    for _ in range(max_ticks):
        if ticks_run >= min_ticks and all(saw.values()):
            break
        tick()
        ticks_run += 1
        saw = _demo_conditions()
    ok = all(saw.values())
    if ok:
        _log(
            f"🎬 Guided Demo complete after {ticks_run} tick(s): "
            "flagged vessel ✓ · debris assignment ✓ · reroute ✓"
        )
    else:
        missing = [k for k, v in saw.items() if not v]
        _log(
            f"🎬 Guided Demo timed out after {ticks_run} ticks — still missing: "
            f"{', '.join(missing)}. Use Next Tick or Hurricane Ida replay."
        )
    return {"ok": ok, "ticks": ticks_run, "seen": saw}
