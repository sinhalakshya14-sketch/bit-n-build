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
from agents.investigator import investigate_vessel

logger = logging.getLogger(__name__)

# Route under management: Galveston Entrance → Mississippi River South Pass
ROUTE_ORIGIN = (29.30, -94.80)
ROUTE_DESTINATION = (29.90, -90.10)

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
    _log(f"🚨 Dark vessel scan: {len(flagged)} flagged (P {m['precision']:.0%} / R {m['recall']:.0%})")
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
        _log(f"🕵️ Investigator briefed {det['vessel_id']} ({briefs[key].get('model')})")
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


def _update_route(reason: str) -> bool:
    """Cross-agent reroute: recompute Agent 1's path with hazard cost nodes."""
    if not _state.get("current_route"):
        return False
    near = _flagged_near_route(_state["current_route"]["waypoints"])
    extra = sorted({h["node"] for h in near}) + _storm_cost_nodes()
    if not extra:
        return False
    new_route = get_route(
        ROUTE_ORIGIN, ROUTE_DESTINATION, _state["weather_df"], extra_cost_nodes=extra
    )
    if new_route["waypoints"] != _state["current_route"]["waypoints"]:
        _state["current_route"] = new_route
        if new_route.get("naive_waypoints"):
            _state["baseline_route"] = {"waypoints": new_route["naive_waypoints"]}
        _state["metrics"]["fuel_savings_pct"] = new_route["savings_pct"]
        _state["metrics"]["reroute_count"] += 1
        vids = ", ".join(h["det"]["vessel_id"] for h in near[:4]) or "storm-eye nodes"
        if near:
            _log(
                f"🔗 Route Agent recalculated the Galveston→NOLA corridor in response to "
                f"Dark Vessel Agent flagging {vids} within {FLAGGED_RADIUS_KM:.0f} km of the managed route "
                f"({reason}; savings {new_route['savings_pct']}%)"
            )
        else:
            _log(
                f"🔗 Route Agent recalculated the Galveston→NOLA corridor in response to "
                f"storm-eye cost nodes ({reason}; savings {new_route['savings_pct']}%)"
            )
        return True
    return False


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
        _log(
            f"🔗 Debris Agent assigned collector {cid} to hotspot {hs['hotspot_id']} "
            f"in response to Dark Vessel Agent flagging {p['det']['vessel_id']} "
            f"({p['dist_km']:.0f} km from the hotspot)"
        )
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

    # Agent 3 — debris clustering & collector assignment (priority from Dark Vessel Agent)
    _run_debris_tick(spawn_new=False)
    _log(
        f"🗑️ {len(_state['debris_hotspots'])} debris hotspots clustered, collectors assigned"
    )

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
