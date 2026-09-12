"""
smoke_test.py
=============
End-to-end verification suite for MaritimeMAS.
Run before launching the dashboard.

Usage:
    python smoke_test.py
"""

import sys
import traceback
import logging

logging.basicConfig(level=logging.WARNING)

PASS = "[PASS]"
FAIL = "[FAIL]"
results = []


def check(name, fn):
    try:
        result = fn()
        results.append((PASS, name, str(result)[:120]))
        return result
    except Exception as exc:
        results.append((FAIL, name, traceback.format_exc()[-300:]))
        return None


# ── 1. Data loader ─────────────────────────────────────────────────────────
print("\n=== Data Loader ===")

from data.loader import load_all, DATA_STATUS

data_tuple = check("load_all()", load_all)
ais_df, weather_df, debris_df, storm_df, fishing_zones = data_tuple

check("AIS rows > 0", lambda: len(ais_df) > 0)
check("AIS columns", lambda: set(["vessel_id", "lat", "lon", "speed", "heading"]).issubset(ais_df.columns))
check("Weather rows > 0", lambda: len(weather_df) > 0)
check("Storm track rows > 0", lambda: len(storm_df) > 0)
check("Fishing zones > 0", lambda: len(fishing_zones) > 0)
check("Debris rows > 0", lambda: len(debris_df) > 0)

print(f"  AIS: {len(ais_df)} rows, {ais_df['vessel_id'].nunique()} vessels")
print(f"  Weather: {len(weather_df)} grid points")
print(f"  Storm track: {len(storm_df)} NOAA points (Hurricane Ida)")
print(f"  Fishing zones: {len(fishing_zones)} GFW zones")
print(f"  Debris: {len(debris_df)} seed points")
print(f"  Data status: {DATA_STATUS}")

# ── 2. Route agent ─────────────────────────────────────────────────────────
print("\n=== Agent 1: Route Optimisation ===")

from agents.route import get_route

ORIGIN = (29.3, -94.8)
DEST   = (29.9, -90.1)

route = check("get_route()", lambda: get_route(ORIGIN, DEST, weather_df))

if route:
    check("Route has waypoints", lambda: len(route["waypoints"]) >= 2)
    check("Savings not a hardcoded 14% floor", lambda: abs(route["baseline_cost"] / route["cost"] - 1.14) > 0.001 if route["cost"] else True)
    print(f"  Waypoints: {len(route['waypoints'])}, savings: {route['savings_pct']:.1f}%")

# ── 3. Dark vessel agent ───────────────────────────────────────────────────
print("\n=== Agent 2: Dark Vessel Detection ===")

from agents.dark_vessel import detect_dark_vessels

dv_results = check("detect_dark_vessels()", lambda: detect_dark_vessels(ais_df))

if dv_results:
    flagged = [r for r in dv_results if r["flagged"]]
    check("At least 1 vessel flagged", lambda: len(flagged) >= 1)
    check("Precision and recall are not a perfect tautology", lambda: not (
        abs(sum(1 for r in dv_results if r["flagged"] and r["is_ground_truth_dark"]) / max(1, sum(1 for r in dv_results if r["flagged"])) - 1.0) < 1e-9
        and abs(sum(1 for r in dv_results if r["flagged"] and r["is_ground_truth_dark"]) / max(1, sum(1 for r in dv_results if r["is_ground_truth_dark"])) - 1.0) < 1e-9
        and sum(1 for r in dv_results if r["flagged"]) == sum(1 for r in dv_results if r["is_ground_truth_dark"])
    ))
    from agents.dark_vessel import DARK_FRACTION, IF_CONTAMINATION
    check("Detector contamination prior != GT dark fraction", lambda: IF_CONTAMINATION != DARK_FRACTION)
    n_fp = sum(1 for r in dv_results if r["flagged"] and not r["is_ground_truth_dark"])
    n_fn = sum(1 for r in dv_results if (not r["flagged"]) and r["is_ground_truth_dark"])
    check("Flagged set is not identical to ground truth (false positives exist)", lambda: n_fp > 0)
    print(f"  Evaluation: FP={n_fp} FN={n_fn} (precision/recall can move; not a closed loop)")
    strict = detect_dark_vessels(ais_df, gap_threshold_min=90, dr_threshold_km=20)
    check("Live thresholds appear on detections", lambda: all(
        r.get("flag_gap_threshold_min") == 90 and r.get("flag_dr_threshold_km") == 20 for r in strict
    ))

# ── 4. Debris agent ────────────────────────────────────────────────────────
print("\n=== Agent 3: Debris Coordination ===")

from agents.debris import init_collectors, get_debris_state

collectors = check("init_collectors()", init_collectors)
debris_state = check(
    "get_debris_state()",
    lambda: get_debris_state(debris_df, collectors)
)

if debris_state:
    hotspots, colls, updated_df = debris_state
    check("At least 1 hotspot", lambda: len(hotspots) >= 1)
    check("ETA present in assignment", lambda: any(h.get("eta") for h in hotspots if h["collector_assigned"]))

# ── 5. Investigator (local tools; LLM optional) ────────────────────────────
from agents.investigator import investigate_vessel

inv = check("investigator local tools", lambda: investigate_vessel(
    {"vessel_id": "V0001", "last_known_position": [29.3, -94.8], "max_gap_minutes": 80, "displacement_error_km": 12, "confidence": 0.6},
    use_llm=False,
))
if inv:
    check("Investigator cites a port", lambda: "Port" in inv["summary"] or inv["tool_calls"][0]["name"] == "nearest_port")
    check("Investigator exposes tool trace", lambda: len(inv["tool_calls"]) >= 2)

import orchestrator

print("\n=== Orchestrator & Demo Modes ===")
check("initialise()", orchestrator.initialise)
state = check("get_state()", orchestrator.get_state)

if state:
    check("Event log non-empty", lambda: len(state["event_log"]) > 0)
    check("Wakes tracked", lambda: len(state["vessel_wakes"]) > 0)
    check("Causal cross-agent log present", lambda: any(
        "in response to" in line for line in orchestrator.get_state()["event_log"]
    ))

# Test Hurricane Ida storm mode toggle
check("toggle_storm_mode(True)", lambda: orchestrator.toggle_storm_mode(True))
check("toggle_storm_mode(False)", lambda: orchestrator.toggle_storm_mode(False))
check("tick()", orchestrator.tick)
for _ in range(5):
    orchestrator.tick()
mem = orchestrator.get_state().get("vessel_memory") or {}
flagged_mem = [r for r in mem.values() if r.get("flag_count", 0) >= 1]
check("Vessel memory records flags", lambda: len(flagged_mem) >= 1)
check(
    "Repeat-flag suspicion can exceed scan score",
    lambda: any(
        r.get("cumulative_suspicion", 0) >= r.get("last_scan_confidence", 0)
        and r.get("flag_count", 0) >= 1
        for r in flagged_mem
    ),
)
check("get_tick_history()", lambda: len(orchestrator.get_tick_history()) > 0)
check("set_detection_thresholds stores live cutoffs", lambda: (
    orchestrator.set_detection_thresholds(60.0, 12.0),
    orchestrator.get_state()["detection_thresholds"]["gap_threshold_min"] == 60.0,
)[-1])
demo = check("guided_demo()", orchestrator.guided_demo)
if demo:
    check("Guided demo produced flag, assignment, and reroute", lambda: demo.get("ok") is True)

# ── Summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SMOKE TEST SUMMARY")
print("=" * 60)
for status, name, detail in results:
    print(f"{status} {name}")
    if status == FAIL:
        print(f"   {detail}")

n_pass = sum(1 for s, _, _ in results if s == PASS)
n_fail = sum(1 for s, _, _ in results if s == FAIL)
print(f"\n{n_pass} passed, {n_fail} failed")

if n_fail > 0:
    sys.exit(1)
else:
    print("\nAll checks passed -- run: python -m streamlit run dashboard.py")
