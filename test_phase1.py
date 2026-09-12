import sys
import os
import json
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
load_dotenv()

import orchestrator
import storage

def test_tradeoff_and_feedback():
    print("=================================================================")
    print("   TESTING MULTI-AGENT TRADEOFF ENGINE & FEEDBACK LOOP")
    print("=================================================================\n")

    # Step 1: Initialize orchestrator
    print("[1] Initializing Orchestrator state...")
    orchestrator.initialise()
    state = orchestrator.get_state()

    # Step 2: Validate Candidate Routes
    candidates = state.get("candidate_routes")
    assert candidates is not None, "candidate_routes must be present in state"
    cand_a = candidates["candidate_a"]
    cand_b = candidates["candidate_b"]
    print(f"  Candidate A: {cand_a['name']} | Cost: {cand_a['cost']} | Fuel Saved: {cand_a['savings_pct']}%")
    print(f"  Candidate B: {cand_b['name']} | Cost: {cand_b['cost']} | Fuel Saved: {cand_b['savings_pct']}%")
    print(f"  Fuel Variance: +{candidates['fuel_diff_pct']:.1f}%")

    # Step 3: Validate Tradeoff Decision
    decision = state.get("tradeoff_decision")
    assert decision is not None, "tradeoff_decision must be present in state"
    print(f"\n[2] Orchestrator Tradeoff Decision:")
    print(f"  Selected: {decision['chosen_name']}")
    print(f"  Reason:   {decision['reason']}")
    print(f"  Risk Threshold Used: {decision['threshold_used']}%")
    if decision.get("debris_bonus"):
        db = decision["debris_bonus"]
        print(f"  Opportunistic Debris: Hotspot {db['hotspot_id']} ({db['dist_km']} km away, collector {db['collector']})")

    # Step 4: Validate Live Reasoning Trace
    trace = state.get("reasoning_trace")
    assert trace and len(trace) >= 4, "reasoning_trace must contain entries from Surveillance, Route Planner, Debris, and Orchestrator"
    print(f"\n[3] Live Reasoning Trace ({len(trace)} lines):")
    for t in trace:
        print(f"  [{t.get('agent', 'Agent')}]: {t.get('text', '')}")

    # Step 5: Validate Persistent SQLite Zone Memory (Phase 2A)
    print(f"\n[4] Testing Persistent SQLite Zone Memory (Phase 2A)...")
    zone_stats = storage.get_zone_rolling_stats(window_ticks=30, current_tick=state["tick"])
    print(f"  Logged zones in SQLite: {len(zone_stats)} sectors tracked")
    for zid, zdata in list(zone_stats.items())[:3]:
        print(f"    - {zid} ({zdata['zone_name']}): {zdata['count']} flag events -> Risk: {zdata['risk_label'].upper()}")

    # Step 6: Test Feedback Loop (Phase 2B & 2C)
    print(f"\n[5] Testing Dispatcher Override Feedback Loop (Phase 2B & 2C)...")
    initial_threshold = state.get("risk_threshold", 5.0)
    print(f"  Initial risk-avoidance threshold: {initial_threshold}%")

    # Force 3 dispatcher overrides in SQLite
    print("  Simulating 3 dispatcher overrides in favor of fuel budget...")
    for i in range(3):
        storage.log_route_decision(
            rec_id=f"test_override_{i}",
            action="overridden",
            reason=f"Fuel budget restriction {i+1}",
            threshold=initial_threshold,
        )

    override_count = storage.get_recent_override_count(limit=10)
    print(f"  Recent override count in SQLite: {override_count}")
    assert override_count >= 3, "Recent overrides must be at least 3"

    # Re-run tradeoff engine
    orchestrator._run_route_tradeoff("dispatcher override feedback test")
    new_state = orchestrator.get_state()
    new_threshold = new_state.get("risk_threshold")
    new_decision = new_state.get("tradeoff_decision")
    new_trace = new_state.get("reasoning_trace")

    print(f"  Updated risk-avoidance threshold: {new_threshold}%")
    assert new_threshold == 7.0, f"Expected threshold to adjust to 7.0%, got {new_threshold}"

    # Confirm feedback note in trace
    feedback_entries = [t for t in new_trace if t.get("agent") == "Feedback Loop"]
    assert len(feedback_entries) > 0, "Trace must include a Feedback Loop entry"
    print(f"  Feedback Loop Trace Confirmation:")
    print(f"    {feedback_entries[0]['text']}")

    print("\n=================================================================")
    print("   ALL PHASES (1A, 1B, 1C, 1D, 2A, 2B, 2C) PASSED SUCCESSFULLY!")
    print("=================================================================\n")

if __name__ == "__main__":
    test_tradeoff_and_feedback()
