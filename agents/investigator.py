"""
agents/investigator.py
======================
LLM-as-tool-using agent for flagged-vessel briefs.

Lookups hit local CSVs only. The Claude call is optional: skipped unless
ANTHROPIC_API_KEY is set and the caller enables the feature flag.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any

from groq import Groq

from data.reference import load_companies, load_lighthouses, load_ports

MODEL = "openai/gpt-oss-120b"

# Tools definition moved to local function call


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def nearest_port(lat: float, lon: float) -> dict[str, Any]:
    df = load_ports()
    best = None
    best_d = 1e9
    for rec in df.to_dict("records"):
        d = _haversine(lat, lon, float(rec["lat"]), float(rec["lon"]))
        if d < best_d:
            best_d = d
            best = rec
    out = dict(best or {})
    out["distance_km"] = round(best_d, 1)
    out["distance_nm"] = round(best_d / 1.852, 1)
    return out


def nearest_lighthouse(lat: float, lon: float) -> dict[str, Any]:
    df = load_lighthouses()
    best = None
    best_d = 1e9
    for rec in df.to_dict("records"):
        d = _haversine(lat, lon, float(rec["lat"]), float(rec["lon"]))
        if d < best_d:
            best_d = d
            best = rec
    out = dict(best or {})
    out["distance_km"] = round(best_d, 1)
    out["distance_nm"] = round(best_d / 1.852, 1)
    return out


def companies_at_port(port_name: str) -> list[dict[str, Any]]:
    df = load_companies()
    hits = df[df["hq_port"].str.lower() == str(port_name).lower()]
    if hits.empty:
        # nearest HQ by name containment
        hits = df[df["hq_port"].str.lower().str.contains(str(port_name).lower().split(",")[0][:12], na=False)]
    return hits.to_dict("records")[:5]


def _dispatch(name: str, args: dict) -> Any:
    if name == "nearest_port":
        return nearest_port(float(args["lat"]), float(args["lon"]))
    if name == "nearest_lighthouse":
        return nearest_lighthouse(float(args["lat"]), float(args["lon"]))
    if name == "companies_at_port":
        return companies_at_port(str(args["port_name"]))
    return {"error": f"unknown tool {name}"}


def llm_configured() -> bool:
    return bool(os.getenv("GROQ_API_KEY"))


def investigate_vessel(vessel: dict[str, Any], *, use_llm: bool = True) -> dict[str, Any]:
    """
    Returns {summary, tool_calls: [{name, input, output}], model, used_llm}.
    Never raises on missing API key.
    """
    latlon = vessel.get("last_known_position") or [vessel.get("lat"), vessel.get("lon")]
    lat, lon = float(latlon[0]), float(latlon[1])
    tool_calls: list[dict[str, Any]] = []

    if not use_llm or not llm_configured():
        port = nearest_port(lat, lon)
        light = nearest_lighthouse(lat, lon)
        cos = companies_at_port(str(port.get("name", "")))
        tool_calls = [
            {"name": "nearest_port", "input": {"lat": lat, "lon": lon}, "output": port},
            {"name": "nearest_lighthouse", "input": {"lat": lat, "lon": lon}, "output": light},
            {"name": "companies_at_port", "input": {"port_name": port.get("name")}, "output": cos},
        ]
        ops = ", ".join(c.get("name", "?") for c in cos) or "no illustrative operator on file"
        is_flagged = vessel.get("is_flagged", False)
        if is_flagged:
            conf = vessel.get("confidence", 0)
            summary = (
                f"🚨 {vessel.get('vessel_id')} flagged for dark activity (Confidence: {conf:.0%}). Gap duration: {vessel.get('max_gap_minutes')} min, "
                f"DR discrepancy: {vessel.get('displacement_error_km')} km. "
                f"It went dark ~{port.get('distance_nm')} nm from {port.get('name')} "
                f"({port.get('state')}). Nearest light: {light.get('name')} "
                f"({light.get('distance_nm')} nm). Operators tied to that HQ port: {ops}. "
                f"(Deterministic lookup — set GROQ_API_KEY and enable LLM briefs for Groq.)"
            )
        else:
            summary = (
                f"✅ {vessel.get('vessel_id')} is operating normally. No anomaly detected. "
                f"Current position: {lat:.4f}, {lon:.4f}. Speed: {vessel.get('speed', 'N/A')} kts, Heading: {vessel.get('heading', 'N/A')}°. "
                f"Nearest port is {port.get('name')} ({port.get('state')}) at {port.get('distance_nm')} nm. "
                f"(Deterministic lookup — set GROQ_API_KEY and enable LLM briefs for Groq.)"
            )
        return {"summary": summary, "tool_calls": tool_calls, "model": "local-lookup", "used_llm": False}

    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    user = (
        f"Investigate flagged vessel {vessel.get('vessel_id')}. "
        f"Last known position {lat:.4f}, {lon:.4f}. "
        f"Max AIS gap {vessel.get('max_gap_minutes')} min, "
        f"dead-reckoning error {vessel.get('displacement_error_km')} km, "
        f"scan confidence {vessel.get('scan_confidence', vessel.get('confidence'))}. "
        "Use tools to find nearest port, nearest lighthouse, and any company at that port. "
        "Explicitly mention the confidence score ({vessel.get('scan_confidence', vessel.get('confidence', 0)):.1%}) in the summary text. "
        "Write 3-5 sentences smoothly integrating the facts (include nm distances). "
        "CRITICAL INSTRUCTION: DO NOT mention any tool names (like 'nearest_port', 'nearest_lighthouse', etc) and DO NOT say 'According to the tool...'. Just state the facts directly as a human analyst."
    )
    messages: list[dict[str, Any]] = [{"role": "user", "content": user}]
    summary = ""

    # Groq OpenAI-compatible tools parameter format
    groq_tools = [
        {
            "type": "function",
            "function": {
                "name": "nearest_port",
                "description": "Finds the nearest major commercial port to a given latitude and longitude.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "lat": {"type": "number"},
                        "lon": {"type": "number"}
                    },
                    "required": ["lat", "lon"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "nearest_lighthouse",
                "description": "Finds the nearest navigation light/lighthouse.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "lat": {"type": "number"},
                        "lon": {"type": "number"}
                    },
                    "required": ["lat", "lon"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "companies_at_port",
                "description": "Lists maritime companies operating near a specific port city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "port_name": {"type": "string"}
                    },
                    "required": ["port_name"]
                }
            }
        }
    ]

    for _ in range(4):
        resp = client.chat.completions.create(
            model=MODEL,
            max_tokens=800,
            tools=groq_tools,
            messages=messages,
        )
        msg = resp.choices[0].message
        
        if msg.content:
            summary = msg.content
            
        tool_uses = msg.tool_calls
        if not tool_uses:
            break
            
        messages.append(msg)
        for tu in tool_uses:
            args = json.loads(tu.function.arguments)
            out = _dispatch(tu.function.name, args)
            tool_calls.append({"name": tu.function.name, "input": args, "output": out})
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tu.id,
                    "content": json.dumps(out, default=str),
                }
            )

    if not summary:
        summary = "Groq returned no narrative after tool use."
    return {"summary": summary, "tool_calls": tool_calls, "model": MODEL, "used_llm": True}


def generate_vessel_brief(vessel_id: str, state: dict) -> dict:
    """
    Returns an investigative brief for any vessel ID in the state, in the requested format.
    """
    vessels = state.get("vessel_positions", [])
    vessel = next((v for v in vessels if str(v.get("vessel_id")) == str(vessel_id)), None)
    
    if not vessel:
        return {
            "vessel_id": vessel_id,
            "summary": "",
            "tool_calls": [],
            "is_flagged": False,
            "confidence": None,
            "error": f"Vessel '{vessel_id}' not found in current state."
        }
        
    flagged_vessels = state.get("flagged_vessels", [])
    det = next((fv for fv in flagged_vessels if str(fv.get("vessel_id")) == str(vessel_id)), None)
    
    is_flagged = bool(det)
    confidence = det.get("confidence") if det else None
    
    target_vessel = dict(vessel)
    target_vessel["last_known_position"] = [vessel.get("lat"), vessel.get("lon")]
    target_vessel["is_flagged"] = is_flagged
    
    if det:
        target_vessel["max_gap_minutes"] = det.get("max_gap_minutes", 0)
        target_vessel["displacement_error_km"] = det.get("displacement_error_km", 0)
        target_vessel["confidence"] = det.get("confidence", 0)
        target_vessel["scan_confidence"] = det.get("scan_confidence", 0)
    else:
        target_vessel["max_gap_minutes"] = 0
        target_vessel["displacement_error_km"] = 0
        target_vessel["confidence"] = 0
        target_vessel["scan_confidence"] = 0
        
    try:
        res = investigate_vessel(target_vessel, use_llm=True)
        tool_calls_formatted = []
        for tc in res.get("tool_calls", []):
            tool_calls_formatted.append({
                "tool": tc.get("name"),
                "input": tc.get("input"),
                "result": tc.get("output")
            })
            
        return {
            "vessel_id": vessel_id,
            "summary": res.get("summary", ""),
            "tool_calls": tool_calls_formatted,
            "is_flagged": is_flagged,
            "confidence": confidence,
            "error": None
        }
    except Exception as e:
        return {
            "vessel_id": vessel_id,
            "summary": "",
            "tool_calls": [],
            "is_flagged": is_flagged,
            "confidence": confidence,
            "error": f"Failed to generate brief: {str(e)}"
        }
