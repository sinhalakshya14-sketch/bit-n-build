"""
dashboard.py
============
Production-grade Dashboard for Maritime Multi-Agent System (MaritimeMAS).

Features:
  - Real Satellite Imagery (Google Maps Satellite Hybrid & Esri World Imagery)
  - MarineTraffic Light Nautical Basemap option
  - Real-time animated vessel markers with heading vectors & wake trails
  - Interactive popup tooltips for dark vessels & debris collector ETAs
  - Collapsible 'Why was this flagged?' plain-language explainability panel
  - Before vs. After Route Comparison toggle (Naive straight-line vs. Optimized A*)
  - Hurricane Ida Demo Mode (NOAA HURDAT2 Cat 4 storm avoidance)
  - Global Fishing Watch (GFW) active fishing zone overlay
  - Timeline Scrubber for past simulation state replay

Run:
  python -m streamlit run dashboard.py
"""

import time
import sys
import os
import json
import base64

import streamlit as st
import streamlit.components.v1 as components
import folium
from folium.plugins import MousePosition
from streamlit_folium import st_folium
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import orchestrator
import storage
from agents.route import get_route
from data.reference import load_companies, load_lighthouses, load_ports

# ─────────────────────────────────────────────────────────────────────────── #
#  Page Config                                                                #
# ─────────────────────────────────────────────────────────────────────────── #
st.set_page_config(
    page_title="MaritimeMAS — Global Operations",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────── #
#  Glassmorphic Professional CSS                                              #
# ─────────────────────────────────────────────────────────────────────────── #
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
    .stApp { background: #000000; color: #DDDDDD; }
    
    /* Animations */
    @keyframes fadein {
        from { opacity: 0; transform: translateY(10px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    
    .stApp > div {
        animation: fadein 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }

    /* Header Banner */
    .mas-header {
        background: #000000;
        border: 1px solid #333333;
        border-radius: 12px;
        padding: 20px 28px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 10px;
        transition: border-color 0.3s ease;
    }
    .mas-header:hover { border-color: #555555; }
    .mas-title-group { display: flex; align-items: center; gap: 16px; }
    .mas-title-group .mas-icon {
        width: 48px; height: 48px; border-radius: 12px; flex-shrink: 0;
        background: #111111;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.5rem; 
        border: 1px solid #333333;
    }
    .mas-title-group h1 { color: #FFFFFF; font-size: 1.6rem; font-weight: 800; margin: 0; }
    .mas-title-group p  { color: #AAAAAA; font-size: 0.85rem; margin: 4px 0 0 0; font-weight: 500; }
    
    .mas-header-right { display: flex; align-items: center; gap: 12px; }
    .mas-tick-pill {
        font-weight: 600; font-size: 0.75rem; color: #FFFFFF;
        background: #111111; border: 1px solid #333333;
        padding: 6px 12px; border-radius: 20px;
    }
    
    .pulse-dot {
        width: 8px; height: 8px; border-radius: 50%; background: #FFFFFF; display: inline-block;
        margin-right: 6px; box-shadow: 0 0 0 rgba(255,255,255,0.6); animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%   { box-shadow: 0 0 0 0 rgba(255,255,255,0.4); }
        70%  { box-shadow: 0 0 0 6px rgba(255,255,255,0); }
        100% { box-shadow: 0 0 0 0 rgba(255,255,255,0); }
    }

    /* KPI Strip */
    .kpi-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 20px; }
    .kpi-card {
        background: #000000;
        border: 1px solid #333333;
        border-radius: 12px;
        padding: 18px 22px;
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    .kpi-card:hover { transform: translateY(-4px); border-color: #777777; }
    .kpi-label { color: #AAAAAA; font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; display:flex; align-items:center; gap:6px; }
    .kpi-value { color: #FFFFFF; font-size: 2rem; font-weight: 800; margin-top: 6px; line-height: 1.1; }
    .kpi-sub { color: #888888; font-size: 0.75rem; margin-top: 4px; font-weight: 500; }

    /* Sidebar Metric Cards */
    .metric-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .metric-card {
        background: #000000;
        border: 1px solid #333333;
        border-radius: 12px;
        padding: 14px;
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    .metric-card:hover { border-color: #666666; transform: translateY(-2px); }
    .metric-label { color: #AAAAAA; font-size: 0.65rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-value { color: #FFFFFF; font-size: 1.4rem; font-weight: 800; line-height: 1.2; margin-top: 4px; }
    .metric-sub   { color: #888888; font-size: 0.7rem; margin-top: 4px; line-height: 1.3; font-weight: 500; }

    /* Section Headings */
    .section-heading {
        display: flex; align-items: center; gap: 10px;
        color: #FFFFFF; font-size: 1.1rem; font-weight: 800;
        margin: 8px 0 14px 0;
        padding-bottom: 10px;
        border-bottom: 1px solid #333333;
    }
    .section-heading .tag {
        font-size: 0.65rem; font-weight: 700; color: #DDDDDD;
        background: #222222; padding: 4px 10px; border-radius: 20px;
        text-transform: uppercase; letter-spacing: 0.05em;
    }

    /* Map Legend */
    .legend-bar {
        display: flex; gap: 20px; flex-wrap: wrap; align-items: center;
        margin-bottom: 12px; padding: 12px 20px;
        background: #000000; border: 1px solid #333333;
        border-radius: 12px;
    }
    .legend-item { display: flex; align-items: center; gap: 8px; font-size: 0.8rem; color: #BBBBBB; font-weight: 500; }
    .legend-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
    .legend-line { width: 20px; height: 0; display: inline-block; border-top: 3px solid; }
    .legend-line.dashed { border-top-style: dashed; }
    .legend-sq { width: 12px; height: 12px; display: inline-block; border-radius: 3px; }

    /* Route Comparison Bar */
    .route-bar {
        background: #000000; border: 1px solid #333333;
        border-radius: 12px; padding: 16px 22px; margin-top: 12px;
        display: flex; justify-content: space-between; align-items: center;
        flex-wrap: wrap; gap: 12px; font-size: 0.85rem; color: #CCCCCC;
    }
    .route-bar b { color: #FFFFFF; font-weight: 700; }

    /* Explainability Panel */
    .explain-box {
        background: #000000;
        border: 1px solid #444444;
        border-left: 4px solid #FFFFFF;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 14px;
        font-size: 0.85rem;
        color: #DDDDDD;
    }
    .explain-metric { font-weight: 700; color: #FFFFFF; }

    /* Event Feed */
    .event-item {
        background: #000000;
        border: 1px solid #333333;
        border-left: 3px solid #777777;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
        font-size: 0.85rem;
        color: #CCCCCC;
        line-height: 1.5;
        display: flex; justify-content: space-between; align-items: flex-start; gap: 12px;
        font-weight: 500;
        transition: background 0.3s ease;
    }
    .event-item:hover { background: #0A0A0A; }
    .event-item.warn { border-left-color: #AAAAAA; }
    .event-item.crit { border-left-color: #FFFFFF; }
    .event-count {
        flex-shrink: 0; font-size: 0.7rem; font-weight: 800; color: #000000;
        background: #FFFFFF; border-radius: 12px; padding: 2px 8px; white-space: nowrap;
    }

    /* Status Badges */
    .badge {
        display: inline-block; border-radius: 16px;
        padding: 4px 12px; font-size: 0.7rem; font-weight: 700;
        letter-spacing: 0.05em;
    }
    .badge-real { background: #111111; color: #FFFFFF; border: 1px solid #444444; }
    .badge-syn  { background: #222222; color: #DDDDDD; border: 1px solid #555555; }

    /* Data Authenticity Rows */
    .data-row {
        display: flex; align-items: flex-start; gap: 10px;
        padding: 8px 0; border-bottom: 1px solid #222222;
    }
    .data-row:last-child { border-bottom: none; }
    .data-row-text { font-size: 0.8rem; color: #AAAAAA; line-height: 1.4; font-weight: 500; }
    .data-row-text b { color: #FFFFFF; font-size: 0.85rem; font-weight: 700; }

    /* Overlay sidebar */
    [data-testid="stSidebar"] {
        background: #000000 !important;
        border-right: 1px solid #333333;
        z-index: 400;
    }
    [data-testid="stSidebar"] * {
        color: #DDDDDD !important;
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] .stMarkdown {
        color: #DDDDDD !important;
        opacity: 1 !important;
    }
    [data-testid="stSidebar"] .stButton button {
        border-radius: 8px; font-weight: 700; font-size: 0.85rem;
        background: #000000;
        color: #FFFFFF !important;
        border: 1px solid #555555;
        transition: all 0.2s ease;
    }
    [data-testid="stSidebar"] .stButton button:hover {
        border-color: #FFFFFF; background: #111111;
    }
    div[data-testid="stAppViewContainer"] > section.main {
        margin-left: 0 !important;
    }
    [data-testid="stBottom"] {
        background: #000000;
        border-top: 1px solid #333333;
        color: #888888;
        font-size: 0.8rem;
        font-weight: 500;
    }

    /* Streamlit overrides for inputs and expanders */
    .stTextInput input {
        border-radius: 8px; border: 1px solid #444444; background: #000000; color: #FFFFFF; font-weight: 500;
    }
    .stTextInput input:focus { border-color: #FFFFFF; box-shadow: 0 0 0 1px #FFFFFF; }
    .stExpander { border: 1px solid #333333; border-radius: 12px; background: #000000; }
    .stExpander summary { color: #FFFFFF !important; font-weight: 700; }

    /* Footer */
    .mas-footer {
        text-align: center; color: #666666; font-size: 0.75rem; padding: 20px 0 10px 0; font-weight: 500;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────── #
#  Bootstrap Session State                                                    #
# ─────────────────────────────────────────────────────────────────────────── #
NAV_LIVE = "Live Map"
NAV_PORTS = "Ports"
NAV_COMPANIES = "Companies"
NAV_LIGHTHOUSES = "Lighthouses"
NAV_PLANNER = "Route Planner"
NAV_OPTIONS = [NAV_LIVE, NAV_PORTS, NAV_COMPANIES, NAV_LIGHTHOUSES, NAV_PLANNER]
GULF_CENTER = (22.0, -30.0)
GULF_ZOOM = 3

if "initialised" not in st.session_state:
    st.session_state.initialised = False
    st.session_state.auto_play = False
    st.session_state.playback_speed = 3
    st.session_state.show_gfw = True
    st.session_state.show_before_after = True
    st.session_state.map_style_choice = " CartoDB Dark Matter"
    st.session_state.planner_route = None
    st.session_state.planner_baseline = None
    st.session_state.planner_labels = None
    st.session_state.planner_origin = "Port of Houston"
    st.session_state.planner_dest = "Port of Tampa"
    st.session_state.nav_view = NAV_LIVE
    if "search_focus_vessel" not in st.session_state:
        st.session_state.search_focus_vessel = None
    if "active_search_vessel" not in st.session_state:
        st.session_state.active_search_vessel = None

# Drop any leftover Folium HTML previously stuffed into session_state (P1).
st.session_state.pop("_folium_map_cache", None)
st.session_state["_main_script_runs"] = st.session_state.get("_main_script_runs", 0) + 1


@st.cache_data
def _ports_ref() -> pd.DataFrame:
    return load_ports()


@st.cache_data
def _lighthouses_ref() -> pd.DataFrame:
    return load_lighthouses()


@st.cache_data
def _companies_ref() -> pd.DataFrame:
    return load_companies()


def _apply_basemap(folium_map, choice: str) -> None:
    if "Google Maps" in choice:
        folium.TileLayer(
            tiles="https://mt1.google.com/vt/lyrs=s,h&x={x}&y={y}&z={z}",
            attr="Google Maps Satellite",
            name="Google Satellite Hybrid",
            overlay=False,
            control=True,
        ).add_to(folium_map)
    elif "Esri" in choice:
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri World Imagery",
            name="Esri Satellite",
            overlay=False,
            control=True,
        ).add_to(folium_map)
    elif "CartoDB" in choice:
        folium.TileLayer(
            tiles="cartodbdark_matter",
            name="CartoDB Dark Matter",
            overlay=False,
            control=True,
        ).add_to(folium_map)
    else:
        folium.TileLayer(
            tiles="OpenStreetMap",
            name="OpenStreetMap Marine View",
            overlay=False,
            control=True,
        ).add_to(folium_map)


def _new_gulf_map(choice: str, center=GULF_CENTER, zoom=GULF_ZOOM):
    fmap = folium.Map(
        location=list(center),
        zoom_start=zoom,
        tiles=None,
        control_scale=True,
        prefer_canvas=True,
    )
    _apply_basemap(fmap, choice)
    MousePosition(
        position="bottomleft",
        separator=" | ",
        prefix="Cursor:",
        num_digits=5,
        lat_formatter="function(num) {return L.Util.formatNum(num, 5) + ' N';}",
        lng_formatter="function(num) {return L.Util.formatNum(num, 5) + ' E';}",
    ).add_to(fmap)
    return fmap


def add_route_layers(folium_map, route: dict | None, baseline_route=None, show_before_after: bool = True):
    """Shared route overlay used by Live Map and Route Planner."""
    if show_before_after and baseline_route:
        bl_wps = baseline_route.get("waypoints", []) if isinstance(baseline_route, dict) else baseline_route
        if bl_wps and len(bl_wps) >= 2:
            folium.PolyLine(
                locations=[[wp[0], wp[1]] for wp in bl_wps],
                color="#ef4444",
                weight=3,
                dash_array="8, 8",
                opacity=0.8,
                popup="Naive Straight-Line Baseline Route",
            ).add_to(folium_map)
    route = route or {}
    waypoints = route.get("waypoints", [])
    if len(waypoints) >= 2:
        folium.PolyLine(
            locations=[[wp[0], wp[1]] for wp in waypoints],
            color="#38bdf8",
            weight=5,
            opacity=0.95,
            popup=(
                f"Optimized A* Route (Cost: {route.get('cost', 0):.1f}, "
                f"Savings: {route.get('savings_pct', 0):.1f}%)"
            ),
        ).add_to(folium_map)
    return folium_map


def _request_live_tick() -> None:
    """Sidebar Next Tick: advance sim and rerun the live-ops fragment."""
    st.session_state["_pending_tick"] = True
    st.session_state["_tick_via_callback"] = True
    st.rerun("live_tick_map")


def _sync_analyst_name() -> None:
    """Copy the fragment-owned name widget onto a non-widget key."""
    st.session_state["_analyst_name"] = (
        st.session_state.get("analyst_name_frag")
        or st.session_state.get("analyst_name")
        or ""
    ).strip()


def _current_analyst() -> str:
    return (
        st.session_state.get("_analyst_name")
        or st.session_state.get("analyst_name_frag")
        or st.session_state.get("analyst_name")
        or ""
    ).strip()


def _watchlist_add(vid: str) -> None:
    storage.add_to_watchlist(_current_analyst(), vid)


def _watchlist_remove(vid: str) -> None:
    storage.remove_from_watchlist(_current_analyst(), vid)


def _render_analyst_watchlist_bar() -> None:
    """Name + hint sit in the fragment body (under the map), not an outside container.

    A new widget key is required so Streamlit treats Enter as a fragment rerun;
    the old `analyst_name` key was owned by the main script and rebuilt the tables.
    """
    st.markdown(
        '<div class="section-heading"> Analyst watchlist<span class="tag">press Enter to commit</span></div>',
        unsafe_allow_html=True,
    )
    st.text_input(
        "Your name",
        key="analyst_name_frag",
        placeholder="e.g. Jordan Chen",
        on_change=_sync_analyst_name,
    )
    _sync_analyst_name()
    if not _current_analyst():
        st.info("Type your name and press Enter — your watchlist and  Add buttons appear right here, under the map.")
    else:
        st.success(f"Signed in as {_current_analyst()}. Star flagged vessels in the expander below.")


def _render_header(state: dict) -> None:
    st.markdown(
        f"""
        <div class="mas-header">
          <div class="mas-title-group">
            <div class="mas-icon"></div>
            <div>
              <h1>MaritimeMAS — Global Fleet Operations</h1>
              <p>Live AIS density · NOAA weather · IsolationForest dark-vessel scan</p>
            </div>
          </div>
          <div class="mas-header-right">
            <span class="badge badge-real"><span class="pulse-dot"></span>LIVE ENGINE</span>
            <span class="mas-tick-pill">TICK #{state['tick']}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_kpi_strip(state: dict) -> None:
    _m = state["metrics"]
    st.markdown(
        f"""
        <div class="kpi-row">
          <div class="kpi-card" title="Calculated dynamically: Optimal A* vs. naive distance-based path for the currently simulated corridor.">
            <div class="kpi-label"> Fuel Savings (Demo Corridor)</div>
            <div class="kpi-value accent-blue">{_m['fuel_savings_pct']:.1f}%</div>
            <div class="kpi-sub">Optimized route vs. naive straight line</div>
          </div>
          <div class="kpi-card" title="Computed live per-tick from the anomaly detection model. Precision/Recall independently scored against random ground-truth masking (not circular).">
            <div class="kpi-label"> Flagged (Current Scan)</div>
            <div class="kpi-value accent-red">{_m['vessels_flagged']}</div>
            <div class="kpi-sub">Precision {_m['precision']:.0%} · Recall {_m['recall']:.0%}</div>
          </div>
          <div class="kpi-card" title="Live count of active debris hotspots currently assigned to collector vessels.">
            <div class="kpi-label"> Debris Hotspots Covered</div>
            <div class="kpi-value accent-amber">{_m['hotspots_covered']}</div>
            <div class="kpi-sub">Nearest-collector assignment</div>
          </div>
          <div class="kpi-card" title="Cumulative count over the entire simulation session of how many times the Route Agent recalculated a path.">
            <div class="kpi-label"> Dynamic Reroutes (Session)</div>
            <div class="kpi-value accent-green">{_m['reroute_count']}</div>
            <div class="kpi-sub">Cross-agent obstacle avoidance</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_event_feed(state: dict) -> None:
    st.markdown('<div class="section-heading"> Real-Time Event Feed</div>', unsafe_allow_html=True)
    event_log = state.get("event_log", [])
    if not event_log:
        st.info("System initializing...")
        return
    grouped = []
    for ev in event_log[:40]:
        msg_body = ev.split("] ", 1)[-1]
        if grouped and grouped[-1]["msg"] == msg_body:
            grouped[-1]["count"] += 1
        else:
            grouped.append({"ev": ev, "msg": msg_body, "count": 1})
        if len(grouped) >= 15:
            break
    for g in grouped:
        ev = g["ev"]
        is_warn = "" in ev or "Rerouted" in ev or "flagged" in ev.lower()
        is_crit = "Hurricane" in ev or "ALERT" in ev or "WATCHLIST RE-ALERT" in ev
        cls = "crit" if is_crit else ("warn" if is_warn else "")
        count_badge = f'<span class="event-count">×{g["count"]}</span>' if g["count"] > 1 else ""
        st.markdown(
            f'<div class="event-item {cls}"><span>{ev}</span>{count_badge}</div>',
            unsafe_allow_html=True,
        )


def _render_single_explain_card(fv: dict, analyst: str, watched_ids: set, state: dict, key_prefix: str = "") -> None:
    vid = fv["vessel_id"]
    on_watch = vid in watched_ids
    conf = fv.get("confidence", 0)
    max_g = fv.get("max_gap_minutes", 0)
    disp = fv.get("displacement_error_km", 0)
    spd = fv.get("speed_change_after_gap", 0)
    hdg = fv.get("heading_change_after_gap", 0)
    expl = fv.get("explanation", "Flagged by anomaly detector.")
    decision = fv.get("if_decision", None)
    z = fv.get("feature_z") or {}
    z_html = ", ".join(f"{k}={v:.1f}" for k, v in z.items()) or "n/a"
    live_th = state.get("detection_thresholds") or {}
    gap_th = fv.get("flag_gap_threshold_min", live_th.get("gap_threshold_min", 45))
    dr_th = fv.get("flag_dr_threshold_km", live_th.get("dr_threshold_km", 8))
    note = fv.get("memory_note") or ""
    scan_c = fv.get("scan_confidence", conf)
    realert_badge = (
        '<span class="badge badge-real"> RE-ALERT</span>' if on_watch else ""
    )
    
    # Optional styling adjustments based on prefix to help differentiate search results vs flagged
    border_color = "#3b82f6" if key_prefix else "#f59e0b"
    
    route_info = orchestrator.get_vessel_origin_dest(vid)
    orig_str = "Unknown"
    dest_str = "Unknown"
    if route_info:
        o = route_info.get("origin") or {}
        d = route_info.get("dest") or {}
        if o:
            orig_str = f"{o.get('name', 'Unknown')} ({o.get('state', '')})"
        if d:
            dest_str = f"{d.get('name', 'Unknown')} ({d.get('state', '')})"

    html = (
        f"<div class='explain-box' style='border-left-color: {border_color}'>"
        f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:6px'>"
        f"<b>Vessel {vid}</b>"
        f"<span class='badge badge-syn'>Confidence {conf:.0%}</span>"
        f"{realert_badge}"
        "</div>"
        f"<p style='margin-bottom:8px'><b>Route:</b> {orig_str} ➔ {dest_str}</p>"
        f"<p>{expl}</p>"
        f"<p style='font-size:0.8rem'><b>Memory:</b> {note} "
        f"Scan-only score was {scan_c:.0%}.</p>"
        "<p style='font-size:0.75rem;color:#fde68a'>"
        f"Confidence is a logistic of the IsolationForest decision score"
        f"{'' if decision is None else f' (decision={decision})'}"
        " — not a min-max rank of this batch. "
        f"Rule threshold: gap ≥ {gap_th:.0f} min (Class A underway normally reports within ~3 min; "
        f"{gap_th:.0f} min allows coastal shadowing) and dead-reckoning error "
        f"≥ {dr_th:.0f} km (projected from last speed/heading across the silence).</p>"
        f"<div style='margin-top:8px;display:flex;flex-wrap:wrap;gap:14px;font-size:0.75rem'>"
        f"<span>Transponder gap: <span class='explain-metric'>{max_g} min</span></span>"
        f"<span>DR discrepancy: <span class='explain-metric'>{disp} km</span></span>"
        f"<span>Speed delta: <span class='explain-metric'>{spd} kts</span></span>"
        f"<span>Heading delta: <span class='explain-metric'>{hdg}°</span></span>"
        f"<span>Top z-scores: <span class='explain-metric'>{z_html}</span></span>"
        "</div></div>"
    )
    st.markdown(html, unsafe_allow_html=True)
    if analyst:
        if on_watch:
            st.button(
                " On watchlist",
                key=f"{key_prefix}wl_on_{vid}",
                on_click=_watchlist_remove,
                args=(vid,),
            )
        else:
            st.button(
                " Add to watchlist",
                key=f"{key_prefix}wl_add_{vid}",
                on_click=_watchlist_add,
                args=(vid,),
            )
    brief = fv.get("investigation")
    if not brief:
        for k, b in (state.get("llm_briefs") or {}).items():
            if k.startswith(str(vid) + "@"):
                brief = b
                break
    if brief:
        with st.expander(f"Investigator trace — {vid}", expanded=True):
            st.caption(f"Model: {brief.get('model')} · LLM={brief.get('used_llm')}")
            for call in brief.get("tool_calls") or []:
                st.markdown(f"**Tool:** `{call.get('name')}`")
                st.json({"input": call.get("input"), "output": call.get("output")})
            st.markdown(brief.get("summary") or "")
    else:
        if st.button(" Generate investigative brief for this vessel", key=f"{key_prefix}gen_brief_{vid}"):
            with st.spinner("Generating brief..."):
                orchestrator.generate_single_brief(vid)
            st.rerun()

def _render_flag_explain(state: dict) -> None:
    st.markdown("---")
    analyst = _current_analyst()
    watched_ids = {r["vessel_id"] for r in storage.get_watchlist(analyst)} if analyst else set()
    
    active_search = st.session_state.get("active_search_vessel")
    if active_search:
        st.markdown(f"#####  Searched Vessel — {active_search}")
        det_by_id = state.get("detection_by_id") or {}
        fv = det_by_id.get(active_search)
        if not fv:
            fv = {
                "vessel_id": active_search,
                "confidence": 0,
                "explanation": "Normal operating vessel (not flagged by dark-vessel anomaly detector).",
                "max_gap_minutes": 0,
                "displacement_error_km": 0,
            }
        _render_single_explain_card(fv, analyst, watched_ids, state, key_prefix="search_")
        st.markdown("<br/>", unsafe_allow_html=True)

    with st.expander(" Add to watchlist — flagged vessels this tick", expanded=True):
        flagged = state.get("flagged_vessels", [])
        if not flagged:
            st.success("No dark vessels flagged in current scan frame.")
            return
        st.markdown("#####  Flagged Dark Vessels — Anomaly Analysis")
        shown = sorted(flagged, key=lambda r: r.get("confidence", 0), reverse=True)[:12]
        for fv in shown:
            _render_single_explain_card(fv, analyst, watched_ids, state, key_prefix="")
            
        if len(flagged) > 12:
            st.caption(f"Showing top 12 of {len(flagged)} flagged vessels by confidence.")


def _render_watchlist_panel(state: dict) -> None:
    analyst = _current_analyst()
    if not analyst:
        return
    st.markdown("---")
    st.markdown(
        f'<div class="section-heading"> My Watchlist<span class="tag">{analyst}</span></div>',
        unsafe_allow_html=True,
    )
    items = storage.get_watchlist(analyst)
    flagged_ids = {r["vessel_id"] for r in (state.get("flagged_vessels") or [])}
    if not items:
        st.caption("No vessels on your watchlist yet. Use  Add to watchlist on a flagged vessel below.")
        return
    for row in items:
        vid = row["vessel_id"]
        flagged_now = vid in flagged_ids
        badge = " RE-ALERT — flagged this tick" if flagged_now else "not flagged this tick"
        st.markdown(f"**{vid}** · added {row['added_at']} UTC · {badge}")
        with st.form(key=f"wl_note_form_{vid}"):
            note_txt = st.text_input("Add a case note")
            save = st.form_submit_button("Save note")
            if save and (note_txt or "").strip():
                storage.add_note(analyst, vid, note_txt)
        for n in storage.get_notes(analyst, vid)[:8]:
            st.caption(f"{n['created_at']} UTC — {n['note_text']}")
        st.button(
            "Remove from watchlist",
            key=f"wl_rm_{vid}",
            on_click=_watchlist_remove,
            args=(vid,),
        )


def _maybe_log_watchlist_realerts(live: dict) -> None:
    analyst = _current_analyst()
    if not analyst:
        return
    watched = {r["vessel_id"] for r in storage.get_watchlist(analyst)}
    if not watched:
        return
    flagged_ids = {r["vessel_id"] for r in (live.get("flagged_vessels") or [])}
    tick = live.get("tick")
    last = st.session_state.setdefault("_wl_realert_tick", {})
    hits = [vid for vid in watched if vid in flagged_ids and last.get(vid) != tick]
    for vid in hits:
        last[vid] = tick
    if hits:
        shown = ", ".join(hits[:8])
        extra = f" (+{len(hits) - 8} more)" if len(hits) > 8 else ""
        orchestrator.append_event(f" WATCHLIST RE-ALERT: {shown}{extra} flagged again this tick")


@st.cache_data
def _static_gfw_geojson(zones: tuple) -> dict:
    """Hashable GFW polygons — rebuilt only when zone geometry changes."""
    features = []
    for name, gear, density, poly in zones:
        ring = [[lon, lat] for lat, lon in poly]
        if ring and ring[0] != ring[-1]:
            ring.append(ring[0])
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [ring]},
                "properties": {
                    "name": name,
                    "gear": gear,
                    "density": density,
                },
            }
        )
    return {"type": "FeatureCollection", "features": features}


@st.cache_data
def _static_anchor_ports() -> tuple:
    """Galveston origin + Mississippi dest — fixed corridor endpoints."""
    return (tuple(orchestrator.ROUTE_ORIGIN), tuple(orchestrator.ROUTE_DESTINATION))


def _gfw_zone_key(zones: list) -> tuple:
    return tuple(
        (
            z.get("name", ""),
            z.get("gear", ""),
            z.get("vessel_density", ""),
            tuple((float(p[0]), float(p[1])) for p in z.get("polygon") or []),
        )
        for z in zones
    )


def _build_live_folium_map(state: dict, choice: str):
    """Dynamic live-map layers on a canvas-backed Folium map."""
    focus_vid = st.session_state.get("search_focus_vessel")
    center = GULF_CENTER
    zoom = GULF_ZOOM
    if focus_vid:
        for v in state.get("vessel_positions", []):
            if v["vessel_id"] == focus_vid:
                center = (v["lat"], v["lon"])
                zoom = 10
                break
        st.session_state["search_focus_vessel"] = None

    folium_map = _new_gulf_map(choice, center=center, zoom=zoom)

    if st.session_state.show_gfw and state.get("fishing_zones"):
        gj = _static_gfw_geojson(_gfw_zone_key(state["fishing_zones"]))
        folium.GeoJson(
            gj,
            name="Global Fishing Watch Zones",
            style_function=lambda _f: {
                "color": "#10b981",
                "weight": 2,
                "fillColor": "#10b981",
                "fillOpacity": 0.22,
            },
            popup=folium.GeoJsonPopup(
                fields=["name", "gear", "density"],
                aliases=["Zone", "Gear", "Density"],
                max_width=250,
            ),
        ).add_to(folium_map)

    if state.get("storm_mode_active") and state.get("storm_eye_pos"):
        se = state["storm_eye_pos"]
        fg_storm = folium.FeatureGroup(name="Hurricane Ida (NOAA Cat 4)")
        folium.Circle(
            location=[se[0], se[1]],
            radius=140000,
            color="#ef4444",
            weight=2,
            fill=True,
            fill_color="#ef4444",
            fill_opacity=0.25,
            popup="Hurricane Ida 140km Gale Wind Radius",
        ).add_to(fg_storm)
        folium.Circle(
            location=[se[0], se[1]],
            radius=45000,
            color="#dc2626",
            weight=3,
            fill=True,
            fill_color="#b91c1c",
            fill_opacity=0.6,
            popup=f"<b>Hurricane Ida Cat 4 Eye</b><br/>Pos: {se[0]:.2f}°N, {abs(se[1]):.2f}°W<br/>Max Wind: 130 kts",
        ).add_to(fg_storm)
        fg_storm.add_to(folium_map)

    add_route_layers(
        folium_map,
        state.get("current_route"),
        state.get("baseline_route"),
        show_before_after=st.session_state.show_before_after,
    )

    for hs in state.get("debris_hotspots", []):
        c = hs["center"]
        folium.Circle(
            location=[c[0], c[1]],
            radius=12000 + hs["sighting_count"] * 4000,
            color="#f59e0b",
            weight=2,
            fill=True,
            fill_color="#fbbf24",
            fill_opacity=0.45,
            popup=folium.Popup(
                f"<b>Debris Hotspot {hs['hotspot_id']}</b><br/>"
                f"Sightings: {hs['sighting_count']}<br/>"
                f"Severity: {hs['severity']}<br/>"
                f"Assigned Collector: {hs.get('collector_assigned') or 'None'}<br/>"
                f"Cleanup ETA: {hs.get('eta', 'N/A')}",
                max_width=260,
            ),
        ).add_to(folium_map)

    hotspot_centers = {h["hotspot_id"]: h["center"] for h in state.get("debris_hotspots", [])}
    for col in state.get("collectors", []):
        clat, clon = col["lat"], col["lon"]
        folium.CircleMarker(
            location=[clat, clon],
            radius=7,
            color="#3b82f6",
            fill=True,
            fill_color="#60a5fa",
            fill_opacity=0.9,
            popup=(
                f"<b>Collector {col['collector_id']}</b><br/>Status: {col['status']}<br/>"
                f"Assigned Hotspot: {col.get('assigned_hotspot') or 'None'}"
            ),
        ).add_to(folium_map)
        hid = col.get("assigned_hotspot")
        if hid and hid in hotspot_centers:
            hc = hotspot_centers[hid]
            folium.PolyLine(
                locations=[[clat, clon], [hc[0], hc[1]]],
                color="#3b82f6",
                weight=2,
                dash_array="5, 5",
                opacity=0.7,
            ).add_to(folium_map)

    vessel_wakes = state.get("vessel_wakes", {})

    def make_ship_icon(heading_deg: float, color: str = "#10b981", size: int = 22):
        svg = (
            f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" '
            f'style="transform: rotate({heading_deg}deg); transform-origin: center;">'
            f'<path d="M12 2 L19 21 L12 17 L5 21 Z" fill="{color}" stroke="#ffffff" stroke-width="1.5"/>'
            f"</svg>"
        )
        return folium.DivIcon(
            html=f'<div style="width:{size}px;height:{size}px;display:flex;align-items:center;justify-content:center;">{svg}</div>',
            icon_size=(size, size),
            icon_anchor=(size // 2, size // 2),
        )

    active_features = []
    active_search_vid = st.session_state.get("active_search_vessel")
    for v in state.get("vessel_positions", []):
        vid = v["vessel_id"]
        is_flagged = v.get("flagged", False)
        vlat, vlon = v["lat"], v["lon"]
        heading = v.get("heading", 0)
        
        if vid == active_search_vid:
            folium.Marker(
                location=[vlat, vlon],
                icon=folium.Icon(color="purple", icon="star", prefix="fa"),
                popup=folium.Popup(f"<b>Vessel {vid}</b><br/>Search Target", max_width=280),
                z_index_offset=1000,
            ).add_to(folium_map)

        if not is_flagged:
            active_features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [vlon, vlat]},
                    "properties": {
                        "vessel_id": vid,
                        "vessel_type": v.get("vessel_type", "CARGO"),
                        "status": "Normal",
                        "speed": v.get("speed", 0),
                        "heading": heading,
                        "pos": f"{vlat:.4f}, {vlon:.4f}",
                    },
                }
            )
            continue
        w_pts = vessel_wakes.get(vid, [])
        if len(w_pts) >= 2:
            folium.PolyLine(
                locations=[[pt[0], pt[1]] for pt in w_pts],
                color="#ef4444",
                weight=2,
                opacity=0.65,
            ).add_to(folium_map)
        popup_html = (
            f"<div style='font-family:Inter,sans-serif;font-size:0.8rem;color:#0f172a'>"
            f"<b>Vessel {vid} ({v.get('vessel_type', 'CARGO')})</b><br/>"
            f"<b>Status:</b> FLAGGED DARK<br/>"
            f"<b>Speed:</b> {v.get('speed', 0)} kts | <b>Heading:</b> {heading} deg<br/>"
            f"<b>Pos:</b> {vlat:.4f}, {vlon:.4f}<br/>"
            f"<b>Confidence:</b> {v.get('confidence', 0):.0%} (logistic of IF decision)<br/>"
            f"<b>Gap:</b> {v.get('max_gap_minutes', 0)} min<br/>"
            f"<b>DR error:</b> {v.get('displacement_error_km', 0)} km<br/>"
            "</div>"
        )
        folium.Marker(
            location=[vlat, vlon],
            icon=make_ship_icon(heading, color="#ef4444", size=26),
            popup=folium.Popup(popup_html, max_width=280),
        ).add_to(folium_map)

    if active_features:
        folium.GeoJson(
            {"type": "FeatureCollection", "features": active_features},
            name="Active vessels",
            marker=folium.CircleMarker(
                radius=3,
                color="#16a34a",
                fill=True,
                fill_color="#22c55e",
                fill_opacity=0.75,
                weight=1,
            ),
            popup=folium.GeoJsonPopup(
                fields=["vessel_id", "vessel_type", "status", "speed", "heading", "pos"],
                aliases=["Vessel", "Type", "Status", "Speed (kts)", "Heading", "Pos"],
                max_width=260,
            ),
        ).add_to(folium_map)

    origin, dest = _static_anchor_ports()
    folium.Marker(
        location=list(origin),
        popup="<b>Galveston Entrance Channel Port</b><br/>Route Origin (29.30°N, 94.75°W)",
        icon=folium.Icon(color="green", icon="anchor", prefix="fa"),
    ).add_to(folium_map)
    folium.Marker(
        location=list(dest),
        popup="<b>Mississippi River South Pass Approach</b><br/>Route Destination (29.10°N, 89.50°W)",
        icon=folium.Icon(color="blue", icon="flag", prefix="fa"),
    ).add_to(folium_map)
    return folium_map


_LIVE_FOLIUM_CACHE: dict = {"key": None, "html": None}


def _live_map_cache_key(state: dict, choice: str) -> tuple:
    """Identity of the live Folium map — rebuild only when this changes."""
    return (
        state.get("tick"),
        choice,
        bool(st.session_state.get("show_gfw")),
        bool(st.session_state.get("show_before_after")),
        bool(state.get("storm_mode_active")),
        st.session_state.get("search_focus_vessel"),
        st.session_state.get("active_search_vessel"),
    )


def _get_cached_live_folium_map(state: dict, choice: str) -> tuple:
    """Reuse the last built Folium HTML when the tick (and display layers) are unchanged.

    The Folium object and rendered HTML stay in a process-level dict — never
    st.session_state — so fragment reruns (analyst name, watchlist clicks) do
    not protobuf-serialize ~1MB of map HTML.

    Returns (html, rebuilt: bool, build_ms: float).
    """
    key = _live_map_cache_key(state, choice)
    if _LIVE_FOLIUM_CACHE.get("key") == key and _LIVE_FOLIUM_CACHE.get("html"):
        st.session_state["_map_build_ms"] = 0.0
        st.session_state["_map_rebuilt"] = False
        return _LIVE_FOLIUM_CACHE["html"], False, 0.0
    t0 = time.perf_counter()
    fmap = _build_live_folium_map(state, choice)
    html = fmap.get_root().render()
    build_ms = (time.perf_counter() - t0) * 1000.0
    _LIVE_FOLIUM_CACHE["key"] = key
    _LIVE_FOLIUM_CACHE["html"] = html
    st.session_state["_map_build_ms"] = build_ms
    st.session_state["_map_rebuilt"] = True
    return html, True, build_ms


def _hide_parked_live_map() -> None:
    """Hide the parent-parked Leaflet iframe when Live Map is not the active view."""
    components.html(
        """
        <script>
        (function () {
          try {
            const park = window.parent.document.getElementById("masLiveMapPark");
            if (park) park.style.display = "none";
          } catch (e) {}
        })();
        </script>
        """,
        height=0,
        scrolling=False,
    )


def _show_live_map(map_html: str | None, rebuilt: bool, tick, build_ms: float) -> None:
    """Keep Leaflet in a parent-parked iframe so fragment reruns do not reparse 2,000 markers.

    Rebuild (new tick / layers): send Folium HTML once and assign park.srcdoc.
    Cache hit (e.g. analyst name Enter): send only a spacer + reposition script — no map HTML.
    """
    payload = ""
    if rebuilt and map_html:
        payload = base64.b64encode(map_html.encode("utf-8")).decode("ascii")
    components.html(
        f"""
<div id="mas-map-slot" style="height:780px;width:100%;background:#0b1220;border-radius:8px;"></div>
<script>
(function () {{
  const HEIGHT = 780;
  const PARK_ID = "masLiveMapPark";
  const payload = {json.dumps(payload)};
  function parentDoc() {{
    try {{ return window.parent.document; }} catch (e) {{ return null; }}
  }}
  function place(park) {{
    const slot = document.getElementById("mas-map-slot");
    const fe = window.frameElement;
    if (!slot || !fe) return;
    const fr = fe.getBoundingClientRect();
    const sr = slot.getBoundingClientRect();
    park.style.position = "fixed";
    park.style.left = (fr.left + sr.left) + "px";
    park.style.top = (fr.top + sr.top) + "px";
    park.style.width = Math.max(sr.width, 1) + "px";
    park.style.height = HEIGHT + "px";
    park.style.zIndex = "45";
    park.style.border = "0";
    park.style.display = "block";
    park.style.background = "#0b1220";
    park.style.borderRadius = "8px";
  }}
  const doc = parentDoc();
  if (!doc) return;
  let park = doc.getElementById(PARK_ID);
  if (!park) {{
    park = doc.createElement("iframe");
    park.id = PARK_ID;
    park.title = "Live operations map";
    doc.body.appendChild(park);
  }}
  if (payload) {{
    park.srcdoc = decodeURIComponent(escape(atob(payload)));
  }}
  place(park);
  const pwin = window.parent;
  pwin.addEventListener("resize", function () {{ place(park); }});
  pwin.addEventListener("scroll", function () {{ place(park); }}, true);
  const iv = setInterval(function () {{ place(park); }}, 200);
  setTimeout(function () {{ clearInterval(iv); }}, 5000);
}})();
</script>
        """,
        height=780,
        scrolling=False,
    )
    if rebuilt:
        st.caption(f"Map rebuilt in {build_ms:.0f} ms (tick {tick}).")
    else:
        st.caption(f"Map reused from cache (tick {tick}, 0 ms rebuild).")


@st.fragment(key="live_tick_map")
def _live_ops_panel(header_slot, kpi_slot, map_feed_slot=None, explain_slot=None, analyst_slot=None):
    """Tick + header/KPIs/map/feed/explain. Tables and timeline stay outside."""
    if analyst_slot is not None:
        with analyst_slot:
            st.caption("Analyst name and watchlist are directly under the live map.")
    if st.session_state.pop("_pending_tick", False):
        with st.spinner("Advancing simulation tick…"):
            orchestrator.tick()
            live = orchestrator.get_state()
            _maybe_log_watchlist_realerts(live)
    live = orchestrator.get_state()
    st.session_state["_frag_runs"] = st.session_state.get("_frag_runs", 0) + 1
    _frag_t0 = time.perf_counter()
    with header_slot:
        _render_header(live)
    with kpi_slot:
        _render_kpi_strip(live)
    if map_feed_slot is not None:
        with map_feed_slot:
            map_col, feed_col = st.columns([3.2, 1.3])
            with map_col:
                st.markdown('<div class="section-heading"> Live Operations Map</div>', unsafe_allow_html=True)
                
                with st.form("vessel_search_form"):
                    cols = st.columns([3, 1])
                    with cols[0]:
                        all_vids = sorted([v["vessel_id"] for v in live.get("vessel_positions", [])])
                        search_vid = st.selectbox("Search and center on vessel", [""] + all_vids, index=0)
                    with cols[1]:
                        st.markdown("<div style='margin-top: 28px'></div>", unsafe_allow_html=True)
                        submit_search = st.form_submit_button("Locate", use_container_width=True)
                        
                    if submit_search and search_vid:
                        st.session_state["search_focus_vessel"] = search_vid
                        st.session_state["active_search_vessel"] = search_vid
                        
                st.markdown(
                    """
                    <div class="legend-bar">
                      <span class="legend-item"><span class="legend-dot" style="background:#10b981"></span>Active vessel</span>
                      <span class="legend-item"><span class="legend-dot" style="background:#ef4444"></span>Flagged dark vessel</span>
                      <span class="legend-item"><span class="legend-dot" style="background:#fbbf24"></span>Debris hotspot</span>
                      <span class="legend-item"><span class="legend-dot" style="background:#3b82f6"></span>Collector ship</span>
                      <span class="legend-item"><span class="legend-line" style="border-color:#38bdf8"></span>Optimized route</span>
                      <span class="legend-item"><span class="legend-line dashed" style="border-color:#ef4444"></span>Naive baseline</span>
                      <span class="legend-item"><span class="legend-sq" style="background:#10b981;opacity:0.5"></span>GFW fishing zone</span>
                      <span class="legend-item"><span class="legend-dot" style="background:#dc2626"></span>Hurricane Ida eye</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                choice = st.session_state.get("map_style_choice", " Google Maps Satellite Hybrid")
                map_html, rebuilt, build_ms = _get_cached_live_folium_map(live, choice)
                if rebuilt:
                    with st.spinner("Rebuilding live map (~2,000 vessels)…"):
                        _show_live_map(map_html, True, live.get("tick"), build_ms)
                else:
                    # Do not remount a components.html iframe — that re-parses Leaflet
                    # and was ~10s even when Python rebuild was 0 ms. The parent-parked
                    # map iframe from the last rebuild stays on screen.
                    st.markdown(
                        '<div id="mas-map-slot" style="height:780px;width:100%;background:#0b1220;border-radius:8px;"></div>',
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        f"Map reused from cache (tick {live.get('tick')}, 0 ms rebuild). "
                        f"frag-runs {st.session_state.get('_frag_runs')} · "
                        f"main-script-runs {st.session_state.get('_main_script_runs')}"
                    )
                route = live.get("current_route") or {}
                if route:
                    bl_cost = route.get("baseline_cost", 0)
                    opt_cost = route.get("cost", 0)
                    sav = route.get("savings_pct", 0)
                    st.markdown(
                        f"""
                        <div class="route-bar">
                          <div> <b>Reference Corridor (Demo):</b> Houston/Galveston (29.3°N, 94.8°W) &nbsp;➔&nbsp; New Orleans (29.9°N, 90.1°W)</div>
                          <div><b>Naive:</b> <span style="color:#f87171">{bl_cost:.1f}</span> &nbsp;·&nbsp; <b>Optimized:</b> <span style="color:#38bdf8">{opt_cost:.1f}</span> &nbsp;·&nbsp; <b>Fuel saved:</b> <span style="color:#34d399;font-weight:700">{sav:.1f}%</span></div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            with feed_col:
                _render_event_feed(live)
    elif map_feed_slot is None:
        _hide_parked_live_map()
    if explain_slot is not None:
        with explain_slot:
            _render_analyst_watchlist_bar()
            _render_watchlist_panel(live)
            _render_flag_explain(live)
    _frag_ms = (time.perf_counter() - _frag_t0) * 1000.0
    try:
        with open(
            os.path.join(os.path.dirname(__file__), ".frag_profile.txt"),
            "a",
            encoding="utf-8",
        ) as _pf:
            _pf.write(
                f"frag={st.session_state.get('_frag_runs')} python_ms={_frag_ms:.0f} "
                f"rebuilt={st.session_state.get('_map_rebuilt')} "
                f"analyst={_current_analyst()!r}\n"
            )
    except OSError:
        pass


def _reference_points_map(df: pd.DataFrame, choice: str, color: str, kind: str):
    fmap = _new_gulf_map(choice)
    for rec in df.to_dict("records"):
        if rec.get("lat") is None or rec.get("lon") is None or pd.isna(rec.get("lat")) or pd.isna(rec.get("lon")):
            continue
        lat, lon = float(rec["lat"]), float(rec["lon"])
        name = rec.get("name", "")
        extra = ""
        if rec.get("state"):
            extra += f"<br/>State: {rec['state']}"
        if rec.get("type"):
            extra += f"<br/>Type: {rec['type']}"
        if rec.get("hq_port"):
            extra += f"<br/>HQ port: {rec['hq_port']}"
        if rec.get("fleet_size") is not None and not (
            isinstance(rec.get("fleet_size"), float) and pd.isna(rec.get("fleet_size"))
        ):
            extra += f"<br/>Fleet size: {rec['fleet_size']}"
        popup = (
            f"<b>{name}</b><br/>{kind}<br/>"
            f"{lat:.4f}°N, {abs(lon):.4f}°W{extra}"
        )
        folium.CircleMarker(
            location=[lat, lon],
            radius=7,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            popup=folium.Popup(popup, max_width=260),
            tooltip=name,
        ).add_to(fmap)
    return fmap


def _render_reference_page(title: str, df: pd.DataFrame, kind: str, color: str, filename: str, caption: str | None = None):
    st.markdown(f'<div class="section-heading">{title}</div>', unsafe_allow_html=True)
    if caption:
        st.caption(caption)
    shown = df.sort_values(df.columns[0]).reset_index(drop=True)
    st.download_button(
        label="Download CSV",
        data=shown.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        key=f"download_{kind}_csv",
    )
    st.dataframe(shown, width="stretch", hide_index=True)
    choice = st.session_state.get("map_style_choice", " Google Maps Satellite Hybrid")
    st_folium(
        _reference_points_map(shown.dropna(subset=["lat", "lon"]), choice, color, kind),
        use_container_width=True,
        height=420,
        key=f"ref_map_{kind}",
    )


def _render_route_planner(state: dict) -> None:
    """Origin/destination search that calls Agent 1 get_route() and reuses add_route_layers."""
    st.markdown('<div class="section-heading"> Route Planner</div>', unsafe_allow_html=True)
    st.caption("Select two Gulf Coast ports, then compute an A* route with the same Agent 1 path used on the Live Map.")

    ports = _ports_ref()
    names = ports["name"].tolist()
    coords = {
        str(rec["name"]): (float(rec["lat"]), float(rec["lon"]))
        for rec in ports.to_dict("records")
    }

    with st.form("route_planner_form"):
        col_o, col_d = st.columns(2)
        with col_o:
            origin_name = st.selectbox("Origin", names, key="planner_origin")
        with col_d:
            dest_name = st.selectbox("Destination", names, key="planner_dest")
        submitted = st.form_submit_button("Compute Route", type="primary")

    if submitted:
        if origin_name == dest_name:
            st.warning("Origin and destination are the same port. Choose two different ports to compute a route.")
        else:
            origin = coords[origin_name]
            dest = coords[dest_name]
            with st.spinner(f"Computing route {origin_name} → {dest_name}…"):
                computed = get_route(origin, dest, state.get("weather_df"))
            st.session_state.planner_route = computed
            st.session_state.planner_baseline = {"waypoints": [list(origin), list(dest)]}
            st.session_state.planner_labels = (origin_name, dest_name)

    route = st.session_state.get("planner_route")
    if not route:
        st.info("Pick an origin and destination, then click Compute Route.")
        return

    m1, m2, m3 = st.columns(3)
    m1.metric("Route cost", f"{route.get('cost', 0):.1f}")
    m2.metric("Baseline cost", f"{route.get('baseline_cost', 0):.1f}")
    m3.metric("Savings", f"{route.get('savings_pct', 0):.1f}%")

    labels = st.session_state.get("planner_labels") or (
        st.session_state.get("planner_origin"),
        st.session_state.get("planner_dest"),
    )
    origin_name, dest_name = labels[0], labels[1]
    origin = coords.get(origin_name)
    dest = coords.get(dest_name)

    choice = st.session_state.get("map_style_choice", " Google Maps Satellite Hybrid")
    fmap = _new_gulf_map(choice)
    add_route_layers(
        fmap,
        route,
        st.session_state.get("planner_baseline"),
        show_before_after=True,
    )
    if origin:
        folium.Marker(
            location=list(origin),
            popup=f"<b>{origin_name}</b><br/>Origin<br/>{origin[0]:.4f}°N, {abs(origin[1]):.4f}°W",
            icon=folium.Icon(color="green", icon="anchor", prefix="fa"),
        ).add_to(fmap)
    if dest:
        folium.Marker(
            location=list(dest),
            popup=f"<b>{dest_name}</b><br/>Destination<br/>{dest[0]:.4f}°N, {abs(dest[1]):.4f}°W",
            icon=folium.Icon(color="blue", icon="flag", prefix="fa"),
        ).add_to(fmap)

    st_folium(fmap, use_container_width=True, height=480, key="planner_map")

if not st.session_state.initialised:
    with st.spinner(" Initialising MaritimeMAS Engine — fetching Open-Meteo & NOAA data..."):
        orchestrator.initialise()
    st.session_state.initialised = True

# ─────────────────────────────────────────────────────────────────────────── #
#  Header Bar                                                                 #
# ─────────────────────────────────────────────────────────────────────────── #
header_slot = st.container()
kpi_slot = st.container()

state = orchestrator.get_state()

# ─────────────────────────────────────────────────────────────────────────── #
#  Sidebar Controls                                                           #
# ─────────────────────────────────────────────────────────────────────────── #
with st.sidebar:
    analyst_slot = st.container()

    st.markdown('<div class="section-heading"> Simulation Controls</div>', unsafe_allow_html=True)

    col_play1, col_play2 = st.columns(2)
    with col_play1:
        tick_click = st.button(" Next Tick", width="stretch", on_click=_request_live_tick)
    with col_play2:
        play_click = st.button(
            " Pause" if st.session_state.auto_play else " Auto-Play",
            type="primary" if st.session_state.auto_play else "secondary",
            width='stretch',
        )

    if play_click:
        st.session_state.auto_play = not st.session_state.auto_play

    st.session_state.playback_speed = st.slider("Playback speed (sec / tick)", 1, 6, 2)
    if st.session_state.auto_play:
        st.caption(" Auto-play running — the simulation advances automatically.")

    st.caption("Detection thresholds (Dark Vessel Agent rule)")
    if "gap_threshold_min" not in st.session_state:
        st.session_state.gap_threshold_min = int((state.get("detection_thresholds") or {}).get("gap_threshold_min", 45))
    if "dr_threshold_km" not in st.session_state:
        st.session_state.dr_threshold_km = int((state.get("detection_thresholds") or {}).get("dr_threshold_km", 8))
    gap_slider = st.slider(
        "gap_threshold_min (minutes)",
        min_value=15,
        max_value=120,
        help="IMO-style AIS silence cutoff used by the rule flag.",
        key="gap_threshold_min",
    )
    dr_slider = st.slider(
        "dr_threshold_km",
        min_value=2,
        max_value=30,
        help="Dead-reckoning disagreement needed to confirm a gap as a course-change-while-dark.",
        key="dr_threshold_km",
    )
    orchestrator.set_detection_thresholds(float(gap_slider), float(dr_slider))

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown('<div class="section-heading"> Special Demo Modes</div>', unsafe_allow_html=True)

    storm_active = state.get("storm_mode_active", False)
    if st.button(
        " Disable Storm Replay" if storm_active else " Replay Hurricane Ida (Cat 4)",
        type="primary" if not storm_active else "secondary",
        width='stretch',
        help="Replays NOAA HURDAT2 historical track data for Hurricane Ida (Aug 2021).",
    ):
        orchestrator.toggle_storm_mode()
        st.rerun()
    if storm_active:
        st.caption(" Storm mode active — Agent 1 is dynamically routing around the Cat 4 eye.")

    if st.button(
        " Guided Demo",
        width="stretch",
        help="Advances the live simulation until a flagged vessel, a debris assignment, and a reroute are all visible.",
    ):
        st.session_state.guided_demo_result = orchestrator.guided_demo()
        st.rerun()
    demo = st.session_state.get("guided_demo_result")
    if demo:
        if demo.get("ok"):
            st.caption(
                f"Guided Demo finished in {demo.get('ticks')} tick(s): "
                "flag  · collector assignment  · reroute "
            )
        else:
            missing = ", ".join(k for k, v in (demo.get("seen") or {}).items() if not v) or "unknown"
            st.warning(f"Guided Demo timed out after {demo.get('ticks')} ticks. Still missing: {missing}.")

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown('<div class="section-heading"> Data Authenticity<span class="tag">source map</span></div>', unsafe_allow_html=True)
    rows_html = ""
    for k, v in state.get("data_status", {}).items():
        is_real = "REAL" in v.upper()
        b_class = "badge-real" if is_real else "badge-syn"
        rows_html += f"""
        <div class="data-row">
          <span class="badge {b_class}">{"REAL" if is_real else "SYNTHETIC"}</span>
          <span class="data-row-text"><b>{k}</b><br/>{v}</span>
        </div>
        """
    st.markdown(rows_html, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown('<div class="section-heading"> Directory</div>', unsafe_allow_html=True)
    nav = st.radio("Workspace", NAV_OPTIONS, key="nav_view")

    if nav == NAV_LIVE:
        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown('<div class="section-heading"> Map & Layers</div>', unsafe_allow_html=True)
        st.session_state.map_style_choice = st.selectbox(
            "Basemap",
            [
                " CartoDB Dark Matter",
                " Google Maps Satellite Hybrid",
                " Esri World Imagery (High-Res)",
                " OpenStreetMap Marine View",
            ],
        )
        st.session_state.show_gfw = st.checkbox("Show Global Fishing Watch zones", value=st.session_state.show_gfw)
        st.session_state.show_before_after = st.checkbox("Show naive vs. optimized route", value=st.session_state.show_before_after)

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown('<div class="section-heading"> Live Operations Metrics</div>', unsafe_allow_html=True)

    m = state["metrics"]

    st.markdown(
        f"""
        <div class="metric-grid">
          <div class="metric-card">
            <div class="metric-label"> Fuel Savings</div>
            <div class="metric-value">{m['fuel_savings_pct']:.1f}%</div>
            <div class="metric-sub">vs. naive route</div>
          </div>
          <div class="metric-card">
            <div class="metric-label"> Flagged</div>
            <div class="metric-value">{m['vessels_flagged']}</div>
            <div class="metric-sub">P {m['precision']:.0%} · R {m['recall']:.0%}</div>
          </div>
          <div class="metric-card">
            <div class="metric-label"> Hotspots</div>
            <div class="metric-value">{m['hotspots_covered']}</div>
            <div class="metric-sub">covered by collectors</div>
          </div>
          <div class="metric-card">
            <div class="metric-label"> Reroutes</div>
            <div class="metric-value">{m['reroute_count']}</div>
            <div class="metric-sub">cross-agent triggers</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────── #
#  Handle Simulation Advance                                                   #
# ─────────────────────────────────────────────────────────────────────────── #
if st.session_state.auto_play:
    with st.spinner("Auto-play: advancing tick and rebuilding the live map…"):
        orchestrator.tick()
        time.sleep(st.session_state.playback_speed)
    st.rerun()

state = orchestrator.get_state()

map_feed_slot = None
explain_slot = None
if nav == NAV_LIVE:
    map_feed_slot = st.container()
    explain_slot = st.container()
    if tick_click and not st.session_state.pop("_tick_via_callback", False):
        st.session_state["_pending_tick"] = True

    # ─────────────────────────────────────────────────────────────────────────── #
    #  Tabbed data view (same orchestrator state as the map)                      #
    # ─────────────────────────────────────────────────────────────────────────── #

    def _fmt_latlon(value) -> str:

        if isinstance(value, (list, tuple)) and len(value) >= 2:

            return f"{value[0]}, {value[1]}"

        return ""

    def _vessels_df(state: dict) -> pd.DataFrame:

        src = (state.get("data_status") or {}).get("AIS", "")

        rows = [

            {

                "vessel_id": v.get("vessel_id"),

                "lat": v.get("lat"),

                "lon": v.get("lon"),

                "speed": v.get("speed"),

                "heading": v.get("heading"),

                "data_source": src,

            }

            for v in state.get("vessel_positions", [])

        ]

        df = pd.DataFrame(rows, columns=["vessel_id", "lat", "lon", "speed", "heading", "data_source"])

        if df.empty:

            return df

        return df.sort_values("vessel_id", kind="mergesort").reset_index(drop=True)

    def _flagged_df(state: dict) -> pd.DataFrame:

        src = (state.get("data_status") or {}).get("AIS", "")

        rows = []

        for fv in state.get("flagged_vessels", []):

            gap = fv.get("gap_start_time", "")

            rows.append(

                {

                    "vessel_id": fv.get("vessel_id"),

                    "confidence": fv.get("confidence"),

                    "last_known_position": _fmt_latlon(fv.get("last_known_position")),

                    "gap_start_time": gap if gap not in (None, "") else "",

                    "data_source": src,

                }

            )

        df = pd.DataFrame(

            rows,

            columns=["vessel_id", "confidence", "last_known_position", "gap_start_time", "data_source"],

        )

        if df.empty:

            return df

        return df.sort_values("confidence", ascending=False, kind="mergesort").reset_index(drop=True)

    def _hotspots_df(state: dict) -> pd.DataFrame:

        rows = [

            {

                "hotspot_id": hs.get("hotspot_id"),

                "center": _fmt_latlon(hs.get("center")),

                "sighting_count": hs.get("sighting_count"),

                "collector_assigned": hs.get("collector_assigned") or "",

                "status": hs.get("status"),

            }

            for hs in state.get("debris_hotspots", [])

        ]

        df = pd.DataFrame(

            rows,

            columns=["hotspot_id", "center", "sighting_count", "collector_assigned", "status"],

        )

        if df.empty:

            return df

        return df.sort_values("sighting_count", ascending=False, kind="mergesort").reset_index(drop=True)

    def _collectors_df(state: dict) -> pd.DataFrame:

        rows = []

        for col in state.get("collectors", []):

            assigned = col.get("assigned_hotspot")

            rows.append(

                {

                    "collector_id": col.get("collector_id"),

                    "current_position": _fmt_latlon([col.get("lat"), col.get("lon")]),

                    "assigned_hotspot_id": assigned if assigned else "",

                    "status": col.get("status"),

                }

            )

        df = pd.DataFrame(

            rows,

            columns=["collector_id", "current_position", "assigned_hotspot_id", "status"],

        )

        if df.empty:

            return df

        return df.sort_values("collector_id", kind="mergesort").reset_index(drop=True)

    def _render_data_tab(df: pd.DataFrame, *, filename: str, empty_msg: str, download_key: str) -> None:

        if df.empty:

            st.info(empty_msg)

            return

        st.download_button(

            label="Download CSV",

            data=df.to_csv(index=False).encode("utf-8"),

            file_name=filename,

            mime="text/csv",

            key=download_key,

        )

        st.dataframe(df, width="stretch")

    st.markdown("---")

    st.markdown('<div class="section-heading">📋 Operations data<span class="tag">live tables</span></div>', unsafe_allow_html=True)

    _tick = state["tick"]

    tab_vessels, tab_flagged, tab_hotspots, tab_collectors = st.tabs(

        ["Vessels", "Flagged Vessels", "Debris Hotspots", "Collectors"]

    )

    with tab_vessels:

        _render_data_tab(

            _vessels_df(state),

            filename=f"vessels_tick_{_tick}.csv",

            empty_msg="No vessels yet — advance the simulation.",

            download_key="download_vessels_csv",

        )

    with tab_flagged:

        _render_data_tab(

            _flagged_df(state),

            filename=f"flagged_vessels_tick_{_tick}.csv",

            empty_msg="No flagged vessels yet — advance the simulation.",

            download_key="download_flagged_csv",

        )

    with tab_hotspots:

        _render_data_tab(

            _hotspots_df(state),

            filename=f"debris_hotspots_tick_{_tick}.csv",

            empty_msg="No debris hotspots yet — advance the simulation.",

            download_key="download_hotspots_csv",

        )

    with tab_collectors:

        _render_data_tab(

            _collectors_df(state),

            filename=f"collectors_tick_{_tick}.csv",

            empty_msg="No collectors yet — advance the simulation.",

            download_key="download_collectors_csv",

        )

    # ─────────────────────────────────────────────────────────────────────────── #

    #  Timeline Scrubber (Past State Playback)                                     #

    # ─────────────────────────────────────────────────────────────────────────── #

    st.markdown("---")

    st.markdown('<div class="section-heading">⏱️ Simulation Timeline Scrubber</div>', unsafe_allow_html=True)

    history = orchestrator.get_tick_history()

    if history:

        max_t = len(history) - 1

        if max_t > 0:

            selected_t = st.slider("Scrub timeline to replay past state snapshots", 0, max_t, max_t)

            if selected_t != max_t:

                snap = history[selected_t]

                st.info(f"⏪ Replaying state from **Tick #{snap['tick']}** — {len(snap['vessel_positions'])} vessels tracked. Move the slider to the far right to return to live.")

        else:

            st.caption("Advance the simulation to build timeline history.")

    else:

        st.caption("No snapshots yet — advance the simulation to build timeline history.")

    tab_v, tab_d, tab_s = st.tabs(["🛳️ Tracked Vessels Table", " Debris & Collector Status", " Storm & Weather Data"])
    with tab_v:
        if state["vessel_positions"]:
            vdf = pd.DataFrame(state["vessel_positions"])
            st.dataframe(vdf, width='stretch', height=260)
    with tab_d:
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.subheader("Debris Hotspots (DBSCAN Clustered)")
            if state["debris_hotspots"]:
                st.dataframe(pd.DataFrame(state["debris_hotspots"]), width='stretch')
        with col_d2:
            st.subheader("Collector Fleet Status")
            if state["collectors"]:
                st.dataframe(pd.DataFrame(state["collectors"]), width='stretch')
    with tab_s:
        if state.get("storm_df") is not None:
            st.subheader("NOAA HURDAT2 Historical Storm Track (Hurricane Ida, Aug 2021)")
            st.dataframe(state["storm_df"], width='stretch')

elif nav == NAV_PORTS:
    _render_reference_page(
        ' Ports',
        _ports_ref(),
        'port',
        '#38bdf8',
        'gulf_ports.csv',
        'Real named Gulf Coast ports with published coordinates. Static local reference — not simulated.',
    )
elif nav == NAV_LIGHTHOUSES:
    _render_reference_page(
        '💡 Lighthouses',
        _lighthouses_ref(),
        'lighthouse',
        '#fbbf24',
        'gulf_lighthouses.csv',
        'Historic Gulf Coast lights. Static local reference — not simulated.',
    )
elif nav == NAV_COMPANIES:
    ports = _ports_ref()
    companies = _companies_ref().merge(
        ports.rename(columns={'name': 'hq_port', 'lat': 'lat', 'lon': 'lon'})[['hq_port', 'lat', 'lon']],
        on='hq_port',
        how='left',
    )
    _render_reference_page(
        '🏢 Companies',
        companies,
        'company',
        '#a78bfa',
        'gulf_companies.csv',
        'Illustrative shipping companies for demo only — not a live commercial registry.',
    )
elif nav == NAV_PLANNER:
    _render_route_planner(state)

_live_ops_panel(header_slot, kpi_slot, map_feed_slot, explain_slot, analyst_slot)

zoom = st.session_state.get("map_zoom", GULF_ZOOM)
click = st.session_state.get("map_last_click") or {}
clat = click.get("lat")
clon = click.get("lng")
click_txt = f"{clat:.5f}, {clon:.5f}" if clat is not None and clon is not None else "click map for a point"
st.caption(
    f"`scale bar on map` · zoom **{zoom}** · last click `{click_txt}` · "
    f"cursor lat/lon is the Leaflet readout at bottom-left of the map · "
    f"main-script-runs {st.session_state.get('_main_script_runs')}"
)

st.markdown(
    """
    <div class="mas-footer">
      MaritimeMAS · Weather via Open-Meteo Marine API · Storm track via NOAA HURDAT2 ·
      Fishing zones via Global Fishing Watch · AIS &amp; debris data synthetic for demo purposes
    </div>
    """,
    unsafe_allow_html=True,
)