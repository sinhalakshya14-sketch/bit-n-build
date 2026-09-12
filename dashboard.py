
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
from dotenv import load_dotenv
load_dotenv(override=True)

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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: #060a12; color: #e2e8f0; }

    /* Give the custom header room to breathe below Streamlit's own top bar
       so it never collides with the "Connecting…/Deploy" toolbar. */
    .block-container { padding-top: 0.6rem !important; padding-bottom: 0.4rem; max-width: 100% !important; padding-left: 0.6rem; padding-right: 0.6rem; }
    #MainMenu, footer { visibility: hidden; }

    h1, h2, h3, h4 { letter-spacing: -0.01em; }
    hr { border-color: rgba(148, 163, 184, 0.12); margin: 1.4rem 0; }

    /* ── Header Banner ────────────────────────────────────────────────── */
    .mas-header {
        background: linear-gradient(135deg, rgba(13, 27, 62, 0.97) 0%, rgba(15, 35, 75, 0.97) 50%, rgba(10, 45, 90, 0.97) 100%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 14px;
        padding: 18px 26px;
        margin-bottom: 18px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 10px;
        box-shadow: 0 10px 36px rgba(0, 90, 220, 0.18);
    }
    .mas-title-group { display: flex; align-items: center; gap: 14px; }
    .mas-title-group .mas-icon {
        width: 44px; height: 44px; border-radius: 10px; flex-shrink: 0;
        background: linear-gradient(135deg, #0ea5e9, #1d4ed8);
        display: flex; align-items: center; justify-content: center;
        font-size: 1.35rem; box-shadow: 0 4px 14px rgba(14,165,233,0.4);
    }
    .mas-title-group h1 { color: #f8fafc; font-size: 1.4rem; font-weight: 700; margin: 0; }
    .mas-title-group p  { color: #93a5c2; font-size: 0.8rem; margin: 3px 0 0 0; }
    .mas-header-right { display: flex; align-items: center; gap: 10px; }
    .mas-tick-pill {
        font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #cbd5e1;
        background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.1);
        padding: 4px 10px; border-radius: 8px;
    }
    .pulse-dot {
        width: 8px; height: 8px; border-radius: 50%; background: #34d399; display: inline-block;
        margin-right: 6px; box-shadow: 0 0 0 rgba(52,211,153,0.6); animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%   { box-shadow: 0 0 0 0 rgba(52,211,153,0.55); }
        70%  { box-shadow: 0 0 0 7px rgba(52,211,153,0); }
        100% { box-shadow: 0 0 0 0 rgba(52,211,153,0); }
    }

    /* ── KPI Strip (top of main content) ─────────────────────────────── */
    .kpi-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 18px; }
    .kpi-card {
        background: rgba(13, 27, 56, 0.75);
        border: 1px solid rgba(56, 189, 248, 0.16);
        border-radius: 12px;
        padding: 14px 18px;
        transition: border-color 0.15s ease, transform 0.15s ease;
    }
    .kpi-card:hover { border-color: rgba(56,189,248,0.5); transform: translateY(-2px); }
    .kpi-label { color: #8ea2c2; font-size: 0.68rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.07em; display:flex; align-items:center; gap:6px; }
    .kpi-value { color: #f1f5f9; font-size: 1.7rem; font-weight: 700; margin-top: 4px; line-height: 1.2; }
    .kpi-value.accent-blue  { color: #38bdf8; }
    .kpi-value.accent-red   { color: #f87171; }
    .kpi-value.accent-amber { color: #fbbf24; }
    .kpi-value.accent-green { color: #34d399; }
    .kpi-sub { color: #64748b; font-size: 0.7rem; margin-top: 3px; }

    /* ── Sidebar Metric Cards (2-col grid) ───────────────────────────── */
    .metric-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .metric-card {
        background: rgba(13, 27, 56, 0.85);
        border: 1px solid rgba(30, 80, 160, 0.4);
        border-radius: 10px;
        padding: 10px 12px;
        box-shadow: 0 4px 16px rgba(0, 40, 100, 0.15);
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .metric-card:hover { border-color: #3b82f6; transform: translateY(-1px); }
    .metric-label { color: #94a3b8; font-size: 0.63rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; }
    .metric-value { color: #38bdf8; font-size: 1.25rem; font-weight: 700; line-height: 1.25; margin-top: 2px; }
    .metric-sub   { color: #64748b; font-size: 0.62rem; margin-top: 2px; line-height: 1.3; }

    /* ── Section Headings ────────────────────────────────────────────── */
    .section-heading {
        display: flex; align-items: center; gap: 8px;
        color: #e2e8f0; font-size: 0.95rem; font-weight: 700;
        margin: 4px 0 10px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(148, 163, 184, 0.14);
    }
    .section-heading .tag {
        font-size: 0.62rem; font-weight: 600; color: #64748b;
        background: rgba(148,163,184,0.08); padding: 2px 8px; border-radius: 6px;
        text-transform: uppercase; letter-spacing: 0.05em;
    }

    /* ── Map Legend ───────────────────────────────────────────────────── */
    .legend-bar {
        display: flex; gap: 18px; flex-wrap: wrap; align-items: center;
        margin-bottom: 8px; padding: 8px 16px;
        background: rgba(13,27,56,0.6); border: 1px solid rgba(56,189,248,0.12);
        border-radius: 8px;
    }
    .legend-item { display: flex; align-items: center; gap: 6px; font-size: 0.74rem; color: #cbd5e1; }
    .legend-dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }
    .legend-line { width: 18px; height: 0; display: inline-block; border-top: 3px solid; }
    .legend-line.dashed { border-top-style: dashed; }
    .legend-sq { width: 10px; height: 10px; display: inline-block; border-radius: 2px; }

    /* ── Route Comparison Bar ────────────────────────────────────────── */
    .route-bar {
        background: rgba(13,27,56,0.85); border: 1px solid rgba(59,130,246,0.28);
        border-radius: 10px; padding: 12px 18px; margin-top: 8px;
        display: flex; justify-content: space-between; align-items: center;
        flex-wrap: wrap; gap: 8px; font-size: 0.8rem; color: #cbd5e1;
    }
    .route-bar b { color: #e2e8f0; }

    /* ── Explainability Panel ────────────────────────────────────────── */
    .explain-box {
        background: rgba(26, 18, 9, 0.9);
        border-left: 4px solid #f59e0b;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin-bottom: 10px;
        font-size: 0.82rem;
        color: #fef3c7;
    }
    .explain-metric { font-family: 'JetBrains Mono', monospace; color: #fbbf24; font-weight: 600; }

    /* ── Event Feed ───────────────────────────────────────────────────── */
    .event-item {
        background: rgba(13, 27, 56, 0.75);
        border-left: 3px solid #3b82f6;
        border-radius: 0 6px 6px 0;
        padding: 8px 12px;
        margin-bottom: 6px;
        font-size: 0.78rem;
        color: #cbd5e1;
        line-height: 1.4;
        display: flex; justify-content: space-between; align-items: flex-start; gap: 8px;
    }
    .event-item.warn { border-left-color: #f59e0b; color: #fde68a; background: rgba(35, 25, 10, 0.75); }
    .event-item.crit { border-left-color: #ef4444; color: #fca5a5; background: rgba(40, 15, 15, 0.75); }
    .event-count {
        flex-shrink: 0; font-size: 0.65rem; font-weight: 700; color: #0f172a;
        background: #94a3b8; border-radius: 8px; padding: 1px 7px; white-space: nowrap;
    }

    /* ── Status Badges ────────────────────────────────────────────────── */
    .badge {
        display: inline-block; border-radius: 12px;
        padding: 2px 10px; font-size: 0.68rem; font-weight: 600;
        letter-spacing: 0.04em;
    }
    .badge-real { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }
    .badge-syn  { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }

    /* ── Data Authenticity Rows ───────────────────────────────────────── */
    .data-row {
        display: flex; align-items: flex-start; gap: 8px;
        padding: 6px 0; border-bottom: 1px solid rgba(148,163,184,0.08);
    }
    .data-row:last-child { border-bottom: none; }
    .data-row-text { font-size: 0.72rem; color: #94a3b8; line-height: 1.35; }
    .data-row-text b { color: #cbd5e1; font-size: 0.76rem; }

    /* Overlay sidebar: does not shrink the map. Collapsed by default; hover/click expands. */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1220 0%, #101a30 100%) !important;
        border-right: 1px solid rgba(56, 189, 248, 0.35);
        z-index: 400;
    }
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] .stMarkdown {
        color: #e2e8f0 !important;
        opacity: 1 !important;
    }
    [data-testid="stSidebar"] .stButton button {
        border-radius: 8px; font-weight: 600; font-size: 0.82rem;
        color: #f8fafc !important;
    }
    div[data-testid="stAppViewContainer"] > section.main {
        margin-left: 0 !important;
    }
    [data-testid="stBottom"] {
        background: rgba(8, 14, 28, 0.92);
        border-top: 1px solid rgba(56, 189, 248, 0.25);
        color: #cbd5e1;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
    }

    /* Footer */
    .mas-footer {
        text-align: center; color: #4b5c78; font-size: 0.7rem; padding: 18px 0 4px 0;
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
    st.session_state.show_debris_zones = True
    st.session_state.show_before_after = True
    st.session_state.map_style_choice = "🌌 Esri Dark Nautical Canvas"
    st.session_state.planner_route = None
    st.session_state.planner_baseline = None
    st.session_state.planner_labels = None
    st.session_state.planner_origin = "Port of Houston"
    st.session_state.planner_dest = "Port of Tampa"
    st.session_state.nav_view = NAV_LIVE
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
    elif "Imagery" in choice:
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri World Imagery",
            name="Esri Satellite",
            overlay=False,
            control=True,
        ).add_to(folium_map)
    elif "Dark" in choice:
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
            attr="Esri World Dark Gray Canvas",
            name="Esri Dark Nautical Canvas",
            overlay=False,
            control=True,
        ).add_to(folium_map)
    elif "Ocean" in choice:
        folium.TileLayer(
            tiles="https://services.arcgisonline.com/arcgis/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}",
            attr="Esri Ocean Basemap",
            name="Esri Ocean Bathymetry",
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


def add_route_layers(folium_map, route: dict | None, baseline_route=None, show_before_after: bool = True, candidate_routes: dict | None = None):
    """Shared route overlay with Candidate A (Fuel-optimal), Candidate B (Risk-avoiding), and Candidate C (Balanced)."""
    if show_before_after and baseline_route:
        bl_wps = baseline_route.get("waypoints", []) if isinstance(baseline_route, dict) else baseline_route
        if bl_wps and len(bl_wps) >= 2:
            folium.PolyLine(
                locations=[[wp[0], wp[1]] for wp in bl_wps],
                color="#ef4444",
                weight=3,
                dash_array="8, 8",
                opacity=0.75,
                popup="Naive Baseline Corridor (Unoptimized straight path)",
            ).add_to(folium_map)

    cand_dict = candidate_routes or {}
    cand_a = cand_dict.get("candidate_a")
    cand_b = cand_dict.get("candidate_b")
    cand_c = cand_dict.get("candidate_c")
    chosen_id = cand_dict.get("chosen_id", (route or {}).get("chosen_candidate", "A"))

    if cand_a and cand_b:
        is_a_chosen = (chosen_id == "A")
        is_b_chosen = (chosen_id == "B")
        is_c_chosen = (chosen_id == "C")

        # Candidate A (Fuel-optimal)
        folium.PolyLine(
            locations=[[wp[0], wp[1]] for wp in cand_a["waypoints"]],
            color="#38bdf8",
            weight=5 if is_a_chosen else 3,
            dash_array=None if is_a_chosen else "6, 6",
            opacity=0.95 if is_a_chosen else 0.70,
            popup=f"Candidate A (Fuel-Optimal): Cost {cand_a['cost']:.1f}, Fuel Saved {cand_a['savings_pct']:.1f}%" + (" [CHOSEN ROUTE]" if is_a_chosen else " [EVALUATED]"),
        ).add_to(folium_map)

        # Candidate C (Balanced)
        if cand_c:
            folium.PolyLine(
                locations=[[wp[0], wp[1]] for wp in cand_c["waypoints"]],
                color="#f59e0b",
                weight=5 if is_c_chosen else 3,
                dash_array=None if is_c_chosen else "6, 6",
                opacity=0.95 if is_c_chosen else 0.70,
                popup=f"Candidate C (Balanced): Cost {cand_c['cost']:.1f}, Fuel Saved {cand_c['savings_pct']:.1f}%" + (" [CHOSEN ROUTE]" if is_c_chosen else " [EVALUATED]"),
            ).add_to(folium_map)

        # Candidate B (Risk-avoiding)
        folium.PolyLine(
            locations=[[wp[0], wp[1]] for wp in cand_b["waypoints"]],
            color="#10b981",
            weight=5 if is_b_chosen else 3,
            dash_array=None if is_b_chosen else "6, 6",
            opacity=0.95 if is_b_chosen else 0.70,
            popup=f"Candidate B (Risk-Avoidance): Cost {cand_b['cost']:.1f}, Fuel Saved {cand_b['savings_pct']:.1f}%" + (" [CHOSEN ROUTE]" if is_b_chosen else " [EVALUATED]"),
        ).add_to(folium_map)
    else:
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


def _generate_codename() -> str:
    """Generate a memorable anonymous codename: Adjective-Noun-NN."""
    import random
    _ADJ = [
        "Silent", "Quiet", "Swift", "Bold", "Steady", "Iron",
        "Deep", "Calm", "Bright", "Lone", "Keen", "Dusk",
    ]
    _NOUN = [
        "Compass", "Harbor", "Anchor", "Beacon", "Horizon", "Rudder",
        "Current", "Vessel", "Tide", "Helm", "Reef", "Stern",
    ]
    adj = random.choice(_ADJ)
    noun = random.choice(_NOUN)
    num = random.randint(10, 99)
    return f"{adj}-{noun}-{num}"


def _sync_analyst_name() -> None:
    """Copy the fragment-owned name widget onto a non-widget key."""
    entered = (
        st.session_state.get("analyst_name_frag")
        or st.session_state.get("analyst_name")
        or ""
    ).strip()
    if entered:
        st.session_state["_analyst_name"] = entered
    # If nothing was typed, keep whatever auto-generated codename is already set


def _current_analyst() -> str:
    return (
        st.session_state.get("_analyst_name")
        or st.session_state.get("analyst_name_frag")
        or st.session_state.get("analyst_name")
        or ""
    ).strip()


def _ensure_codename() -> str:
    """Auto-generate a codename on first visit if one doesn't exist."""
    current = _current_analyst()
    if not current:
        codename = _generate_codename()
        st.session_state["_analyst_name"] = codename
        st.session_state["_codename_is_new"] = True
        return codename
    return current


def _watchlist_add(vid: str) -> None:
    storage.add_to_watchlist(_current_analyst(), vid)


def _watchlist_remove(vid: str) -> None:
    storage.remove_from_watchlist(_current_analyst(), vid)


def _render_analyst_watchlist_bar() -> None:
    """Anonymous codename identity + resume flow (replaces free-text name entry)."""
    codename = _ensure_codename()
    is_new = st.session_state.get("_codename_is_new", False)

    st.markdown(
        '<div class="section-heading">\U0001f575\ufe0f Anonymous Analyst Identity</div>',
        unsafe_allow_html=True,
    )

    # ── Display current codename prominently ──
    status_label = "🆕 New session — save this codename!" if is_new else "✅ Active session"
    st.markdown(
        f"""
        <div style="background:rgba(14,165,233,0.12);border:1.5px solid rgba(56,189,248,0.35);
                    border-radius:8px;padding:12px 14px;margin:6px 0 10px 0;">
          <div style="font-size:0.78rem;color:#94a3b8;margin-bottom:4px;">{status_label}</div>
          <div style="font-size:1.2rem;font-weight:800;color:#38bdf8;letter-spacing:0.02em;
                      font-family:'JetBrains Mono',monospace;">{codename}</div>
          <div style="font-size:0.75rem;color:#94a3b8;margin-top:4px;">
            This is your anonymous ID — save it to access your watchlist again later.
            No real name is collected or stored.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Copy-friendly display
    st.code(codename, language=None)

    # ── Resume flow for returning analysts ──
    with st.expander("🔑 Returning? Enter your codename", expanded=False):
        def _on_resume_codename():
            entered = st.session_state.get("resume_codename_input", "").strip()
            if entered:
                st.session_state["_analyst_name"] = entered
                st.session_state["_codename_is_new"] = False
        st.text_input(
            "Paste your codename",
            key="resume_codename_input",
            placeholder="e.g. Silent-Compass-42",
            on_change=_on_resume_codename,
        )

    st.success(f"Signed in as **{codename}**. Star flagged vessels in the expander below.")


def _render_header(state: dict) -> None:
    st.markdown(
        f"""
        <div class="mas-header">
          <div class="mas-title-group">
            <div class="mas-icon"></div>
            <div>
              <h1>MaritimeMAS — Global Fleet Operations</h1>
              <p style="font-weight: bold; color: #38bdf8;">Primary User: Fleet Dispatcher</p>
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


def _render_reasoning_trace(state: dict) -> None:
    st.markdown('<div class="section-heading">🧠 Live Tradeoff Reasoning Trace</div>', unsafe_allow_html=True)
    trace = state.get("reasoning_trace", [])
    if not trace:
        st.info("Orchestrator tradeoff engine initializing deliberation...")
        return

    st.markdown("""
        <style>
        .reasoning-trace-box {
            font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            font-size: 0.82rem;
            background: linear-gradient(180deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.95) 100%);
            border: 1px solid rgba(56, 189, 248, 0.35);
            border-radius: 8px;
            padding: 10px 12px;
            max-height: 250px;
            overflow-y: auto;
            margin-bottom: 14px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
        }
        .trace-row {
            padding: 5px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.07);
            display: flex;
            align-items: flex-start;
            line-height: 1.35;
        }
        .trace-row:last-child {
            border-bottom: none;
        }
        .trace-badge {
            font-size: 0.72rem;
            font-weight: 700;
            padding: 2px 6px;
            border-radius: 4px;
            margin-right: 8px;
            white-space: nowrap;
            display: inline-block;
        }
        .badge-surveillance { background: rgba(239, 68, 68, 0.2); color: #fca5a5; border: 1px solid rgba(239, 68, 68, 0.4); }
        .badge-route { background: rgba(56, 189, 248, 0.2); color: #7dd3fc; border: 1px solid rgba(56, 189, 248, 0.4); }
        .badge-debris { background: rgba(16, 185, 129, 0.2); color: #6ee7b7; border: 1px solid rgba(16, 185, 129, 0.4); }
        .badge-orchestrator { background: rgba(168, 85, 247, 0.25); color: #d8b4fe; border: 1px solid rgba(168, 85, 247, 0.5); font-weight: 800; }
        .badge-feedback { background: rgba(245, 158, 11, 0.2); color: #fcd34d; border: 1px solid rgba(245, 158, 11, 0.4); }
        .trace-text {
            color: #cbd5e1;
            flex-grow: 1;
        }
        .trace-text.decision {
            color: #f8fafc;
            font-weight: 600;
        }
        </style>
    """, unsafe_allow_html=True)

    badge_classes = {
        "Surveillance Agent": "badge-surveillance",
        "Route Planner": "badge-route",
        "Debris Agent": "badge-debris",
        "Orchestrator": "badge-orchestrator",
        "Feedback Loop": "badge-feedback",
    }

    rows_html = '<div class="reasoning-trace-box">'
    for item in trace:
        agent = item.get("agent", "Agent")
        icon = item.get("icon", "🔹")
        text = item.get("text", "")
        item_type = item.get("type", "")
        b_cls = badge_classes.get(agent, "badge-route")
        is_dec = "decision" if item_type == "decision" else ""
        rows_html += (
            f'<div class="trace-row">'
            f'<span class="trace-badge {b_cls}">{icon} {agent}</span>'
            f'<span class="trace-text {is_dec}">{text}</span>'
            f'</div>'
        )
    rows_html += '</div>'
    st.markdown(rows_html, unsafe_allow_html=True)


def _render_event_feed(state: dict) -> None:
    st.markdown('<div class="section-heading">⚡ System Event Stream</div>', unsafe_allow_html=True)
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

    # Custom CSS for the trace
    st.markdown("""
        <style>
        .agent-trace {
            font-family: 'Courier New', Courier, monospace;
            font-size: 0.85rem;
            background: rgba(15, 23, 42, 0.9);
            border: 1px solid rgba(56, 189, 248, 0.2);
            border-radius: 8px;
            padding: 12px;
            max-height: 250px;
            overflow-y: auto;
            margin-bottom: 12px;
        }
        .agent-trace-msg {
            margin-bottom: 6px;
            color: #94a3b8;
        }
        .agent-trace-msg.crit { color: #f87171; font-weight: bold; }
        .agent-trace-msg.warn { color: #fbbf24; }
        .agent-trace-sender { color: #38bdf8; font-weight: bold; }
        </style>
        <div class="agent-trace">
    """, unsafe_allow_html=True)

    trace_html = ""
    for g in grouped:
        ev = g["ev"]
        is_warn = "->" in ev or "flagged" in ev.lower()
        is_crit = "Hurricane" in ev or "ALERT" in ev or "WATCHLIST RE-ALERT" in ev
        cls = "crit" if is_crit else ("warn" if is_warn else "")
        count_badge = f'<span class="event-count">×{g["count"]}</span>' if g["count"] > 1 else ""

        # Colorize [Agent] tags
        import re
        ev_colored = re.sub(r'(\[.*?\])', r'<span class="agent-trace-sender">\1</span>', ev)

        trace_html += f'<div class="agent-trace-msg {cls}">{ev_colored} {count_badge}</div>'

    st.markdown(trace_html + "</div>", unsafe_allow_html=True)


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
        with st.expander(f"Investigator Brief — {vid}", expanded=True):
            st.caption(f"Model: {brief.get('model')} · LLM={brief.get('used_llm')}")
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
    focus_vid = st.session_state.get("active_search_vessel")
    center = GULF_CENTER
    zoom = GULF_ZOOM
    if focus_vid:
        for v in state.get("vessel_positions", []):
            if v["vessel_id"] == focus_vid:
                center = (v["lat"], v["lon"])
                zoom = 10
                break
    elif state.get("active_route_origin") and state.get("active_route_destination"):
        o = state["active_route_origin"]
        d = state["active_route_destination"]
        # If outside the Gulf box, center on route midpoint with global zoom
        if not (24 <= o[0] <= 32 and -98 <= o[1] <= -80 and 24 <= d[0] <= 32 and -98 <= d[1] <= -80):
            center = ((o[0] + d[0]) / 2.0, (o[1] + d[1]) / 2.0)
            zoom = 3

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
        candidate_routes=state.get("candidate_routes"),
    )

    # ── Real Oceanic Marine Debris Accumulation Zones (NOAA MDMAP & Ocean Gyres) ──
    if st.session_state.get("show_debris_zones", True):
        for dz in state.get("debris_zones", []):
            poly_coords = [[p[0], p[1]] for p in dz["polygon"]]
            folium.Polygon(
                locations=poly_coords,
                color="#d97706",
                weight=2,
                dash_array="6, 6",
                fill=True,
                fill_color="#f59e0b",
                fill_opacity=0.20,
                popup=folium.Popup(
                    f"<div style='font-family:Inter,sans-serif;min-width:240px;'>"
                    f"<div style='color:#d97706;font-weight:700;font-size:0.92rem;'>🌊 {dz['name']}</div>"
                    f"<hr style='margin:4px 0;'/>"
                    f"<b>Density:</b> {dz['density_desc']}<br/>"
                    f"<b>Severity:</b> <span style='color:{'#ef4444' if dz['severity']=='High' else '#d97706'};font-weight:700'>{dz['severity']}</span><br/>"
                    f"<b>Primary Materials:</b> {', '.join(dz.get('primary_materials', []))}<br/>"
                    f"<b>Scientific Citation:</b> <i>{dz.get('citation', 'NOAA Marine Debris Program')}</i>"
                    f"</div>",
                    max_width=320,
                ),
            ).add_to(folium_map)

    # ── Real Marine Debris Clustered Hotspots (Zoom-Resilient Markers) ──
    for hs in state.get("debris_hotspots", []):
        c = hs["center"]
        types_summary = ", ".join([f"{k} ({v})" for k, v in hs.get("debris_types", {}).items()][:3]) or "Mixed Plastics"

        # Outer glowing halo (guarantees visibility at world scale zoom 2-4)
        folium.CircleMarker(
            location=[c[0], c[1]],
            radius=15,
            color="#f59e0b",
            weight=1.5,
            fill=True,
            fill_color="#fde68a",
            fill_opacity=0.35,
        ).add_to(folium_map)

        # Core high-visibility beacon marker
        folium.CircleMarker(
            location=[c[0], c[1]],
            radius=8,
            color="#b45309",
            weight=2,
            fill=True,
            fill_color="#fbbf24",
            fill_opacity=0.95,
            popup=folium.Popup(
                f"<div style='font-family:Inter,sans-serif;min-width:240px;'>"
                f"<div style='color:#b45309;font-weight:700;font-size:0.92rem;'>🟡 Marine Debris Hotspot {hs['hotspot_id']}</div>"
                f"<hr style='margin:4px 0;'/>"
                f"<b>Field Sightings:</b> {hs['sighting_count']} verified monitoring points<br/>"
                f"<b>Severity:</b> <span style='font-weight:700;color:{'#ef4444' if hs['severity']=='High' else '#d97706'}'>{hs['severity']}</span><br/>"
                f"<b>Debris Types:</b> {types_summary}<br/>"
                f"<b>Assigned Collector:</b> <span style='color:#2563eb;font-weight:700'>{hs.get('collector_assigned') or 'None'}</span><br/>"
                f"<b>Cleanup ETA:</b> {hs.get('eta', 'N/A')}<br/>"
                f"<b>Dataset Origin:</b> <i>NOAA NCEI Marine Debris & Ocean Gyre Survey</i>"
                f"</div>",
                max_width=300,
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
            
            # Draw dashed purple line from origin to destination
            v_origin = state.get("vessel_origins", {}).get(vid)
            if v_origin:
                v_dest_dict = orchestrator.get_vessel_origin_dest(vid).get("dest", {})
                if v_dest_dict and v_dest_dict.get("lat") and v_dest_dict.get("lon"):
                    v_dest = (float(v_dest_dict["lat"]), float(v_dest_dict["lon"]))
                    folium.PolyLine(
                        locations=[[v_origin[0], v_origin[1]], [v_dest[0], v_dest[1]]],
                        color="purple",
                        weight=3,
                        dash_array="5, 10",
                        opacity=0.8,
                        popup="Searched Vessel Origin → Destination",
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

    origin = state.get("active_route_origin", orchestrator.ROUTE_ORIGIN)
    dest = state.get("active_route_destination", orchestrator.ROUTE_DESTINATION)
    orig_label = state.get("active_origin_name", "Route Origin")
    dest_label = state.get("active_dest_name", "Route Destination")
    folium.Marker(
        location=list(origin),
        popup=f"<b>{orig_label}</b><br/>Route Origin ({origin[0]:.2f}°, {origin[1]:.2f}°)",
        icon=folium.Icon(color="green", icon="anchor", prefix="fa"),
    ).add_to(folium_map)
    folium.Marker(
        location=list(dest),
        popup=f"<b>{dest_label}</b><br/>Route Destination ({dest[0]:.2f}°, {dest[1]:.2f}°)",
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
        bool(st.session_state.get("show_debris_zones", True)),
        bool(state.get("storm_mode_active")),
        st.session_state.get("active_search_vessel"),
        state.get("active_origin_name"),
        state.get("active_dest_name"),
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
    pass


def _show_live_map(map_html: str | None, rebuilt: bool, tick, build_ms: float) -> None:
    """Render the live Folium map directly inside Streamlit using components.html."""
    if not map_html:
        st.warning("Map is generating...")
        return
    components.html(
        map_html,
        height=780,
        scrolling=False,
    )
    if rebuilt:
        st.caption(f"Map updated in {build_ms:.0f} ms (tick {tick}).")
    else:
        st.caption(f"Map cached (tick {tick}).")


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
            map_col = st.container()
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
                        
                    if submit_search:
                        if search_vid:
                            st.session_state["search_focus_vessel"] = search_vid
                            st.session_state["active_search_vessel"] = search_vid
                            from agents.investigator import generate_vessel_brief
                            with st.spinner("Generating investigative brief..."):
                                brief = generate_vessel_brief(search_vid, live)
                                st.session_state["active_search_brief"] = brief
                        else:
                            st.session_state["search_focus_vessel"] = None
                            st.session_state["active_search_vessel"] = None
                            st.session_state.pop("active_search_brief", None)

                active_search = st.session_state.get("active_search_vessel")
                brief = st.session_state.get("active_search_brief")
                if active_search and brief and str(brief.get("vessel_id")) == str(active_search):
                    st.markdown(f"##### 🔍 Investigative Brief: {active_search}")
                    if brief.get("error"):
                        st.error(brief["error"])
                    else:
                        if brief.get("is_flagged"):
                            flagged_vessels = live.get("flagged_vessels", [])
                            fv = next((v for v in flagged_vessels if str(v.get("vessel_id")) == str(active_search)), {})
                            conf = fv.get("confidence", 0)
                            max_g = fv.get("max_gap_minutes", 0)
                            disp = fv.get("displacement_error_km", 0)
                            spd = fv.get("speed_change_after_gap", 0)
                            hdg = fv.get("heading_change_after_gap", 0)
                            
                            st.error(f"**🚨 Flagged Dark Vessel** (Confidence: {conf:.0%})")
                            st.markdown(
                                f"<div style='font-size:0.8rem; margin-bottom:10px;'>"
                                f"<b>Gap:</b> {max_g} min | <b>DR Error:</b> {disp} km | "
                                f"<b>Speed Δ:</b> {spd} kts | <b>Heading Δ:</b> {hdg}°"
                                f"</div>",
                                unsafe_allow_html=True
                            )
                        else:
                            st.success("✅ Normal operating vessel")
                            
                        st.markdown(f"**Summary:** {brief.get('summary')}")
                        
                        analyst = _current_analyst()
                        if analyst:
                            watched_ids = {r["vessel_id"] for r in storage.get_watchlist(analyst)}
                            if active_search in watched_ids:
                                st.button("✓ On Watchlist", key=f"wl_on_inline_{active_search}", on_click=_watchlist_remove, args=(active_search,))
                            else:
                                st.button("⭐ Add to Watchlist", key=f"wl_add_inline_{active_search}", on_click=_watchlist_add, args=(active_search,))
                        else:
                            st.info("Save your codename below to use the Watchlist.")
                        
                        st.markdown("<br/>", unsafe_allow_html=True)
                
                st.markdown(
                    """
                    <div class="legend-bar">
                      <span class="legend-item"><span class="legend-dot" style="background:#10b981"></span>Active vessel</span>
                      <span class="legend-item"><span class="legend-dot" style="background:#ef4444"></span>Flagged dark vessel</span>
                      <span class="legend-item"><span class="legend-dot" style="background:#fbbf24"></span>Debris hotspot</span>
                      <span class="legend-item"><span class="legend-sq" style="background:#f59e0b;opacity:0.4;border:1px dashed #d97706"></span>Debris gyre zone</span>
                      <span class="legend-item"><span class="legend-dot" style="background:#3b82f6"></span>Collector ship</span>
                      <span class="legend-item"><span class="legend-line" style="border-color:#38bdf8"></span>Optimized route</span>
                      <span class="legend-item"><span class="legend-line dashed" style="border-color:#ef4444"></span>Naive baseline</span>
                      <span class="legend-item"><span class="legend-sq" style="background:#10b981;opacity:0.5"></span>GFW fishing zone</span>
                      <span class="legend-item"><span class="legend-dot" style="background:#dc2626"></span>Hurricane Ida eye</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                choice = st.session_state.get("map_style_choice", "🌌 Esri Dark Nautical Canvas")
                map_html, rebuilt, build_ms = _get_cached_live_folium_map(live, choice)
                _show_live_map(map_html, rebuilt, live.get("tick"), build_ms)
                route = live.get("current_route") or {}
                if active_search:
                    # Show vessel-specific route info
                    v = next((x for x in live.get("vessel_positions", []) if x["vessel_id"] == active_search), None)
                    if v:
                        v_origin = live.get("vessel_origins", {}).get(active_search)
                        v_dest_dict = orchestrator.get_vessel_origin_dest(active_search).get("dest", {})
                        if v_origin and v_dest_dict and v_dest_dict.get("lat") and v_dest_dict.get("lon"):
                            v_dest = (float(v_dest_dict["lat"]), float(v_dest_dict["lon"]))
                            
                            # Cache the route computation for this vessel
                            v_route_key = f"route_{active_search}"
                            if v_route_key not in st.session_state:
                                st.session_state[v_route_key] = get_route(v_origin, v_dest, live.get("weather_df"))
                            
                            v_route = st.session_state.get(v_route_key, {})
                            bl_cost = v_route.get("baseline_cost", 0)
                            opt_cost = v_route.get("cost", 0)
                            sav = v_route.get("savings_pct", 0)
                            
                            current_pos = (v["lat"], v["lon"])
                            from agents.investigator import _haversine
                            dist_rem_km = _haversine(current_pos[0], current_pos[1], v_dest[0], v_dest[1])
                            dist_rem_nm = dist_rem_km / 1.852
                            speed = v.get("speed", 0.1)
                            speed = speed if speed > 0 else 0.1
                            eta_hrs = dist_rem_nm / speed
                            
                            status_str = f"<span style='color:#ef4444'>FLAGGED</span>" if v.get("flagged") else f"<span style='color:#34d399'>NORMAL</span>"
                            dest_name = v_dest_dict.get('name', 'Unknown')
                            
                            st.markdown(
                                f"""
                                <div class="route-bar">
                                  <div>🧭 <b>{active_search} Corridor:</b> Origin ({v_origin[0]:.2f}, {v_origin[1]:.2f}) &nbsp;➔&nbsp; {dest_name} ({v_dest[0]:.2f}, {v_dest[1]:.2f})</div>
                                  <div><b>Pos:</b> {current_pos[0]:.2f}, {current_pos[1]:.2f} &nbsp;·&nbsp; <b>Spd:</b> {v.get('speed', 0)} kts &nbsp;·&nbsp; <b>Hdg:</b> {v.get('heading', 0)}° &nbsp;·&nbsp; <b>Status:</b> {status_str}</div>
                                  <div><b>Dist Rem:</b> {dist_rem_nm:.1f} nm &nbsp;·&nbsp; <b>ETA:</b> {eta_hrs:.1f} hrs</div>
                                  <div><b>Fuel saved:</b> <span style="color:#34d399;font-weight:700">{sav:.1f}%</span> (Naive: {bl_cost:.1f} · Opt: {opt_cost:.1f})</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                            
                            # ── Task 2: Per-vessel route re-optimization check ──
                            REOPT_IMPROVEMENT_THRESHOLD_PCT = 2.0  # named constant — tune as needed
                            
                            if st.button("🔄 Check for optimized route", key=f"reopt_check_{active_search}"):
                                with st.spinner("Computing fresh route with current weather…"):
                                    fresh_route = get_route(
                                        (v["lat"], v["lon"]),  # current position as origin
                                        v_dest,
                                        live.get("weather_df"),  # fresh weather, not cached
                                    )
                                    st.session_state["reopt_candidate"] = {
                                        "vessel_id": active_search,
                                        "current_cost": opt_cost,
                                        "new_cost": fresh_route.get("cost", 0),
                                        "new_route": fresh_route,
                                        "improvement_pct": ((opt_cost - fresh_route.get("cost", 0)) / opt_cost * 100) if opt_cost > 0 else 0,
                                    }
                            
                            # Render comparison popup if one exists for this vessel
                            reopt = st.session_state.get("reopt_candidate")
                            if reopt and reopt.get("vessel_id") == active_search:
                                imp = reopt["improvement_pct"]
                                if imp < REOPT_IMPROVEMENT_THRESHOLD_PCT:
                                    st.success(f"✅ Current route for **{active_search}** is already optimized. No change needed (improvement would be only {imp:.1f}%).")
                                    if st.button("Dismiss", key="reopt_dismiss_ok"):
                                        st.session_state.pop("reopt_candidate", None)
                                        st.rerun()
                                else:
                                    st.warning(f"🔀 A better route exists for **{active_search}**!")
                                    rc1, rc2, rc3 = st.columns(3)
                                    rc1.metric("Current cost", f"{reopt['current_cost']:.1f}")
                                    rc2.metric("New cost", f"{reopt['new_cost']:.1f}")
                                    rc3.metric("Improvement", f"{imp:.1f}%")
                                    
                                    ac1, ac2, _ = st.columns([1.5, 1.5, 3])
                                    with ac1:
                                        if st.button("✅ Apply new route", key="reopt_apply"):
                                            # Update the cached route for this vessel
                                            st.session_state[f"route_{active_search}"] = reopt["new_route"]
                                            # Clear the map cache so it re-renders
                                            _LIVE_FOLIUM_CACHE.clear()
                                            orchestrator.append_event(
                                                f"[Dispatcher] -> [Route Planner]: Re-optimized route for {active_search} — {imp:.1f}% improvement applied"
                                            )
                                            st.session_state.pop("reopt_candidate", None)
                                            st.toast(f"Route for {active_search} updated!")
                                            st.rerun()
                                    with ac2:
                                        if st.button("❌ Keep current route", key="reopt_keep"):
                                            orchestrator.append_event(
                                                f"[Dispatcher] -> [Orchestrator]: Declined re-optimization for {active_search}"
                                            )
                                            st.session_state.pop("reopt_candidate", None)
                                            st.rerun()

                elif route:
                    bl_cost = route.get("baseline_cost", 0)
                    opt_cost = route.get("cost", 0)
                    sav = route.get("savings_pct", 0)
                    decision = live.get("tradeoff_decision") or {}
                    candidates = live.get("candidate_routes") or {}
                    cand_a = candidates.get("candidate_a") or {}
                    cand_b = candidates.get("candidate_b") or {}
                    cand_c = candidates.get("candidate_c") or {}
                    chosen_id = decision.get("chosen_candidate", "A")
                    risk_thresh = live.get("risk_threshold", 5.0)

                    # ── Origin / Destination Port Selector (Part 1A, 1B) ──
                    port_catalog = orchestrator.PORT_CATALOG
                    port_names = [p["name"] for p in port_catalog]
                    active_orig_name = decision.get("origin_name") or live.get("active_origin_name", "Houston / Galveston (US)")
                    active_dest_name = decision.get("dest_name") or live.get("active_dest_name", "New Orleans / South Pass (US)")

                    orig_idx = port_names.index(active_orig_name) if active_orig_name in port_names else 0
                    dest_idx = port_names.index(active_dest_name) if active_dest_name in port_names else 1

                    st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)
                    st.markdown("##### ⚓ Port-to-Port Commercial Route Planner")

                    sel_c1, sel_c2, sel_c3 = st.columns([2.2, 2.2, 1.2])
                    with sel_c1:
                        def _on_origin_select_change():
                            o = st.session_state.get("sel_port_origin")
                            d = st.session_state.get("sel_port_dest")
                            if o and d:
                                orchestrator.set_active_route_ports(o, d)
                                _LIVE_FOLIUM_CACHE.clear()
                        sel_orig = st.selectbox("Origin Port", port_names, index=orig_idx, key="sel_port_origin", on_change=_on_origin_select_change)
                    with sel_c2:
                        def _on_dest_select_change():
                            o = st.session_state.get("sel_port_origin")
                            d = st.session_state.get("sel_port_dest")
                            if o and d:
                                orchestrator.set_active_route_ports(o, d)
                                _LIVE_FOLIUM_CACHE.clear()
                        sel_dest = st.selectbox("Destination Port", port_names, index=dest_idx, key="sel_port_dest", on_change=_on_dest_select_change)
                    with sel_c3:
                        st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
                        def _do_replan_ports():
                            o = st.session_state.get("sel_port_origin", active_orig_name)
                            d = st.session_state.get("sel_port_dest", active_dest_name)
                            orchestrator.set_active_route_ports(o, d)
                            _LIVE_FOLIUM_CACHE.clear()
                            st.rerun()
                        st.button("🔄 Plan Route", key="btn_plan_selected_corridor", on_click=_do_replan_ports, use_container_width=True)

                    # ── Who's Actually Traveling This Route (Part 1D) ──
                    assigned = decision.get("assigned_vessel")
                    if assigned:
                        v_id = assigned.get("vessel_id", "Unknown")
                        v_type = assigned.get("vessel_type", "CARGO")
                        v_dist = assigned.get("dist_km", 0.0)
                        v_speed = assigned.get("speed", 0.0)
                        vessel_header_html = f"""
                        <div style="background: rgba(14, 165, 233, 0.14); border-left: 4px solid #38bdf8; border-radius: 6px; padding: 10px 14px; margin: 10px 0 12px 0; display: flex; justify-content: space-between; align-items: center;">
                          <div>
                            <span style="font-size: 1.15rem; margin-right: 8px;">🚢</span>
                            <span style="font-weight: 700; color: #38bdf8; font-size: 0.95rem;">Vessel {v_id} ({v_type}, currently near {sel_orig} — {v_dist:.1f} km away)</span>
                            <span style="color: #cbd5e1; font-size: 0.9rem;"> — planning route to <b>{sel_dest}</b></span>
                            <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 2px;">
                              Current Speed: {v_speed:.1f} kts &nbsp;·&nbsp; Telemetry: Active Normal AIS Tracking &nbsp;·&nbsp; Sector Scan: Clear
                            </div>
                          </div>
                          <span style="background: #0284c7; color: #ffffff; padding: 3px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 700; white-space: nowrap;">ASSIGNED ACTIVE FLEET</span>
                        </div>
                        """
                    else:
                        vessel_header_html = f"""
                        <div style="background: rgba(100, 116, 139, 0.15); border-left: 4px solid #94a3b8; border-radius: 6px; padding: 10px 14px; margin: 10px 0 12px 0; display: flex; justify-content: space-between; align-items: center;">
                          <div>
                            <span style="font-size: 1.15rem; margin-right: 8px;">📍</span>
                            <span style="color: #cbd5e1; font-size: 0.9rem; font-style: italic;">No active vessel currently near this origin — showing corridor planning only.</span>
                            <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 2px;">
                              Nearest non-flagged fleet unit is &gt; 300 km from origin. Corridor benchmark generated for route feasibility assessment.
                            </div>
                          </div>
                          <span style="background: rgba(255,255,255,0.1); color: #94a3b8; padding: 3px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; white-space: nowrap;">CORRIDOR ONLY</span>
                        </div>
                        """
                    st.markdown(vessel_header_html, unsafe_allow_html=True)

                    st.markdown(
                        f"""
                        <div class="route-bar" style="flex-direction:column;align-items:stretch;">
                          <div style="font-size:0.85rem;margin-bottom:4px;"> <b>Managed Commercial Corridor:</b> {sel_orig} ➔ {sel_dest}</div>
                          <div style="display:flex;flex-wrap:wrap;gap:6px 14px;font-size:0.8rem;">
                            <span><b>Naive Baseline:</b> <span style="color:#f87171">{bl_cost:.1f}</span></span>
                            <span><b>Selected Route ({chosen_id}):</b> <span style="color:#38bdf8">{opt_cost:.1f}</span></span>
                            <span><b>Fuel Saved:</b> <span style="color:#34d399;font-weight:700">{sav:.1f}%</span></span>
                            <span><b>Risk Threshold:</b> <span style="color:#fbbf24;font-weight:700">{risk_thresh:.1f}%</span></span>
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # ── Multi-Agent Candidate Tradeoff Matrix (Part 1C, 1E: 3 Candidate Paths) ──
                    st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)
                    tc1, tc2, tc3 = st.columns(3)
                    with tc1:
                        is_a_sel = (chosen_id == "A")
                        border_color_a = "#38bdf8" if is_a_sel else "rgba(255,255,255,0.1)"
                        badge_a = '<span style="background:#0284c7;color:#fff;padding:2px 6px;border-radius:4px;font-size:0.75rem;font-weight:700">ACTIVE RECOMMENDATION</span>' if is_a_sel else '<span style="background:rgba(255,255,255,0.1);color:#94a3b8;padding:2px 6px;border-radius:4px;font-size:0.75rem">ALTERNATIVE</span>'
                        st.markdown(
                            f"""
                            <div style="background:rgba(15,23,42,0.8);border:1.5px solid {border_color_a};border-radius:8px;padding:10px 12px;height:100%;">
                              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                                <span style="font-weight:700;color:#38bdf8;font-size:0.88rem;">Candidate A (Fuel-Optimal)</span>
                                {badge_a}
                              </div>
                              <div style="font-size:0.83rem;color:#cbd5e1;line-height:1.45;">
                                <b>Fuel Cost:</b> {cand_a.get('cost', opt_cost):.1f} &nbsp;·&nbsp; <b>Saved:</b> <span style="color:#34d399;font-weight:700">{cand_a.get('savings_pct', sav):.1f}%</span><br/>
                                <b>Est. Transit:</b> {cand_a.get('transit_hours', 18.2)} hrs ({cand_a.get('distance_km', 460)} km)<br/>
                                <b>Route Objective:</b> Direct transit via optimal sea conditions
                              </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    with tc2:
                        is_c_sel = (chosen_id == "C")
                        border_color_c = "#f59e0b" if is_c_sel else "rgba(255,255,255,0.1)"
                        badge_c = '<span style="background:#d97706;color:#fff;padding:2px 6px;border-radius:4px;font-size:0.75rem;font-weight:700">ACTIVE RECOMMENDATION</span>' if is_c_sel else '<span style="background:rgba(255,255,255,0.1);color:#94a3b8;padding:2px 6px;border-radius:4px;font-size:0.75rem">ALTERNATIVE</span>'
                        pen_c = cand_c.get("fuel_penalty_pct", 0.0)
                        st.markdown(
                            f"""
                            <div style="background:rgba(15,23,42,0.8);border:1.5px solid {border_color_c};border-radius:8px;padding:10px 12px;height:100%;">
                              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                                <span style="font-weight:700;color:#f59e0b;font-size:0.88rem;">Path C (Balanced)</span>
                                {badge_c}
                              </div>
                              <div style="font-size:0.83rem;color:#cbd5e1;line-height:1.45;">
                                <b>Fuel Cost:</b> {cand_c.get('cost', opt_cost):.1f} &nbsp;·&nbsp; <b>Variance:</b> <span style="color:{'#f59e0b' if pen_c > 0 else '#94a3b8'}">+{pen_c:.1f}% vs A</span><br/>
                                <b>Est. Transit:</b> {cand_c.get('transit_hours', 18.3)} hrs ({cand_c.get('distance_km', 465)} km)<br/>
                                <b>Route Objective:</b> Blended clearance buffer & fuel balance
                              </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    with tc3:
                        is_b_sel = (chosen_id == "B")
                        border_color_b = "#10b981" if is_b_sel else "rgba(255,255,255,0.1)"
                        badge_b = '<span style="background:#059669;color:#fff;padding:2px 6px;border-radius:4px;font-size:0.75rem;font-weight:700">ACTIVE RECOMMENDATION</span>' if is_b_sel else '<span style="background:rgba(255,255,255,0.1);color:#94a3b8;padding:2px 6px;border-radius:4px;font-size:0.75rem">ALTERNATIVE</span>'
                        pen_b = cand_b.get("fuel_penalty_pct", 0.0)
                        st.markdown(
                            f"""
                            <div style="background:rgba(15,23,42,0.8);border:1.5px solid {border_color_b};border-radius:8px;padding:10px 12px;height:100%;">
                              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                                <span style="font-weight:700;color:#10b981;font-size:0.88rem;">Candidate B (Risk-Avoidance)</span>
                                {badge_b}
                              </div>
                              <div style="font-size:0.83rem;color:#cbd5e1;line-height:1.45;">
                                <b>Fuel Cost:</b> {cand_b.get('cost', opt_cost):.1f} &nbsp;·&nbsp; <b>Variance:</b> <span style="color:{'#fbbf24' if pen_b > 0 else '#94a3b8'}">+{pen_b:.1f}% vs A</span><br/>
                                <b>Est. Transit:</b> {cand_b.get('transit_hours', 18.5)} hrs ({cand_b.get('distance_km', 475)} km)<br/>
                                <b>Route Objective:</b> Full detour around flagged hazard zones
                              </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    # ── Orchestrator Decision Callout ──
                    st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
                    st.markdown("##### 🤖 Orchestrator Tradeoff Recommendation")
                    rec_reason = decision.get("reason", f"Use Optimized Corridor (Fuel saved: {sav:.1f}%)")
                    st.info(f"**Recommendation:** **{decision.get('chosen_name', 'Candidate ' + chosen_id)}** — {rec_reason}")

                    if decision.get("debris_bonus"):
                        db = decision["debris_bonus"]
                        st.caption(f"🌊 **Opportunistic Debris Proximity:** Corridor passes within **{db['dist_km']} km** of Debris Hotspot **{db['hotspot_id']}** (Collector **{db['collector']}** on station).")

                    if decision.get("feedback_applied"):
                        st.warning(f"🔄 **Adaptive Dispatcher Feedback Applied:** Risk threshold adjusted from 5.0% to {risk_thresh:.1f}% based on recent dispatcher overrides.")

                    # ── Condensed Multi-Agent Decision Trace (collapsed by default) ──
                    trace = live.get("reasoning_trace") or []
                    if trace:
                        with st.expander("🔎 Show full multi-agent reasoning trace", expanded=False):
                            trace_rows_html = """
                            <div style="background:rgba(15,23,42,0.85);border:1.5px solid rgba(56,189,248,0.25);border-radius:8px;padding:12px 14px;margin-top:4px;margin-bottom:6px;">
                              <div style="font-size:0.84rem;font-weight:700;color:#38bdf8;margin-bottom:8px;display:flex;align-items:center;">
                                <span style="margin-right:6px;">🧠</span> Multi-Agent Decision &amp; Tradeoff Sequence
                              </div>
                              <div style="font-size:0.82rem;color:#cbd5e1;line-height:1.6;">
                            """
                            for item in trace:
                                ag = item.get("agent", "Agent")
                                ic = item.get("icon", "•")
                                tx = item.get("text", "")
                                if ag == "Surveillance Agent":
                                    ag_color = "#fca5a5"
                                elif ag == "Route Planner":
                                    ag_color = "#7dd3fc"
                                elif ag == "Debris Agent":
                                    ag_color = "#6ee7b7"
                                elif ag == "Orchestrator":
                                    ag_color = "#d8b4fe"
                                elif ag == "Feedback Loop":
                                    ag_color = "#fcd34d"
                                else:
                                    ag_color = "#e2e8f0"
                                trace_rows_html += f'<div style="margin-bottom:3px;">{ic} <span style="font-weight:700;color:{ag_color};">[{ag}]</span> {tx}</div>'
                            trace_rows_html += "</div></div>"
                            st.markdown(trace_rows_html, unsafe_allow_html=True)

                    # ── Working Approve / Override Controls ──
                    route_status = st.session_state.get("route_decision_status")

                    if route_status == "approved":
                        st.success("✅ **Route Approved** — Corridor confirmed by dispatcher. Logged to fleet operations memory.")
                        if st.button("↩ Reset decision", key="reset_route_decision"):
                            st.session_state.pop("route_decision_status", None)
                            st.session_state.pop("route_override_reason_text", None)
                            st.rerun()
                    elif route_status == "overridden":
                        override_reason = st.session_state.get("route_override_reason_text", "No reason given")
                        st.error(f"⛔ **Route Overridden** — Dispatcher overridden recommendation. Reason: _{override_reason}_")
                        if st.button("↩ Reset decision", key="reset_route_decision"):
                            st.session_state.pop("route_decision_status", None)
                            st.session_state.pop("route_override_reason_text", None)
                            st.rerun()
                    else:
                        c1, c2, _ = st.columns([1.5, 1.5, 3])
                        with c1:
                            def _approve_route():
                                st.session_state["route_decision_status"] = "approved"
                                storage.log_route_decision(f"tick_{live.get('tick', 0)}", "approved", "Dispatcher approved recommendation", risk_thresh)
                                orchestrator.append_event(f"[Dispatcher] -> [Orchestrator]: Approved {decision.get('chosen_name', 'route')} (savings: {sav:.1f}%)")
                            st.button("✅ Approve", key="btn_approve_route", on_click=_approve_route, use_container_width=True)
                        with c2:
                            def _toggle_override():
                                st.session_state["show_override_panel"] = not st.session_state.get("show_override_panel", False)
                            st.button("❌ Override", key="btn_override_route", on_click=_toggle_override, use_container_width=True)

                        if st.session_state.get("show_override_panel"):
                            st.markdown("---")
                            st.markdown("**Select a reason for override (feeds into adaptive risk threshold):**")
                            override_reasons = [
                                "The route is too costly for current fuel budget",
                                "Preferred corridor has better port access",
                                "Dispatcher has local weather intel not in model",
                                "Vessel crew requested alternate heading",
                            ]
                            selected_reason = st.session_state.get("selected_override_reason", "")
                            for i, r in enumerate(override_reasons):
                                btn_type = "primary" if selected_reason == r else "secondary"
                                def _select_reason(reason=r):
                                    st.session_state["selected_override_reason"] = reason
                                st.button(r, key=f"reason_btn_{i}", on_click=_select_reason, type=btn_type, use_container_width=True)

                            custom = st.text_input("Or enter a custom reason:", key="custom_override_input")

                            def _submit_override():
                                reason_text = st.session_state.get("selected_override_reason", "") or st.session_state.get("custom_override_input", "The route is too costly")
                                st.session_state["route_decision_status"] = "overridden"
                                st.session_state["route_override_reason_text"] = reason_text
                                st.session_state["show_override_panel"] = False
                                st.session_state.pop("selected_override_reason", None)
                                storage.log_route_decision(f"tick_{live.get('tick', 0)}", "overridden", reason_text, risk_thresh)
                                orchestrator.append_event(f"[Dispatcher] -> [Orchestrator]: OVERRIDE — {reason_text}")
                                orchestrator._run_route_tradeoff("dispatcher override")

                            st.button("📤 Submit Override", key="btn_submit_override", on_click=_submit_override, type="primary", use_container_width=True)
    elif map_feed_slot is None:
        _hide_parked_live_map()
    if explain_slot is not None:
        with explain_slot:
            _render_analyst_watchlist_bar()
            _render_watchlist_panel(live)
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
                "🌌 Esri Dark Nautical Canvas",
                "🌊 Esri Ocean Bathymetry",
                "🌍 OpenStreetMap Marine View",
                "🛰️ Google Maps Satellite Hybrid",
                "🛰️ Esri World Imagery (High-Res)",
            ],
        )
        st.session_state.show_debris_zones = st.checkbox("Show real ocean debris gyre zones (NOAA/TOC)", value=st.session_state.get("show_debris_zones", True))
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
if st.session_state.get("auto_play", False):
    orchestrator.tick()
    _live = orchestrator.get_state()
    _maybe_log_watchlist_realerts(_live)

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
                "severity": hs.get("severity", "Medium"),
                "sighting_count": hs.get("sighting_count"),
                "primary_types": ", ".join(list(hs.get("debris_types", {}).keys())[:2]) if hs.get("debris_types") else "Plastics",
                "collector_assigned": hs.get("collector_assigned") or "None",
                "status": hs.get("status"),
                "cleanup_eta": hs.get("eta", "N/A"),
                "source": "NOAA MDMAP & Ocean Gyre Survey",
            }
            for hs in state.get("debris_hotspots", [])
        ]
        df = pd.DataFrame(
            rows,
            columns=["hotspot_id", "center", "severity", "sighting_count", "primary_types", "collector_assigned", "status", "cleanup_eta", "source"],
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

    st.markdown('<div class="section-heading">📋 Operations data<span class="tag">live tables · Fleet Dispatcher</span></div>', unsafe_allow_html=True)

    _tick = state["tick"]

    tab_vessels, tab_flagged, tab_hotspots, tab_collectors = st.tabs(

        ["Vessels", "Flagged Vessels [SIMULATED DATA]", "Debris Hotspots [SIMULATED DATA]", "Collectors"]

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
      Fishing zones via Global Fishing Watch · Marine Debris via NOAA NCEI &amp; Global Gyres
    </div>
    """,
    unsafe_allow_html=True,
)

if st.session_state.get("auto_play", False):
    time.sleep(st.session_state.get("playback_speed", 2))
    st.rerun()