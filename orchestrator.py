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

import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import orchestrator

# ─────────────────────────────────────────────────────────────────────────── #
#  Page Config                                                                #
# ─────────────────────────────────────────────────────────────────────────── #
st.set_page_config(
    page_title="MaritimeMAS — Gulf of Mexico Operations",
    page_icon="⚓",
    layout="wide",
    initial_sidebar_state="expanded",
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
    .block-container { padding-top: 2.6rem !important; padding-bottom: 2rem; max-width: 1500px; }
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

    /* ── Sidebar ──────────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #040812 0%, #081020 100%);
        border-right: 1px solid rgba(30, 60, 120, 0.3);
    }
    [data-testid="stSidebar"] .stButton button {
        border-radius: 8px; font-weight: 600; font-size: 0.82rem;
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
if "initialised" not in st.session_state:
    st.session_state.initialised = False
    st.session_state.auto_play = False
    st.session_state.playback_speed = 3
    st.session_state.show_gfw = True
    st.session_state.show_before_after = True
    st.session_state.map_style_choice = "🛰️ Google Maps Satellite Hybrid"

if not st.session_state.initialised:
    with st.spinner("🌊 Initialising MaritimeMAS Engine — fetching Open-Meteo & NOAA data..."):
        orchestrator.initialise()
    st.session_state.initialised = True

# ─────────────────────────────────────────────────────────────────────────── #
#  Header Bar                                                                 #
# ─────────────────────────────────────────────────────────────────────────── #
state = orchestrator.get_state()

st.markdown(
    f"""
    <div class="mas-header">
      <div class="mas-title-group">
        <div class="mas-icon">⚓</div>
        <div>
          <h1>MaritimeMAS — Gulf of Mexico Operations</h1>
          <p>Real-time multi-agent system · Satellite imagery · NOAA & Global Fishing Watch data</p>
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

# ─────────────────────────────────────────────────────────────────────────── #
#  Top KPI Strip — key metrics visible without opening the sidebar           #
# ─────────────────────────────────────────────────────────────────────────── #
_m = state["metrics"]
st.markdown(
    f"""
    <div class="kpi-row">
      <div class="kpi-card">
        <div class="kpi-label">⛽ Fuel Savings</div>
        <div class="kpi-value accent-blue">{_m['fuel_savings_pct']:.1f}%</div>
        <div class="kpi-sub">Optimized route vs. naive straight line</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">🚨 Dark Vessels Flagged</div>
        <div class="kpi-value accent-red">{_m['vessels_flagged']}</div>
        <div class="kpi-sub">Precision {_m['precision']:.0%} · Recall {_m['recall']:.0%}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">🗑️ Debris Hotspots Covered</div>
        <div class="kpi-value accent-amber">{_m['hotspots_covered']}</div>
        <div class="kpi-sub">Nearest-collector assignment</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">🔄 Dynamic Reroutes</div>
        <div class="kpi-value accent-green">{_m['reroute_count']}</div>
        <div class="kpi-sub">Cross-agent obstacle avoidance</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────── #
#  Sidebar Controls                                                           #
# ─────────────────────────────────────────────────────────────────────────── #
with st.sidebar:
    st.markdown('<div class="section-heading">🎛️ Simulation Controls</div>', unsafe_allow_html=True)

    col_play1, col_play2 = st.columns(2)
    with col_play1:
        tick_click = st.button("⏭️ Next Tick", use_container_width=True)
    with col_play2:
        play_click = st.button(
            "⏸️ Pause" if st.session_state.auto_play else "▶️ Auto-Play",
            type="primary" if st.session_state.auto_play else "secondary",
            use_container_width=True,
        )

    if play_click:
        st.session_state.auto_play = not st.session_state.auto_play

    st.session_state.playback_speed = st.slider("Playback speed (sec / tick)", 1, 6, 2)
    if st.session_state.auto_play:
        st.caption("🟢 Auto-play running — the simulation advances automatically.")

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown('<div class="section-heading">🌀 Special Demo Modes</div>', unsafe_allow_html=True)

    storm_active = state.get("storm_mode_active", False)
    if st.button(
        "🌩️ Disable Storm Replay" if storm_active else "🌀 Replay Hurricane Ida (Cat 4)",
        type="primary" if not storm_active else "secondary",
        use_container_width=True,
        help="Replays NOAA HURDAT2 historical track data for Hurricane Ida (Aug 2021).",
    ):
        orchestrator.toggle_storm_mode()
        st.rerun()
    if storm_active:
        st.caption("⚠️ Storm mode active — Agent 1 is dynamically routing around the Cat 4 eye.")

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown('<div class="section-heading">🗺️ Map & Layers</div>', unsafe_allow_html=True)
    st.session_state.map_style_choice = st.selectbox(
        "Basemap",
        [
            "🛰️ Google Maps Satellite Hybrid",
            "🛰️ Esri World Imagery (High-Res)",
            "🌊 MarineTraffic Light Nautical",
            "🌍 OpenStreetMap Marine View",
        ],
        index=0,
    )
    st.session_state.show_gfw = st.checkbox("Show Global Fishing Watch zones", value=st.session_state.show_gfw)
    st.session_state.show_before_after = st.checkbox("Show naive vs. optimized route", value=st.session_state.show_before_after)

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown('<div class="section-heading">📊 Live Operations Metrics</div>', unsafe_allow_html=True)

    m = state["metrics"]

    st.markdown(
        f"""
        <div class="metric-grid">
          <div class="metric-card">
            <div class="metric-label">⛽ Fuel Savings</div>
            <div class="metric-value">{m['fuel_savings_pct']:.1f}%</div>
            <div class="metric-sub">vs. naive route</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">🚨 Flagged</div>
            <div class="metric-value">{m['vessels_flagged']}</div>
            <div class="metric-sub">P {m['precision']:.0%} · R {m['recall']:.0%}</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">🗑️ Hotspots</div>
            <div class="metric-value">{m['hotspots_covered']}</div>
            <div class="metric-sub">covered by collectors</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">🔄 Reroutes</div>
            <div class="metric-value">{m['reroute_count']}</div>
            <div class="metric-sub">cross-agent triggers</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown('<div class="section-heading">📡 Data Authenticity<span class="tag">source map</span></div>', unsafe_allow_html=True)
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

# ─────────────────────────────────────────────────────────────────────────── #
#  Handle Simulation Advance                                                   #
# ─────────────────────────────────────────────────────────────────────────── #
if tick_click:
    orchestrator.tick()

if st.session_state.auto_play:
    orchestrator.tick()
    time.sleep(st.session_state.playback_speed)
    st.rerun()

state = orchestrator.get_state()

# ─────────────────────────────────────────────────────────────────────────── #
#  Construct Folium Satellite & Nautical Map                                  #
# ─────────────────────────────────────────────────────────────────────────── #
choice = st.session_state.get("map_style_choice", "🛰️ Google Maps Satellite Hybrid")

# Gulf of Mexico Map Centre
m_lat, m_lon = 28.5, -91.8
folium_map = folium.Map(
    location=[m_lat, m_lon],
    zoom_start=6,
    tiles=None,
    control_scale=True,
)

# 1. Tile Basemap Layer Configuration
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
elif "MarineTraffic" in choice:
    folium.TileLayer(
        tiles="https://basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png",
        attr="Carto Voyager MarineTraffic",
        name="MarineTraffic Light Nautical",
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

# 2. Global Fishing Watch (GFW) Active Zones Layer
if st.session_state.show_gfw and state.get("fishing_zones"):
    fg_gfw = folium.FeatureGroup(name="Global Fishing Watch Zones")
    for fz in state["fishing_zones"]:
        poly_coords = [[p[0], p[1]] for p in fz["polygon"]]
        folium.Polygon(
            locations=poly_coords,
            color="#10b981",
            weight=2,
            fill=True,
            fill_color="#10b981",
            fill_opacity=0.22,
            popup=folium.Popup(
                f"<b>{fz['name']}</b><br/>Gear: {fz['gear']}<br/>Density: {fz['vessel_density']}",
                max_width=250,
            ),
        ).add_to(fg_gfw)
    fg_gfw.add_to(folium_map)

# 3. Hurricane Ida Storm Layer
if state.get("storm_mode_active") and state.get("storm_eye_pos"):
    se = state["storm_eye_pos"]
    fg_storm = folium.FeatureGroup(name="Hurricane Ida (NOAA Cat 4)")
    # Outer Gale Force Radius
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
    # Cat 4 Eye
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

# 4. Naive Straight-Line Baseline Route
if st.session_state.show_before_after and state.get("baseline_route"):
    bl_wps = state["baseline_route"].get("waypoints", [])
    if len(bl_wps) >= 2:
        folium.PolyLine(
            locations=[[wp[0], wp[1]] for wp in bl_wps],
            color="#ef4444",
            weight=3,
            dash_array="8, 8",
            opacity=0.8,
            popup="Naive Straight-Line Baseline Route",
        ).add_to(folium_map)

# 5. Optimized Weather Route Line
route = state.get("current_route", {})
waypoints = route.get("waypoints", [])
if len(waypoints) >= 2:
    folium.PolyLine(
        locations=[[wp[0], wp[1]] for wp in waypoints],
        color="#38bdf8",
        weight=5,
        opacity=0.95,
        popup=f"Optimized A* Route (Cost: {route.get('cost', 0):.1f}, Savings: {route.get('savings_pct', 0):.1f}%)",
    ).add_to(folium_map)

# 6. Debris Hotspots & Collectors
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

for col in state.get("collectors", []):
    clat, clon = col["lat"], col["lon"]
    folium.CircleMarker(
        location=[clat, clon],
        radius=7,
        color="#3b82f6",
        fill=True,
        fill_color="#60a5fa",
        fill_opacity=0.9,
        popup=f"<b>Collector {col['collector_id']}</b><br/>Status: {col['status']}<br/>Assigned Hotspot: {col.get('assigned_hotspot') or 'None'}",
    ).add_to(folium_map)

    # Line to assigned hotspot
    if col.get("assigned_hotspot") and col["assigned_hotspot"] in {h["hotspot_id"]: h["center"] for h in state.get("debris_hotspots", [])}:
        hc = {h["hotspot_id"]: h["center"] for h in state.get("debris_hotspots", [])}[col["assigned_hotspot"]]
        folium.PolyLine(
            locations=[[clat, clon], [hc[0], hc[1]]],
            color="#3b82f6",
            weight=2,
            dash_array="5, 5",
            opacity=0.7,
        ).add_to(folium_map)

# Helper for Directional Ship SVG Icons
def make_ship_icon(heading_deg: float, color: str = "#10b981", size: int = 22):
    svg = f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" style="transform: rotate({heading_deg}deg); transform-origin: center;">
        <path d="M12 2 L19 21 L12 17 L5 21 Z" fill="{color}" stroke="#ffffff" stroke-width="1.5"/>
    </svg>'''
    return folium.DivIcon(
        html=f'<div style="width:{size}px;height:{size}px;display:flex;align-items:center;justify-content:center;">{svg}</div>',
        icon_size=(size, size),
        icon_anchor=(size // 2, size // 2),
    )

# 7. Vessel Tracks & Directional Ship Icons
vessel_wakes = state.get("vessel_wakes", {})
for v in state.get("vessel_positions", []):
    vid = v["vessel_id"]
    is_flagged = v.get("flagged", False)
    vlat, vlon = v["lat"], v["lon"]
    heading = v.get("heading", 0)

    # Wake Trail
    w_pts = vessel_wakes.get(vid, [])
    if len(w_pts) >= 2:
        folium.PolyLine(
            locations=[[pt[0], pt[1]] for pt in w_pts],
            color="#ef4444" if is_flagged else "#34d399",
            weight=2,
            opacity=0.65,
        ).add_to(folium_map)

    # Vessel Directional Ship Icon
    ship_color = "#ef4444" if is_flagged else "#10b981"
    ship_size = 26 if is_flagged else 20

    popup_html = f"""
    <div style="font-family:Inter,sans-serif;font-size:0.8rem;color:#0f172a">
      <b>Vessel {vid} ({v.get('vessel_type', 'CARGO')})</b><br/>
      <b>Status:</b> {'🚨 FLAGGED DARK' if is_flagged else '🟢 Normal'}<br/>
      <b>Speed:</b> {v.get('speed', 0)} kts &nbsp;|&nbsp; <b>Heading:</b> {heading}°<br/>
      <b>Pos:</b> {vlat:.4f}°N, {vlon:.4f}°W<br/>
    """
    if is_flagged:
        popup_html += f"""
        <hr style="margin:4px 0"/>
        <b>IsolationForest Confidence:</b> {v.get('confidence', 0):.0%}<br/>
        <b>Transponder Gap:</b> {v.get('max_gap_minutes', 0)} min<br/>
        <b>Displacement Error:</b> {v.get('displacement_error_km', 0)} km<br/>
        <b>Reason:</b> {v.get('explanation', '')}
        """
    popup_html += "</div>"

    folium.Marker(
        location=[vlat, vlon],
        icon=make_ship_icon(heading, color=ship_color, size=ship_size),
        popup=folium.Popup(popup_html, max_width=280),
    ).add_to(folium_map)

# 8. Origin / Destination Port Markers (Marine Waters)
folium.Marker(
    location=list(orchestrator.ROUTE_ORIGIN),
    popup="<b>Galveston Entrance Channel Port</b><br/>Route Origin (29.30°N, 94.75°W)",
    icon=folium.Icon(color="green", icon="anchor", prefix="fa"),
).add_to(folium_map)

folium.Marker(
    location=list(orchestrator.ROUTE_DESTINATION),
    popup="<b>Mississippi River South Pass Approach</b><br/>Route Destination (29.10°N, 89.50°W)",
    icon=folium.Icon(color="blue", icon="flag", prefix="fa"),
).add_to(folium_map)

# ─────────────────────────────────────────────────────────────────────────── #
#  Main UI Layout                                                             #
# ─────────────────────────────────────────────────────────────────────────── #
map_col, feed_col = st.columns([3.2, 1.3])

with map_col:
    st.markdown('<div class="section-heading">🗺️ Live Operations Map</div>', unsafe_allow_html=True)

    # Map Legend
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

    # Render Folium Satellite Map
    st_folium(folium_map, use_container_width=True, height=540)

    # Route Comparison Info Bar
    if route:
        bl_cost = route.get("baseline_cost", 0)
        opt_cost = route.get("cost", 0)
        sav = route.get("savings_pct", 0)
        st.markdown(
            f"""
            <div class="route-bar">
              <div>📍 <b>Origin:</b> Houston/Galveston (29.3°N, 94.8°W) &nbsp;➔&nbsp; 📍 <b>Dest:</b> New Orleans (29.9°N, 90.1°W)</div>
              <div><b>Naive:</b> <span style="color:#f87171">{bl_cost:.1f}</span> &nbsp;·&nbsp; <b>Optimized:</b> <span style="color:#38bdf8">{opt_cost:.1f}</span> &nbsp;·&nbsp; <b>Fuel saved:</b> <span style="color:#34d399;font-weight:700">{sav:.1f}%</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with feed_col:
    st.markdown('<div class="section-heading">📡 Real-Time Event Feed</div>', unsafe_allow_html=True)
    event_log = state.get("event_log", [])
    if not event_log:
        st.info("System initializing...")

    # Collapse consecutive duplicate messages (strip the leading timestamp so
    # repeats of the same underlying event are grouped as "seen ×N" rather
    # than flooding the feed with identical lines.
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
        is_warn = "⚠️" in ev or "Rerouted" in ev or "flagged" in ev.lower()
        is_crit = "Hurricane" in ev or "ALERT" in ev
        cls = "crit" if is_crit else ("warn" if is_warn else "")
        count_badge = f'<span class="event-count">×{g["count"]}</span>' if g["count"] > 1 else ""
        st.markdown(
            f'<div class="event-item {cls}"><span>{ev}</span>{count_badge}</div>',
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────────────────────────────────── #
#  Timeline Scrubber (Past State Playback)                                     #
# ─────────────────────────────────────────────────────────────────────────── #
st.markdown("---")
st.markdown('<div class="section-heading">⏱️ Simulation Timeline Scrubber</div>', unsafe_allow_html=True)

history = orchestrator.get_tick_history()
if history:
    max_t = len(history) - 1
    selected_t = st.slider("Scrub timeline to replay past state snapshots", 0, max_t, max_t)
    if selected_t != max_t:
        snap = history[selected_t]
        st.info(f"⏪ Replaying state from **Tick #{snap['tick']}** — {len(snap['vessel_positions'])} vessels tracked. Move the slider to the far right to return to live.")
else:
    st.caption("No snapshots yet — advance the simulation to build timeline history.")

# ─────────────────────────────────────────────────────────────────────────── #
#  Explainability & Breakdown Panels                                          #
# ─────────────────────────────────────────────────────────────────────────── #
st.markdown("---")
with st.expander("🔍 AI Explainability — \"Why was this flagged?\" & data inspection", expanded=True):
    flagged = state.get("flagged_vessels", [])
    if flagged:
        st.markdown("##### 🚨 Flagged Dark Vessels — Anomaly Analysis")
        for fv in flagged:
            vid = fv["vessel_id"]
            conf = fv.get("confidence", 0)
            max_g = fv.get("max_gap_minutes", 0)
            disp = fv.get("displacement_error_km", 0)
            spd = fv.get("speed_change_after_gap", 0)
            hdg = fv.get("heading_change_after_gap", 0)
            expl = fv.get("explanation", "Flagged by anomaly detector.")

            st.markdown(
                f"""
                <div class="explain-box">
                  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
                    <b>Vessel {vid}</b>
                    <span class="badge badge-syn">Confidence: {conf:.0%}</span>
                  </div>
                  <div>{expl}</div>
                  <div style="margin-top:8px;display:flex;gap:18px;font-size:0.75rem">
                    <span>Transponder Gap: <span class="explain-metric">{max_g} min</span></span>
                    <span>Position Discrepancy: <span class="explain-metric">{disp} km</span></span>
                    <span>Speed Delta: <span class="explain-metric">{spd} kts</span></span>
                    <span>Heading Delta: <span class="explain-metric">{hdg}°</span></span>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.success("No dark vessels flagged in current scan frame.")

    tab_v, tab_d, tab_s = st.tabs(["🛳️ Tracked Vessels Table", "🗑️ Debris & Collector Status", "🌩️ Storm & Weather Data"])

    with tab_v:
        if state["vessel_positions"]:
            vdf = pd.DataFrame(state["vessel_positions"])
            st.dataframe(vdf, use_container_width=True, height=260)

    with tab_d:
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.subheader("Debris Hotspots (DBSCAN Clustered)")
            if state["debris_hotspots"]:
                st.dataframe(pd.DataFrame(state["debris_hotspots"]), use_container_width=True)
        with col_d2:
            st.subheader("Collector Fleet Status")
            if state["collectors"]:
                st.dataframe(pd.DataFrame(state["collectors"]), use_container_width=True)

    with tab_s:
        if state.get("storm_df") is not None:
            st.subheader("NOAA HURDAT2 Historical Storm Track (Hurricane Ida, Aug 2021)")
            st.dataframe(state["storm_df"], use_container_width=True)

st.markdown(
    """
    <div class="mas-footer">
      MaritimeMAS · Weather via Open-Meteo Marine API · Storm track via NOAA HURDAT2 ·
      Fishing zones via Global Fishing Watch · AIS &amp; debris data synthetic for demo purposes
    </div>
    """,
    unsafe_allow_html=True,
)