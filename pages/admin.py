"""
Admin Panel — Internal Use Only
================================
Password-gated read-only view of all analyst watchlist and notes data
across all codenames. No edit/delete controls.

Access: Set MARITIMEMAS_ADMIN_TOKEN in .env to enable.
"""
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

import os
import sys
import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import storage

st.set_page_config(
    page_title="MaritimeMAS Admin",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Dark theme CSS to match main dashboard ───
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: #060a12; color: #e2e8f0; }
    .block-container { padding-top: 1.2rem !important; max-width: 100% !important; }
    [data-testid="stToolbar"],
    [data-testid="stToolbarActions"],
    header[data-testid="stHeader"] [data-testid="stToolbar"],
    header[data-testid="stHeader"] [data-testid="stToolbarActions"],
    .stAppDeployButton,
    #MainMenu,
    footer,
    [data-testid="stStatusWidget"],
    [data-testid="stDecoration"],
    .viewerBadge_container__1QSob,
    .viewerBadge_link__1S137,
    [data-testid="manage-app-button"],
    #manage-app-button,
    button[kind="manageApp"],
    button[title*="Manage app" i],
    button[aria-label*="Manage app" i],
    div[class*="viewerBadge"],
    div[class*="manageApp"],
    div[class*="manage-app"],
    .leaflet-bottom.leaflet-right,
    .leaflet-control-attribution {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        width: 0 !important;
        height: 0 !important;
        pointer-events: none !important;
    }
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    [data-testid="stSidebarCollapsedControl"] {
        display: block !important;
        visibility: visible !important;
        z-index: 999999 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Admin Banner ───
st.markdown(
    """
    <div style="background:linear-gradient(135deg,rgba(127,29,29,0.85),rgba(153,27,27,0.85));
                border:1.5px solid rgba(248,113,113,0.4);border-radius:10px;
                padding:16px 22px;margin-bottom:20px;text-align:center;">
      <div style="font-size:1.3rem;font-weight:800;color:#fca5a5;letter-spacing:0.03em;">
        ⚙️ Admin Panel — Internal Use Only
      </div>
      <div style="font-size:0.82rem;color:#fecaca;margin-top:4px;">
        Read-only aggregate view of all analyst watchlist and notes data.
        No real names are stored — all identities are anonymous codenames.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─── Gate behind admin token ───
ADMIN_TOKEN = os.environ.get("MARITIMEMAS_ADMIN_TOKEN", "")

if not ADMIN_TOKEN:
    st.error(
        "**Admin access is disabled.** No `MARITIMEMAS_ADMIN_TOKEN` environment variable is set. "
        "Add it to your `.env` file to enable this panel."
    )
    st.stop()

entered_token = st.text_input(
    "🔒 Enter admin token to continue",
    type="password",
    key="admin_token_input",
    placeholder="Paste the admin token from .env",
)

if not entered_token:
    st.info("Enter the admin token above to access the aggregate data view.")
    st.stop()

if entered_token != ADMIN_TOKEN:
    st.error("❌ **Access denied** — incorrect admin token. Please try again.")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════ #
#  Authenticated — render admin content                                      #
# ═══════════════════════════════════════════════════════════════════════════ #

st.success("✅ Authenticated — admin access granted.")
st.markdown("---")

# ─── Summary Stats ───
stats = storage.get_summary_stats()
st.markdown("### 📊 Summary Statistics")
sc1, sc2, sc3 = st.columns(3)
with sc1:
    st.metric("Total Watchlist Entries", stats["total_watchlist"])
with sc2:
    st.metric("Total Case Notes", stats["total_notes"])
with sc3:
    st.metric("Distinct Codenames", stats["distinct_codenames"])

st.markdown("---")

# ─── All Watchlist Entries ───
st.markdown("### 📋 All Watchlist Entries (across all codenames)")
wl_entries = storage.get_all_watchlist_entries()
if wl_entries:
    wl_df = pd.DataFrame(wl_entries)
    wl_df.columns = ["Codename", "Vessel ID", "Added At (UTC)"]
    st.dataframe(wl_df, use_container_width=True, hide_index=True)
else:
    st.caption("No watchlist entries found.")

st.markdown("---")

# ─── All Notes ───
st.markdown("### 📝 All Case Notes (across all codenames)")
all_notes = storage.get_all_notes()
if all_notes:
    notes_df = pd.DataFrame(all_notes)
    notes_df.columns = ["Note ID", "Codename", "Vessel ID", "Note Text", "Created At (UTC)"]
    st.dataframe(notes_df, use_container_width=True, hide_index=True)
else:
    st.caption("No case notes found.")

st.markdown("---")
st.caption(
    "This is a **read-only** admin view. No edit or delete controls are available. "
    "All analyst identities shown are anonymous codenames — no real names are stored anywhere in the system."
)
