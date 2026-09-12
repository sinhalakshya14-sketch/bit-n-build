"""
agents/dark_vessel.py
=====================
AGENT 2 — DARK VESSEL DETECTION

Process:
1. From AIS data, randomly delete pings for a subset of vessels within a
   chosen time window, simulating "going dark" (AIS transponder off).
2. Engineer features per vessel: gap duration, position displacement error
   (actual reappearance vs. dead-reckoned straight-line), speed/heading change.
3. Run sklearn IsolationForest on all vessels' features.
4. Flag vessels whose anomaly score exceeds a threshold.
5. Report precision/recall against the ground-truth tampered set.

Public API:
    detect_dark_vessels(ais_df) -> list[dict]
"""

import logging
import math
import random
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Geometry helper                                                             #
# --------------------------------------------------------------------------- #
def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


# --------------------------------------------------------------------------- #
#  Step 1 — Inject synthetic gaps                                             #
# --------------------------------------------------------------------------- #
def _inject_dark_windows(
    ais_df: pd.DataFrame,
    dark_fraction: float = 0.25,
    min_gap_pings: int = 3,
    max_gap_pings: int = 8,
    seed: int = 99,
) -> tuple[pd.DataFrame, set[str]]:
    """
    Randomly remove consecutive ping windows for a subset of vessels.
    Returns the modified DataFrame and the set of tampered vessel_ids.
    """
    random.seed(seed)
    np.random.seed(seed)
    vessels = ais_df["vessel_id"].unique().tolist()
    n_dark = max(1, int(len(vessels) * dark_fraction))
    dark_vessels = set(random.sample(vessels, n_dark))

    rows_to_drop = []
    for vid in dark_vessels:
        vdf = ais_df[ais_df["vessel_id"] == vid].sort_values("timestamp")
        n = len(vdf)
        gap_size = random.randint(min_gap_pings, min(max_gap_pings, n // 2))
        start_idx = random.randint(1, max(1, n - gap_size - 1))
        rows_to_drop.extend(vdf.index[start_idx : start_idx + gap_size].tolist())

    modified_df = ais_df.drop(index=rows_to_drop).reset_index(drop=True)
    logger.info("Dark vessel injection: %d vessels tampered, %d pings removed",
                len(dark_vessels), len(rows_to_drop))
    return modified_df, dark_vessels


# --------------------------------------------------------------------------- #
#  Step 2 — Feature engineering                                               #
# --------------------------------------------------------------------------- #
def _engineer_features(ais_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-vessel features that characterise suspicious gap behaviour.

    Features:
      max_gap_minutes       - longest consecutive ping gap (minutes)
      mean_gap_minutes      - mean gap between pings
      displacement_error_km - max distance between dead-reckoned and actual
                              position after a gap
      speed_change_after_gap- mean absolute speed change across gaps
      heading_change_after_gap - mean absolute heading change across gaps
      ping_count            - total number of pings (dark vessels have fewer)
    """
    records = []
    for vid, grp in ais_df.groupby("vessel_id"):
        grp = grp.sort_values("timestamp").reset_index(drop=True)
        timestamps = grp["timestamp"].tolist()
        lats = grp["lat"].tolist()
        lons = grp["lon"].tolist()
        speeds = grp["speed"].tolist()
        headings = grp["heading"].tolist()
        n = len(grp)

        if n < 2:
            records.append(
                {
                    "vessel_id": vid,
                    "max_gap_minutes": 0,
                    "mean_gap_minutes": 0,
                    "displacement_error_km": 0,
                    "speed_change_after_gap": 0,
                    "heading_change_after_gap": 0,
                    "ping_count": n,
                    "last_lat": lats[0] if lats else 0,
                    "last_lon": lons[0] if lons else 0,
                    "last_timestamp": timestamps[0] if timestamps else None,
                }
            )
            continue

        gap_minutes = []
        disp_errors = []
        speed_changes = []
        heading_changes = []

        for i in range(1, n):
            gap_m = (timestamps[i] - timestamps[i - 1]).total_seconds() / 60
            gap_minutes.append(gap_m)

            if gap_m > 15:  # only meaningful gaps
                # Dead-reckon: project position using last speed (kts) & heading
                spd_km_per_min = speeds[i - 1] * 1.852 / 60
                dr_lat = lats[i - 1] + (spd_km_per_min * gap_m * math.cos(math.radians(headings[i - 1]))) / 111.0
                dr_lon = lons[i - 1] + (spd_km_per_min * gap_m * math.sin(math.radians(headings[i - 1]))) / (
                    111.0 * math.cos(math.radians(lats[i - 1]))
                )
                err = _haversine(dr_lat, dr_lon, lats[i], lons[i])
                disp_errors.append(err)
                speed_changes.append(abs(speeds[i] - speeds[i - 1]))
                heading_changes.append(min(abs(headings[i] - headings[i - 1]), 360 - abs(headings[i] - headings[i - 1])))

        records.append(
            {
                "vessel_id": vid,
                "max_gap_minutes": max(gap_minutes, default=0),
                "mean_gap_minutes": float(np.mean(gap_minutes)) if gap_minutes else 0,
                "displacement_error_km": float(np.max(disp_errors)) if disp_errors else 0,
                "speed_change_after_gap": float(np.mean(speed_changes)) if speed_changes else 0,
                "heading_change_after_gap": float(np.mean(heading_changes)) if heading_changes else 0,
                "ping_count": n,
                "last_lat": lats[-1],
                "last_lon": lons[-1],
                "last_timestamp": timestamps[-1],
            }
        )

    return pd.DataFrame(records)


# --------------------------------------------------------------------------- #
#  Step 3 — IsolationForest detection                                        #
# --------------------------------------------------------------------------- #
FEATURE_COLS = [
    "max_gap_minutes",
    "mean_gap_minutes",
    "displacement_error_km",
    "speed_change_after_gap",
    "heading_change_after_gap",
    "ping_count",
]


def _run_isolation_forest(feat_df: pd.DataFrame, contamination: float = 0.25) -> pd.DataFrame:
    X = feat_df[FEATURE_COLS].fillna(0).values
    clf = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
    clf.fit(X)
    scores = clf.decision_function(X)  # lower = more anomalous
    preds = clf.predict(X)             # -1 = anomaly

    feat_df = feat_df.copy()
    feat_df["anomaly_score"] = scores
    feat_df["flagged"] = preds == -1
    # Normalise score to [0, 1] confidence (invert so 1 = most suspicious)
    min_s, max_s = scores.min(), scores.max()
    if max_s > min_s:
        feat_df["confidence"] = 1 - (scores - min_s) / (max_s - min_s)
    else:
        feat_df["confidence"] = 0.5
    return feat_df


# --------------------------------------------------------------------------- #
#  Public API                                                                  #
# --------------------------------------------------------------------------- #
_cached_ground_truth: set[str] = set()


def detect_dark_vessels(ais_df: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Full pipeline: inject gaps → engineer features → IsolationForest → report.

    Returns list of dicts:
      vessel_id, flagged (bool), confidence (0-1), last_known_position,
      gap_start_time, max_gap_minutes
    Also prints precision/recall to the logger.
    """
    global _cached_ground_truth

    modified_df, dark_vessels = _inject_dark_windows(ais_df)
    _cached_ground_truth = dark_vessels

    feat_df = _engineer_features(modified_df)
    feat_df = _run_isolation_forest(feat_df)

    # Precision / recall against ground truth
    y_true = feat_df["vessel_id"].isin(dark_vessels).astype(int).tolist()
    y_pred = feat_df["flagged"].astype(int).tolist()
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    logger.info("Dark vessel detection — Precision: %.2f  Recall: %.2f", prec, rec)
    print(f"[DarkVessel] Precision={prec:.2f}  Recall={rec:.2f}  "
          f"(ground truth dark: {len(dark_vessels)}, flagged: {feat_df['flagged'].sum()})")

    # Build last-known position from modified (gap-injected) AIS
    last_pos = (
        modified_df.sort_values("timestamp")
        .groupby("vessel_id")
        .last()
        [["lat", "lon", "timestamp"]]
        .rename(columns={"timestamp": "last_timestamp"})
    )

    results = []
    for _, row in feat_df.iterrows():
        vid = row["vessel_id"]
        pos = last_pos.loc[vid] if vid in last_pos.index else None
        flagged = bool(row["flagged"])
        max_gap = round(float(row["max_gap_minutes"]), 1)
        disp_err = round(float(row["displacement_error_km"]), 1)
        spd_chg = round(float(row["speed_change_after_gap"]), 1)
        hdg_chg = round(float(row["heading_change_after_gap"]), 1)
        
        if flagged:
            explanation = (
                f"Flagged by IsolationForest: AIS transponder inactive for {max_gap} min. "
                f"Reappeared {disp_err} km off dead-reckoning trajectory with {spd_chg} kt speed jump "
                f"and {hdg_chg}° heading shift."
            )
        else:
            explanation = "Normal transmission pattern within expected operational limits."

        results.append(
            {
                "vessel_id": vid,
                "flagged": flagged,
                "confidence": round(float(row["confidence"]), 3),
                "last_known_position": (
                    [round(float(pos["lat"]), 5), round(float(pos["lon"]), 5)] if pos is not None else [0, 0]
                ),
                "last_timestamp": str(pos["last_timestamp"]) if pos is not None else "",
                "max_gap_minutes": max_gap,
                "displacement_error_km": disp_err,
                "speed_change_after_gap": spd_chg,
                "heading_change_after_gap": hdg_chg,
                "explanation": explanation,
                "is_ground_truth_dark": vid in dark_vessels,
            }
        )

    flagged_count = sum(1 for r in results if r["flagged"])
    logger.info("Dark vessel results: %d/%d flagged", flagged_count, len(results))
    return results


def get_precision_recall(ais_df: pd.DataFrame) -> tuple[float, float]:
    """Convenience for dashboard metrics."""
    results = detect_dark_vessels(ais_df)
    y_true = [int(r["is_ground_truth_dark"]) for r in results]
    y_pred = [int(r["flagged"]) for r in results]
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    return prec, rec
