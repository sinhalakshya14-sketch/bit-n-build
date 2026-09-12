"""
agents/dark_vessel.py
=====================
AGENT 2 — DARK VESSEL DETECTION

Ground-truth generation and detection are independent:

  * Ground truth (labels only): `_inject_dark_windows` deletes ping windows
    for a labelled subset. A second process injects *benign* coverage holes
    that are NOT labelled dark. Detection never reads these labels.

  * Detection: IsolationForest is fit on a random vessel subset with a
    contamination prior that is *not* the injection rate, then every vessel
    is scored. A separate IMO-style gap rule is OR-ed in. Precision/recall
    are computed afterwards against held-out labels.

Public API:
    detect_dark_vessels(ais_df) -> list[dict]
    get_precision_recall(ais_df) -> (precision, recall)
"""

from __future__ import annotations

import logging
import math
import random
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score

logger = logging.getLogger(__name__)

# --- Ground-truth injection (labels). Detector must not copy these rates. ---
DARK_FRACTION = 0.12          # labelled "went dark"
BENIGN_GAP_FRACTION = 0.10    # coverage holes, NOT labelled dark
DARK_GAP_PINGS = (10, 20)     # ~100–200 min at 6 pings/hour
BENIGN_GAP_PINGS = (2, 4)     # ~20–40 min — below IMO flag threshold
INJECT_SEED = 99

# --- Detection (independent of the fractions above) ---
# Class A AIS underway typically reports every 2–10 s (fast) to ~3 min (slow).
# A 45-minute silence while supposedly underway is outside normal reporting
# even allowing for coastal shadowing; 15 min is used only to compute
# dead-reckoning error, not to flag.
DR_GAP_MINUTES = 15.0
FLAG_GAP_MINUTES = 45.0
FLAG_DR_ERROR_KM = 8.0
IF_CONTAMINATION = 0.06       # model prior — not DARK_FRACTION
IF_TRAIN_FRACTION = 0.65
IF_RANDOM_STATE = 42


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _drop_ping_window(ais_df: pd.DataFrame, vid: str, gap_lo: int, gap_hi: int, rng: random.Random) -> list:
    vdf = ais_df[ais_df["vessel_id"] == vid].sort_values("timestamp")
    n = len(vdf)
    if n < 8:
        return []
    gap_size = rng.randint(gap_lo, min(gap_hi, n // 3))
    start_idx = rng.randint(1, max(1, n - gap_size - 1))
    return vdf.index[start_idx : start_idx + gap_size].tolist()


def inject_ground_truth(
    ais_df: pd.DataFrame,
    dark_fraction: float = DARK_FRACTION,
    benign_fraction: float = BENIGN_GAP_FRACTION,
    seed: int = INJECT_SEED,
) -> tuple[pd.DataFrame, set[str], set[str]]:
    """
    Label-producing simulator only.

    Returns modified AIS, the dark vessel_id set (positives), and the benign
    coverage-hole set (true negatives that still have irregular gaps).
    """
    rng = random.Random(seed)
    vessels = ais_df["vessel_id"].unique().tolist()
    n_dark = max(1, int(len(vessels) * dark_fraction))
    n_benign = max(1, int(len(vessels) * benign_fraction))
    shuffled = vessels[:]
    rng.shuffle(shuffled)
    dark_vessels = set(shuffled[:n_dark])
    benign_vessels = set(shuffled[n_dark : n_dark + n_benign])

    rows_to_drop: list = []
    for vid in dark_vessels:
        rows_to_drop.extend(_drop_ping_window(ais_df, vid, DARK_GAP_PINGS[0], DARK_GAP_PINGS[1], rng))
    for vid in benign_vessels:
        rows_to_drop.extend(_drop_ping_window(ais_df, vid, BENIGN_GAP_PINGS[0], BENIGN_GAP_PINGS[1], rng))

    modified_df = ais_df.drop(index=rows_to_drop).reset_index(drop=True)
    logger.info(
        "GT inject: %d dark, %d benign holes, %d pings removed",
        len(dark_vessels),
        len(benign_vessels),
        len(rows_to_drop),
    )
    return modified_df, dark_vessels, benign_vessels


def _engineer_features(ais_df: pd.DataFrame) -> pd.DataFrame:
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
                    "max_gap_minutes": 0.0,
                    "mean_gap_minutes": 0.0,
                    "displacement_error_km": 0.0,
                    "speed_change_after_gap": 0.0,
                    "heading_change_after_gap": 0.0,
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
            if gap_m <= DR_GAP_MINUTES:
                continue
            spd_km_per_min = speeds[i - 1] * 1.852 / 60
            dr_lat = lats[i - 1] + (spd_km_per_min * gap_m * math.cos(math.radians(headings[i - 1]))) / 111.0
            dr_lon = lons[i - 1] + (spd_km_per_min * gap_m * math.sin(math.radians(headings[i - 1]))) / (
                111.0 * math.cos(math.radians(lats[i - 1]))
            )
            disp_errors.append(_haversine(dr_lat, dr_lon, lats[i], lons[i]))
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


FEATURE_COLS = [
    "max_gap_minutes",
    "mean_gap_minutes",
    "displacement_error_km",
    "speed_change_after_gap",
    "heading_change_after_gap",
    "ping_count",
]


def _run_isolation_forest(feat_df: pd.DataFrame) -> pd.DataFrame:
    """Fit IF on a random vessel subset; score the full population."""
    X = feat_df[FEATURE_COLS].fillna(0).to_numpy(dtype=float)
    n = len(feat_df)
    rng = np.random.RandomState(IF_RANDOM_STATE)
    train_n = max(20, int(n * IF_TRAIN_FRACTION))
    train_idx = rng.choice(n, size=min(train_n, n), replace=False)

    clf = IsolationForest(
        n_estimators=120,
        contamination=IF_CONTAMINATION,
        random_state=IF_RANDOM_STATE,
    )
    clf.fit(X[train_idx])
    decision = clf.decision_function(X)  # >0 inlier; <0 anomaly
    pred = clf.predict(X)

    train_mean = X[train_idx].mean(axis=0)
    train_std = np.clip(X[train_idx].std(axis=0), 1e-6, None)
    z = (X - train_mean) / train_std

    out = feat_df.copy()
    out["if_decision"] = decision
    out["if_flagged"] = pred == -1
    # Confidence from the model's signed score, not a batch min-max rank.
    out["confidence"] = 1.0 / (1.0 + np.exp(np.clip(decision * 10.0, -20, 20)))
    for i, col in enumerate(FEATURE_COLS):
        out[f"z_{col}"] = z[:, i]
    return out


def _rule_flag(row: pd.Series) -> tuple[bool, str]:
    """IMO-style silence rule — independent of IsolationForest."""
    if row["max_gap_minutes"] >= FLAG_GAP_MINUTES and row["displacement_error_km"] >= FLAG_DR_ERROR_KM:
        return True, (
            f"Rule: gap {row['max_gap_minutes']:.0f} min ≥ {FLAG_GAP_MINUTES:.0f} min "
            f"(Class A underway normally reports within ~3 min; 45 min is used as a "
            f"coastal-shadowing allowance) and reappearance is "
            f"{row['displacement_error_km']:.1f} km off dead-reckoning "
            f"(threshold {FLAG_DR_ERROR_KM:.0f} km)."
        )
    if row["max_gap_minutes"] >= FLAG_GAP_MINUTES:
        return False, (
            f"Gap {row['max_gap_minutes']:.0f} min exceeds {FLAG_GAP_MINUTES:.0f} min but "
            f"reappearance is only {row['displacement_error_km']:.1f} km off DR "
            f"(<{FLAG_DR_ERROR_KM:.0f} km) — consistent with a coverage hole, not a course change while dark."
        )
    return False, ""


def _top_features(row: pd.Series, k: int = 3) -> list[tuple[str, float]]:
    ranked = sorted(
        ((col, float(row[f"z_{col}"])) for col in FEATURE_COLS),
        key=lambda t: abs(t[1]),
        reverse=True,
    )
    return ranked[:k]


def detect_dark_vessels(ais_df: pd.DataFrame) -> list[dict[str, Any]]:
    modified_df, dark_vessels, _benign = inject_ground_truth(ais_df)
    feat_df = _engineer_features(modified_df)
    feat_df = _run_isolation_forest(feat_df)

    last_pos = (
        modified_df.sort_values("timestamp")
        .groupby("vessel_id")
        .last()[["lat", "lon", "timestamp"]]
        .rename(columns={"timestamp": "last_timestamp"})
    )

    results = []
    for _, row in feat_df.iterrows():
        vid = row["vessel_id"]
        pos = last_pos.loc[vid] if vid in last_pos.index else None
        rule_hit, rule_text = _rule_flag(row)
        if_hit = bool(row["if_flagged"])
        flagged = if_hit or rule_hit
        reasons = []
        if if_hit:
            reasons.append(
                f"IsolationForest (trained on {IF_TRAIN_FRACTION:.0%} of vessels, "
                f"contamination prior {IF_CONTAMINATION:.0%}, not the label rate) "
                f"decision={row['if_decision']:.3f} (<0 ⇒ anomaly)."
            )
        if rule_text:
            reasons.append(rule_text)

        top = _top_features(row)
        contrib = ", ".join(f"{name} z={z:.1f}" for name, z in top)
        max_gap = round(float(row["max_gap_minutes"]), 1)
        disp_err = round(float(row["displacement_error_km"]), 1)
        spd_chg = round(float(row["speed_change_after_gap"]), 1)
        hdg_chg = round(float(row["heading_change_after_gap"]), 1)

        if flagged:
            explanation = (
                f"AIS silence {max_gap} min; reappeared {disp_err} km from the dead-reckoned "
                f"position (last speed/heading projected across the gap). "
                + " ".join(reasons)
                + f" Largest feature z-scores vs train split: {contrib}."
            )
        else:
            explanation = (
                f"Reporting pattern within limits (max gap {max_gap} min, DR error {disp_err} km). "
                f"IF decision={row['if_decision']:.3f}."
            )

        results.append(
            {
                "vessel_id": vid,
                "flagged": flagged,
                "if_flagged": if_hit,
                "rule_flagged": rule_hit,
                "confidence": round(float(row["confidence"]), 3),
                "if_decision": round(float(row["if_decision"]), 4),
                "last_known_position": (
                    [round(float(pos["lat"]), 5), round(float(pos["lon"]), 5)] if pos is not None else [0, 0]
                ),
                "last_timestamp": str(pos["last_timestamp"]) if pos is not None else "",
                "max_gap_minutes": max_gap,
                "displacement_error_km": disp_err,
                "speed_change_after_gap": spd_chg,
                "heading_change_after_gap": hdg_chg,
                "feature_z": {name: round(z, 2) for name, z in top},
                "flag_gap_threshold_min": FLAG_GAP_MINUTES,
                "flag_dr_threshold_km": FLAG_DR_ERROR_KM,
                "explanation": explanation,
                "is_ground_truth_dark": vid in dark_vessels,
            }
        )

    y_true = [int(r["is_ground_truth_dark"]) for r in results]
    y_pred = [int(r["flagged"]) for r in results]
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    flagged_count = sum(y_pred)
    logger.info("Dark vessel detection — Precision: %.2f  Recall: %.2f", prec, rec)
    print(
        f"[DarkVessel] Precision={prec:.2f}  Recall={rec:.2f}  "
        f"(ground truth dark: {len(dark_vessels)}, flagged: {flagged_count})"
    )
    return results


def get_precision_recall(ais_df: pd.DataFrame) -> tuple[float, float]:
    results = detect_dark_vessels(ais_df)
    y_true = [int(r["is_ground_truth_dark"]) for r in results]
    y_pred = [int(r["flagged"]) for r in results]
    return (
        float(precision_score(y_true, y_pred, zero_division=0)),
        float(recall_score(y_true, y_pred, zero_division=0)),
    )
