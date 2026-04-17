"""
Arrival / departure direction analysis.

For each departure, look FORWARD in the AIS track to the first position that
is ≥ min_dist_km from the port center, then compute the bearing from port to
that point (where the ship is heading).

For each arrival, look BACKWARD in the AIS track to the most recent position
that was ≥ min_dist_km from the port center, then compute the bearing (where
the ship came from).

Bearings bin into 8 compass sectors (N, NE, E, SE, S, SW, W, NW). Per-port
semantic labels live in PORT_DIRECTION_MAP — currently only SGSIN is populated.
For ports without a mapping, the bare compass sector is used.
"""
import math
import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================================
# Compass sectors (8-way) and per-port semantic labels
# ============================================================================

# Each entry: (label, center_deg). Half-width is 22.5°.
COMPASS_8 = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
COMPASS_CENTERS = {name: i * 45.0 for i, name in enumerate(COMPASS_8)}

# Singapore (SGSIN ~ 1.27°N 103.85°E) — major shipping lanes radiating out.
PORT_DIRECTION_MAP = {
    "SGSIN": {
        "N":  "Peninsular Malaysia",
        "NE": "South China Sea (HK / S. China)",
        "E":  "S. China Sea → Japan / Korea / Taiwan",
        "SE": "Indonesia (Borneo) / Pacific",
        "S":  "Indonesia (Java)",
        "SW": "Sunda Strait → Indian Ocean / Australia",
        "W":  "Sumatra coast",
        "NW": "Malacca → India / Middle East / Europe",
    },
}

SECTOR_COLORS = {
    "N":  "#e74c3c", "NE": "#e67e22", "E":  "#f39c12", "SE": "#27ae60",
    "S":  "#16a085", "SW": "#2980b9", "W":  "#8e44ad", "NW": "#c0392b",
}


# ============================================================================
# Geo helpers (haversine + bearing)
# ============================================================================

_EARTH_R_KM = 6371.0088


def _haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km. Vectorized over (lat2, lon2)."""
    p1 = math.radians(lat1)
    p2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = (np.sin(dphi / 2) ** 2
         + math.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2)
    return 2 * _EARTH_R_KM * np.arcsin(np.sqrt(a))


def _initial_bearing_deg(lat1, lon1, lat2, lon2):
    """Forward bearing in degrees (0–360, N=0, clockwise)."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlmb = math.radians(lon2 - lon1)
    y = math.sin(dlmb) * math.cos(p2)
    x = (math.cos(p1) * math.sin(p2)
         - math.sin(p1) * math.cos(p2) * math.cos(dlmb))
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def _bearing_to_sector(bearing_deg):
    """Bin a bearing into one of 8 compass sectors (centered every 45°)."""
    idx = int(((bearing_deg + 22.5) % 360) // 45)
    return COMPASS_8[idx]


# ============================================================================
# Core: compute bearings for each departure
# ============================================================================

def compute_post_departure_bearings(
    departures_df: pd.DataFrame,
    tracking_data: pd.DataFrame,
    port_list_df: pd.DataFrame,
    port_code: Optional[str] = None,
    min_dist_km: float = 100.0,
    max_lookforward_hours: float = 72.0,
) -> pd.DataFrame:
    """
    For each departure, find the first AIS point ≥ min_dist_km from the port
    center within max_lookforward_hours, and compute the bearing.

    Parameters
    ----------
    departures_df : DataFrame
        Output of detect_port_events — must have VESSEL_IMO, port_name,
        TIMESTAMP (or TIMESTAMP_UTC) = exit time, and vessel_type.
    tracking_data : DataFrame
        Raw AIS data with VESSEL_IMO, TIMESTAMP(_UTC), LATITUDE, LONGITUDE.
    port_list_df : DataFrame
        Port list with port, lat, lon.
    port_code : str | None
        If given, only compute for departures from this port.
    min_dist_km : float
        Distance threshold for the "outbound" position.
    max_lookforward_hours : float
        Don't search beyond this window after the departure.

    Returns
    -------
    DataFrame with one row per departure: VESSEL_IMO, port_name, exit_time,
    vessel_type, out_lat, out_lon, out_time, dist_km, bearing_deg, sector,
    direction_label, status.
        status ∈ {"ok", "no_track_after_exit", "stayed_within_radius"}.
    """
    dep_tc = ("TIMESTAMP_UTC" if "TIMESTAMP_UTC" in departures_df.columns
              else "TIMESTAMP")
    trk_tc = ("TIMESTAMP_UTC" if "TIMESTAMP_UTC" in tracking_data.columns
              else "TIMESTAMP")

    dep = departures_df.copy()
    dep[dep_tc] = pd.to_datetime(dep[dep_tc])
    if port_code is not None:
        dep = dep[dep["port_name"] == port_code].copy()
        if dep.empty:
            print(f"  [warn] No departures found for port {port_code}")
            return pd.DataFrame()

    trk = tracking_data.rename(
        columns={"SHIP_ID": "VESSEL_IMO",
                 "LAT": "LATITUDE", "LON": "LONGITUDE"}
    ).copy()
    trk[trk_tc] = pd.to_datetime(trk[trk_tc])

    # Restrict tracking to the vessels we care about
    vessels = dep["VESSEL_IMO"].unique()
    trk = (trk[trk["VESSEL_IMO"].isin(vessels)]
           .sort_values(["VESSEL_IMO", trk_tc]))

    # Index AIS rows by vessel for fast lookup
    by_vessel = {v: g.reset_index(drop=True)
                 for v, g in trk.groupby("VESSEL_IMO")}
    port_lookup = (port_list_df.set_index("port")[["lat", "lon"]]
                   .to_dict("index"))

    rows = []
    cutoff = pd.Timedelta(hours=max_lookforward_hours)

    for _, d in dep.iterrows():
        port = d["port_name"]
        if port not in port_lookup:
            continue
        plat, plon = port_lookup[port]["lat"], port_lookup[port]["lon"]

        exit_t = d[dep_tc]
        track = by_vessel.get(d["VESSEL_IMO"])
        base = {
            "VESSEL_IMO": d["VESSEL_IMO"], "port_name": port,
            "exit_time": exit_t, "vessel_type": d.get("vessel_type"),
        }

        if track is None or track.empty:
            rows.append({**base, "status": "no_track_after_exit"})
            continue

        future = track[(track[trk_tc] > exit_t)
                       & (track[trk_tc] <= exit_t + cutoff)]
        if future.empty:
            rows.append({**base, "status": "no_track_after_exit"})
            continue

        dists = _haversine_km(plat, plon,
                              future["LATITUDE"].values,
                              future["LONGITUDE"].values)
        far_idx = np.where(dists >= min_dist_km)[0]
        if far_idx.size == 0:
            rows.append({**base, "status": "stayed_within_radius",
                         "max_dist_km": float(dists.max())})
            continue

        i = far_idx[0]
        out_lat = float(future["LATITUDE"].iloc[i])
        out_lon = float(future["LONGITUDE"].iloc[i])
        out_time = future[trk_tc].iloc[i]
        bearing = _initial_bearing_deg(plat, plon, out_lat, out_lon)
        sector = _bearing_to_sector(bearing)
        label = PORT_DIRECTION_MAP.get(port, {}).get(sector, sector)

        rows.append({**base, "out_lat": out_lat, "out_lon": out_lon,
                     "out_time": out_time, "dist_km": float(dists[i]),
                     "bearing_deg": bearing, "sector": sector,
                     "direction_label": label, "status": "ok"})

    return pd.DataFrame(rows)


# ============================================================================
# Core: compute bearings for each arrival (where the ship came from)
# ============================================================================

def compute_pre_arrival_bearings(
    arrivals_df: pd.DataFrame,
    tracking_data: pd.DataFrame,
    port_list_df: pd.DataFrame,
    port_code: Optional[str] = None,
    min_dist_km: float = 100.0,
    max_lookback_hours: float = 72.0,
) -> pd.DataFrame:
    """
    For each arrival, find the MOST RECENT AIS point ≥ min_dist_km from the
    port within max_lookback_hours BEFORE the arrival, and compute the bearing
    from the port to that point (the direction the ship came from).

    Returns
    -------
    DataFrame with one row per arrival: VESSEL_IMO, port_name, arrive_time,
    vessel_type, in_lat, in_lon, in_time, dist_km, bearing_deg, sector,
    direction_label, status.
        status ∈ {"ok", "no_track_before_arrival", "stayed_within_radius"}.
    """
    arr_tc = ("TIMESTAMP_UTC" if "TIMESTAMP_UTC" in arrivals_df.columns
              else "TIMESTAMP")
    trk_tc = ("TIMESTAMP_UTC" if "TIMESTAMP_UTC" in tracking_data.columns
              else "TIMESTAMP")

    arr = arrivals_df.copy()
    arr[arr_tc] = pd.to_datetime(arr[arr_tc])
    if port_code is not None:
        arr = arr[arr["port_name"] == port_code].copy()
        if arr.empty:
            print(f"  [warn] No arrivals found for port {port_code}")
            return pd.DataFrame()

    trk = tracking_data.rename(
        columns={"SHIP_ID": "VESSEL_IMO",
                 "LAT": "LATITUDE", "LON": "LONGITUDE"}
    ).copy()
    trk[trk_tc] = pd.to_datetime(trk[trk_tc])

    vessels = arr["VESSEL_IMO"].unique()
    trk = (trk[trk["VESSEL_IMO"].isin(vessels)]
           .sort_values(["VESSEL_IMO", trk_tc]))

    by_vessel = {v: g.reset_index(drop=True)
                 for v, g in trk.groupby("VESSEL_IMO")}
    port_lookup = (port_list_df.set_index("port")[["lat", "lon"]]
                   .to_dict("index"))

    rows = []
    cutoff = pd.Timedelta(hours=max_lookback_hours)

    for _, a in arr.iterrows():
        port = a["port_name"]
        if port not in port_lookup:
            continue
        plat, plon = port_lookup[port]["lat"], port_lookup[port]["lon"]

        arrive_t = a[arr_tc]
        track = by_vessel.get(a["VESSEL_IMO"])
        base = {
            "VESSEL_IMO": a["VESSEL_IMO"], "port_name": port,
            "arrive_time": arrive_t, "vessel_type": a.get("vessel_type"),
        }

        if track is None or track.empty:
            rows.append({**base, "status": "no_track_before_arrival"})
            continue

        past = track[(track[trk_tc] < arrive_t)
                     & (track[trk_tc] >= arrive_t - cutoff)]
        if past.empty:
            rows.append({**base, "status": "no_track_before_arrival"})
            continue

        dists = _haversine_km(plat, plon,
                              past["LATITUDE"].values,
                              past["LONGITUDE"].values)
        far_idx = np.where(dists >= min_dist_km)[0]
        if far_idx.size == 0:
            rows.append({**base, "status": "stayed_within_radius",
                         "max_dist_km": float(dists.max())})
            continue

        # Most recent far point = last index (past is time-ascending)
        i = far_idx[-1]
        in_lat = float(past["LATITUDE"].iloc[i])
        in_lon = float(past["LONGITUDE"].iloc[i])
        in_time = past[trk_tc].iloc[i]
        bearing = _initial_bearing_deg(plat, plon, in_lat, in_lon)
        sector = _bearing_to_sector(bearing)
        label = PORT_DIRECTION_MAP.get(port, {}).get(sector, sector)

        rows.append({**base, "in_lat": in_lat, "in_lon": in_lon,
                     "in_time": in_time, "dist_km": float(dists[i]),
                     "bearing_deg": bearing, "sector": sector,
                     "direction_label": label, "status": "ok"})

    return pd.DataFrame(rows)


# ============================================================================
# Visualization
# ============================================================================

def _save(fig, save_dir, filename, dpi=300):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, filename)
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {path}")


def plot_departure_directions(
    bearings_df: pd.DataFrame,
    port_code: str,
    title_suffix: str = "",
    save_dir: Optional[str] = None,
):
    """
    Two-panel figure for departures from one port:
      - Left: polar rose of bearings (bin width = 22.5°), per vessel group.
      - Right: bar chart of named-region counts (uses PORT_DIRECTION_MAP).

    Returns the figure.
    """
    df = bearings_df[(bearings_df["port_name"] == port_code) &
                     (bearings_df["status"] == "ok")].copy()
    if df.empty:
        print(f"  [warn] No usable bearings for {port_code}")
        return None

    # Vessel grouping (keep consistent with visualization._map_vessel_group)
    def _vg(v):
        if pd.isna(v):
            return "Other"
        u = str(v).upper()
        if "CONTAINER" in u:
            return "Container"
        if any(k in u for k in ["TANKER", "CRUDE", "LNG", "GAS", "CHEMICAL"]):
            return "Tanker"
        return "Other"
    df["vessel_group"] = df["vessel_type"].apply(_vg)
    vg_order = ["Container", "Tanker", "Other"]
    vg_colors = {"Container": "#2980b9",
                 "Tanker": "#e74c3c", "Other": "#95a5a6"}

    fig = plt.figure(figsize=(18, 8))
    n_total = len(df)
    skipped = (bearings_df["port_name"] == port_code).sum() - n_total
    fig.suptitle(
        f"{port_code} departure directions {title_suffix}  "
        f"(n={n_total}, skipped={skipped})",
        fontsize=15, fontweight="bold", y=1.02,
    )

    # ---- LEFT: polar rose, stacked by vessel group ----
    ax_r = fig.add_subplot(1, 2, 1, projection="polar")
    ax_r.set_theta_zero_location("N")
    ax_r.set_theta_direction(-1)  # clockwise so 90° = E
    # 8 bins of 45°, each bar = one compass sector, centered on its tick
    centers_deg = np.array([COMPASS_CENTERS[s] for s in COMPASS_8])
    centers = np.deg2rad(centers_deg)
    width = np.deg2rad(45.0)

    # Shift bearings by +22.5° so histogram bins map 1-to-1 onto sectors.
    bottom = np.zeros(8)
    for vg in vg_order:
        sub = df[df["vessel_group"] == vg]
        if sub.empty:
            continue
        s_shift = (sub["bearing_deg"].values + 22.5) % 360
        counts, _ = np.histogram(s_shift, bins=np.linspace(0, 360, 9))
        ax_r.bar(centers, counts, width=width, bottom=bottom,
                 label=f"{vg} ({len(sub)})", color=vg_colors[vg],
                 alpha=0.85, edgecolor="white", linewidth=0.5)
        bottom += counts

    ax_r.set_xticks(centers)
    ax_r.set_xticklabels(COMPASS_8, fontsize=11, fontweight="bold")
    ax_r.set_title("Bearing rose (count per 45° sector)",
                   fontsize=12, fontweight="bold", pad=15)
    ax_r.legend(loc="upper right", bbox_to_anchor=(1.25, 1.05), fontsize=9)
    ax_r.grid(True, alpha=0.4)

    # ---- RIGHT: named-region bars ----
    ax_b = fig.add_subplot(1, 2, 2)
    label_counts = (df.groupby(["sector", "direction_label"])
                      .size().reset_index(name="count"))
    # Order sectors clockwise from N
    sector_order = {s: i for i, s in enumerate(COMPASS_8)}
    label_counts["order"] = label_counts["sector"].map(sector_order)
    label_counts = label_counts.sort_values("order")

    colors = [SECTOR_COLORS[s] for s in label_counts["sector"]]
    bars = ax_b.barh(range(len(label_counts)),
                     label_counts["count"].values,
                     color=colors, alpha=0.85, edgecolor="white")
    ax_b.set_yticks(range(len(label_counts)))
    ax_b.set_yticklabels(
        [f"{r['sector']:>2}  {r['direction_label']}"
         for _, r in label_counts.iterrows()],
        fontsize=10,
    )
    ax_b.invert_yaxis()
    ax_b.set_xlabel("Departure count")
    ax_b.set_title("Bearing → named region", fontsize=12, fontweight="bold")
    ax_b.grid(axis="x", ls="--", alpha=0.4)
    for bar, c in zip(bars, label_counts["count"].values):
        ax_b.text(bar.get_width() + max(label_counts["count"]) * 0.01,
                  bar.get_y() + bar.get_height() / 2,
                  f"  {int(c)}  ({c / n_total * 100:.0f}%)",
                  va="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    _save(fig, save_dir, f"departure_directions_{port_code}.png")
    plt.show()
    return fig


# ============================================================================
# Combined plot: arrival origin + departure destination for one port
# ============================================================================

def _vessel_group(v):
    if pd.isna(v):
        return "Other"
    u = str(v).upper()
    if "CONTAINER" in u:
        return "Container"
    if any(k in u for k in ["TANKER", "CRUDE", "LNG", "GAS", "CHEMICAL"]):
        return "Tanker"
    return "Other"


_VG_ORDER = ["Container", "Tanker", "Other"]
_VG_COLORS = {"Container": "#2980b9",
              "Tanker": "#e74c3c", "Other": "#95a5a6"}


def _draw_rose(ax, df, port_code, title):
    """Polar rose of bearings, stacked by vessel group."""
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    centers_deg = np.array([COMPASS_CENTERS[s] for s in COMPASS_8])
    centers = np.deg2rad(centers_deg)
    width = np.deg2rad(45.0)

    bottom = np.zeros(8)
    for vg in _VG_ORDER:
        sub = df[df["vessel_group"] == vg]
        if sub.empty:
            continue
        s_shift = (sub["bearing_deg"].values + 22.5) % 360
        counts, _ = np.histogram(s_shift, bins=np.linspace(0, 360, 9))
        ax.bar(centers, counts, width=width, bottom=bottom,
               label=f"{vg} ({len(sub)})", color=_VG_COLORS[vg],
               alpha=0.85, edgecolor="white", linewidth=0.5)
        bottom += counts

    ax.set_xticks(centers)
    ax.set_xticklabels(COMPASS_8, fontsize=10, fontweight="bold")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=12)
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.08), fontsize=8)
    ax.grid(True, alpha=0.4)


def _draw_bars(ax, df, title, n_total):
    """Horizontal bar chart of named-region counts."""
    label_counts = (df.groupby(["sector", "direction_label"])
                      .size().reset_index(name="count"))
    order = {s: i for i, s in enumerate(COMPASS_8)}
    label_counts["order"] = label_counts["sector"].map(order)
    label_counts = label_counts.sort_values("order")

    colors = [SECTOR_COLORS[s] for s in label_counts["sector"]]
    bars = ax.barh(range(len(label_counts)),
                   label_counts["count"].values,
                   color=colors, alpha=0.85, edgecolor="white")
    ax.set_yticks(range(len(label_counts)))
    ax.set_yticklabels(
        [f"{r['sector']:>2}  {r['direction_label']}"
         for _, r in label_counts.iterrows()],
        fontsize=9,
    )
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(axis="x", ls="--", alpha=0.4)
    if len(label_counts):
        max_c = max(label_counts["count"])
        for bar, c in zip(bars, label_counts["count"].values):
            ax.text(bar.get_width() + max_c * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"  {int(c)}  ({c / n_total * 100:.0f}%)",
                    va="center", fontsize=8, fontweight="bold")


def plot_port_directions(
    arrival_bearings: pd.DataFrame,
    departure_bearings: pd.DataFrame,
    port_code: str,
    title_suffix: str = "",
    save_dir: Optional[str] = None,
):
    """
    2×2 figure for one port:
      Top row    = arrivals (where ships came from)
      Bottom row = departures (where ships went)
      Left col   = polar rose by vessel group
      Right col  = bar chart by named region

    Returns the figure (or None if no usable rows in either frame).
    """
    arr = arrival_bearings[(arrival_bearings["port_name"] == port_code) &
                           (arrival_bearings["status"] == "ok")].copy()
    dep = departure_bearings[(departure_bearings["port_name"] == port_code) &
                             (departure_bearings["status"] == "ok")].copy()
    if arr.empty and dep.empty:
        print(f"  [warn] No usable bearings for {port_code}")
        return None

    arr["vessel_group"] = arr["vessel_type"].apply(_vessel_group)
    dep["vessel_group"] = dep["vessel_type"].apply(_vessel_group)

    arr_total_port = (arrival_bearings["port_name"] == port_code).sum()
    dep_total_port = (departure_bearings["port_name"] == port_code).sum()
    n_arr = len(arr)
    n_dep = len(dep)

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        f"{port_code} arrival origins & departure destinations {title_suffix}  "
        f"(arrivals n={n_arr}/{arr_total_port}, "
        f"departures n={n_dep}/{dep_total_port})",
        fontsize=14, fontweight="bold", y=1.00,
    )

    ax_ar = fig.add_subplot(2, 2, 1, projection="polar")
    if n_arr:
        _draw_rose(ax_ar, arr, port_code, "Arrivals — origin bearing rose")
    else:
        ax_ar.set_title("Arrivals — (no data)", fontsize=11, fontweight="bold")

    ax_ab = fig.add_subplot(2, 2, 2)
    if n_arr:
        _draw_bars(ax_ab, arr, "Arrivals — origin by named region", n_arr)
    else:
        ax_ab.set_title("Arrivals — (no data)", fontsize=11, fontweight="bold")
        ax_ab.axis("off")

    ax_dr = fig.add_subplot(2, 2, 3, projection="polar")
    if n_dep:
        _draw_rose(ax_dr, dep, port_code, "Departures — destination bearing rose")
    else:
        ax_dr.set_title("Departures — (no data)", fontsize=11, fontweight="bold")

    ax_db = fig.add_subplot(2, 2, 4)
    if n_dep:
        _draw_bars(ax_db, dep, "Departures — destination by named region", n_dep)
    else:
        ax_db.set_title("Departures — (no data)", fontsize=11, fontweight="bold")
        ax_db.axis("off")

    plt.tight_layout()
    _save(fig, save_dir, f"port_directions_{port_code}.png")
    plt.show()
    return fig


def summarize_port_directions(
    arrival_bearings: pd.DataFrame,
    departure_bearings: pd.DataFrame,
    port_code: Optional[str] = None,
):
    """Print a combined text summary of arrival + departure directions."""
    print("\n" + "#" * 72)
    print(f"#  PORT: {port_code or 'ALL PORTS'}")
    print("#" * 72)
    print("\n-- ARRIVALS (where ships came from) --")
    summarize_directions(arrival_bearings, port_code)
    print("\n-- DEPARTURES (where ships went) --")
    summarize_directions(departure_bearings, port_code)


# ============================================================================
# Area estimation: summarize the geographic area each sector points to
# ============================================================================

def estimate_direction_areas(
    arrival_bearings: Optional[pd.DataFrame],
    departure_bearings: Optional[pd.DataFrame],
    port_code: Optional[str] = None,
) -> pd.DataFrame:
    """
    Summarize the geographic endpoint cluster for each
    (port, flow, sector) combination.

    Returns one row per group with:
      port_name, flow ("arrival"|"departure"), sector, direction_label,
      count, centroid_lat, centroid_lon, lat_min/max, lon_min/max,
      mean_dist_km.
    """
    def _agg(df, flow, lat_col, lon_col):
        d = df[df["status"] == "ok"].copy()
        if port_code is not None:
            d = d[d["port_name"] == port_code]
        if d.empty:
            return pd.DataFrame()
        d = d.rename(columns={lat_col: "lat", lon_col: "lon"})
        g = (d.groupby(["port_name", "sector", "direction_label"])
               .agg(count=("lat", "size"),
                    centroid_lat=("lat", "mean"),
                    centroid_lon=("lon", "mean"),
                    lat_min=("lat", "min"), lat_max=("lat", "max"),
                    lon_min=("lon", "min"), lon_max=("lon", "max"),
                    mean_dist_km=("dist_km", "mean"))
               .reset_index())
        g.insert(1, "flow", flow)
        return g

    parts = []
    if arrival_bearings is not None and not arrival_bearings.empty:
        parts.append(_agg(arrival_bearings, "arrival", "in_lat", "in_lon"))
    if departure_bearings is not None and not departure_bearings.empty:
        parts.append(_agg(departure_bearings, "departure",
                          "out_lat", "out_lon"))
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


# ============================================================================
# World-map view: folium routes from port to sector endpoints
# ============================================================================

def plot_port_routes_on_map(
    arrival_bearings: pd.DataFrame,
    departure_bearings: pd.DataFrame,
    port_list_df: pd.DataFrame,
    port_code: Optional[str] = None,
    save_path: Optional[str] = None,
    show_endpoints: bool = True,
    show_area_boxes: bool = True,
):
    """
    Draw an interactive folium map with routes from each port to the
    centroid of every compass sector's endpoint cluster.

    - Arrival lines (dashed) point INTO the port from the origin centroid.
    - Departure lines (solid) point OUT of the port to the destination
      centroid. Line width scales with the number of ships in the sector.
    - show_endpoints: drop a tiny circle marker at every AIS endpoint.
    - show_area_boxes: draw a bounding rectangle per sector showing the
      geographic area that sector's ships pointed to.

    If port_code is None, all ports present in the bearings are drawn.
    Saves an HTML file if save_path is provided.
    """
    try:
        import folium
    except ImportError:
        print("  [warn] folium not installed — pip install folium")
        return None

    areas = estimate_direction_areas(
        arrival_bearings, departure_bearings, port_code=port_code)
    if areas.empty:
        print("  [warn] No 'ok' bearings to map.")
        return None

    port_lookup = (port_list_df.set_index("port")[["lat", "lon"]]
                   .to_dict("index"))
    ports = sorted(areas["port_name"].unique())

    # Center map
    if port_code and port_code in port_lookup:
        center = [port_lookup[port_code]["lat"],
                  port_lookup[port_code]["lon"]]
        zoom = 4
    else:
        center = [20, 60]
        zoom = 2

    m = folium.Map(location=center, zoom_start=zoom,
                   tiles="CartoDB positron", control_scale=True)

    fg_arr = folium.FeatureGroup(name="Arrivals (origin)", show=True)
    fg_dep = folium.FeatureGroup(name="Departures (destination)", show=True)
    fg_box = folium.FeatureGroup(name="Area boxes", show=show_area_boxes)
    fg_pts = folium.FeatureGroup(name="Endpoints", show=show_endpoints)

    # Port markers
    for p in ports:
        if p not in port_lookup:
            continue
        folium.Marker(
            [port_lookup[p]["lat"], port_lookup[p]["lon"]],
            icon=folium.Icon(color="black", icon="anchor", prefix="fa"),
            tooltip=f"<b>{p}</b>",
        ).add_to(m)

    # Routes & boxes per (port, flow, sector)
    max_count = max(areas["count"].max(), 1)
    for _, r in areas.iterrows():
        if r["port_name"] not in port_lookup:
            continue
        plat = port_lookup[r["port_name"]]["lat"]
        plon = port_lookup[r["port_name"]]["lon"]
        col = SECTOR_COLORS.get(r["sector"], "#555")
        # Width scales 2–8 by count
        w = 2 + 6 * (r["count"] / max_count)
        is_arr = r["flow"] == "arrival"
        # Line: arrival = endpoint → port; departure = port → endpoint
        coords = ([[r["centroid_lat"], r["centroid_lon"]], [plat, plon]]
                  if is_arr
                  else [[plat, plon], [r["centroid_lat"], r["centroid_lon"]]])
        tooltip = (f"<b>{r['port_name']}</b> — {r['flow']}<br>"
                   f"Sector: {r['sector']} ({r['direction_label']})<br>"
                   f"Count: {int(r['count'])}<br>"
                   f"Mean dist from port: {r['mean_dist_km']:.0f} km<br>"
                   f"Area: lat [{r['lat_min']:.2f}, {r['lat_max']:.2f}], "
                   f"lon [{r['lon_min']:.2f}, {r['lon_max']:.2f}]")
        folium.PolyLine(
            coords, color=col, weight=w, opacity=0.85,
            dash_array=("8 6" if is_arr else None), tooltip=tooltip,
        ).add_to(fg_arr if is_arr else fg_dep)

        if show_area_boxes:
            folium.Rectangle(
                bounds=[[r["lat_min"], r["lon_min"]],
                        [r["lat_max"], r["lon_max"]]],
                color=col, weight=1, fill=True, fill_opacity=0.12,
                tooltip=tooltip,
            ).add_to(fg_box)

    # Scatter of raw endpoints
    if show_endpoints:
        def _scatter(df, flow, lat_col, lon_col):
            d = df[df["status"] == "ok"]
            if port_code is not None:
                d = d[d["port_name"] == port_code]
            for _, row in d.iterrows():
                col = SECTOR_COLORS.get(row.get("sector", ""), "#555")
                folium.CircleMarker(
                    [row[lat_col], row[lon_col]],
                    radius=2, color=col, fill=True,
                    fill_color=col, fill_opacity=0.5, weight=0,
                    tooltip=f"{row['port_name']} {flow} → {row['sector']}",
                ).add_to(fg_pts)

        if arrival_bearings is not None and not arrival_bearings.empty:
            _scatter(arrival_bearings, "arrival", "in_lat", "in_lon")
        if departure_bearings is not None and not departure_bearings.empty:
            _scatter(departure_bearings, "departure", "out_lat", "out_lon")

    fg_arr.add_to(m)
    fg_dep.add_to(m)
    fg_box.add_to(m)
    fg_pts.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        m.save(save_path)
        print(f"  Saved: {save_path}")
    return m


# ============================================================================
# Convenience: print summary for a port
# ============================================================================

def summarize_directions(
    bearings_df: pd.DataFrame, port_code: Optional[str] = None,
):
    """Print a compact text summary of direction outcomes."""
    df = bearings_df.copy()
    if port_code is not None:
        df = df[df["port_name"] == port_code]
    if df.empty:
        print("  (no rows)")
        return

    n = len(df)
    print("=" * 72)
    print(f"  Direction summary — {port_code or 'ALL PORTS'}  (n={n})")
    print("=" * 72)
    status_counts = df["status"].value_counts()
    for s, c in status_counts.items():
        print(f"  {s:<25} {c:>5}  ({c / n * 100:5.1f}%)")
    ok = df[df["status"] == "ok"]
    if not ok.empty:
        print("  --- bearing distribution (ok rows) ---")
        sec_counts = (ok["sector"].value_counts()
                      .reindex(COMPASS_8, fill_value=0))
        for s in COMPASS_8:
            c = sec_counts[s]
            label = PORT_DIRECTION_MAP.get(port_code, {}).get(s, "")
            print(f"  {s:>2}  {c:>5}  {label}")
    print("=" * 72)
