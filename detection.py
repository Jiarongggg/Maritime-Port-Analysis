"""
Core port event detection: arrivals, departures, and stays.
Logic matches the original detect_multi_port_events notebook function exactly.
"""
import os
import warnings

import numpy as np
import pandas as pd


def detect_port_events(
    tracking_data: pd.DataFrame,
    port_list_df: pd.DataFrame,
    radius_deg: float = 1.0,
    min_stay_duration_minutes: int = 30,
    gap_tolerance_minutes: int = 60,
    output_dir: str | None = None,
    output_suffix: str = "",
    verbose: bool = True,
) -> dict:
    """
    Detect port stays, arrivals, and departures from AIS tracking data.

    Logic (matches original notebook exactly):
    1. Flags records in any port zone
    2. Groups consecutive in-zone records into stays
       - First record per vessel: prev_in_zone defaults to True (fillna(True))
         so ships already in port at T=0 get zone_group=0 → is_initial_stay
    3. Merges stays with small gaps (flickering fix)
    4. Assigns port by mean position during stay
    5. Filters valid stays (duration + speed check)
    6. Arrival = start of a valid stay (excludes ships already in port at T=0)
    7. Departure = end of a valid stay (includes ships already in port at T=0)
       - Only confirmed if vessel has records AFTER the stay ended

    Parameters
    ----------
    tracking_data : pd.DataFrame
        Preprocessed AIS data (must have VESSEL_IMO, TIMESTAMP, LATITUDE,
        LONGITUDE, SPEED, VESSEL_TYPE).
    port_list_df : pd.DataFrame
        Port list with 'port', 'lat', 'lon' columns.
        Optional 'radius' column for per-port radius (degrees).
    radius_deg : float
        Default radius in degrees when no per-port radius is specified.
    min_stay_duration_minutes : int
        Minimum stay to count as a valid port call.
    gap_tolerance_minutes : int
        Maximum gap between in-zone records to merge (flickering fix).
    output_dir : str | None
        Directory to save CSVs. None = don't save.
    output_suffix : str
        Suffix for output filenames.
    verbose : bool
        Print summary report.

    Returns
    -------
    dict with keys: 'arrivals', 'departures', 'reassignments', 'stays', 'summary'
    """
    warnings.filterwarnings("ignore", category=FutureWarning)

    df = tracking_data.copy()
    time_col = "TIMESTAMP_UTC" if "TIMESTAMP_UTC" in df.columns else "TIMESTAMP"
    df[time_col] = pd.to_datetime(df[time_col])

    # Standardise names
    df = df.rename(columns={"SHIP_ID": "VESSEL_IMO", "LAT": "LATITUDE", "LON": "LONGITUDE"})
    df = df.sort_values(["VESSEL_IMO", time_col]).reset_index(drop=True)

    # Speed check
    df["speed_stopped"] = (df["SPEED"] <= 0.5) | (df["SPEED"].isna())

    total_recs = len(df)
    nan_speed = df["SPEED"].isna().sum()

    # Resolve per-port radius
    ports = port_list_df.copy()
    if "radius" not in ports.columns:
        ports["radius"] = radius_deg
    else:
        ports["radius"] = ports["radius"].fillna(radius_deg)

    # =========================================================================
    # STEP 1 — Flag records that fall in ANY port zone (per-port radius)
    # =========================================================================
    df["in_any_zone"] = False
    for _, row in ports.iterrows():
        c_lat, c_lon, r = row["lat"], row["lon"], row["radius"]
        mask = df["LATITUDE"].between(c_lat - r, c_lat + r) & df["LONGITUDE"].between(c_lon - r, c_lon + r)
        df.loc[mask, "in_any_zone"] = True

    # =========================================================================
    # STEP 2 — Detect entry/exit from any zone per vessel
    # CRITICAL: fillna(True) means first record per vessel is treated as
    # "was already in this state" → no zone_change → zone_group stays 0
    # for ships already in port at T=0
    # =========================================================================
    df["prev_in_zone"] = df.groupby("VESSEL_IMO")["in_any_zone"].shift(1).fillna(True).astype(bool)
    df["zone_change"] = df["in_any_zone"] != df["prev_in_zone"]
    df["zone_group"] = df.groupby("VESSEL_IMO")["zone_change"].cumsum()

    # =========================================================================
    # STEP 3 — Group in-zone records into raw stays with stats
    # =========================================================================
    in_zone = df[df["in_any_zone"]].copy()
    if in_zone.empty:
        if verbose:
            print("No records found in any port zone.")
        return {"arrivals": pd.DataFrame(), "departures": pd.DataFrame(),
                "reassignments": pd.DataFrame(), "stays": pd.DataFrame(), "summary": {}}

    raw_stays = in_zone.groupby(["VESSEL_IMO", "zone_group"]).agg(
        start_time=(time_col, "min"), end_time=(time_col, "max"),
        lat_sum=("LATITUDE", "sum"), lon_sum=("LONGITUDE", "sum"),
        record_count=("LATITUDE", "count"),
        has_stopped=("speed_stopped", "any"),
        vessel_type=("VESSEL_TYPE", "first"),
    ).reset_index()

    # NOTE: Do NOT remove zone_group=0 here — those ships were already in port
    # at T=0 and can still have valid departures. We filter them out only for arrivals later.

    # =========================================================================
    # STEP 4 — Merge stays with small gaps (FLICKERING FIX)
    # =========================================================================
    merged = []
    for _, group in raw_stays.groupby("VESSEL_IMO"):
        group = group.sort_values("start_time").reset_index(drop=True)
        cur = group.iloc[0].to_dict()
        for i in range(1, len(group)):
            nxt = group.iloc[i]
            gap = (nxt["start_time"] - cur["end_time"]).total_seconds() / 60
            if gap <= gap_tolerance_minutes:
                # Merge: extend end time, accumulate position sums, combine flags
                cur["end_time"] = nxt["end_time"]
                cur["lat_sum"] += nxt["lat_sum"]
                cur["lon_sum"] += nxt["lon_sum"]
                cur["record_count"] += nxt["record_count"]
                cur["has_stopped"] = cur["has_stopped"] or nxt["has_stopped"]
                # Keep the original zone_group (from the first stay in the merge)
            else:
                merged.append(cur)
                cur = nxt.to_dict()
        merged.append(cur)
    stays = pd.DataFrame(merged)

    # =========================================================================
    # STEP 5 — Compute mean position and assign to closest port
    # =========================================================================
    stays["mean_lat"] = stays["lat_sum"] / stays["record_count"]
    stays["mean_lon"] = stays["lon_sum"] / stays["record_count"]
    stays["stay_duration_mins"] = (stays["end_time"] - stays["start_time"]).dt.total_seconds() / 60

    p_lats, p_lons, p_names = ports["lat"].values, ports["lon"].values, ports["port"].values

    def _closest_port(lat, lon):
        d2 = (lat - p_lats) ** 2 + (lon - p_lons) ** 2
        return p_names[np.argmin(d2)]

    stays["port_name"] = stays.apply(lambda r: _closest_port(r["mean_lat"], r["mean_lon"]), axis=1)

    # Also compute entry-point port for comparison
    entry_pos = in_zone.groupby(["VESSEL_IMO", "zone_group"]).agg(
        entry_lat=("LATITUDE", "first"), entry_lon=("LONGITUDE", "first")
    ).reset_index()

    entry_ports = []
    for _, s in stays.iterrows():
        m = entry_pos[(entry_pos["VESSEL_IMO"] == s["VESSEL_IMO"]) & (entry_pos["zone_group"] == s["zone_group"])]
        if not m.empty:
            entry_ports.append(_closest_port(m.iloc[0]["entry_lat"], m.iloc[0]["entry_lon"]))
        else:
            entry_ports.append(s["port_name"])
    stays["entry_point_port"] = entry_ports

    # Mark whether this stay started at T=0 (ship was already in port)
    stays["is_initial_stay"] = stays["zone_group"] == 0

    # =========================================================================
    # STEP 6 — Detect confirmed departures
    # A departure is confirmed only if the vessel has records AFTER the stay,
    # meaning the ship actually left (not just data ended while in port)
    # =========================================================================
    last_rec = df.groupby("VESSEL_IMO")[time_col].max().reset_index()
    last_rec.columns = ["VESSEL_IMO", "vessel_last_record"]
    stays = stays.merge(last_rec, on="VESSEL_IMO")
    stays["confirmed_departure"] = stays["end_time"] < stays["vessel_last_record"]

    # =========================================================================
    # STEP 7 — Filter valid stays
    # =========================================================================
    valid = stays[(stays["stay_duration_mins"] >= min_stay_duration_minutes) & stays["has_stopped"]].copy()

    # =========================================================================
    # STEP 8 — Track reassignment cases (mean-position vs entry-point)
    # Only for non-initial stays (initial stays have no real "entry point")
    # =========================================================================
    non_init = valid[~valid["is_initial_stay"]]
    reassign = non_init[non_init["port_name"] != non_init["entry_point_port"]][
        ["VESSEL_IMO", "start_time", "end_time", "port_name", "entry_point_port",
         "mean_lat", "mean_lon", "stay_duration_mins"]
    ].rename(columns={"port_name": "assigned_port_mean", "entry_point_port": "assigned_port_entry"}).copy()

    # =========================================================================
    # STEP 9 — Prepare outputs
    # =========================================================================

    # --- Arrivals: valid stays EXCLUDING initial stays (no observed arrival) ---
    arr = valid[~valid["is_initial_stay"]][["VESSEL_IMO", "start_time", "port_name", "stay_duration_mins", "vessel_type"]].copy()
    arr = arr.rename(columns={"start_time": time_col}).sort_values(time_col).reset_index(drop=True)

    # --- Departures: valid stays where ship actually left (INCLUDING initial stays) ---
    dep = valid[valid["confirmed_departure"]][
        ["VESSEL_IMO", "end_time", "port_name", "stay_duration_mins", "start_time", "vessel_type"]
    ].copy()
    dep = dep.rename(columns={"end_time": time_col, "start_time": "entry_time"}).sort_values(time_col).reset_index(drop=True)

    # Summary dict
    n_init = int(valid["is_initial_stay"].sum())
    summary = {
        "total_records": total_recs,
        "nan_speed_pct": nan_speed / total_recs if total_recs else 0,
        "raw_stays": len(raw_stays),
        "merged_stays": len(stays),
        "valid_stays": len(valid),
        "arrivals": len(arr),
        "departures": len(dep),
        "reassignments": len(reassign),
        "initial_in_port": n_init,
        "still_in_port": len(valid) - len(valid[valid["confirmed_departure"]]),
        "unique_ships_arr": arr["VESSEL_IMO"].nunique() if not arr.empty else 0,
        "unique_ships_dep": dep["VESSEL_IMO"].nunique() if not dep.empty else 0,
    }

    if verbose:
        _print_summary(summary, ports, radius_deg, min_stay_duration_minutes, gap_tolerance_minutes)

    # Save CSVs
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        arr.to_csv(os.path.join(output_dir, f"arrivals{output_suffix}.csv"), index=False)
        dep.to_csv(os.path.join(output_dir, f"departures{output_suffix}.csv"), index=False)
        if not reassign.empty:
            reassign.to_csv(os.path.join(output_dir, f"reassignments{output_suffix}.csv"), index=False)
        if verbose:
            print(f"  CSVs saved to {output_dir}/")

    warnings.filterwarnings("default", category=FutureWarning)

    return {
        "arrivals": arr,
        "departures": dep,
        "reassignments": reassign,
        "stays": valid,
        "summary": summary,
    }


def calculate_parking_durations(arrivals_df: pd.DataFrame, departures_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate parking durations from pre-computed arrivals / departures.
    Only includes stays where the ship actually departed.
    """
    time_col = "TIMESTAMP_UTC" if "TIMESTAMP_UTC" in arrivals_df.columns else "TIMESTAMP"
    result = arrivals_df.copy().rename(columns={time_col: "entry_time"})
    result["exit_time"] = result["entry_time"] + pd.to_timedelta(result["stay_duration_mins"], unit="min")
    result["total_parked_mins"] = result["stay_duration_mins"].round(2)
    result["total_parked_hours"] = (result["total_parked_mins"] / 60).round(2)

    dep_keys = set(zip(departures_df["VESSEL_IMO"], departures_df["entry_time"]))
    result = result[result.apply(lambda r: (r["VESSEL_IMO"], r["entry_time"]) in dep_keys, axis=1)]

    return result[["VESSEL_IMO", "port_name", "vessel_type", "entry_time",
                    "exit_time", "total_parked_mins", "total_parked_hours"]]


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _print_summary(s, ports, radius_deg, min_stay, gap_tol):
    w = 90
    print("=" * w)
    print(f"{'PORT EVENT DETECTION RESULTS':^{w}}")
    print("=" * w)
    print(f"  Method:  Stay Duration + Speed Check + Mean-Position Assignment")
    print(f"  Params:  min_stay={min_stay}min | speed<=0.5kn | gap_tol={gap_tol}min | default_radius={radius_deg} deg")
    print(f"  Per-port radii:")
    for _, p in ports.iterrows():
        print(f"      {p['port']}: {p['radius']} deg")
    print(f"  Data:    {s['total_records']:,} records | NaN speed: {s['nan_speed_pct']:.1%}")
    print(f"  Stays:   {s['raw_stays']} raw -> {s['merged_stays']} merged -> {s['valid_stays']} valid")
    print(f"  Initial: {s['initial_in_port']} ships already in port at T=0")
    print(f"  Results: {s['arrivals']} arrivals | {s['departures']} departures | "
          f"{s['still_in_port']} still in port | {s['reassignments']} reassigned")
    print(f"  Ships:   {s['unique_ships_arr']} unique (arr) | {s['unique_ships_dep']} unique (dep)")
    print("=" * w)
