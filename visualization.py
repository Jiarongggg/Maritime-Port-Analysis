"""
Visualization functions for maritime port analysis.
All plots return their figure objects and optionally save to disk.
"""
import math
import os
from typing import Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================================
# HELPER: vessel type grouping
# ============================================================================

def _map_vessel_group(vtype):
    if pd.isna(vtype):
        return "Other"
    vt = vtype.upper()
    if "CONTAINER" in vt:
        return "Container"
    elif any(kw in vt for kw in ["TANKER", "CRUDE", "LNG", "GAS", "CHEMICAL"]):
        return "Tanker"
    return "Other"


VG_COLORS = {"Container": "#2980b9", "Tanker": "#e74c3c", "Other": "#95a5a6"}

# ============================================================================
# HELPER: region classification by port name
# ============================================================================

REGION_COLORS = {
    "East Asia":       "#e74c3c",
    "Southeast Asia":  "#2ecc71",
    "South Asia":      "#f39c12",
    "Middle East":     "#e67e22",
    "Europe":          "#3498db",
    "Americas":        "#9b59b6",
    "Africa/Oceania":  "#1abc9c",
}

# Direct port-to-region mapping — edit this dict to adjust groupings
PORT_REGION_MAP = {
    # East Asia
    "CNSHA": "East Asia",    # Shanghai
    "CNNBG": "East Asia",    # Ningbo-Zhoushan
    "CNSZX": "East Asia",    # Shenzhen
    "CNTXG": "East Asia",    # Tianjin
    "CNZZU": "East Asia",    # Fuzhou
    "HKHKG": "East Asia",    # Hong Kong
    "TWKHH": "East Asia",    # Kaohsiung
    "TWKEL": "East Asia",    # Keelung
    "KRPUS": "East Asia",    # Busan
    "JPOSA": "East Asia",    # Osaka
    "JPNGO": "East Asia",    # Nagoya
    "JPTYO": "East Asia",    # Tokyo
    # Southeast Asia
    "SGSIN": "Southeast Asia",  # Singapore
    "MYLPK": "Southeast Asia",  # Port Klang
    "MYTPP": "Southeast Asia",  # Tanjong Pelepas
    "THLCH": "Southeast Asia",  # Laem Chabang
    "VNHPH": "Southeast Asia",  # Hai Phong
    "VNVUT": "Southeast Asia",  # Vung Tau
    "PHMNL": "Southeast Asia",  # Manila
    "IDJKT": "Southeast Asia",  # Jakarta
    # South Asia
    "INNSA": "South Asia",   # Nhava Sheva
    "INMUN": "South Asia",   # Mundra
    "PKKHI": "South Asia",   # Karachi
    "LKCMB": "South Asia",   # Colombo
    "BDCGP": "South Asia",   # Chittagong
    # Middle East
    "AEJEA": "Middle East",  # Jebel Ali
    "AEKHL": "Middle East",  # Khalifa
    "SAJED": "Middle East",  # Jeddah
    "DJJIB": "Middle East",  # Djibouti
    # Europe
    "NLRTM": "Europe",       # Rotterdam
    "BEANR": "Europe",       # Antwerp-Bruges
    "DEHAM": "Europe",       # Hamburg
    "DEBRV": "Europe",       # Bremerhaven
    "FRLEH": "Europe",       # Le Havre
    "GBSOU": "Europe",       # Southampton
    "ESBCN": "Europe",       # Barcelona
    "ESVLC": "Europe",       # Valencia
    "ESALG": "Europe",       # Algeciras
    "ITGOA": "Europe",       # Genoa
    "GRPIR": "Europe",       # Piraeus
    "TRIST": "Europe",       # Istanbul
    "EGPSD": "Europe",       # Port Said
    "MACAS": "Europe",       # Casablanca
    "MAPTM": "Europe",       # Tanger Med
    # Americas
    "USLAX": "Americas",     # Los Angeles / Long Beach
    "USLGB": "Americas",     # Long Beach
    "USNYC": "Americas",     # New York / New Jersey
    "USNWK": "Americas",     # Newark
    "USHOU": "Americas",     # Houston
    "BRSSZ": "Americas",     # Santos
    # Africa / Oceania
    "ZACPT": "Africa/Oceania",  # Cape Town
    "AUMEL": "Africa/Oceania",  # Melbourne
    "AUSYD": "Africa/Oceania",  # Sydney
}


def classify_region(port_code):
    """
    Classify a port into a geographic region by port code.
    Returns 'Unknown' if port not in the mapping.
    Add new ports to PORT_REGION_MAP as needed.
    """
    return PORT_REGION_MAP.get(port_code, "Unknown")


def add_region_to_ports(port_list_df):
    """Add a 'region' column to a port list DataFrame."""
    df = port_list_df.copy()
    df["region"] = df["port"].map(PORT_REGION_MAP).fillna("Unknown")
    return df


def enrich_with_region(events_df, port_list_df=None, port_col="port_name"):
    """
    Add region column to an arrivals/departures/parking DataFrame.
    Maps directly from port_name → region using PORT_REGION_MAP.
    port_list_df is accepted for backwards compatibility but not needed.
    """
    df = events_df.copy()
    df["region"] = df[port_col].map(PORT_REGION_MAP).fillna("Unknown")
    return df


# ============================================================================
# Regional arrival/departure by vessel type
# ============================================================================

def plot_regional_vessel_type(
    arrivals_df, departures_df, port_list_df,
    port_col="port_name", vessel_col="vessel_type",
    title_suffix="", save_dir=None,
):
    """
    Arrival and departure counts by vessel type, grouped by geographic region.

    Produces:
      - Figure 1: Stacked bars per region (arrivals vs departures, coloured by vessel type)
      - Figure 2: Heatmap of vessel type × region (arrivals and departures side-by-side)
      - Figure 3: Per-region subplots showing vessel type breakdown

    Parameters
    ----------
    arrivals_df, departures_df : pd.DataFrame
        Output from detect_port_events.
    port_list_df : pd.DataFrame
        Port list with port, lat, lon columns.
    title_suffix : str
        e.g. '- March 2026'

    Returns
    -------
    dict of figures
    """
    arr = enrich_with_region(arrivals_df, port_list_df, port_col)
    dep = enrich_with_region(departures_df, port_list_df, port_col)

    arr["vessel_group"] = arr[vessel_col].apply(_map_vessel_group)
    dep["vessel_group"] = dep[vessel_col].apply(_map_vessel_group)

    vg_order = ["Container", "Tanker", "Other"]
    region_order = sorted(set(arr["region"].unique()) | set(dep["region"].unique()),
                          key=lambda r: list(REGION_COLORS.keys()).index(r)
                          if r in REGION_COLORS else 99)

    figs = {}

    # ================================================================
    # FIGURE 1: Grouped bars — arrivals vs departures per region,
    #           stacked by vessel type
    # ================================================================
    fig1, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig1.suptitle(f"Arrivals & Departures by Region and Vessel Type {title_suffix}",
                  fontsize=16, fontweight="bold", y=1.02)

    for ax, df_src, label in [(axes[0], arr, "Arrivals"), (axes[1], dep, "Departures")]:
        pivot = df_src.groupby(["region", "vessel_group"]).size().unstack(fill_value=0)
        pivot = pivot.reindex(index=region_order, columns=vg_order, fill_value=0)

        bottom = np.zeros(len(pivot))
        x = np.arange(len(pivot))
        for vg in vg_order:
            vals = pivot[vg].values if vg in pivot.columns else np.zeros(len(pivot))
            ax.bar(x, vals, bottom=bottom, label=vg, color=VG_COLORS.get(vg, "#999"),
                   alpha=0.85, edgecolor="white", linewidth=0.5)
            # value labels
            for i, v in enumerate(vals):
                if v > 0:
                    ax.text(i, bottom[i] + v / 2, str(int(v)),
                            ha="center", va="center", fontsize=9, fontweight="bold", color="white")
            bottom += vals

        # Total labels on top
        totals = pivot.sum(axis=1).values
        for i, t in enumerate(totals):
            ax.text(i, t + max(totals) * 0.02, str(int(t)),
                    ha="center", fontsize=11, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index, rotation=30, ha="right")
        ax.set_ylabel("Vessel Count")
        ax.set_title(label, fontsize=14, fontweight="bold")
        ax.legend(loc="upper right")
        ax.grid(axis="y", ls="--", alpha=0.3)

    plt.tight_layout()
    _save(fig1, save_dir, "regional_vessel_type_bars.png")
    plt.show()
    figs["bars"] = fig1

    # ================================================================
    # FIGURE 2: Side-by-side heatmaps (Arrivals | Departures)
    # ================================================================
    fig2, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig2.suptitle(f"Vessel Type × Region Heatmap {title_suffix}",
                  fontsize=16, fontweight="bold", y=1.04)

    for ax, df_src, label in [(axes[0], arr, "Arrivals"), (axes[1], dep, "Departures")]:
        pivot = df_src.groupby(["region", "vessel_group"]).size().unstack(fill_value=0)
        pivot = pivot.reindex(index=region_order, columns=vg_order, fill_value=0)

        im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(len(vg_order)))
        ax.set_xticklabels(vg_order)
        ax.set_yticks(range(len(region_order)))
        ax.set_yticklabels(region_order)
        ax.set_title(label, fontsize=14, fontweight="bold")
        fig2.colorbar(im, ax=ax, shrink=0.8, label="Count")

        # Annotate cells
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i, j]
                if v > 0:
                    color = "white" if v > pivot.values.max() * 0.55 else "black"
                    ax.text(j, i, str(int(v)), ha="center", va="center",
                            fontsize=11, fontweight="bold", color=color)

    plt.tight_layout()
    _save(fig2, save_dir, "regional_vessel_type_heatmap.png")
    plt.show()
    figs["heatmap"] = fig2

    # ================================================================
    # FIGURE 3: Per-region subplots — arrivals vs departures by type
    # ================================================================
    n_regions = len(region_order)
    ncols = min(3, n_regions)
    nrows = math.ceil(n_regions / ncols)
    fig3, axes_grid = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
    fig3.suptitle(f"Per-Region: Arrivals vs Departures by Vessel Type {title_suffix}",
                  fontsize=16, fontweight="bold", y=1.02)
    axes_flat = np.array(axes_grid).flatten() if n_regions > 1 else [axes_grid]

    for idx, region in enumerate(region_order):
        ax = axes_flat[idx]
        arr_r = arr[arr["region"] == region].groupby("vessel_group").size().reindex(vg_order, fill_value=0)
        dep_r = dep[dep["region"] == region].groupby("vessel_group").size().reindex(vg_order, fill_value=0)

        x = np.arange(len(vg_order))
        w = 0.35
        bars_a = ax.bar(x - w / 2, arr_r.values, w, label="Arrivals",
                        color="#2980b9", alpha=0.8, edgecolor="white")
        bars_d = ax.bar(x + w / 2, dep_r.values, w, label="Departures",
                        color="#e74c3c", alpha=0.8, edgecolor="white")

        # Value labels
        for bars in [bars_a, bars_d]:
            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, h + max(max(arr_r), max(dep_r)) * 0.02,
                            str(int(h)), ha="center", va="bottom", fontsize=9, fontweight="bold")

        total_a = arr_r.sum()
        total_d = dep_r.sum()
        net = total_a - total_d
        net_color = "#27ae60" if net >= 0 else "#e74c3c"

        ax.set_xticks(x)
        ax.set_xticklabels(vg_order)
        ax.set_ylabel("Count")
        ax.set_title(f"{region}\nArr: {total_a}  Dep: {total_d}  Net: {net:+d}",
                     fontsize=12, fontweight="bold",
                     color=REGION_COLORS.get(region, "#333"))
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", ls="--", alpha=0.3)

    # Hide unused axes
    for j in range(len(region_order), len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    _save(fig3, save_dir, "regional_vessel_type_detail.png")
    plt.show()
    figs["detail"] = fig3

    # ================================================================
    # Print summary table
    # ================================================================
    print("\n" + "=" * 95)
    print(f"{'ARRIVALS & DEPARTURES BY REGION AND VESSEL TYPE ' + title_suffix:^95}")
    print("=" * 95)
    print(f"\n{'Region':<20} {'':^5} {'Container':>12} {'Tanker':>12} {'Other':>12} {'Total':>12} {'Net(A-D)':>10}")
    print("-" * 95)
    for region in region_order:
        for event_type, df_src, marker in [("Arr", arr, "↓"), ("Dep", dep, "↑")]:
            r = df_src[df_src["region"] == region]
            counts = r["vessel_group"].value_counts()
            c = counts.get("Container", 0)
            t = counts.get("Tanker", 0)
            o = counts.get("Other", 0)
            total = c + t + o
            if event_type == "Arr":
                print(f"{region:<20} {marker + ' ' + event_type:^5} {c:>12} {t:>12} {o:>12} {total:>12}", end="")
                arr_total = total
            else:
                net = arr_total - total
                net_str = f"{net:+d}"
                print(f"\n{'':20} {marker + ' ' + event_type:^5} {c:>12} {t:>12} {o:>12} {total:>12} {net_str:>10}")
        print("-" * 95)

    # Grand totals
    ga = len(arr)
    gd = len(dep)
    print(f"{'GRAND TOTAL':<20} {'↓ Arr':^5} {(arr['vessel_group']=='Container').sum():>12} "
          f"{(arr['vessel_group']=='Tanker').sum():>12} "
          f"{(arr['vessel_group']=='Other').sum():>12} {ga:>12}")
    print(f"{'':20} {'↑ Dep':^5} {(dep['vessel_group']=='Container').sum():>12} "
          f"{(dep['vessel_group']=='Tanker').sum():>12} "
          f"{(dep['vessel_group']=='Other').sum():>12} {gd:>12} {ga - gd:>+10}")
    print("=" * 95)

    return figs


def _save(fig, directory, filename, dpi=300):
    if directory:
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, filename)
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {path}")


# ============================================================================
# Daily arrivals/departures breakdown by vessel type
# ============================================================================

def plot_daily_by_vessel_type(
    arrivals_df, departures_df,
    timestamp_col="TIMESTAMP", vessel_col="vessel_type",
    cols=3, title_suffix="", save_dir=None,
):
    """
    Daily arrivals vs departures broken down by vessel type (Container, Tanker, Other).
    One row of subplots per vessel type showing daily trend + net flow.

    Returns dict of figures.
    """
    arr = arrivals_df.copy()
    dep = departures_df.copy()
    arr[timestamp_col] = pd.to_datetime(arr[timestamp_col])
    dep_tc = "TIMESTAMP_UTC" if "TIMESTAMP_UTC" in dep.columns else timestamp_col
    dep[dep_tc] = pd.to_datetime(dep[dep_tc])

    arr["day"] = arr[timestamp_col].dt.date
    dep["day"] = dep[dep_tc].dt.date
    arr["vessel_group"] = arr[vessel_col].apply(_map_vessel_group)
    dep["vessel_group"] = dep[vessel_col].apply(_map_vessel_group)

    all_days = sorted(set(arr["day"]) | set(dep["day"]))
    vg_order = ["Container", "Tanker", "Other"]
    figs = {}

    # --- FIGURE 1: Per vessel-type daily trend (arr vs dep) ---
    fig1, axes = plt.subplots(1, 3, figsize=(22, 6))
    fig1.suptitle(f"Daily Arrivals vs Departures by Vessel Type {title_suffix}",
                  fontsize=16, fontweight="bold", y=1.04)

    for i, vg in enumerate(vg_order):
        ax = axes[i]
        arr_vg = arr[arr["vessel_group"] == vg].groupby("day").size().reindex(all_days, fill_value=0)
        dep_vg = dep[dep["vessel_group"] == vg].groupby("day").size().reindex(all_days, fill_value=0)

        ax.plot(range(len(all_days)), arr_vg.values, "o-", color="#2980b9", ms=5, lw=2, label="Arrivals")
        ax.plot(range(len(all_days)), dep_vg.values, "s-", color="#e74c3c", ms=5, lw=2, label="Departures")
        ax.fill_between(range(len(all_days)), arr_vg.values, dep_vg.values, alpha=0.15,
                        color="#27ae60" if arr_vg.sum() >= dep_vg.sum() else "#e74c3c")

        net = arr_vg.sum() - dep_vg.sum()
        net_color = "#27ae60" if net >= 0 else "#e74c3c"
        ax.set_title(f"{vg}\nArr: {arr_vg.sum()}  Dep: {dep_vg.sum()}  Net: {net:+d}",
                     fontsize=13, fontweight="bold", color=VG_COLORS.get(vg, "#333"))
        ax.set_ylabel("Count")
        day_labels = [d.strftime("%b %d") for d in all_days]
        ax.set_xticks(range(len(all_days)))
        ax.set_xticklabels(day_labels, rotation=45, ha="right", fontsize=8)
        ax.legend(fontsize=9)
        ax.grid(True, ls="--", alpha=0.3)

    plt.tight_layout()
    _save(fig1, save_dir, "daily_by_vessel_type.png")
    plt.show()
    figs["daily_vessel_type"] = fig1

    # --- FIGURE 2: Stacked area — all types together ---
    fig2, axes = plt.subplots(1, 2, figsize=(20, 7))
    fig2.suptitle(f"Daily Stacked Arrivals & Departures by Vessel Type {title_suffix}",
                  fontsize=16, fontweight="bold", y=1.02)

    for ax, src, label in [(axes[0], arr, "Arrivals"), (axes[1], dep, "Departures")]:
        bottom = np.zeros(len(all_days))
        for vg in vg_order:
            daily = src[src["vessel_group"] == vg].groupby("day").size().reindex(all_days, fill_value=0)
            ax.bar(range(len(all_days)), daily.values, bottom=bottom,
                   label=vg, color=VG_COLORS.get(vg), alpha=0.85, edgecolor="white", linewidth=0.5)
            bottom += daily.values

        # Total on top
        for j, t in enumerate(bottom):
            if t > 0:
                ax.text(j, t + max(bottom) * 0.02, str(int(t)),
                        ha="center", fontsize=8, fontweight="bold")

        ax.set_title(label, fontsize=14, fontweight="bold")
        ax.set_xticks(range(len(all_days)))
        ax.set_xticklabels(day_labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Count")
        ax.legend(loc="upper right")
        ax.grid(axis="y", ls="--", alpha=0.3)

    plt.tight_layout()
    _save(fig2, save_dir, "daily_stacked_vessel_type.png")
    plt.show()
    figs["stacked_vessel_type"] = fig2

    return figs


# ============================================================================
# Daily arrivals/departures breakdown by region
# ============================================================================

def plot_daily_by_region(
    arrivals_df, departures_df, port_list_df,
    timestamp_col="TIMESTAMP", port_col="port_name",
    title_suffix="", save_dir=None,
):
    """
    Daily arrivals vs departures broken down by geographic region.
    One subplot per region showing daily trend + net.

    Returns dict of figures.
    """
    arr = enrich_with_region(arrivals_df, port_list_df, port_col)
    dep = enrich_with_region(departures_df, port_list_df, port_col)

    arr[timestamp_col] = pd.to_datetime(arr[timestamp_col])
    dep_tc = "TIMESTAMP_UTC" if "TIMESTAMP_UTC" in dep.columns else timestamp_col
    dep[dep_tc] = pd.to_datetime(dep[dep_tc])

    arr["day"] = arr[timestamp_col].dt.date
    dep["day"] = dep[dep_tc].dt.date

    all_days = sorted(set(arr["day"]) | set(dep["day"]))
    day_labels = [d.strftime("%b %d") for d in all_days]
    regions = sorted(set(arr["region"].unique()) | set(dep["region"].unique()),
                     key=lambda r: list(REGION_COLORS.keys()).index(r)
                     if r in REGION_COLORS else 99)

    figs = {}

    # --- FIGURE 1: Per-region daily trend (arr vs dep) ---
    n_regions = len(regions)
    ncols = min(3, n_regions)
    nrows = math.ceil(n_regions / ncols)
    fig1, axes_grid = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
    fig1.suptitle(f"Daily Arrivals vs Departures by Region {title_suffix}",
                  fontsize=16, fontweight="bold", y=1.02)
    axes_flat = np.array(axes_grid).flatten() if n_regions > 1 else [axes_grid]

    for idx, region in enumerate(regions):
        ax = axes_flat[idx]
        arr_r = arr[arr["region"] == region].groupby("day").size().reindex(all_days, fill_value=0)
        dep_r = dep[dep["region"] == region].groupby("day").size().reindex(all_days, fill_value=0)

        ax.plot(range(len(all_days)), arr_r.values, "o-", color="#2980b9", ms=5, lw=2, label="Arrivals")
        ax.plot(range(len(all_days)), dep_r.values, "s-", color="#e74c3c", ms=5, lw=2, label="Departures")
        ax.fill_between(range(len(all_days)), arr_r.values, dep_r.values, alpha=0.15,
                        color="#27ae60" if arr_r.sum() >= dep_r.sum() else "#e74c3c")

        net = arr_r.sum() - dep_r.sum()
        ax.set_title(f"{region}\nArr: {arr_r.sum()}  Dep: {dep_r.sum()}  Net: {net:+d}",
                     fontsize=12, fontweight="bold", color=REGION_COLORS.get(region, "#333"))
        ax.set_ylabel("Count")
        ax.set_xticks(range(len(all_days)))
        ax.set_xticklabels(day_labels, rotation=45, ha="right", fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, ls="--", alpha=0.3)

    for j in range(len(regions), len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    _save(fig1, save_dir, "daily_by_region.png")
    plt.show()
    figs["daily_region"] = fig1

    # --- FIGURE 2: Stacked area — all regions together ---
    fig2, axes = plt.subplots(1, 2, figsize=(20, 7))
    fig2.suptitle(f"Daily Stacked Arrivals & Departures by Region {title_suffix}",
                  fontsize=16, fontweight="bold", y=1.02)

    for ax, src, label in [(axes[0], arr, "Arrivals"), (axes[1], dep, "Departures")]:
        bottom = np.zeros(len(all_days))
        for region in regions:
            daily = src[src["region"] == region].groupby("day").size().reindex(all_days, fill_value=0)
            ax.bar(range(len(all_days)), daily.values, bottom=bottom,
                   label=region, color=REGION_COLORS.get(region, "#999"), alpha=0.85,
                   edgecolor="white", linewidth=0.5)
            bottom += daily.values

        for j, t in enumerate(bottom):
            if t > 0:
                ax.text(j, t + max(bottom) * 0.02, str(int(t)),
                        ha="center", fontsize=8, fontweight="bold")

        ax.set_title(label, fontsize=14, fontweight="bold")
        ax.set_xticks(range(len(all_days)))
        ax.set_xticklabels(day_labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Count")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(axis="y", ls="--", alpha=0.3)

    plt.tight_layout()
    _save(fig2, save_dir, "daily_stacked_region.png")
    plt.show()
    figs["stacked_region"] = fig2

    return figs


# ============================================================================
# 1. Daily arrivals — overlaid lines
# ============================================================================

def plot_daily_arrivals(arrivals_df, timestamp_col="TIMESTAMP",
                        port_col="port_name", save_dir=None):
    df = arrivals_df.copy()
    df["DATE"] = pd.to_datetime(df[timestamp_col]).dt.date
    daily = df.groupby(["DATE", port_col]).size().reset_index(name="N")
    daily["DATE"] = pd.to_datetime(daily["DATE"])

    fig, ax = plt.subplots(figsize=(14, 8))
    for port in daily[port_col].unique():
        sub = daily[daily[port_col] == port]
        ax.plot(sub["DATE"], sub["N"], marker="o", markersize=3, linewidth=1.5, label=port)
    ax.set(xlabel="Date", ylabel="Arrivals", title="Daily Ship Arrivals by Port")
    ax.legend(title="Port", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    _save(fig, save_dir, "daily_arrivals_overlay.png")
    plt.show()

    stats = daily.groupby(port_col)["N"].agg(["mean", "max", "sum"]).sort_values("mean", ascending=False)
    stats.columns = ["Avg Daily", "Max Daily", "Total"]
    return stats, fig


# ============================================================================
# 2. Daily arrivals — grid (one subplot per port)
# ============================================================================

def plot_daily_arrivals_grid(arrivals_df, timestamp_col="TIMESTAMP",
                              port_col="port_name", cols=2,
                              title="", save_dir=None):
    df = arrivals_df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df["DATE"] = df[timestamp_col].dt.date
    daily = df.groupby(["DATE", port_col]).size().reset_index(name="N")
    daily["DATE"] = pd.to_datetime(daily["DATE"])

    ports = sorted(daily[port_col].unique())
    rows = math.ceil(len(ports) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    axes = np.array(axes).flatten()

    for i, port in enumerate(ports):
        sub = daily[daily[port_col] == port].sort_values("DATE")
        ax = axes[i]
        ax.plot(sub["DATE"], sub["N"], marker="o", markersize=3, color="steelblue")
        mean_val = sub["N"].mean()
        ax.axhline(mean_val, color="red", linestyle="--", alpha=0.4, label=f"Avg: {mean_val:.1f}")
        ax.set_title(f"Port: {port}", fontweight="bold")
        ax.set_ylabel("Arrivals")
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=45)
        ax.legend(fontsize="small", loc="upper right")

    for j in range(len(ports), len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(title or "Daily Port Arrivals Grid", fontsize=16, y=1.00, fontweight="bold")
    plt.tight_layout()
    _save(fig, save_dir, "daily_arrivals_grid.png")
    plt.show()
    return fig


# ============================================================================
# 3. Arrival ranking bar chart
# ============================================================================

def plot_arrival_ranking(arrivals_df, port_col="port_name",
                          title_suffix="", save_dir=None):
    counts = arrivals_df.groupby(port_col).size().reset_index(name="TOTAL").sort_values("TOTAL", ascending=False)
    fig, ax = plt.subplots(figsize=(18, 7))
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(counts)))
    bars = ax.bar(counts[port_col], counts["TOTAL"], color=colors, edgecolor="black", alpha=0.8)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1, str(int(h)),
                ha="center", va="bottom", fontweight="bold", fontsize=8.5)
    ax.set(title=f"Total Ship Arrivals by Port {title_suffix}", xlabel="Port", ylabel="Total Arrivals")
    plt.xticks(rotation=45)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    _save(fig, save_dir, "arrival_ranking.png")
    plt.show()
    return counts, fig


# ============================================================================
# 4. Daily departure trends grid
# ============================================================================

def plot_daily_departures_grid(departures_df, timestamp_col="TIMESTAMP",
                                port_col="port_name", cols=3, title="", save_dir=None):
    df = departures_df.copy()
    df["DATE"] = pd.to_datetime(df[timestamp_col]).dt.date
    daily = df.groupby(["DATE", port_col]).size().reset_index(name="N")
    daily["DATE"] = pd.to_datetime(daily["DATE"])

    ports = sorted(daily[port_col].unique())
    rows = math.ceil(len(ports) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    axes = np.array(axes).flatten()

    for i, port in enumerate(ports):
        sub = daily[daily[port_col] == port].sort_values("DATE")
        ax = axes[i]
        ax.plot(sub["DATE"], sub["N"], color="coral", marker=".", markersize=4, linewidth=1)
        avg = sub["N"].mean()
        ax.axhline(avg, color="red", linestyle="--", alpha=0.5, label=f"Avg: {avg:.1f}")
        ax.set_title(f"Port: {port}", fontsize=14, fontweight="bold")
        ax.set_ylabel("Departures")
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=45)
        ax.legend(fontsize="small", loc="upper right")

    for j in range(len(ports), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(title or "Daily Departure Trends per Port", fontsize=22, fontweight="bold")
    plt.tight_layout(h_pad=3.0)
    plt.subplots_adjust(top=0.95)
    _save(fig, save_dir, "daily_departures_grid.png")
    plt.show()
    return fig


# ============================================================================
# 5. Arrival vs departure comparison (grid + totals)
# ============================================================================

def plot_arrival_departure_comparison(arrivals_df, departures_df,
                                       timestamp_col="TIMESTAMP",
                                       port_col="port_name", cols=3,
                                       title_suffix="", save_dir=None):
    df_a = arrivals_df.copy()
    df_d = departures_df.copy()
    df_a["DATE"] = pd.to_datetime(df_a[timestamp_col]).dt.date
    dep_tc = "TIMESTAMP_UTC" if "TIMESTAMP_UTC" in df_d.columns else timestamp_col
    df_d["DATE"] = pd.to_datetime(df_d[dep_tc]).dt.date

    daily_a = df_a.groupby(["DATE", port_col]).size().reset_index(name="ARRIVALS")
    daily_d = df_d.groupby(["DATE", port_col]).size().reset_index(name="DEPARTURES")
    combined = pd.merge(daily_a, daily_d, on=["DATE", port_col], how="outer").fillna(0)
    combined["DATE"] = pd.to_datetime(combined["DATE"])

    ports = sorted(combined[port_col].unique())
    rows = math.ceil(len(ports) / cols)

    # Grid
    fig1, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    axes = np.array(axes).flatten()
    for i, port in enumerate(ports):
        sub = combined[combined[port_col] == port].sort_values("DATE")
        ax = axes[i]
        ax.plot(sub["DATE"], sub["ARRIVALS"], label="Arrivals", color="#008080", marker="o", markersize=3)
        ax.plot(sub["DATE"], sub["DEPARTURES"], label="Departures", color="#FF7F50", marker="x", markersize=3)
        ax.set_title(f"Port activity: {port}", fontweight="bold")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=45)
        ax.legend(fontsize="small", loc="upper right")
    for j in range(len(ports), len(axes)):
        fig1.delaxes(axes[j])
    plt.suptitle(f"Arrival vs Departure Comparison {title_suffix}", fontsize=22, y=1.02, fontweight="bold")
    plt.tight_layout()
    _save(fig1, save_dir, "arrival_vs_departure_grid.png")
    plt.show()

    # Totals bar
    totals = combined.groupby(port_col).agg({"ARRIVALS": "sum", "DEPARTURES": "sum"}).sort_values("ARRIVALS", ascending=False)
    fig2 = plt.figure(figsize=(15, 8))
    x = range(len(totals))
    w = 0.35
    plt.bar([i - w / 2 for i in x], totals["ARRIVALS"], w, label="Total Arrivals", color="#008080", alpha=0.8)
    plt.bar([i + w / 2 for i in x], totals["DEPARTURES"], w, label="Total Departures", color="#FF7F50", alpha=0.8)
    plt.xticks(list(x), totals.index, rotation=45)
    plt.ylabel("Total Ship Count")
    plt.title(f"Total Arrival vs Departure Volume {title_suffix}", fontsize=16, pad=20)
    plt.legend(fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    _save(fig2, save_dir, "arrival_vs_departure_totals.png")
    plt.show()

    return combined, fig1, fig2


# ============================================================================
# 6. Parking duration analysis (stats + 4-panel viz)
# ============================================================================

def analyze_parking_stats(parking_df, port_name=None, output_dir=None, output_suffix=""):
    df = parking_df.copy()
    if port_name:
        df = df[df["port_name"] == port_name]
    if len(df) == 0:
        print("No parking data.")
        return {}

    label = f" - {port_name}" if port_name else " - All Ports"
    overall = {
        "count": len(df), "unique_ships": df["VESSEL_IMO"].nunique(),
        "mean_mins": df["total_parked_mins"].mean(),
        "median_mins": df["total_parked_mins"].median(),
        "q25": df["total_parked_mins"].quantile(0.25),
        "q75": df["total_parked_mins"].quantile(0.75),
        "q90": df["total_parked_mins"].quantile(0.90),
    }

    port_stats = df.groupby("port_name").agg(
        count=("total_parked_mins", "count"), unique_ships=("VESSEL_IMO", "nunique"),
        mean_mins=("total_parked_mins", "mean"), median_mins=("total_parked_mins", "median"),
    ).round(2).sort_values("count", ascending=False)

    bins = [0, 30, 60, 120, 360, 720, 1440, 2880, float("inf")]
    labels_b = ["<30m", "30-60m", "1-2h", "2-6h", "6-12h", "12-24h", "1-2d", ">2d"]
    df["bucket"] = pd.cut(df["total_parked_mins"], bins=bins, labels=labels_b)
    bucket_counts = df["bucket"].value_counts().sort_index()

    print("=" * 80)
    print(f"{'PARKING DURATION STATISTICS' + label:^80}")
    print("=" * 80)
    for k, v in overall.items():
        print(f"  {k}: {v:,.2f}" if isinstance(v, float) else f"  {k}: {v:,}")
    print(f"\n  By port:\n{port_stats.to_string()}")
    print(f"\n  Duration buckets:\n{bucket_counts.to_string()}")
    print("=" * 80)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        port_stats.to_csv(os.path.join(output_dir, f"parking_stats_by_port{output_suffix}.csv"))
        bucket_counts.to_frame("count").to_csv(os.path.join(output_dir, f"parking_buckets{output_suffix}.csv"))

    return {"overall": overall, "by_port": port_stats, "buckets": bucket_counts}


def plot_parking_duration(parking_df, port_name=None, max_mins=5000, save_dir=None):
    df = parking_df.copy()
    if port_name:
        df = df[df["port_name"] == port_name]
    label = f" - {port_name}" if port_name else " - All Ports"
    dfv = df[df["total_parked_mins"] <= max_mins]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Histogram
    ax = axes[0, 0]
    ax.hist(dfv["total_parked_mins"], bins=50, color="steelblue", alpha=0.7, edgecolor="black")
    ax.axvline(dfv["total_parked_mins"].mean(), color="red", ls="--", lw=2,
               label=f"Mean: {dfv['total_parked_mins'].mean():.1f}")
    ax.axvline(dfv["total_parked_mins"].median(), color="orange", ls="--", lw=2,
               label=f"Median: {dfv['total_parked_mins'].median():.1f}")
    ax.set(xlabel="Duration (min)", ylabel="Frequency", title=f"Parking Duration{label}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Box by port (top 10)
    ax = axes[0, 1]
    top10 = df["port_name"].value_counts().head(10).index.tolist()
    box_data = [dfv[dfv["port_name"] == p]["total_parked_mins"].values for p in top10]
    bp = ax.boxplot(box_data, labels=top10, patch_artist=True, vert=False)
    for patch in bp["boxes"]:
        patch.set_facecolor("steelblue")
        patch.set_alpha(0.7)
    ax.set(xlabel="Duration (min)", title="By Port (Top 10)")
    ax.grid(True, alpha=0.3)

    # Monthly trend
    ax = axes[1, 0]
    df["month"] = pd.to_datetime(df["entry_time"]).dt.to_period("M").astype(str)
    monthly = df.groupby("month").agg(mean=("total_parked_mins", "mean"),
                                       median=("total_parked_mins", "median"),
                                       count=("total_parked_mins", "count"))
    x = range(len(monthly))
    ax.bar(x, monthly["count"], color="lightgray", alpha=0.7)
    ax2 = ax.twinx()
    ax2.plot(x, monthly["mean"], "ro-", lw=2, ms=8, label="Mean")
    ax2.plot(x, monthly["median"], "bs--", lw=2, ms=8, label="Median")
    ax.set_xticks(list(x))
    ax.set_xticklabels(monthly.index, rotation=45)
    ax.set(xlabel="Month", ylabel="Count")
    ax2.set_ylabel("Duration (min)")
    ax2.legend(loc="upper right")
    ax.set_title("Monthly Trend")
    ax.grid(True, alpha=0.3)

    # Buckets
    ax = axes[1, 1]
    bins = [0, 30, 60, 120, 360, 720, 1440, 2880, float("inf")]
    labs = ["<30m", "30-60m", "1-2h", "2-6h", "6-12h", "12-24h", "1-2d", ">2d"]
    df["bucket"] = pd.cut(df["total_parked_mins"], bins=bins, labels=labs)
    bc = df["bucket"].value_counts().sort_index()
    ax.bar(range(len(bc)), bc.values, color="steelblue", alpha=0.7)
    ax.set_xticks(range(len(bc)))
    ax.set_xticklabels(bc.index)
    total = bc.sum()
    for i, v in enumerate(bc.values):
        ax.text(i, v + total * 0.01, f"{v / total * 100:.1f}%", ha="center", fontsize=9)
    ax.set(xlabel="Duration Bucket", ylabel="Count", title="Duration Buckets")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, save_dir, "parking_duration.png")
    plt.show()
    return fig


# ============================================================================
# 7. Parking box plot by port
# ============================================================================

def plot_parking_boxplot_by_port(parking_df, title_suffix="", save_dir=None, ylim=None):
    try:
        import seaborn as sns
    except ImportError:
        print("  [warn] seaborn not installed — skipping boxplot.")
        return None, None

    df = parking_df.copy()
    order = df.groupby("port_name")["total_parked_mins"].median().sort_values(ascending=False).index
    fig = plt.figure(figsize=(20, 10))
    sns.boxplot(data=df, x="port_name", y="total_parked_mins", order=order, palette="viridis", showfliers=True)
    plt.title(f"Parking Duration by Port {title_suffix}", fontsize=20, fontweight="bold", pad=20)
    plt.xlabel("Port")
    plt.ylabel("Duration (min)")
    plt.xticks(rotation=90)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    if ylim:
        plt.ylim(ylim)
    plt.tight_layout()
    _save(fig, save_dir, "parking_boxplot.png")
    plt.show()

    stats = df.groupby("port_name")["total_parked_mins"].agg(
        Avg="mean", Median="median", Max="max", Count="count"
    ).sort_values("Avg", ascending=False).round(1)
    return stats, fig


# ============================================================================
# 8. Day-over-day comparison heatmap + changes
# ============================================================================

def plot_day_over_day(df, date_col="entry_time", port_col="port_name",
                       sector="Events", top_n=20, vessel_filter=None,
                       vessel_col="vessel_type", save_dir=None):
    data = df.copy()
    data[date_col] = pd.to_datetime(data[date_col])
    data["day"] = data[date_col].dt.date
    if vessel_filter:
        data = data[data[vessel_col].isin(vessel_filter)]

    day_order = sorted(data["day"].unique())
    n_days = len(day_order)
    day_labels = [d.strftime("%b %d") for d in day_order]
    if n_days < 2:
        print("Need ≥2 days for comparison.")
        return {}

    daily = data.groupby(["day", port_col]).size().unstack(fill_value=0)

    # Heatmap
    top_ports = daily.sum().nlargest(top_n).index
    hm = daily[top_ports].T
    hm.columns = day_labels

    fig1, axes = plt.subplots(2, 1, figsize=(18, 14), gridspec_kw={"height_ratios": [1, 2]})
    totals = daily.sum(axis=1).reindex(day_order, fill_value=0)
    ax = axes[0]
    ax.plot(range(n_days), totals.values, marker="o", lw=2.5, color="steelblue", ms=8)
    ax.fill_between(range(n_days), totals.values, alpha=0.15, color="steelblue")
    for i, v in enumerate(totals.values):
        ax.text(i, v + totals.max() * 0.02, str(int(v)), ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(range(n_days))
    ax.set_xticklabels(day_labels, rotation=45, ha="right", fontsize=8)
    ax.set(ylabel=f"Total {sector}", title=f"Daily Total {sector}")
    ax.grid(axis="y", ls="--", alpha=0.4)

    ax2 = axes[1]
    im = ax2.imshow(hm.values, aspect="auto", cmap="YlOrRd")
    ax2.set_xticks(range(n_days))
    ax2.set_xticklabels(day_labels, rotation=45, ha="right", fontsize=8)
    ax2.set_yticks(range(len(top_ports)))
    ax2.set_yticklabels(top_ports, fontsize=9)
    ax2.set_title(f"Top {top_n} Ports — Daily {sector}", fontweight="bold")
    fig1.colorbar(im, ax=ax2, shrink=0.6, label=sector)
    for i in range(hm.shape[0]):
        for j in range(hm.shape[1]):
            v = hm.values[i, j]
            color = "white" if v > hm.values.max() * 0.6 else "black"
            ax2.text(j, i, str(int(v)), ha="center", va="center", fontsize=8, color=color)
    fig1.tight_layout()
    _save(fig1, save_dir, f"day_over_day_{sector.replace(' ', '_')}.png")
    plt.show()
    return fig1


# ============================================================================
# 9. Single-port deep-dive
# ============================================================================

def analyze_single_port(arrivals_df, departures_df, port_code,
                         parking_df=None, time_col="TIMESTAMP",
                         port_col="port_name", vessel_col="vessel_type",
                         save_dir=None):
    arr = arrivals_df.copy()
    dep = departures_df.copy()
    arr[time_col] = pd.to_datetime(arr[time_col])
    dep[time_col] = pd.to_datetime(dep[time_col])
    arr_p = arr[arr[port_col] == port_code].copy()
    dep_p = dep[dep[port_col] == port_code].copy()

    if arr_p.empty and dep_p.empty:
        print(f"No data for port {port_code}.")
        return None

    arr_p["day"] = arr_p[time_col].dt.date
    arr_p["hour"] = arr_p[time_col].dt.hour
    dep_p["day"] = dep_p[time_col].dt.date
    if "stay_duration_mins" in arr_p.columns:
        arr_p["duration_hours"] = arr_p["stay_duration_mins"] / 60

    all_days = sorted(set(arr_p["day"]) | set(dep_p["day"]))
    day_labels = [d.strftime("%b %d") for d in all_days]
    n_days = len(all_days)

    arr_p["vessel_group"] = arr_p[vessel_col].apply(_map_vessel_group)
    dep_p["vessel_group"] = dep_p[vessel_col].apply(_map_vessel_group)

    has_park = parking_df is not None
    if has_park:
        park = parking_df.copy()
        park["entry_time"] = pd.to_datetime(park["entry_time"])
        park_p = park[park[port_col] == port_code].copy()
        park_p["day"] = park_p["entry_time"].dt.date
        park_p["vessel_group"] = park_p[vessel_col].apply(_map_vessel_group)
        has_park = not park_p.empty
        if has_park:
            park_daily = park_p.groupby("day").size().reindex(all_days, fill_value=0)

    n_rows = 3 + (1 if has_park else 0)
    fig, axes = plt.subplots(n_rows, 2, figsize=(22, 7 * n_rows))
    fig.suptitle(f"{port_code} — Deep Dive\nArrivals: {len(arr_p)}  Departures: {len(dep_p)}"
                 + (f"  Parking: {len(park_p)}" if has_park else ""),
                 fontsize=18, fontweight="bold", y=1.01)

    # Row 0: stacked bars
    def _stacked(ax, src, day_col, title):
        d = src.groupby([day_col, "vessel_group"]).size().unstack(fill_value=0).reindex(all_days, fill_value=0)
        bottom = np.zeros(len(d))
        for vg in ["Container", "Tanker", "Other"]:
            if vg in d.columns:
                ax.bar(range(len(d)), d[vg], bottom=bottom, label=vg, color=VG_COLORS.get(vg), alpha=0.85)
                bottom += d[vg].values
        for i, total in enumerate(d.sum(axis=1)):
            if total > 0:
                ax.text(i, total + 0.5, str(int(total)), ha="center", fontweight="bold", fontsize=10)
        ax.set_xticks(range(len(d)))
        ax.set_xticklabels(day_labels, rotation=45, ha="right", fontsize=8)
        ax.set(ylabel="Count", title=title)
        ax.legend(fontsize=9)
        ax.grid(axis="y", ls="--", alpha=0.3)
        return d

    daily_arr = _stacked(axes[0, 0], arr_p, "day", "Daily Arrivals by Type")
    daily_dep = _stacked(axes[0, 1], dep_p, "day", "Daily Departures by Type")

    # Row 1: net flow + cumulative
    ax = axes[1, 0]
    at = daily_arr.sum(axis=1).reindex(all_days, fill_value=0)
    dt = daily_dep.sum(axis=1).reindex(all_days, fill_value=0)
    x = np.arange(n_days)
    w = 0.35
    ax.bar(x - w / 2, at.values, w, label="Arrivals", color="#2980b9", alpha=0.8)
    ax.bar(x + w / 2, dt.values, w, label="Departures", color="#e74c3c", alpha=0.8)
    for i in range(n_days):
        net = int(at.values[i] - dt.values[i])
        c = "#27ae60" if net >= 0 else "#e74c3c"
        ax.text(i, max(at.values[i], dt.values[i]) + 1, f"net {net:+d}", ha="center", fontsize=9, color=c, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(day_labels, rotation=45, ha="right", fontsize=8)
    ax.set(ylabel="Count", title="Net Flow")
    ax.legend()
    ax.grid(axis="y", ls="--", alpha=0.3)
    ax_twin = ax.twinx()
    cum = np.cumsum([int(at.get(d, 0) - dt.get(d, 0)) for d in all_days])
    ax_twin.plot(range(n_days), cum, marker="D", lw=2.5, color="#8e44ad", ms=7, label="Cumulative")
    ax_twin.set_ylabel("Cumulative Net", color="#8e44ad")
    ax_twin.legend(loc="upper left", fontsize=9)

    axes[1, 1].axis("off")

    # Parking rows
    row_offset = 2
    if has_park:
        ax = axes[row_offset, 0]
        ax.plot(range(n_days), at.values, marker="o", lw=2.5, color="#2980b9", ms=7, label="Arrivals")
        ax.plot(range(n_days), park_daily.values, marker="D", lw=2.5, color="#27ae60", ms=7, label="Parking")
        ax.plot(range(n_days), dt.values, marker="s", lw=2.5, color="#e74c3c", ms=7, label="Departures")
        ax.fill_between(range(n_days), at.values, park_daily.values, alpha=0.15, color="#f39c12")
        ax.set_xticks(range(n_days))
        ax.set_xticklabels(day_labels, rotation=45, ha="right", fontsize=8)
        ax.set(ylabel="Count", title="Arrivals vs Parking vs Departures")
        ax.legend(fontsize=9)
        ax.grid(True, ls="--", alpha=0.3)

        ax = axes[row_offset, 1]
        for vg in ["Container", "Tanker", "Other"]:
            vg_d = park_p[park_p["vessel_group"] == vg].groupby("day").size().reindex(all_days, fill_value=0)
            if vg_d.sum() > 0:
                ax.plot(range(n_days), vg_d.values, marker="o", lw=2, color=VG_COLORS.get(vg), ms=6, label=vg)
        ax.set_xticks(range(n_days))
        ax.set_xticklabels(day_labels, rotation=45, ha="right", fontsize=8)
        ax.set(ylabel="Count", title="Parking by Vessel Type")
        ax.legend(fontsize=8)
        ax.grid(True, ls="--", alpha=0.3)
        row_offset += 1

    # Duration box + hourly heatmap
    dur_ax = axes[row_offset, 0]
    heat_ax = axes[row_offset, 1]

    if "duration_hours" in arr_p.columns:
        groups = [vg for vg in ["Container", "Tanker", "Other"] if vg in arr_p["vessel_group"].values]
        bd = [arr_p[arr_p["vessel_group"] == vg]["duration_hours"].dropna().values for vg in groups]
        bp = dur_ax.boxplot(bd, labels=groups, vert=True, patch_artist=True, showfliers=False)
        for patch, vg in zip(bp["boxes"], groups):
            patch.set_facecolor(VG_COLORS.get(vg, "#95a5a6"))
            patch.set_alpha(0.7)
        dur_ax.set(ylabel="Stay (hours)", title="Duration by Type")
        dur_ax.grid(axis="y", ls="--", alpha=0.3)
    else:
        dur_ax.text(0.5, 0.5, "No duration data", ha="center", va="center", transform=dur_ax.transAxes)

    hourly = arr_p.groupby(["day", "hour"]).size().unstack(fill_value=0).reindex(index=all_days, columns=range(24), fill_value=0)
    hourly.index = day_labels
    im = heat_ax.imshow(hourly.values, aspect="auto", cmap="YlOrRd")
    heat_ax.set_xticks(range(24))
    heat_ax.set_xticklabels(range(24), fontsize=8)
    heat_ax.set_yticks(range(len(day_labels)))
    heat_ax.set_yticklabels(day_labels)
    heat_ax.set(xlabel="Hour (UTC)", title="Hourly Arrival Heatmap")
    fig.colorbar(im, ax=heat_ax, shrink=0.7, label="Arrivals")

    plt.tight_layout()
    _save(fig, save_dir, f"single_port_{port_code}.png")
    plt.show()

    # Print summary
    print("=" * 80)
    print(f"  {port_code}: {len(arr_p)} arr | {len(dep_p)} dep | net {len(arr_p) - len(dep_p):+d}")
    if has_park:
        print(f"  Parking: {len(park_p)} | Still in port: ~{len(arr_p) - len(park_p)}")
    print("=" * 80)

    return {"arrivals": arr_p, "departures": dep_p, "figure": fig}


# ============================================================================
# 10. Interactive Folium map
# ============================================================================

def create_interactive_map(arrivals_df, departures_df, parking_df,
                            port_list_df, focus_ports=None,
                            time_col="TIMESTAMP", port_col="port_name",
                            vessel_col="vessel_type", save_path=None,
                            title="Interactive Port Analysis",
                            show_routes=True, show_chokepoints=True):
    """
    Create an interactive Folium map with arrival/departure/parking popups,
    shipping route polylines, chokepoint markers, legend, and title overlay.

    Parameters
    ----------
    arrivals_df, departures_df, parking_df : pd.DataFrame
        Output from detect_port_events / calculate_parking_durations.
    port_list_df : pd.DataFrame
        Port list with port, lat, lon columns.
    focus_ports : list[str] | None
        If set, only show these ports on the map.
    time_col, port_col, vessel_col : str
        Column names.
    save_path : str | None
        Path to save the HTML file (e.g. 'output/map.html').
    title : str
        Title shown in the map overlay.
    show_routes : bool
        Draw Suez / Cape / East Asia shipping route polylines.
    show_chokepoints : bool
        Show chokepoint markers (Hormuz, Suez, Bab el-Mandeb, Cape).

    Returns
    -------
    folium.Map or None (if folium not installed)
    """
    try:
        import folium
    except ImportError:
        print("  [warn] folium not installed — skipping interactive map. pip install folium")
        return None

    arr = arrivals_df.copy()
    dep = departures_df.copy()
    park = parking_df.copy()
    ports_df = port_list_df.copy()

    if focus_ports:
        ports_df = ports_df[ports_df["port"].isin(focus_ports)]
        arr = arr[arr[port_col].isin(focus_ports)]
        dep = dep[dep[port_col].isin(focus_ports)]
        park = park[park[port_col].isin(focus_ports)]

    arr[time_col] = pd.to_datetime(arr[time_col])
    dep[time_col] = pd.to_datetime(dep[time_col])
    park["entry_time"] = pd.to_datetime(park["entry_time"])
    arr["day"] = arr[time_col].dt.strftime("%b %d")
    dep["day"] = dep[time_col].dt.strftime("%b %d")
    park["day"] = park["entry_time"].dt.strftime("%b %d")

    def _vg(vt):
        if pd.isna(vt): return "Other"
        v = vt.upper()
        if "CONTAINER" in v: return "Container"
        if any(k in v for k in ["TANKER", "CRUDE", "LNG", "GAS", "CHEMICAL"]): return "Tanker"
        return "Other"

    arr["vg"] = arr[vessel_col].apply(_vg)
    dep["vg"] = dep[vessel_col].apply(_vg)

    all_days = sorted(arr["day"].unique())
    arr_port = arr.groupby(port_col).size().to_dict()
    dep_port = dep.groupby(port_col).size().to_dict()
    park_port = park.groupby(port_col).size().to_dict()
    arr_daily = arr.groupby([port_col, "day"]).size().unstack(fill_value=0)
    dep_daily = dep.groupby([port_col, "day"]).size().unstack(fill_value=0)
    park_daily = park.groupby([port_col, "day"]).size().unstack(fill_value=0)
    arr_vg = arr.groupby([port_col, "vg"]).size().unstack(fill_value=0)
    dep_vg = dep.groupby([port_col, "vg"]).size().unstack(fill_value=0)

    # ==================================================================
    # Create map
    # ==================================================================
    m = folium.Map(location=[20, 60], zoom_start=3, tiles="CartoDB positron", control_scale=True)

    # ==================================================================
    # Shipping routes
    # ==================================================================
    if show_routes:
        # Suez route (disrupted)
        suez_coords = [
            [22, 120], [10, 114], [3, 108], [1.3, 103.8], [4, 98], [5, 92],
            [5, 86], [6, 80], [8, 77], [10, 73], [14, 68], [16, 62],
            [15, 57], [13, 52], [12, 48], [12, 45], [12.5, 43.5], [14.5, 42],
            [17, 40], [21, 38], [25, 35.5], [28, 33.5], [30, 32.5],
            [31.3, 32.3], [32, 30], [34, 25], [35, 20], [36, 15],
            [37, 10], [38, 5], [37, 0], [36, -5.5], [37, -9.5], [45, -5], [50, 0],
        ]
        folium.PolyLine(
            suez_coords, color="#e74c3c", weight=3, opacity=0.7,
            dash_array="10 5", tooltip="Suez Route (DISRUPTED)"
        ).add_to(m)

        # Cape route (active)
        cape_coords = [
            [22, 120], [10, 114], [3, 108], [1.3, 103.8], [4, 98], [5, 92],
            [5, 86], [6, 80], [8, 77], [8, 73], [6, 68], [2, 62],
            [-5, 55], [-10, 48], [-14, 43], [-20, 38], [-28, 33],
            [-34, 26], [-34.5, 18.5], [-30, 14], [-22, 12], [-12, 10],
            [-5, 8], [-2, 3], [0, -5], [3, -12], [8, -17], [15, -18],
            [21, -17], [28, -15], [34, -12], [37, -9.5], [45, -5], [50, 0],
        ]
        folium.PolyLine(
            cape_coords, color="#2ecc71", weight=3, opacity=0.7,
            tooltip="Cape of Good Hope Route (ACTIVE)"
        ).add_to(m)

        # East Asia branch
        branch_coords = [[22, 120], [31.2, 121.3], [33, 125], [34, 130], [34.7, 135], [35.5, 139.6]]
        folium.PolyLine(
            branch_coords, color="#3498db", weight=2, opacity=0.5,
            tooltip="East Asia Connections"
        ).add_to(m)

    # ==================================================================
    # Chokepoints
    # ==================================================================
    if show_chokepoints:
        chokepoints = [
            {"name": "Strait of Hormuz", "lat": 26.5, "lon": 56.5, "status": "CLOSED", "color": "red"},
            {"name": "Suez Canal", "lat": 30.5, "lon": 32.3, "status": "Disrupted", "color": "orange"},
            {"name": "Bab el-Mandeb", "lat": 12.5, "lon": 43.3, "status": "Disrupted", "color": "orange"},
            {"name": "Cape of Good Hope", "lat": -34.5, "lon": 18.5, "status": "Active", "color": "green"},
        ]
        for cp in chokepoints:
            folium.Marker(
                [cp["lat"], cp["lon"]],
                popup=f"<b>{cp['name']}</b><br>Status: {cp['status']}",
                tooltip=f"{cp['name']} ({cp['status']})",
                icon=folium.Icon(
                    color=cp["color"],
                    icon="warning-sign" if cp["status"] != "Active" else "ok-sign",
                    prefix="glyphicon",
                ),
            ).add_to(m)

    # ==================================================================
    # Port markers with rich popups
    # ==================================================================
    for _, row in ports_df.iterrows():
        port = row["port"]
        lat, lon = row["lat"], row["lon"]
        a = arr_port.get(port, 0)
        d = dep_port.get(port, 0)
        p = park_port.get(port, 0)
        net = a - d
        gap = a - p

        if a == 0 and d == 0:
            continue

        # Status label
        if net > 5:
            net_label = "&#x1F534; Accumulating"
        elif net < -5:
            net_label = "&#x1F7E2; Draining"
        else:
            net_label = "&#x26AA; Balanced"

        # Daily breakdown rows
        daily_rows = ""
        for day in all_days:
            ad = int(arr_daily.loc[port, day]) if port in arr_daily.index and day in arr_daily.columns else 0
            dd = int(dep_daily.loc[port, day]) if port in dep_daily.index and day in dep_daily.columns else 0
            pd_ = int(park_daily.loc[port, day]) if port in park_daily.index and day in park_daily.columns else 0
            nd = ad - dd
            nc = "#27ae60" if nd >= 0 else "#e74c3c"
            daily_rows += (
                f'<tr><td>{day}</td>'
                f'<td style="text-align:right">{ad}</td>'
                f'<td style="text-align:right">{dd}</td>'
                f'<td style="text-align:right">{pd_}</td>'
                f'<td style="text-align:right;color:{nc};font-weight:bold">{nd:+d}</td></tr>'
            )

        # Totals row
        net_color = "#27ae60" if net >= 0 else "#e74c3c"
        daily_rows += (
            f'<tr style="background:#eee;font-weight:bold;">'
            f'<td style="padding:4px;">Total</td>'
            f'<td style="padding:4px;text-align:right">{a}</td>'
            f'<td style="padding:4px;text-align:right">{d}</td>'
            f'<td style="padding:4px;text-align:right">{p}</td>'
            f'<td style="padding:4px;text-align:right;color:{net_color}">{net:+d}</td></tr>'
        )

        # Vessel type rows
        vtype_rows = ""
        for vg in ["Container", "Tanker", "Other"]:
            av = int(arr_vg.loc[port, vg]) if port in arr_vg.index and vg in arr_vg.columns else 0
            dv = int(dep_vg.loc[port, vg]) if port in dep_vg.index and vg in dep_vg.columns else 0
            vtype_rows += (
                f'<tr><td>{vg}</td>'
                f'<td style="text-align:right">{av}</td>'
                f'<td style="text-align:right">{dv}</td>'
                f'<td style="text-align:right;font-weight:bold">{av - dv:+d}</td></tr>'
            )

        popup_html = f"""
        <div style="font-family:Arial;width:380px;font-size:12px;">
            <h3 style="margin:0;padding:5px;background:#1a3a5c;color:white;border-radius:5px 5px 0 0;text-align:center;">
                {port}
            </h3>
            <div style="padding:8px;background:#f8f9fa;border:1px solid #ddd;">
                <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
                    <div style="text-align:center;flex:1;">
                        <div style="font-size:20px;font-weight:bold;color:#2980b9;">{a}</div>
                        <div style="font-size:10px;color:#666;">Arrivals</div>
                    </div>
                    <div style="text-align:center;flex:1;">
                        <div style="font-size:20px;font-weight:bold;color:#e74c3c;">{d}</div>
                        <div style="font-size:10px;color:#666;">Departures</div>
                    </div>
                    <div style="text-align:center;flex:1;">
                        <div style="font-size:20px;font-weight:bold;color:#27ae60;">{p}</div>
                        <div style="font-size:10px;color:#666;">Parking</div>
                    </div>
                    <div style="text-align:center;flex:1;">
                        <div style="font-size:20px;font-weight:bold;color:{net_color}">{net:+d}</div>
                        <div style="font-size:10px;color:#666;">Net Flow</div>
                    </div>
                </div>
                <div style="text-align:center;margin-bottom:8px;font-size:11px;">
                    Status: <b>{net_label}</b> &nbsp;|&nbsp; Gap (A-P): <b>{gap}</b> vessels not yet departed
                </div>
                <table style="width:100%;border-collapse:collapse;font-size:11px;margin-bottom:8px;">
                    <tr style="background:#2980b9;color:white;">
                        <th style="padding:4px;">Day</th>
                        <th style="padding:4px;text-align:right;">Arr</th>
                        <th style="padding:4px;text-align:right;">Dep</th>
                        <th style="padding:4px;text-align:right;">Park</th>
                        <th style="padding:4px;text-align:right;">Net</th>
                    </tr>
                    {daily_rows}
                </table>
                <table style="width:100%;border-collapse:collapse;font-size:11px;">
                    <tr style="background:#e67e22;color:white;">
                        <th style="padding:4px;">Vessel Type</th>
                        <th style="padding:4px;text-align:right;">Arr</th>
                        <th style="padding:4px;text-align:right;">Dep</th>
                        <th style="padding:4px;text-align:right;">Net</th>
                    </tr>
                    {vtype_rows}
                </table>
            </div>
        </div>
        """

        # Circle size based on total arrivals
        radius = max(4, min(15, a / 30))

        # Color based on net flow
        if net > 10:
            color = "#e74c3c"       # accumulating
        elif net < -10:
            color = "#2ecc71"       # draining
        else:
            color = "#3498db"       # balanced

        tooltip_text = f"{port}: {a} arr / {d} dep / {p} park (net {net:+d})"

        folium.CircleMarker(
            [lat, lon], radius=radius, color=color, fill=True, fill_color=color,
            fill_opacity=0.6, weight=2,
            popup=folium.Popup(popup_html, max_width=400),
            tooltip=tooltip_text,
        ).add_to(m)

        # Port label
        folium.Marker(
            [lat, lon],
            icon=folium.DivIcon(
                html=f'<div style="font-size:9px;font-weight:bold;color:#2c3e50;'
                     f'text-shadow:1px 1px 2px white;white-space:nowrap;">{port}</div>',
                icon_size=(60, 15), icon_anchor=(30, -10),
            ),
        ).add_to(m)

    # ==================================================================
    # Legend overlay
    # ==================================================================
    # Derive date range from the data
    date_range = ""
    if all_days:
        date_range = f"{all_days[0]}–{all_days[-1]}"

    legend_html = f"""
    <div style="position:fixed;bottom:30px;right:30px;z-index:1000;
                background:rgba(255,255,255,0.95);padding:15px;border-radius:8px;
                font-size:12px;color:#333;font-family:Arial;
                border:1px solid #ccc;box-shadow:0 2px 6px rgba(0,0,0,0.15);">
        <b style="font-size:14px;">Port Analysis Map</b><br>
        <b>{date_range}</b><br><br>
        <span style="color:#e74c3c;">&#9679;</span> Accumulating (net &gt; +10)<br>
        <span style="color:#3498db;">&#9679;</span> Balanced (-10 to +10)<br>
        <span style="color:#2ecc71;">&#9679;</span> Draining (net &lt; -10)<br>
        <span style="font-size:8px;">&#9675;</span> Circle size = arrival volume<br><br>
        <span style="color:#e74c3c;">- - -</span> Suez Route (Disrupted)<br>
        <span style="color:#2ecc71;">&#9472;&#9472;&#9472;</span> Cape Route (Active)<br><br>
        <i>Click any port for details</i>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # ==================================================================
    # Title overlay
    # ==================================================================
    title_html = f"""
    <div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);
                z-index:1000;background:rgba(255,255,255,0.95);padding:10px 25px;
                border-radius:8px;font-family:Arial;text-align:center;
                border:1px solid #ccc;box-shadow:0 2px 6px rgba(0,0,0,0.15);">
        <span style="color:#2c3e50;font-size:18px;font-weight:bold;">
            {title}
        </span><br>
        <span style="color:#777;font-size:12px;">
            {date_range} | Click ports for daily breakdown
        </span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        m.save(save_path)
        print(f"  Interactive map saved to: {save_path}")
    return m
