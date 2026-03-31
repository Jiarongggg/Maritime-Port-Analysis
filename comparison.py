"""
Cross-period comparison functions.
All functions take TWO datasets and produce comparative visualizations.
"""
import math
import os
from typing import Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _save(fig, directory, filename, dpi=300):
    if directory:
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, filename)
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {path}")


# ============================================================================
# 1. compare_parking_distributions
# ============================================================================

def compare_parking_distributions(
    parking_df_1, parking_df_2,
    label_1="Dataset 1", label_2="Dataset 2",
    port_col="port_name", duration_col="total_parked_mins",
    use_minutes=True, top_n_ports=15, ylim=None,
    save_dir=None,
):
    """
    Compare parking duration distributions between two datasets.
    Produces 4 panels: side-by-side boxplot, histogram overlay,
    mean duration bars, and visit count bars.

    Returns (comparison_stats_df, figure).
    """
    df1 = parking_df_1.copy()
    df2 = parking_df_2.copy()
    df1["dataset"] = label_1
    df2["dataset"] = label_2
    combined = pd.concat([df1, df2], ignore_index=True)

    if not use_minutes:
        combined["dur"] = combined[duration_col] / 60
        dur_label = "Duration (Hours)"
    else:
        combined["dur"] = combined[duration_col]
        dur_label = "Duration (Minutes)"

    top_ports = combined[port_col].value_counts().head(top_n_ports).index.tolist()
    comb_top = combined[combined[port_col].isin(top_ports)]
    port_order = comb_top.groupby(port_col)["dur"].median().sort_values(ascending=False).index

    try:
        import seaborn as sns
        has_sns = True
    except ImportError:
        has_sns = False

    fig, axes = plt.subplots(2, 2, figsize=(20, 12))

    # 1a — Side-by-side boxplot
    ax = axes[0, 0]
    if has_sns:
        sns.boxplot(data=comb_top, x=port_col, y="dur", hue="dataset",
                    order=port_order, palette=["steelblue", "coral"],
                    showfliers=False, ax=ax)
    else:
        ax.text(0.5, 0.5, "seaborn required for boxplot", transform=ax.transAxes, ha="center")
    ax.set_title(f"Parking Duration: {label_1} vs {label_2}", fontweight="bold")
    ax.set_xlabel("Port")
    ax.set_ylabel(dur_label)
    ax.tick_params(axis="x", rotation=90)
    ax.grid(axis="y", ls="--", alpha=0.3)
    if ylim:
        ax.set_ylim(ylim)

    # 1b — Histogram overlay
    ax = axes[0, 1]
    max_val = combined["dur"].quantile(0.95)
    d1 = combined.loc[(combined["dataset"] == label_1) & (combined["dur"] <= max_val), "dur"]
    d2 = combined.loc[(combined["dataset"] == label_2) & (combined["dur"] <= max_val), "dur"]
    ax.hist(d1, bins=50, alpha=0.5, label=f"{label_1} (mean {d1.mean():.1f})", color="steelblue", edgecolor="black")
    ax.hist(d2, bins=50, alpha=0.5, label=f"{label_2} (mean {d2.mean():.1f})", color="coral", edgecolor="black")
    ax.axvline(d1.mean(), color="steelblue", ls="--", lw=2)
    ax.axvline(d2.mean(), color="coral", ls="--", lw=2)
    ax.set_title("Duration Distribution Comparison", fontweight="bold")
    ax.set_xlabel(dur_label)
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 1c — Mean duration bars
    ax = axes[1, 0]
    mean_by = comb_top.groupby([port_col, "dataset"])["dur"].mean().unstack().reindex(port_order)
    x = range(len(mean_by))
    w = 0.35
    ax.bar([i - w / 2 for i in x], mean_by.get(label_1, 0), w, label=label_1, color="steelblue", alpha=0.7)
    ax.bar([i + w / 2 for i in x], mean_by.get(label_2, 0), w, label=label_2, color="coral", alpha=0.7)
    ax.set_title("Mean Parking Duration by Port", fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(mean_by.index, rotation=90)
    ax.set_ylabel(f"Mean {dur_label}")
    ax.legend()
    ax.grid(axis="y", ls="--", alpha=0.3)

    # 1d — Visit count bars
    ax = axes[1, 1]
    cnt_by = comb_top.groupby([port_col, "dataset"]).size().unstack().reindex(port_order).fillna(0)
    ax.bar([i - w / 2 for i in x], cnt_by.get(label_1, 0), w, label=label_1, color="steelblue", alpha=0.7)
    ax.bar([i + w / 2 for i in x], cnt_by.get(label_2, 0), w, label=label_2, color="coral", alpha=0.7)
    ax.set_title("Visit Count by Port", fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(cnt_by.index, rotation=90)
    ax.set_ylabel("Visits")
    ax.legend()
    ax.grid(axis="y", ls="--", alpha=0.3)

    plt.tight_layout()
    _save(fig, save_dir, f"compare_parking_{label_1}_vs_{label_2}.png".replace(" ", "_"))
    plt.show()

    # Build comparison stats table
    s1 = df1.groupby(port_col)[duration_col].agg(count_1="count", mean_1="mean", median_1="median").round(1)
    s2 = df2.groupby(port_col)[duration_col].agg(count_2="count", mean_2="mean", median_2="median").round(1)
    stats = s1.join(s2, how="outer").fillna(0)
    stats["count_diff"] = stats["count_2"] - stats["count_1"]
    stats["mean_diff"] = stats["mean_2"] - stats["mean_1"]
    stats["count_diff_pct"] = np.where(stats["count_1"] > 0,
                                        ((stats["count_2"] - stats["count_1"]) / stats["count_1"] * 100).round(1),
                                        np.inf)
    stats = stats.sort_values("count_1", ascending=False)

    # Print summary
    print("\n" + "=" * 100)
    print(f"{'COMPARISON: ' + label_1 + ' vs ' + label_2:^100}")
    print("=" * 100)
    print(f"  {'Metric':<30} {label_1:>20} {label_2:>20} {'Diff':>20}")
    print("  " + "-" * 90)
    print(f"  {'Total Visits':<30} {len(df1):>20,} {len(df2):>20,} {len(df2) - len(df1):>+20,}")
    print(f"  {'Unique Ships':<30} {df1['VESSEL_IMO'].nunique():>20,} {df2['VESSEL_IMO'].nunique():>20,} {df2['VESSEL_IMO'].nunique() - df1['VESSEL_IMO'].nunique():>+20,}")
    print(f"  {'Mean Duration (mins)':<30} {df1[duration_col].mean():>20.1f} {df2[duration_col].mean():>20.1f} {df2[duration_col].mean() - df1[duration_col].mean():>+20.1f}")
    print(f"  {'Median Duration (mins)':<30} {df1[duration_col].median():>20.1f} {df2[duration_col].median():>20.1f} {df2[duration_col].median() - df1[duration_col].median():>+20.1f}")
    print("=" * 100)

    return stats, fig


# ============================================================================
# 2. compare_parking_dashboard
# ============================================================================

def compare_parking_dashboard(
    parking_df_1, parking_df_2,
    label_1="Dataset 1", label_2="Dataset 2",
    port_col="port_name", duration_col="total_parked_mins",
    entry_time_col="entry_time", exit_time_col="exit_time",
    vessel_type_col="vessel_type",
    outlier_threshold=5000, top_ports=15,
    save_dir=None,
):
    """
    Multi-figure comparison dashboard: peak hours, duration & port analysis,
    peak day I/O, and day-of-week patterns.

    Returns dict of figures.
    """
    df1 = parking_df_1.copy()
    df2 = parking_df_2.copy()
    df1["dataset"], df2["dataset"] = label_1, label_2

    for df in [df1, df2]:
        df[entry_time_col] = pd.to_datetime(df[entry_time_col])
        df[exit_time_col] = pd.to_datetime(df[exit_time_col])
        df["entry_hour"] = df[entry_time_col].dt.hour
        df["exit_hour"] = df[exit_time_col].dt.hour
        df["entry_day"] = df[entry_time_col].dt.day_name()
        df["entry_date"] = df[entry_time_col].dt.date
        df["exit_date"] = df[exit_time_col].dt.date

    combined = pd.concat([df1, df2], ignore_index=True)
    f1 = df1[df1[duration_col] < outlier_threshold]
    f2 = df2[df2[duration_col] < outlier_threshold]

    figs = {}
    hours = range(24)
    w = 0.35
    x24 = np.arange(24)

    # ---- FIGURE 1: Peak Hours ----
    fig1, axes = plt.subplots(2, 2, figsize=(18, 12))
    e1 = df1.groupby("entry_hour").size()
    e2 = df2.groupby("entry_hour").size()
    ex1 = df1.groupby("exit_hour").size()
    ex2 = df2.groupby("exit_hour").size()

    for ax, data_a, data_b, title in [
        (axes[0, 0], e1, e2, "Entry Hour Distribution"),
        (axes[0, 1], ex1, ex2, "Exit Hour Distribution"),
    ]:
        ax.bar(x24 - w / 2, [data_a.get(h, 0) for h in hours], w, label=label_1, color="steelblue", alpha=0.7)
        ax.bar(x24 + w / 2, [data_b.get(h, 0) for h in hours], w, label=label_2, color="coral", alpha=0.7)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Hour")
        ax.set_ylabel("Count")
        ax.set_xticks(range(0, 24, 2))
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    for ax, data_e, data_x, lbl in [
        (axes[1, 0], e1, ex1, label_1),
        (axes[1, 1], e2, ex2, label_2),
    ]:
        ax.plot(hours, [data_e.get(h, 0) for h in hours], "o-", color="steelblue", lw=2, ms=5, label="Entry")
        ax.plot(hours, [data_x.get(h, 0) for h in hours], "s-", color="coral", lw=2, ms=5, label="Exit")
        ax.set_title(f"Entry vs Exit — {lbl}", fontweight="bold")
        ax.set_xlabel("Hour")
        ax.set_ylabel("Count")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig1.suptitle(f"Peak Hours Analysis: {label_1} vs {label_2}", fontsize=16, y=1.02)
    plt.tight_layout()
    _save(fig1, save_dir, "dashboard_01_peak_hours.png")
    plt.show()
    figs["peak_hours"] = fig1

    # ---- FIGURE 2: Duration & Ports ----
    fig2, axes = plt.subplots(2, 2, figsize=(18, 12))

    ax = axes[0, 0]
    w1 = np.ones_like(f1[duration_col].values, dtype=float) / len(f1) * 100
    w2 = np.ones_like(f2[duration_col].values, dtype=float) / len(f2) * 100
    ax.hist(f1[duration_col], bins=50, alpha=0.5, label=label_1, color="steelblue", weights=w1)
    ax.hist(f2[duration_col], bins=50, alpha=0.5, label=label_2, color="coral", weights=w2)
    ax.set_title("Duration Distribution (Normalized %)", fontweight="bold")
    ax.set_xlabel("Duration (min)")
    ax.set_ylabel("%")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    ax = axes[0, 1]
    top_v = combined[vessel_type_col].value_counts().head(10).index.tolist()
    v1 = f1[f1[vessel_type_col].isin(top_v)].groupby(vessel_type_col)[duration_col].median()
    v2 = f2[f2[vessel_type_col].isin(top_v)].groupby(vessel_type_col)[duration_col].median()
    pd.DataFrame({label_1: v1, label_2: v2}).fillna(0).sort_values(label_1).plot(
        kind="barh", ax=ax, color=["steelblue", "coral"], alpha=0.8, width=0.8)
    ax.set_title("Median Duration by Vessel Type", fontweight="bold")
    ax.set_xlabel("Median (min)")
    ax.grid(axis="x", alpha=0.3)

    top_list = combined[port_col].value_counts().head(top_ports).index.tolist()
    for ax, agg_fn, title in [
        (axes[1, 0], lambda d, col: d[d[port_col].isin(top_list)].groupby(port_col)[col].median(), f"Median Duration (Top {top_ports})"),
        (axes[1, 1], lambda d, col: d[d[port_col].isin(top_list)].groupby(port_col).size(), f"Visit Count (Top {top_ports})"),
    ]:
        if "Duration" in title:
            r1, r2 = agg_fn(f1, duration_col), agg_fn(f2, duration_col)
        else:
            r1, r2 = agg_fn(df1, port_col), agg_fn(df2, port_col)
        pd.DataFrame({label_1: r1, label_2: r2}).fillna(0).sort_values(label_1).plot(
            kind="barh", ax=ax, color=["steelblue", "coral"], alpha=0.8, width=0.8)
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

    fig2.suptitle(f"Duration & Port Analysis: {label_1} vs {label_2}", fontsize=16, y=1.02)
    plt.tight_layout()
    _save(fig2, save_dir, "dashboard_02_duration_ports.png")
    plt.show()
    figs["duration_ports"] = fig2

    # ---- FIGURE 3: Peak Day I/O ----
    fig3, axes = plt.subplots(2, 2, figsize=(18, 12))

    for idx, (df, lbl) in enumerate([(df1, label_1), (df2, label_2)]):
        di = df.groupby("entry_date").size().reset_index(name="input")
        do = df.groupby("exit_date").size().reset_index(name="output")
        merged = pd.merge(di, do, left_on="entry_date", right_on="exit_date", how="outer")
        merged["date"] = merged["entry_date"].combine_first(merged["exit_date"])
        merged = merged[["date", "input", "output"]].fillna(0).sort_values("date")

        ax = axes[0, idx]
        ax.plot(merged["date"], merged["input"], "o-", color="steelblue", lw=2, ms=4, label="Input")
        ax.plot(merged["date"], merged["output"], "s-", color="coral", lw=2, ms=4, label="Output")
        peak_in = merged.loc[merged["input"].idxmax()]
        peak_out = merged.loc[merged["output"].idxmax()]
        ax.axhline(peak_in["input"], color="darkblue", ls="--", alpha=0.5, label=f"Peak In: {peak_in['date']}")
        ax.axhline(peak_out["output"], color="darkred", ls="--", alpha=0.5, label=f"Peak Out: {peak_out['date']}")
        ax.set_title(f"Daily I/O — {lbl}", fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Vessels")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=45)

    # Peak comparison bars
    for idx, (metric, ylabel) in enumerate([("input", "Arrivals"), ("output", "Departures")]):
        ax = axes[1, idx]
        peaks = []
        for df, lbl in [(df1, label_1), (df2, label_2)]:
            col = "entry_date" if metric == "input" else "exit_date"
            daily = df.groupby(col).size()
            peak_date = daily.idxmax()
            peaks.append({"label": lbl, "date": peak_date, "count": daily.max()})
        bars = ax.bar(range(2), [p["count"] for p in peaks], color=["steelblue", "coral"], alpha=0.8)
        ax.set_title(f"Peak {ylabel} Day", fontweight="bold")
        ax.set_ylabel(f"Number of {ylabel}")
        ax.set_xticks(range(2))
        ax.set_xticklabels([p["label"] for p in peaks])
        for i, bar in enumerate(bars):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{int(peaks[i]["count"])}\n{peaks[i]["date"]}',
                    ha="center", va="bottom", fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    fig3.suptitle(f"Peak Day Analysis: {label_1} vs {label_2}", fontsize=16, y=1.02)
    plt.tight_layout()
    _save(fig3, save_dir, "dashboard_03_peak_days.png")
    plt.show()
    figs["peak_days"] = fig3

    # ---- FIGURE 4: Day of Week ----
    fig4, axes = plt.subplots(1, 2, figsize=(18, 6))
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    xd = np.arange(len(day_order))

    ax = axes[0]
    da1 = df1.groupby("entry_day").size().reindex(day_order, fill_value=0)
    da2 = df2.groupby("entry_day").size().reindex(day_order, fill_value=0)
    ax.bar(xd - w / 2, da1, w, label=label_1, color="steelblue", alpha=0.8)
    ax.bar(xd + w / 2, da2, w, label=label_2, color="coral", alpha=0.8)
    ax.set_title("Arrivals by Day of Week", fontweight="bold")
    ax.set_xticks(xd)
    ax.set_xticklabels(day_order, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    plot_data, positions, colors = [], [], []
    for i, day in enumerate(day_order):
        for df_f, offset, color in [(f1, -0.15, "steelblue"), (f2, 0.15, "coral")]:
            vals = df_f[df_f["entry_day"] == day][duration_col].dropna().values
            if len(vals) > 0:
                plot_data.append(vals)
                positions.append(i + offset)
                colors.append(color)
    if plot_data:
        bp = ax.boxplot(plot_data, positions=positions, patch_artist=True, widths=0.25, showfliers=False)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
    ax.set_title("Duration by Day of Week", fontweight="bold")
    ax.set_xticks(range(len(day_order)))
    ax.set_xticklabels(day_order, rotation=45, ha="right")
    ax.set_ylabel("Duration (min)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="steelblue", alpha=0.7, label=label_1),
                       Patch(facecolor="coral", alpha=0.7, label=label_2)])
    ax.grid(axis="y", alpha=0.3)

    fig4.suptitle(f"Day of Week: {label_1} vs {label_2}", fontsize=16, y=1.02)
    plt.tight_layout()
    _save(fig4, save_dir, "dashboard_04_day_of_week.png")
    plt.show()
    figs["day_of_week"] = fig4

    return figs


# ============================================================================
# 3. compare_port_changes
# ============================================================================

def compare_port_changes(
    df_period1, df_period2,
    port_col="port_name",
    period1_name="Period 1", period2_name="Period 2",
    sector="Events", top_n=20,
    save_dir=None,
):
    """
    Compare port counts between two periods. Identifies ports with
    greatest absolute and percentage changes.

    Returns comparison DataFrame.
    """
    c1 = df_period1.groupby(port_col).size().reset_index(name="count_p1")
    c2 = df_period2.groupby(port_col).size().reset_index(name="count_p2")
    comp = c1.merge(c2, on=port_col, how="outer").fillna(0)
    comp["count_p1"] = comp["count_p1"].astype(int)
    comp["count_p2"] = comp["count_p2"].astype(int)
    comp["abs_change"] = comp["count_p2"] - comp["count_p1"]
    comp["pct_change"] = np.where(
        comp["count_p1"] > 0,
        ((comp["count_p2"] - comp["count_p1"]) / comp["count_p1"] * 100).round(1),
        np.where(comp["count_p2"] > 0, np.inf, 0),
    )
    comp["abs_mag"] = comp["abs_change"].abs()
    comp = comp.sort_values("abs_mag", ascending=False).reset_index(drop=True)

    # Print summary
    print("=" * 80)
    print(f"{'PORT ' + sector.upper() + ': ' + period1_name + ' vs ' + period2_name:^80}")
    print("=" * 80)
    print(f"  Total {sector}: {comp['count_p1'].sum()} → {comp['count_p2'].sum()} "
          f"({comp['count_p2'].sum() - comp['count_p1'].sum():+d})")

    top_inc = comp[(comp["abs_change"] > 0) & (comp["pct_change"] != np.inf)].nlargest(10, "pct_change")
    if not top_inc.empty:
        print(f"\n  Top Increases:")
        for _, r in top_inc.iterrows():
            print(f"    {r[port_col]:<12} {r['count_p1']:>5} → {r['count_p2']:>5}  ({r['abs_change']:+d}, {r['pct_change']:+.1f}%)")

    top_dec = comp[comp["abs_change"] < 0].nsmallest(10, "pct_change")
    if not top_dec.empty:
        print(f"\n  Top Decreases:")
        for _, r in top_dec.iterrows():
            print(f"    {r[port_col]:<12} {r['count_p1']:>5} → {r['count_p2']:>5}  ({r['abs_change']:+d}, {r['pct_change']:+.1f}%)")
    print("=" * 80)

    # Plot 4-panel
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))

    # Absolute change
    ax = axes[0, 0]
    top_abs = comp.head(top_n)
    colors = ["green" if x > 0 else "red" for x in top_abs["abs_change"]]
    ax.barh(top_abs[port_col], top_abs["abs_change"], color=colors, edgecolor="black", alpha=0.7)
    ax.axvline(0, color="black", lw=0.8)
    ax.invert_yaxis()
    ax.set_title(f"Top {top_n} — Absolute {sector} Change", fontweight="bold")
    ax.set_xlabel(f"Absolute Change")
    ax.grid(axis="x", ls="--", alpha=0.5)

    # Percentage change
    ax = axes[0, 1]
    pct_valid = comp[(comp["pct_change"] != np.inf) & (comp["abs_change"] != 0) & (comp["count_p1"] > 0)].copy()
    pct_valid["abs_pct"] = pct_valid["pct_change"].abs()
    top_pct = pct_valid.nlargest(top_n, "abs_pct")
    colors = ["green" if x > 0 else "red" for x in top_pct["pct_change"]]
    ax.barh(top_pct[port_col], top_pct["pct_change"], color=colors, edgecolor="black", alpha=0.7)
    ax.axvline(0, color="black", lw=0.8)
    ax.invert_yaxis()
    ax.set_title(f"Top {top_n} — Percentage {sector} Change", fontweight="bold")
    ax.set_xlabel("% Change")
    ax.grid(axis="x", ls="--", alpha=0.5)

    # Grouped bar
    ax = axes[1, 0]
    top_total = comp.nlargest(top_n, "count_p2")
    x = np.arange(len(top_total))
    w = 0.35
    ax.bar(x - w / 2, top_total["count_p1"], w, label=period1_name, color="steelblue", alpha=0.8)
    ax.bar(x + w / 2, top_total["count_p2"], w, label=period2_name, color="darkorange", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(top_total[port_col], rotation=45, ha="right")
    ax.set_title(f"Top {top_n} Ports — {sector} Comparison", fontweight="bold")
    ax.set_ylabel(sector)
    ax.legend()
    ax.grid(axis="y", ls="--", alpha=0.5)

    # Scatter
    ax = axes[1, 1]
    if not pct_valid.empty:
        sc_colors = ["green" if x > 0 else "red" for x in pct_valid["abs_change"]]
        ax.scatter(pct_valid["abs_change"], pct_valid["pct_change"],
                   c=sc_colors, alpha=0.6, s=80, edgecolors="black", lw=0.5)
        for _, r in pct_valid.nlargest(5, "abs_pct").iterrows():
            ax.annotate(r[port_col], (r["abs_change"], r["pct_change"]),
                        fontsize=8, xytext=(5, 5), textcoords="offset points",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title(f"Absolute vs Percentage {sector} Change", fontweight="bold")
    ax.set_xlabel(f"Absolute Change")
    ax.set_ylabel("% Change")
    ax.grid(ls="--", alpha=0.5)

    plt.tight_layout()
    _save(fig, save_dir, f"port_changes_{sector}_{period1_name}_vs_{period2_name}.png".replace(" ", "_"))
    plt.show()

    return comp


# ============================================================================
# 4. view_ship_events — detailed per-ship event log
# ============================================================================

def view_ship_events(vessel_imo, arrivals_df, departures_df,
                     tracking_data, reassignments_df=None):
    """
    Print all records for a specific ship with ENTRY / EXIT / IN_PORT / AT_SEA labels.
    Returns labelled DataFrame.
    """
    time_col = "TIMESTAMP_UTC" if "TIMESTAMP_UTC" in arrivals_df.columns else "TIMESTAMP"
    ship_arr = arrivals_df[arrivals_df["VESSEL_IMO"] == vessel_imo].copy()
    ship_dep = departures_df[departures_df["VESSEL_IMO"] == vessel_imo].copy()

    df = tracking_data.copy()
    raw_tc = "TIMESTAMP_UTC" if "TIMESTAMP_UTC" in df.columns else "TIMESTAMP"
    df[raw_tc] = pd.to_datetime(df[raw_tc])
    df = df.rename(columns={"SHIP_ID": "VESSEL_IMO", "LAT": "LATITUDE", "LON": "LONGITUDE"})
    ship = df[df["VESSEL_IMO"] == vessel_imo].sort_values(raw_tc).reset_index(drop=True)

    print("=" * 100)
    print(f"SHIP DETAIL: {vessel_imo}")
    print("=" * 100)
    if ship.empty:
        print("No data found.")
        return pd.DataFrame()

    entry_times = {pd.to_datetime(r[time_col]): r["port_name"] for _, r in ship_arr.iterrows()}
    exit_times = {pd.to_datetime(r[time_col]): r["port_name"] for _, r in ship_dep.iterrows()}
    intervals = []
    for _, a in ship_arr.iterrows():
        et = pd.to_datetime(a[time_col])
        intervals.append({"entry": et, "exit": et + pd.Timedelta(minutes=a["stay_duration_mins"]), "port": a["port_name"]})

    labels, ports = [], []
    for _, rec in ship.iterrows():
        t = rec[raw_tc]
        if t in entry_times:
            labels.append("ENTRY"); ports.append(entry_times[t])
        elif t in exit_times:
            labels.append("EXIT"); ports.append(exit_times[t])
        else:
            found = False
            for iv in intervals:
                if iv["entry"] <= t <= iv["exit"]:
                    labels.append("IN_PORT"); ports.append(iv["port"]); found = True; break
            if not found:
                labels.append("AT_SEA"); ports.append("")

    ship["EVENT"] = labels
    ship["port_name"] = ports
    ship["PARKED"] = (ship["SPEED"] <= 0.5) | (ship["SPEED"].isna())

    cols = [raw_tc, "EVENT", "port_name", "SPEED", "LONGITUDE", "LATITUDE", "PARKED"]

    print(f"Records: {len(ship)} | Entries: {len(ship_arr)} | Exits: {len(ship_dep)}")
    all_ports = set(ship_arr["port_name"]) | set(ship_dep["port_name"])
    for p in sorted(all_ports):
        ea = (ship_arr["port_name"] == p).sum()
        ed = (ship_dep["port_name"] == p).sum()
        dur = ship_arr.loc[ship_arr["port_name"] == p, "stay_duration_mins"].sum()
        print(f"  {p:10s} | Entries: {ea} | Exits: {ed} | Duration: {dur:.0f} min")

    if reassignments_df is not None and not reassignments_df.empty:
        sr = reassignments_df[reassignments_df["VESSEL_IMO"] == vessel_imo]
        if not sr.empty:
            print(f"\nReassigned stays:")
            for _, r in sr.iterrows():
                print(f"  {r['assigned_port_entry']} → {r['assigned_port_mean']} | {r['stay_duration_mins']:.0f} min")

    print("=" * 100)
    print(ship[cols].to_string())
    return ship[cols]


# ============================================================================
# Import helpers from visualization module
# ============================================================================
from .visualization import (
    _map_vessel_group, VG_COLORS, REGION_COLORS,
    classify_region, enrich_with_region,
)


# ============================================================================
# 5. compare_regional_vessel_type — side-by-side regional breakdown
# ============================================================================

def compare_regional_vessel_type(
    arrivals_1, departures_1, arrivals_2, departures_2,
    port_list_df,
    label_1="Period 1", label_2="Period 2",
    port_col="port_name", vessel_col="vessel_type",
    save_dir=None,
):
    """
    Side-by-side regional × vessel type comparison.
    Shared y-axis for direct visual comparison.

    Figure 1: Stacked bars per region (4 panels: Arr P1, Arr P2, Dep P1, Dep P2)
    Figure 2: Per-region subplots — grouped bars (P1 vs P2) for arr and dep by vessel type
    """
    vg_order = ["Container", "Tanker", "Other"]

    # Enrich all with region and vessel group
    datasets = {}
    for key, df_src in [("arr1", arrivals_1), ("arr2", arrivals_2),
                         ("dep1", departures_1), ("dep2", departures_2)]:
        d = enrich_with_region(df_src, port_list_df, port_col)
        d["vessel_group"] = d[vessel_col].apply(_map_vessel_group)
        datasets[key] = d

    all_regions = sorted(
        set().union(*(d["region"].unique() for d in datasets.values())),
        key=lambda r: list(REGION_COLORS.keys()).index(r) if r in REGION_COLORS else 99
    )

    # ---- FIGURE 1: 2×2 stacked bars (Arrivals top, Departures bottom) ----
    fig1, axes = plt.subplots(2, 2, figsize=(22, 14))
    fig1.suptitle(f"Regional Vessel Type: {label_1} vs {label_2}", fontsize=16, fontweight="bold", y=1.02)

    panels = [
        (axes[0, 0], datasets["arr1"], f"Arrivals — {label_1}"),
        (axes[0, 1], datasets["arr2"], f"Arrivals — {label_2}"),
        (axes[1, 0], datasets["dep1"], f"Departures — {label_1}"),
        (axes[1, 1], datasets["dep2"], f"Departures — {label_2}"),
    ]

    # Compute shared y-max per row
    def _get_max(d):
        pivot = d.groupby(["region", "vessel_group"]).size().unstack(fill_value=0)
        return pivot.reindex(index=all_regions, columns=vg_order, fill_value=0).sum(axis=1).max()

    ymax_arr = max(_get_max(datasets["arr1"]), _get_max(datasets["arr2"])) * 1.15
    ymax_dep = max(_get_max(datasets["dep1"]), _get_max(datasets["dep2"])) * 1.15

    for ax, df_src, title in panels:
        pivot = df_src.groupby(["region", "vessel_group"]).size().unstack(fill_value=0)
        pivot = pivot.reindex(index=all_regions, columns=vg_order, fill_value=0)

        x = np.arange(len(all_regions))
        bottom = np.zeros(len(all_regions))
        for vg in vg_order:
            vals = pivot[vg].values if vg in pivot.columns else np.zeros(len(all_regions))
            ax.bar(x, vals, bottom=bottom, label=vg, color=VG_COLORS.get(vg), alpha=0.85, edgecolor="white")
            for i, v in enumerate(vals):
                if v > 0:
                    ax.text(i, bottom[i] + v / 2, str(int(v)), ha="center", va="center", fontsize=9, fontweight="bold", color="white")
            bottom += vals

        for i, t in enumerate(bottom):
            if t > 0:
                ax.text(i, t + max(bottom) * 0.02, str(int(t)), ha="center", fontsize=10, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(all_regions, rotation=30, ha="right")
        ax.set_ylabel("Count")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(axis="y", ls="--", alpha=0.3)
        ax.set_ylim(0, ymax_arr if "Arrivals" in title else ymax_dep)

    plt.tight_layout()
    _save(fig1, save_dir, f"compare_regional_stacked_{label_1}_vs_{label_2}.png".replace(" ", "_"))
    plt.show()

    # ---- FIGURE 2: Per-region subplots — grouped bars P1 vs P2 ----
    n_regions = len(all_regions)
    ncols = min(3, n_regions)
    nrows = math.ceil(n_regions / ncols)
    fig2, axes_grid = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows))
    fig2.suptitle(f"Per-Region Arr/Dep by Vessel Type: {label_1} vs {label_2}",
                  fontsize=16, fontweight="bold", y=1.02)
    axes_flat = np.array(axes_grid).flatten() if n_regions > 1 else [axes_grid]

    for idx, region in enumerate(all_regions):
        ax = axes_flat[idx]
        x = np.arange(len(vg_order))
        w = 0.2

        bars_data = []
        bar_labels = [f"Arr {label_1}", f"Arr {label_2}", f"Dep {label_1}", f"Dep {label_2}"]
        bar_colors = ["#2980b9", "#85C1E9", "#e74c3c", "#F1948A"]

        for i, (key, lbl) in enumerate([("arr1", label_1), ("arr2", label_2),
                                          ("dep1", label_1), ("dep2", label_2)]):
            d = datasets[key]
            counts = d[d["region"] == region]["vessel_group"].value_counts()
            vals = [counts.get(vg, 0) for vg in vg_order]
            bars_data.append(vals)
            offset = (i - 1.5) * w
            bars = ax.bar(x + offset, vals, w, label=bar_labels[i], color=bar_colors[i], alpha=0.85, edgecolor="white")
            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, h, str(int(h)),
                            ha="center", va="bottom", fontsize=7, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(vg_order)
        ax.set_ylabel("Count")
        ax.set_title(region, fontsize=13, fontweight="bold", color=REGION_COLORS.get(region, "#333"))
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(axis="y", ls="--", alpha=0.3)

    for j in range(len(all_regions), len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    _save(fig2, save_dir, f"compare_regional_detail_{label_1}_vs_{label_2}.png".replace(" ", "_"))
    plt.show()

    return {"stacked": fig1, "detail": fig2}


# ============================================================================
# 6. compare_daily_by_vessel_type — shared y-axis daily trends
# ============================================================================

def compare_daily_by_vessel_type(
    arrivals_1, departures_1, arrivals_2, departures_2,
    label_1="Period 1", label_2="Period 2",
    timestamp_col="TIMESTAMP", vessel_col="vessel_type",
    save_dir=None,
):
    """
    Per vessel type: daily arrivals and departures for two periods.
    One row per vessel type, two columns (Period 1 | Period 2), shared y-axis.
    """
    vg_order = ["Container", "Tanker", "Other"]

    def _prep(arr_df, dep_df):
        a = arr_df.copy()
        d = dep_df.copy()
        a[timestamp_col] = pd.to_datetime(a[timestamp_col])
        dep_tc = "TIMESTAMP_UTC" if "TIMESTAMP_UTC" in d.columns else timestamp_col
        d[dep_tc] = pd.to_datetime(d[dep_tc])
        a["day"] = a[timestamp_col].dt.date
        d["day"] = d[dep_tc].dt.date
        a["vessel_group"] = a[vessel_col].apply(_map_vessel_group)
        d["vessel_group"] = d[vessel_col].apply(_map_vessel_group)
        return a, d

    arr1, dep1 = _prep(arrivals_1, departures_1)
    arr2, dep2 = _prep(arrivals_2, departures_2)

    fig, axes = plt.subplots(3, 2, figsize=(22, 15))
    fig.suptitle(f"Daily Arrivals vs Departures by Vessel Type: {label_1} vs {label_2}",
                 fontsize=16, fontweight="bold", y=1.02)

    for row, vg in enumerate(vg_order):
        # Compute shared y-max for this row
        ymaxes = []
        for a, d in [(arr1, dep1), (arr2, dep2)]:
            all_days = sorted(set(a["day"]) | set(d["day"]))
            av = a[a["vessel_group"] == vg].groupby("day").size().reindex(all_days, fill_value=0)
            dv = d[d["vessel_group"] == vg].groupby("day").size().reindex(all_days, fill_value=0)
            ymaxes.append(max(av.max(), dv.max()) if len(av) > 0 else 0)
        ymax = max(ymaxes) * 1.15 if max(ymaxes) > 0 else 10

        for col, (a, d, lbl) in enumerate([(arr1, dep1, label_1), (arr2, dep2, label_2)]):
            ax = axes[row, col]
            all_days = sorted(set(a["day"]) | set(d["day"]))
            day_labels = [dd.strftime("%b %d") for dd in all_days]

            av = a[a["vessel_group"] == vg].groupby("day").size().reindex(all_days, fill_value=0)
            dv = d[d["vessel_group"] == vg].groupby("day").size().reindex(all_days, fill_value=0)

            ax.plot(range(len(all_days)), av.values, "o-", color="#2980b9", ms=4, lw=1.5, label="Arrivals")
            ax.plot(range(len(all_days)), dv.values, "s-", color="#e74c3c", ms=4, lw=1.5, label="Departures")
            ax.fill_between(range(len(all_days)), av.values, dv.values, alpha=0.1,
                            color="#27ae60" if av.sum() >= dv.sum() else "#e74c3c")

            net = av.sum() - dv.sum()
            ax.set_title(f"{vg} — {lbl} (Arr:{av.sum()} Dep:{dv.sum()} Net:{net:+d})",
                         fontsize=11, fontweight="bold", color=VG_COLORS.get(vg, "#333"))
            ax.set_xticks(range(len(all_days)))
            ax.set_xticklabels(day_labels, rotation=45, ha="right", fontsize=7)
            ax.set_ylabel("Count")
            ax.set_ylim(0, ymax)
            ax.legend(fontsize=8)
            ax.grid(True, ls="--", alpha=0.3)

    plt.tight_layout()
    _save(fig, save_dir, f"compare_daily_vessel_type_{label_1}_vs_{label_2}.png".replace(" ", "_"))
    plt.show()
    return fig


# ============================================================================
# 7. compare_daily_by_region — shared y-axis daily trends per region
# ============================================================================

def compare_daily_by_region(
    arrivals_1, departures_1, arrivals_2, departures_2,
    port_list_df,
    label_1="Period 1", label_2="Period 2",
    timestamp_col="TIMESTAMP", port_col="port_name",
    save_dir=None,
):
    """
    Per region: daily arrivals and departures for two periods.
    One row per region, two columns (Period 1 | Period 2), shared y-axis.
    """
    def _prep(arr_df, dep_df):
        a = enrich_with_region(arr_df, port_list_df, port_col)
        d = enrich_with_region(dep_df, port_list_df, port_col)
        a[timestamp_col] = pd.to_datetime(a[timestamp_col])
        dep_tc = "TIMESTAMP_UTC" if "TIMESTAMP_UTC" in d.columns else timestamp_col
        d[dep_tc] = pd.to_datetime(d[dep_tc])
        a["day"] = a[timestamp_col].dt.date
        d["day"] = d[dep_tc].dt.date
        return a, d

    arr1, dep1 = _prep(arrivals_1, departures_1)
    arr2, dep2 = _prep(arrivals_2, departures_2)

    all_regions = sorted(
        set(arr1["region"].unique()) | set(arr2["region"].unique()) |
        set(dep1["region"].unique()) | set(dep2["region"].unique()),
        key=lambda r: list(REGION_COLORS.keys()).index(r) if r in REGION_COLORS else 99
    )

    n_regions = len(all_regions)
    fig, axes = plt.subplots(n_regions, 2, figsize=(22, 5 * n_regions))
    fig.suptitle(f"Daily Arrivals vs Departures by Region: {label_1} vs {label_2}",
                 fontsize=16, fontweight="bold", y=1.01)

    if n_regions == 1:
        axes = axes.reshape(1, -1)

    for row, region in enumerate(all_regions):
        # Shared y-max
        ymaxes = []
        for a, d in [(arr1, dep1), (arr2, dep2)]:
            all_days = sorted(set(a["day"]) | set(d["day"]))
            av = a[a["region"] == region].groupby("day").size().reindex(all_days, fill_value=0)
            dv = d[d["region"] == region].groupby("day").size().reindex(all_days, fill_value=0)
            ymaxes.append(max(av.max(), dv.max()) if len(av) > 0 else 0)
        ymax = max(ymaxes) * 1.15 if max(ymaxes) > 0 else 10

        for col, (a, d, lbl) in enumerate([(arr1, dep1, label_1), (arr2, dep2, label_2)]):
            ax = axes[row, col]
            all_days = sorted(set(a["day"]) | set(d["day"]))
            day_labels = [dd.strftime("%b %d") for dd in all_days]

            av = a[a["region"] == region].groupby("day").size().reindex(all_days, fill_value=0)
            dv = d[d["region"] == region].groupby("day").size().reindex(all_days, fill_value=0)

            ax.plot(range(len(all_days)), av.values, "o-", color="#2980b9", ms=4, lw=1.5, label="Arrivals")
            ax.plot(range(len(all_days)), dv.values, "s-", color="#e74c3c", ms=4, lw=1.5, label="Departures")
            ax.fill_between(range(len(all_days)), av.values, dv.values, alpha=0.1,
                            color="#27ae60" if av.sum() >= dv.sum() else "#e74c3c")

            net = av.sum() - dv.sum()
            ax.set_title(f"{region} — {lbl} (Arr:{av.sum()} Dep:{dv.sum()} Net:{net:+d})",
                         fontsize=11, fontweight="bold", color=REGION_COLORS.get(region, "#333"))
            ax.set_xticks(range(len(all_days)))
            ax.set_xticklabels(day_labels, rotation=45, ha="right", fontsize=7)
            ax.set_ylabel("Count")
            ax.set_ylim(0, ymax)
            ax.legend(fontsize=8)
            ax.grid(True, ls="--", alpha=0.3)

    plt.tight_layout()
    _save(fig, save_dir, f"compare_daily_region_{label_1}_vs_{label_2}.png".replace(" ", "_"))
    plt.show()
    return fig


# ============================================================================
# 8. compare_single_port — per-port side-by-side deep dive
# ============================================================================

def compare_single_port(
    arrivals_1, departures_1, arrivals_2, departures_2,
    port_code, port_list_df=None,
    parking_1=None, parking_2=None,
    label_1="Period 1", label_2="Period 2",
    time_col="TIMESTAMP", port_col="port_name", vessel_col="vessel_type",
    save_dir=None,
):
    """
    Side-by-side deep dive for a single port across two periods.
    Shared y-axis per row for direct comparison.

    Row 1: Stacked arrivals by vessel type (P1 | P2)
    Row 2: Stacked departures by vessel type (P1 | P2)
    Row 3: Net flow + cumulative (P1 | P2)
    """
    def _prep(arr, dep, park, label):
        a = arr[arr[port_col] == port_code].copy()
        d = dep[dep[port_col] == port_code].copy()
        a[time_col] = pd.to_datetime(a[time_col])
        dep_tc = "TIMESTAMP_UTC" if "TIMESTAMP_UTC" in d.columns else time_col
        d[dep_tc] = pd.to_datetime(d[dep_tc])
        a["day"] = a[time_col].dt.date
        d["day"] = d[dep_tc].dt.date
        a["vessel_group"] = a[vessel_col].apply(_map_vessel_group)
        d["vessel_group"] = d[vessel_col].apply(_map_vessel_group)
        all_days = sorted(set(a["day"]) | set(d["day"]))
        day_labels = [dd.strftime("%b %d") for dd in all_days]

        p = None
        if park is not None:
            pk = park[park[port_col] == port_code].copy()
            if not pk.empty:
                pk["entry_time"] = pd.to_datetime(pk["entry_time"])
                pk["day"] = pk["entry_time"].dt.date
                p = pk

        return a, d, p, all_days, day_labels

    a1, d1, p1, days1, dl1 = _prep(arrivals_1, departures_1, parking_1, label_1)
    a2, d2, p2, days2, dl2 = _prep(arrivals_2, departures_2, parking_2, label_2)

    if a1.empty and d1.empty and a2.empty and d2.empty:
        print(f"No data for {port_code} in either period.")
        return None

    vg_order = ["Container", "Tanker", "Other"]
    n_rows = 3
    fig, axes = plt.subplots(n_rows, 2, figsize=(22, 5 * n_rows))

    total_a1, total_d1 = len(a1), len(d1)
    total_a2, total_d2 = len(a2), len(d2)
    fig.suptitle(
        f"{port_code} — {label_1} vs {label_2}\n"
        f"{label_1}: Arr {total_a1} / Dep {total_d1} / Net {total_a1 - total_d1:+d}    |    "
        f"{label_2}: Arr {total_a2} / Dep {total_d2} / Net {total_a2 - total_d2:+d}",
        fontsize=15, fontweight="bold", y=1.02)

    # Helper: stacked bar
    def _stacked(ax, src, all_days, day_labels, title, ymax=None):
        d_grp = src.groupby(["day", "vessel_group"]).size().unstack(fill_value=0).reindex(all_days, fill_value=0)
        d_grp = d_grp.reindex(columns=vg_order, fill_value=0)
        bottom = np.zeros(len(all_days))
        for vg in vg_order:
            vals = d_grp[vg].values
            ax.bar(range(len(all_days)), vals, bottom=bottom, label=vg,
                   color=VG_COLORS.get(vg), alpha=0.85, edgecolor="white", linewidth=0.5)
            bottom += vals
        for i, t in enumerate(bottom):
            if t > 0:
                ax.text(i, t + 0.5, str(int(t)), ha="center", fontsize=8, fontweight="bold")
        ax.set_xticks(range(len(all_days)))
        ax.set_xticklabels(day_labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Count")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(axis="y", ls="--", alpha=0.3)
        if ymax:
            ax.set_ylim(0, ymax)
        return d_grp

    # Row 0: Arrivals stacked — shared y
    ymax_arr = max(
        a1.groupby("day").size().max() if not a1.empty else 0,
        a2.groupby("day").size().max() if not a2.empty else 0,
    ) * 1.2
    _stacked(axes[0, 0], a1, days1, dl1, f"Arrivals — {label_1}", ymax_arr)
    _stacked(axes[0, 1], a2, days2, dl2, f"Arrivals — {label_2}", ymax_arr)

    # Row 1: Departures stacked — shared y
    ymax_dep = max(
        d1.groupby("day").size().max() if not d1.empty else 0,
        d2.groupby("day").size().max() if not d2.empty else 0,
    ) * 1.2
    _stacked(axes[1, 0], d1, days1, dl1, f"Departures — {label_1}", ymax_dep)
    _stacked(axes[1, 1], d2, days2, dl2, f"Departures — {label_2}", ymax_dep)

    # Row 2: Net flow + cumulative — shared y
    ymax_net = 0
    for ax, a_src, d_src, all_days, day_labels, lbl in [
        (axes[2, 0], a1, d1, days1, dl1, label_1),
        (axes[2, 1], a2, d2, days2, dl2, label_2),
    ]:
        at = a_src.groupby("day").size().reindex(all_days, fill_value=0)
        dt = d_src.groupby("day").size().reindex(all_days, fill_value=0)
        x = np.arange(len(all_days))
        w = 0.35
        ax.bar(x - w / 2, at.values, w, label="Arrivals", color="#2980b9", alpha=0.8)
        ax.bar(x + w / 2, dt.values, w, label="Departures", color="#e74c3c", alpha=0.8)

        for i in range(len(all_days)):
            net = int(at.values[i] - dt.values[i])
            c = "#27ae60" if net >= 0 else "#e74c3c"
            y = max(at.values[i], dt.values[i]) + 1
            ax.text(i, y, f"{net:+d}", ha="center", fontsize=7, color=c, fontweight="bold")

        ymax_net = max(ymax_net, max(at.max(), dt.max()) if len(at) > 0 else 0)

        ax.set_xticks(x)
        ax.set_xticklabels(day_labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Count")
        ax.set_title(f"Net Flow — {lbl}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(axis="y", ls="--", alpha=0.3)

        # Cumulative on twin axis
        ax2 = ax.twinx()
        cum = np.cumsum([int(at.get(d, 0) - dt.get(d, 0)) for d in all_days])
        ax2.plot(range(len(all_days)), cum, "D-", color="#8e44ad", lw=2, ms=5, label="Cumulative")
        ax2.set_ylabel("Cumulative Net", color="#8e44ad")
        ax2.tick_params(axis="y", labelcolor="#8e44ad")
        ax2.legend(loc="upper left", fontsize=7)

    # Apply shared y to net flow row
    axes[2, 0].set_ylim(0, ymax_net * 1.2)
    axes[2, 1].set_ylim(0, ymax_net * 1.2)

    plt.tight_layout()
    _save(fig, save_dir, f"compare_port_{port_code}_{label_1}_vs_{label_2}.png".replace(" ", "_"))
    plt.show()

    # Print summary
    print(f"  {port_code}: {label_1} → Arr {total_a1} / Dep {total_d1} / Net {total_a1 - total_d1:+d}")
    print(f"  {port_code}: {label_2} → Arr {total_a2} / Dep {total_d2} / Net {total_a2 - total_d2:+d}")

    return fig


# ============================================================================
# 9. compare_all_ports — loop compare_single_port for all focus ports
# ============================================================================

def compare_all_ports(
    arrivals_1, departures_1, arrivals_2, departures_2,
    port_list_df,
    parking_1=None, parking_2=None,
    label_1="Period 1", label_2="Period 2",
    focus_ports=None, save_dir=None,
):
    """
    Run compare_single_port for every focus port.
    If focus_ports is None, uses all ports found in both datasets.
    """
    if focus_ports is None:
        focus_ports = sorted(
            set(arrivals_1["port_name"].unique()) | set(arrivals_2["port_name"].unique())
        )

    figs = {}
    for port in focus_ports:
        print(f"\n{'=' * 70}")
        print(f"  Comparing {port}: {label_1} vs {label_2}")
        print(f"{'=' * 70}")
        fig = compare_single_port(
            arrivals_1, departures_1, arrivals_2, departures_2,
            port_code=port, port_list_df=port_list_df,
            parking_1=parking_1, parking_2=parking_2,
            label_1=label_1, label_2=label_2,
            save_dir=save_dir,
        )
        if fig:
            figs[port] = fig

    return figs
