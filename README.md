# Maritime Port Analysis

AIS vessel tracking data — port event detection and visualization toolkit.

---

## Quick Start — Single Period Analysis

```python
from maritime_port_analysis import MaritimePortAnalyzer

results = MaritimePortAnalyzer(
    ais_data="original AIS data/202603/*.csv",
    port_list="port_list_15.csv",
    output_suffix="_202603",
    focus_ports=["SGSIN", "NLRTM", "CNSHA"],
).run()
```

---

## Cross-Period Comparison

```python
from maritime_port_analysis import (
    MaritimePortAnalyzer,
    compare_parking_distributions,
    compare_parking_dashboard,
    compare_port_changes,
    compare_regional_vessel_type,
    compare_daily_by_vessel_type,
    compare_daily_by_region,
    compare_single_port,
    compare_all_ports,
    view_ship_events,
)

# Step 1: Run two periods separately
r1 = MaritimePortAnalyzer(ais_data="data/AIS_202509*.csv", port_list="port_list_15.csv").run()
r2 = MaritimePortAnalyzer(ais_data="data/AIS_202601*.csv", port_list="port_list_15.csv").run()

# Step 2: Compare parking durations (4-panel: boxplot, histogram, mean bars, count bars)
stats, fig = compare_parking_distributions(
    r1.parking, r2.parking,
    label_1="Sep 2025", label_2="Jan 2026",
)

# Step 3: Full comparison dashboard (peak hours, vessel types, peak days, day-of-week)
figs = compare_parking_dashboard(
    r1.parking, r2.parking,
    label_1="Sep 2025", label_2="Jan 2026",
    save_dir="comparison_output",
)

# Step 4: Compare port arrival/departure/parking changes
compare_port_changes(r1.arrivals, r2.arrivals,
                     period1_name="Sep 2025", period2_name="Jan 2026",
                     sector="Arrivals")

compare_port_changes(r1.departures, r2.departures,
                     period1_name="Sep 2025", period2_name="Jan 2026",
                     sector="Departures")

compare_port_changes(r1.parking, r2.parking,
                     period1_name="Sep 2025", period2_name="Jan 2026",
                     sector="Parking")

# Step 5: Regional & vessel type comparisons
compare_regional_vessel_type(
    r1.arrivals, r1.departures, r2.arrivals, r2.departures,
    port_list_df=r1.port_list,
    label_1="Sep 2025", label_2="Jan 2026",
)

compare_daily_by_vessel_type(
    r1.arrivals, r1.departures, r2.arrivals, r2.departures,
    label_1="Sep 2025", label_2="Jan 2026",
)

compare_daily_by_region(
    r1.arrivals, r1.departures, r2.arrivals, r2.departures,
    port_list_df=r1.port_list,
    label_1="Sep 2025", label_2="Jan 2026",
)

# Step 6: Per-port deep dive (single port or all focus ports)
compare_single_port(
    r1.arrivals, r1.departures, r2.arrivals, r2.departures,
    port_code="SGSIN",
    parking_1=r1.parking, parking_2=r2.parking,
    label_1="Sep 2025", label_2="Jan 2026",
)

compare_all_ports(
    r1.arrivals, r1.departures, r2.arrivals, r2.departures,
    port_list_df=r1.port_list,
    parking_1=r1.parking, parking_2=r2.parking,
    label_1="Sep 2025", label_2="Jan 2026",
    focus_ports=["SGSIN", "NLRTM", "CNSHA"],
)

# Step 7: Inspect a specific ship's journey
view_ship_events(
    vessel_imo=9858292,
    arrivals_df=r1.arrivals,
    departures_df=r1.departures,
    tracking_data=r1.ais_data,
    reassignments_df=r1.reassignments,
)
```

---

## Step-by-Step Execution

```python
analyzer = MaritimePortAnalyzer(
    ais_data=my_dataframe,
    port_list="port_list_15.csv",
    extra_ports=[{"port": "INNSA", "lat": 18.952, "lon": 72.948}],
    radius_deg=1.0,
    output_suffix="_202603",
    focus_ports=["SGSIN", "NLRTM", "CNSHA"],
)
analyzer.detect()           # Detect arrival/departure events
analyzer.compute_parking()  # Calculate parking (stay) durations
analyzer.visualize()        # Generate all charts
analyzer.create_map()       # Generate interactive Folium map
results = analyzer.results
```

---

## Command Line

```bash
python -m maritime_port_analysis.run \
    --ais "original AIS data/202603/*.csv" \
    --ports "port_list_15.csv" \
    --output "output_202603" \
    --suffix "_202603" \
    --focus SGSIN NLRTM CNSHA
```

---

## Per-Port Radius

```csv
port,lat,lon,radius
SGSIN,1.283,103.85,0.5
NLRTM,51.917,4.5,1.2
```

If the `radius` column is absent, the `radius_deg` default is used.

---

## Package Structure

```
maritime_port_analysis/
├── __init__.py        # Public API
├── config.py          # AnalysisConfig parameters
├── utils.py           # Data loading
├── detection.py       # Core detection algorithm
├── visualization.py   # Single-period visualization & regional helpers
├── comparison.py      # Cross-period comparison (9 functions)
├── pipeline.py        # MaritimePortAnalyzer orchestrator
└── run.py             # CLI entry point
```

### All Visualization & Analysis Functions

| Module | Function | Description |
|--------|----------|-------------|
| visualization | `plot_daily_arrivals` | Daily arrivals overlay line chart |
| visualization | `plot_daily_arrivals_grid` | Per-port arrivals sub-plot grid |
| visualization | `plot_arrival_ranking` | Port arrival ranking bar chart |
| visualization | `plot_daily_departures_grid` | Daily departures grid |
| visualization | `plot_arrival_departure_comparison` | Arrivals vs departures comparison (grid + totals) |
| visualization | `analyze_parking_stats` | Parking statistics (by port / vessel type / month) |
| visualization | `plot_parking_duration` | Parking duration 4-panel chart |
| visualization | `plot_parking_boxplot_by_port` | Box plot by port |
| visualization | `plot_day_over_day` | Day-over-day heatmap |
| visualization | `analyze_single_port` | Single-port deep-dive analysis |
| visualization | `create_interactive_map` | Folium interactive map |
| visualization | `plot_regional_vessel_type` | Regional arrivals/departures by vessel type |
| visualization | `plot_daily_by_vessel_type` | Daily arrivals/departures broken down by vessel type |
| visualization | `plot_daily_by_region` | Daily arrivals/departures broken down by region |

### Regional Helpers

| Module | Function | Description |
|--------|----------|-------------|
| visualization | `PORT_REGION_MAP` | Port-code → region mapping dictionary |
| visualization | `classify_region` | Classify a single port code into its region |
| visualization | `add_region_to_ports` | Add a `region` column to a port list DataFrame |
| visualization | `enrich_with_region` | Add a `region` column to an events DataFrame |

### Cross-Period Comparison Functions

| Module | Function | Description |
|--------|----------|-------------|
| comparison | `compare_parking_distributions` | Two-period parking distribution comparison (4-panel) |
| comparison | `compare_parking_dashboard` | Multi-dimension comparison dashboard (peak hours, vessel types, peak days, day-of-week) |
| comparison | `compare_port_changes` | Port-level change analysis (absolute & percentage) |
| comparison | `compare_regional_vessel_type` | Side-by-side regional × vessel type comparison |
| comparison | `compare_daily_by_vessel_type` | Daily vessel type trends across two periods |
| comparison | `compare_daily_by_region` | Daily regional trends across two periods |
| comparison | `compare_single_port` | Per-port side-by-side deep dive (arrivals, departures, net flow) |
| comparison | `compare_all_ports` | Loop `compare_single_port` across all focus ports |
| comparison | `view_ship_events` | Single-vessel event timeline viewer |
