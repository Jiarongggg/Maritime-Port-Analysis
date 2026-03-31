"""
Maritime Port Analysis — AIS-based port event detection and visualization.

Quick start (single period):
    from maritime_port_analysis import MaritimePortAnalyzer
    results = MaritimePortAnalyzer(
        ais_data="data/AIS_202603*.csv",
        port_list="port_list_15.csv",
    ).run()

Cross-period comparison:
    from maritime_port_analysis.comparison import (
        compare_parking_distributions,
        compare_parking_dashboard,
        compare_port_changes,
        view_ship_events,
    )
    compare_parking_distributions(parking_sep, parking_jan,
                                  label_1="Sep 2025", label_2="Jan 2026")
    compare_port_changes(arrivals_sep, arrivals_jan,
                         period1_name="Sep 2025", period2_name="Jan 2026",
                         sector="Arrivals")
"""

from .config import AnalysisConfig
from .pipeline import MaritimePortAnalyzer, AnalysisResults
from .detection import detect_port_events, calculate_parking_durations
from .utils import load_ais_data, load_port_list

# Regional helpers
from .visualization import (
    PORT_REGION_MAP,
    classify_region,
    add_region_to_ports,
    enrich_with_region,
    plot_regional_vessel_type,
    plot_daily_by_vessel_type,
    plot_daily_by_region,
)

# Comparison functions (cross-period)
from .comparison import (
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

__all__ = [
    # Core pipeline
    "MaritimePortAnalyzer",
    "AnalysisConfig",
    "AnalysisResults",
    "detect_port_events",
    "calculate_parking_durations",
    "load_ais_data",
    "load_port_list",
    # Regional / daily breakdown
    "PORT_REGION_MAP",
    "classify_region",
    "add_region_to_ports",
    "enrich_with_region",
    "plot_regional_vessel_type",
    "plot_daily_by_vessel_type",
    "plot_daily_by_region",
    # Comparison
    "compare_parking_distributions",
    "compare_parking_dashboard",
    "compare_port_changes",
    "compare_regional_vessel_type",
    "compare_daily_by_vessel_type",
    "compare_daily_by_region",
    "compare_single_port",
    "compare_all_ports",
    "view_ship_events",
]
