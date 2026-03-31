"""
Main pipeline orchestrator — the single entry point for the analysis.

Usage:
    from maritime_port_analysis import MaritimePortAnalyzer

    analyzer = MaritimePortAnalyzer(
        ais_data="data/AIS_202603*.csv",       # glob, path, list, or DataFrame
        port_list="data/port_list.csv",         # path or DataFrame
    )
    results = analyzer.run()
"""
import os
from dataclasses import asdict
from typing import List, Optional, Union

import pandas as pd

from .config import AnalysisConfig
from .utils import load_ais_data, load_port_list
from .detection import detect_port_events, calculate_parking_durations
from . import visualization as viz


class AnalysisResults:
    """Container for all analysis outputs."""

    def __init__(self):
        self.ais_data: pd.DataFrame = pd.DataFrame()
        self.port_list: pd.DataFrame = pd.DataFrame()
        self.arrivals: pd.DataFrame = pd.DataFrame()
        self.departures: pd.DataFrame = pd.DataFrame()
        self.parking: pd.DataFrame = pd.DataFrame()
        self.reassignments: pd.DataFrame = pd.DataFrame()
        self.detection_summary: dict = {}
        self.parking_stats: dict = {}
        self.figures: dict = {}

    def __repr__(self):
        return (
            f"AnalysisResults(\n"
            f"  arrivals={len(self.arrivals)} events,\n"
            f"  departures={len(self.departures)} events,\n"
            f"  parking={len(self.parking)} events,\n"
            f"  reassignments={len(self.reassignments)} cases,\n"
            f"  figures={list(self.figures.keys())}\n"
            f")"
        )


class MaritimePortAnalyzer:
    """
    End-to-end maritime port analysis pipeline.

    Parameters
    ----------
    ais_data : str | list[str] | pd.DataFrame
        AIS tracking data source (glob pattern, file path(s), or DataFrame).
    port_list : str | pd.DataFrame
        Port list source (CSV path or DataFrame).
    extra_ports : list[dict] | None
        Additional ports to append to the port list.
    config : AnalysisConfig | None
        Configuration object. None uses defaults.
    **config_overrides
        Any AnalysisConfig field can be overridden as a keyword argument.
        e.g. MaritimePortAnalyzer(..., radius_deg=0.8, output_suffix='_202603')
    """

    def __init__(
        self,
        ais_data: Union[str, List[str], pd.DataFrame],
        port_list: Union[str, pd.DataFrame],
        extra_ports: Optional[List[dict]] = None,
        config: Optional[AnalysisConfig] = None,
        **config_overrides,
    ):
        # Build config
        if config is None:
            config = AnalysisConfig(**config_overrides)
        else:
            # Apply overrides to a copy
            d = asdict(config)
            d.update(config_overrides)
            config = AnalysisConfig(**d)
        self.config = config

        # Load data
        print("=" * 60)
        print("  Loading data...")
        print("=" * 60)
        self._raw_ais = load_ais_data(ais_data, config.useful_columns)
        self._port_list = load_port_list(port_list, extra_ports)
        self.results = AnalysisResults()
        self.results.ais_data = self._raw_ais
        self.results.port_list = self._port_list

    # ------------------------------------------------------------------
    # Step-by-step API
    # ------------------------------------------------------------------

    def detect(self) -> "MaritimePortAnalyzer":
        """Run port event detection (arrivals, departures, stays)."""
        cfg = self.config
        out = detect_port_events(
            tracking_data=self._raw_ais,
            port_list_df=self._port_list,
            radius_deg=cfg.radius_deg,
            min_stay_duration_minutes=cfg.min_stay_duration_minutes,
            gap_tolerance_minutes=cfg.gap_tolerance_minutes,
            output_dir=cfg.output_dir if cfg.save_csv else None,
            output_suffix=cfg.output_suffix,
        )
        self.results.arrivals = out["arrivals"]
        self.results.departures = out["departures"]
        self.results.reassignments = out["reassignments"]
        self.results.detection_summary = out["summary"]
        return self

    def compute_parking(self) -> "MaritimePortAnalyzer":
        """Calculate parking durations from detection results."""
        r = self.results
        if r.arrivals.empty:
            print("  [skip] No arrivals — run detect() first.")
            return self
        r.parking = calculate_parking_durations(r.arrivals, r.departures)
        r.parking_stats = viz.analyze_parking_stats(
            r.parking,
            output_dir=self.config.output_dir if self.config.save_csv else None,
            output_suffix=self.config.output_suffix,
        )
        return self

    def visualize(self) -> "MaritimePortAnalyzer":
        """Generate all standard visualizations."""
        cfg = self.config
        r = self.results
        save = cfg.output_dir if cfg.save_plots else None
        suffix = cfg.output_suffix

        if r.arrivals.empty:
            print("  [skip] No data to visualize.")
            return self

        print("\n" + "=" * 60)
        print("  Generating visualizations...")
        print("=" * 60)

        # Daily arrivals
        stats, fig = viz.plot_daily_arrivals(r.arrivals, save_dir=save)
        r.figures["daily_arrivals"] = fig

        # Arrivals grid
        fig = viz.plot_daily_arrivals_grid(r.arrivals, cols=cfg.grid_cols,
                                            title=f"Daily Arrivals {suffix}", save_dir=save)
        r.figures["arrivals_grid"] = fig

        # Arrival ranking
        _, fig = viz.plot_arrival_ranking(r.arrivals, title_suffix=suffix, save_dir=save)
        r.figures["arrival_ranking"] = fig

        # Departures grid
        fig = viz.plot_daily_departures_grid(r.departures, cols=cfg.grid_cols,
                                              title=f"Daily Departures {suffix}", save_dir=save)
        r.figures["departures_grid"] = fig

        # Arrival vs departure
        _, fig1, fig2 = viz.plot_arrival_departure_comparison(
            r.arrivals, r.departures, title_suffix=suffix, save_dir=save)
        r.figures["arr_vs_dep_grid"] = fig1
        r.figures["arr_vs_dep_totals"] = fig2

        # Regional arrivals/departures by vessel type
        regional_figs = viz.plot_regional_vessel_type(
            r.arrivals, r.departures, r.port_list,
            title_suffix=suffix, save_dir=save)
        r.figures["regional_vessel_type"] = regional_figs

        # Daily arrivals/departures by vessel type
        vtype_daily_figs = viz.plot_daily_by_vessel_type(
            r.arrivals, r.departures,
            title_suffix=suffix, save_dir=save)
        r.figures["daily_by_vessel_type"] = vtype_daily_figs

        # Daily arrivals/departures by region
        region_daily_figs = viz.plot_daily_by_region(
            r.arrivals, r.departures, r.port_list,
            title_suffix=suffix, save_dir=save)
        r.figures["daily_by_region"] = region_daily_figs

        # Parking visualizations (if available)
        if not r.parking.empty:
            fig = viz.plot_parking_duration(r.parking, save_dir=save)
            r.figures["parking_duration"] = fig

            _, fig = viz.plot_parking_boxplot_by_port(r.parking, title_suffix=suffix, save_dir=save)
            r.figures["parking_boxplot"] = fig

            # Day-over-day heatmap (parking events per port)
            fig = viz.plot_day_over_day(r.parking, top_n=cfg.top_n_ports, save_dir=save)
            r.figures["day_over_day"] = fig

        # Focus port deep-dives
        ports_to_analyze = cfg.focus_ports or []
        for port_code in ports_to_analyze:
            out = viz.analyze_single_port(
                r.arrivals, r.departures, port_code=port_code,
                parking_df=r.parking if not r.parking.empty else None,
                save_dir=save,
            )
            if out:
                r.figures[f"port_{port_code}"] = out.get("figure")

        return self

    def create_map(self, save_path: Optional[str] = None) -> "MaritimePortAnalyzer":
        """Generate interactive Folium map."""
        r = self.results
        if r.parking.empty:
            print("  [skip] No parking data — run compute_parking() first.")
            return self
        path = save_path or os.path.join(self.config.output_dir, "interactive_map.html")
        m = viz.create_interactive_map(
            r.arrivals, r.departures, r.parking, r.port_list,
            focus_ports=self.config.focus_ports, save_path=path,
        )
        r.figures["interactive_map"] = m
        return self

    # ------------------------------------------------------------------
    # One-shot API
    # ------------------------------------------------------------------

    def run(self) -> AnalysisResults:
        """
        Run the full pipeline: detect → parking → visualize → map.
        Returns all results in a single object.
        """
        print("\n" + "=" * 60)
        print("  MARITIME PORT ANALYSIS — FULL PIPELINE")
        print("=" * 60)
        self.detect()
        self.compute_parking()
        self.visualize()
        self.create_map()
        print("\n" + "=" * 60)
        print("  PIPELINE COMPLETE")
        print(f"  {self.results}")
        print("=" * 60)
        return self.results
