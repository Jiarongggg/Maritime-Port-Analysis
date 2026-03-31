"""
Configuration for Maritime Port Analysis pipeline.
All tunable parameters in one place.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AnalysisConfig:
    """All parameters that control the analysis pipeline."""

    # --- Detection parameters ---
    radius_deg: float = 1.0
    min_stay_duration_minutes: int = 30
    gap_tolerance_minutes: int = 60

    # --- Useful columns to keep from raw AIS data ---
    useful_columns: Optional[List[str]] = None  # None = auto-detect

    # --- Output ---
    output_dir: str = "output"
    output_suffix: str = ""
    save_csv: bool = True
    save_plots: bool = True
    plot_format: str = "png"
    plot_dpi: int = 300

    # --- Visualization ---
    focus_ports: Optional[List[str]] = None  # None = all ports
    top_n_ports: int = 20
    grid_cols: int = 3
    max_reassignment_examples: int = 6

    # --- Vessel type filters for segmented analysis ---
    tanker_keywords: List[str] = field(
        default_factory=lambda: ["TANKER_CRUDE", "TANKER_PRODUCT", "TANKER_CHEMICALS"]
    )
    container_keywords: List[str] = field(
        default_factory=lambda: ["CONTAINER"]
    )

    def __post_init__(self):
        if self.useful_columns is None:
            self.useful_columns = [
                "VESSEL_IMO", "TIMESTAMP", "SPEED",
                "LONGITUDE", "LATITUDE", "VESSEL_TYPE", "DESTINATION",
            ]
