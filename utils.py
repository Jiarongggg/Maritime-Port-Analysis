"""
Data loading and preprocessing utilities.
Handles flexible input: file paths, glob patterns, lists, or DataFrames.
"""
import glob
import os
from typing import List, Optional, Union

import pandas as pd


def load_ais_data(
    source: Union[str, List[str], pd.DataFrame],
    useful_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load AIS tracking data from flexible sources.

    Parameters
    ----------
    source : str | list[str] | pd.DataFrame
        - pd.DataFrame : used directly
        - str ending with .csv / .parquet : single file
        - str with wildcards (* ?) : glob pattern  (e.g. 'data/AIS_2603*.csv')
        - list[str] : list of file paths
    useful_columns : list[str] | None
        Columns to keep. None keeps everything.

    Returns
    -------
    pd.DataFrame
    """
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    elif isinstance(source, list):
        df = pd.concat([_read_one(f) for f in source], ignore_index=True)
    elif isinstance(source, str):
        if any(c in source for c in ["*", "?"]):
            files = sorted(glob.glob(source))
            if not files:
                raise FileNotFoundError(f"No files matched pattern: {source}")
            df = pd.concat([_read_one(f) for f in files], ignore_index=True)
        else:
            df = _read_one(source)
    else:
        raise TypeError(f"Unsupported source type: {type(source)}")

    # Column name normalisation (handle common variants)
    rename_map = {}
    cols_upper = {c.upper(): c for c in df.columns}
    alias = {
        "SHIP_ID": "VESSEL_IMO",
        "LAT": "LATITUDE",
        "LON": "LONGITUDE",
        "TIMESTAMP_UTC": "TIMESTAMP",
    }
    for old, new in alias.items():
        if old in cols_upper and new not in cols_upper:
            rename_map[cols_upper[old]] = new
    if rename_map:
        df = df.rename(columns=rename_map)

    # Select useful columns
    if useful_columns is not None:
        available = [c for c in useful_columns if c in df.columns]
        missing = [c for c in useful_columns if c not in df.columns]
        if missing:
            print(f"  [warn] Columns not found (skipped): {missing}")
        df = df[available]

    print(f"  Loaded AIS data: {len(df):,} records, {df.columns.tolist()}")
    return df


def load_port_list(
    source: Union[str, pd.DataFrame],
    extra_ports: Optional[List[dict]] = None,
) -> pd.DataFrame:
    """
    Load port list from a CSV or DataFrame.

    Parameters
    ----------
    source : str | pd.DataFrame
        CSV path or DataFrame with at least columns: port, lat, lon.
        Optionally includes 'radius' for per-port detection radius.
    extra_ports : list[dict] | None
        Additional ports to append, e.g.
        [{'port': 'INNSA', 'lat': 18.952, 'lon': 72.948}]

    Returns
    -------
    pd.DataFrame
    """
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    elif isinstance(source, str):
        df = _read_one(source)
    else:
        raise TypeError(f"Unsupported source type: {type(source)}")

    required = {"port", "lat", "lon"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"Port list must contain columns {required}. Got: {df.columns.tolist()}")

    if extra_ports:
        extras = pd.DataFrame(extra_ports)
        df = pd.concat([df, extras], ignore_index=True)
        df = df.drop_duplicates(subset="port", keep="last").reset_index(drop=True)

    print(f"  Port list: {len(df)} ports loaded")
    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_one(path: str) -> pd.DataFrame:
    """Read a single file (csv or parquet)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        return pd.read_parquet(path)
    elif ext in (".csv", ".tsv", ".txt"):
        return pd.read_csv(path)
    else:
        # Try CSV as fallback
        return pd.read_csv(path)
