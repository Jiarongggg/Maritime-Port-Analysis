#!/usr/bin/env python3
"""
Command-line interface for Maritime Port Analysis.

Usage:
    python -m maritime_port_analysis.run \
        --ais "original AIS data/202603/*.csv" \
        --ports "port_list_15.csv" \
        --output "output_202603" \
        --suffix "_202603" \
        --radius 1.0 \
        --focus SGSIN MYLPK MYTPP NLRTM BEANR INNSA CNSHA CNNBG KRPUS CNSZX USLAX USNYC BRSSZ AEJEA USHOU
"""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Maritime Port Analysis — AIS-based port event detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--ais", required=True,
                        help="AIS data source: file path or glob pattern (e.g. 'data/AIS_*.csv')")
    parser.add_argument("--ports", required=True,
                        help="Port list CSV (columns: port, lat, lon, optional: radius)")
    parser.add_argument("--output", default="output",
                        help="Output directory (default: output)")
    parser.add_argument("--suffix", default="",
                        help="Suffix for output filenames (e.g. '_202603')")
    parser.add_argument("--radius", type=float, default=1.0,
                        help="Default detection radius in degrees (default: 1.0)")
    parser.add_argument("--min-stay", type=int, default=30,
                        help="Minimum stay duration in minutes (default: 30)")
    parser.add_argument("--gap-tolerance", type=int, default=60,
                        help="Gap tolerance for flickering fix in minutes (default: 60)")
    parser.add_argument("--focus", nargs="*", default=None,
                        help="Focus port codes for deep-dive analysis (e.g. SGSIN NLRTM)")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip plot generation")
    parser.add_argument("--no-csv", action="store_true",
                        help="Skip CSV output")

    args = parser.parse_args()

    # Lazy import so --help is fast
    from maritime_port_analysis import MaritimePortAnalyzer

    analyzer = MaritimePortAnalyzer(
        ais_data=args.ais,
        port_list=args.ports,
        radius_deg=args.radius,
        min_stay_duration_minutes=args.min_stay,
        gap_tolerance_minutes=args.gap_tolerance,
        output_dir=args.output,
        output_suffix=args.suffix,
        focus_ports=args.focus,
        save_plots=not args.no_plots,
        save_csv=not args.no_csv,
    )
    results = analyzer.run()

    print(f"\nDone. Outputs in: {args.output}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
