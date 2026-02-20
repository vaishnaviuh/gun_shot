#!/usr/bin/env python3
"""
Unified CLI: run full TDOA localization pipeline and generate report.
Connects: positions -> simulate_or_load -> detect -> TDOA -> estimate -> plots -> PDF report.
"""
import argparse
import sys
from pathlib import Path

# Paths so pipeline can import detection/simulation
_triang_dir = Path(__file__).resolve().parent
_gunshot_root = _triang_dir.parent
_detection_root = _gunshot_root / "Gunshot_Simulation && Detection"
sys.path.insert(0, str(_gunshot_root))
sys.path.insert(0, str(_detection_root))

# Default dirs: all WAV/log under gunshot_simulation/data; report under triangulation/plots
DEFAULT_DATA_DIR = str(_detection_root / "gunshot_simulation" / "data")
DEFAULT_PLOTS_DIR = str(_triang_dir / "plots")
# Default WAV for localization (positions read from filename: 3002002, 3502902, 3902502 = 3 shots)
WAV_FILE = str(_detection_root / "gunshot_simulation" / "data" / "combined3002002_3502902_3902502.wav")


def _parse_positions(s: str):
    """Parse positions from string: 'x1 y1 z1; x2 y2 z2' or 'x1 y1 z1'."""
    out = []
    for block in s.split(";"):
        block = block.strip()
        if not block:
            continue
        parts = block.split()
        if len(parts) % 3 != 0:
            raise ValueError(f"Position block must have multiples of 3 values: {block}")
        for i in range(0, len(parts), 3):
            out.append((float(parts[i]), float(parts[i + 1]), float(parts[i + 2])))
    return out


def _load_positions_from_file(path: str):
    """Load positions from file: one line per position, 'x y z' or 'x,y,z'."""
    out = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip().replace(",", " ")
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 3:
                out.append((float(parts[0]), float(parts[1]), float(parts[2])))
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Run full TDOA localization pipeline: simulate/load -> detect -> TDOA -> estimate -> 2D/3D/waveform plots -> PDF report"
    )
    parser.add_argument(
        "--positions",
        "--position",
        type=str,
        dest="positions",
        default=None,
        help="Positions as 'x1 y1 z1 x2 y2 z2 ...' or 'x1 y1 z1; x2 y2 z2' (semicolon-separated)",
    )
    parser.add_argument(
        "--positions-file",
        type=str,
        default=None,
        help="Path to file with one 'x y z' per line",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help="Directory for WAV/trajectory files (default: simulation project gunshot_simulation/data)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_PLOTS_DIR,
        help="Output directory for plots and report.pdf (default: triangulation/plots)",
    )
    parser.add_argument(
        "--skip-simulation",
        action="store_true",
        help="Do not simulate; use existing WAVs in data-dir only",
    )
    parser.add_argument(
        "--window-ms",
        type=float,
        default=20.0,
        help="Window duration in ms for TDOA (default: 20)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=2.0,
        help="Recording duration in seconds per simulation (default: 2)",
    )
    parser.add_argument(
        "--time",
        type=float,
        nargs="+",
        default=None,
        help="Shot time(s) in seconds per position (one per position; default: 0.1 for all)",
    )
    parser.add_argument(
        "--baseline",
        type=float,
        default=30.0,
        help="Sensor array baseline in meters (triangle side length; default: 30)",
    )
    parser.add_argument(
        "--min-gap",
        type=float,
        default=0.4,
        help="Minimum gap between detected events in seconds (default: 0.4). Use ~0.4 to avoid reverberation double-count; lower for rapid fire",
    )
    parser.add_argument(
        "--detect-threshold",
        type=float,
        default=85.0,
        dest="detect_threshold_percentile",
        help="Event detection: energy percentile threshold (default: 85). Lower = more sensitive, use for real recordings without .log",
    )
    parser.add_argument(
        "--detect-prominence",
        type=float,
        default=0.15,
        dest="detect_prominence_ratio",
        help="Event detection: min peak prominence as fraction of energy range (default: 0.15). Lower = detect weaker shots",
    )
    parser.add_argument(
        "--wav-file",
        type=str,
        default=None,
        help=f"WAV file path for localization. Ground truth positions are read from the filename (e.g. combined3002002_3502902_3902502 = 3 shots). Default when no positions given: {WAV_FILE}",
    )
    args = parser.parse_args()
    # Use default WAV for localization when no positions given (no need to write position every time)
    if args.wav_file is None and not args.positions and not args.positions_file:
        args.wav_file = WAV_FILE
        print(f"Using default WAV for localization: {WAV_FILE}")

    if args.positions_file:
        positions = _load_positions_from_file(args.positions_file)
    elif args.positions:
        positions = _parse_positions(args.positions)
    elif args.wav_file:
        # With --wav-file only: detect all gunshots from WAV, no positions needed
        positions = []
        print("No --positions given; will detect all events from WAV and localize each.")
    else:
        positions = [
            (100.0, 200.0, 1.5),
            (150.0, 300.0, 2.0),
            (80.0, 250.0, 1.0),
        ]
        print("No --positions or --positions-file given; using default positions:", positions)

    if not positions and not args.wav_file:
        print("No positions to process. Use --positions, --positions-file, or --wav-file.")
        sys.exit(1)

    if args.time is not None and positions and len(args.time) != len(positions):
        print(f"ERROR: --time has {len(args.time)} value(s) but there are {len(positions)} position(s). Provide one time per position.")
        sys.exit(1)

    from triangulation.pipeline import run_pipeline

    if args.wav_file:
        print("Running localization on WAV:", args.wav_file)
    else:
        print("Running multi-shot TDOA pipeline: ref-channel detection (min gap 100 ms) -> TDOA -> multilateration per event")
    window_s = args.window_ms / 1000.0
    summary = run_pipeline(
        positions=positions,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        skip_simulation=args.skip_simulation or bool(args.wav_file),
        detection_project_root=_detection_root,
        window_duration_s=window_s,
        duration_s=args.duration,
        shot_times=args.time,
        baseline_m=args.baseline,
        min_gap_s=args.min_gap,
        wav_file=args.wav_file,
        detect_threshold_percentile=args.detect_threshold_percentile,
        detect_prominence_ratio=args.detect_prominence_ratio,
    )

    print("\n--- Done ---")
    print("Report:", summary["report_path"])
    print("Plots dir:", summary["plots_dir"])
    print("Estimated positions (TDOA only, no TOA):")
    for i, pos in enumerate(summary.get("estimated_positions", [])):
        name = summary["results"][i]["name"] if i < len(summary["results"]) else f"shot_{i+1}"
        if pos is not None:
            print(f"  {name}: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}] m")
        else:
            print(f"  {name}: (not localized)")
    for r in summary["results"]:
        print(f"  {r['name']}: error = {r.get('error_m', 'N/A'):.3f} m")
    print("  [Note: errors above are from synthetic/clean audio. Real-world errors are larger due to noise, reverberation, multipath.]")


if __name__ == "__main__":
    main()
