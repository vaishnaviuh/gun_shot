#!/usr/bin/env python3
"""
Command-line script to generate simulated gunshot audio.
"""

import argparse
import sys
from pathlib import Path
import numpy as np

# Add project root to path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# Default output under gunshot_simulation/data
_DEFAULT_OUTPUT_DIR = _root / "gunshot_simulation" / "data"
_DEFAULT_WAV = str(_DEFAULT_OUTPUT_DIR / "simulated_gunshot.wav")
_DEFAULT_LOG = str(_DEFAULT_OUTPUT_DIR / "simulated_gunshot.log")

from gunshot_simulation.gunshot_simulation import simulate_gunshots


def main():
    parser = argparse.ArgumentParser(description='Generate simulated gunshot audio')
    parser.add_argument('--output-wav', type=str, default=_DEFAULT_WAV,
                       help='Path to output WAV file')
    parser.add_argument('--trajectory', type=str, default=_DEFAULT_LOG,
                       help='Path to output trajectory log file')
    parser.add_argument(
        '--position',
        type=float,
        nargs='+',
        required=True,
        help=(
            'Gunshot position(s) in meters. '
            'Provide values as X Y Z for one shot, or X1 Y1 Z1 X2 Y2 Z2 ... for multiple shots. '
            'Total number of values must be a multiple of 3.'
        ),
    )
    parser.add_argument(
        '--time',
        type=float,
        nargs='+',
        help=(
            'Time(s) of gunshot(s) in seconds from start. '
            'If omitted, all shots default to 0.1 s. '
            'If provided, the number of times must match the number of gunshots.'
        ),
    )
    parser.add_argument('--duration', type=float, default=2.0,
                       help='Duration of recording in seconds')
    parser.add_argument('--sample-rate', type=int, default=16000,
                       help='Sample rate in Hz')
    parser.add_argument('--no-plots', action='store_true',
                       help='Disable plotting')
    parser.add_argument('--name-by-position', action='store_true',
                       help='Name output WAV/log by gunshot position (e.g. 101_175_289.wav)')
    
    args = parser.parse_args()

    # Flatten positions and validate
    pos_values = args.position
    if len(pos_values) % 3 != 0:
        print(f"[ERROR] --position expects values in groups of 3 (X Y Z). Got {len(pos_values)} values.")
        sys.exit(1)

    num_gunshots = len(pos_values) // 3
    gunshot_positions = np.array(pos_values, dtype=float).reshape(num_gunshots, 3)

    # Times: optional; if not provided, default all to 0.1 s
    if args.time is None:
        gunshot_times = [0.1] * num_gunshots
    else:
        if len(args.time) != num_gunshots:
            print(f"[ERROR] Number of --time values ({len(args.time)}) must match number of gunshots ({num_gunshots}).")
            print("       Each gunshot position (X Y Z) must have a corresponding time.")
            sys.exit(1)
        gunshot_times = [float(t) for t in args.time]

    if args.name_by_position:
        pos_str = "_".join(f"{int(p)}" for p in gunshot_positions[0])
        output_dir = _root / "gunshot_simulation" / "data"
        output_dir.mkdir(parents=True, exist_ok=True)
        args.output_wav = str(output_dir / f"{pos_str}.wav")
        args.trajectory = str(output_dir / f"{pos_str}.log")
    
    results = simulate_gunshots(
        gunshot_positions=gunshot_positions,
        gunshot_times=gunshot_times,
        output_wav_file=args.output_wav,
        trajectory_file=args.trajectory,
        duration_s=args.duration,
        sample_rate=args.sample_rate,
        plot_spectrogram=not args.no_plots,
        plot_2d=not args.no_plots,
        plot_3d=not args.no_plots
    )
    
    return results


if __name__ == "__main__":
    main()

