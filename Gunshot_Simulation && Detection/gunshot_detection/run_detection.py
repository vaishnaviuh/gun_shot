#!/usr/bin/env python3
"""
Command-line script to run gunshot detection on WAV files.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# Default output
_DEFAULT_PLOTS_DIR = _root / "plots"
_DEFAULT_PDF = str(_DEFAULT_PLOTS_DIR / "detection_report.pdf")

from gunshot_detection.gunshot_detection import (
    run_detection_on_file,
    ADAPTIVE_K,
    CUSUM_H,
)


def main():
    parser = argparse.ArgumentParser(description='Gunshot detection on WAV file')
    parser.add_argument('wav', type=str, nargs='?', default=None,
                       help='Path to WAV file (positional)')
    parser.add_argument('--wav-file', type=str, default=None,
                       help='Path to WAV file (alternative to positional)')
    parser.add_argument('--output-pdf', type=str, default=None,
                       help=f'PDF report path (default: {_DEFAULT_PDF})')
    parser.add_argument('--no-kurtosis', action='store_true',
                       help='Disable kurtosis check')
    parser.add_argument('--k', type=float, default=ADAPTIVE_K,
                       help=f'Adaptive threshold k (default: {ADAPTIVE_K})')
    parser.add_argument('--cusum-h', type=float, default=CUSUM_H,
                       help=f'CUSUM threshold (default: {CUSUM_H})')
    parser.add_argument('--no-plots', action='store_true',
                       help='Disable PDF report generation')

    args = parser.parse_args()

    wav_path = args.wav_file or args.wav
    if not wav_path:
        parser.error('Either provide WAV path as positional argument or use --wav-file')

    events = run_detection_on_file(
        wav_path,
        output_pdf=None if args.no_plots else args.output_pdf,
        plot_results=not args.no_plots,
        use_kurtosis=not args.no_kurtosis,
        adaptive_k=args.k,
        cusum_h=args.cusum_h,
    )

    return events


if __name__ == "__main__":
    main()
