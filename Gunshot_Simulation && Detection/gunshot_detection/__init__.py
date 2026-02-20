"""
Gunshot Detection Module

This package contains modules for multi-channel gunshot detection
using energy-based detection with CUSUM validation.
"""

from .gunshot_detection import (
    bandpass_filter,
    compute_energy,
    compute_cusum,
    detect_peaks,
    detect_gunshots,
    plot_results,
    load_audio,
    run_detection_on_file,
)

__all__ = [
    'bandpass_filter',
    'compute_energy',
    'compute_cusum',
    'detect_peaks',
    'detect_gunshots',
    'plot_results',
    'load_audio',
    'run_detection_on_file',
]
