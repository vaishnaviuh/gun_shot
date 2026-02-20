"""
Gunshot Simulation Module

This package contains modules for generating simulated gunshot audio signals.
"""

from .gunshot_simulation import (
    simulate_gunshots,
    generate_gunshot_audio,
    load_sensor_positions,
    save_audio_and_trajectory,
    plot_spectrogram,
    plot_positions_2d,
    plot_positions_3d
)

__all__ = [
    'simulate_gunshots',
    'generate_gunshot_audio',
    'load_sensor_positions',
    'save_audio_and_trajectory',
    'plot_spectrogram',
    'plot_positions_2d',
    'plot_positions_3d'
]

