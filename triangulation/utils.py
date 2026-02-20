"""
Triangulation-local utilities. No dependency on Gunshot_Simulation or gunshot_detection.
WAV loading, sensor geometry, speed of sound — used for cross-correlation → TDOA → localization.
"""
from typing import Optional, Tuple
import math
import numpy as np

try:
    import soundfile as sf
except ImportError:
    sf = None
from scipy.signal import butter, sosfiltfilt


def load_audio(
    input_file: str,
    duration: Optional[Tuple[Optional[float], Optional[float]]] = None,
    bandpass: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, int]:
    """
    Load audio from WAV file and apply optional bandpass filtering.

    Args:
        input_file: Path to WAV file
        duration: Optional (start_sec, end_sec); None means full file
        bandpass: Optional (low_hz, high_hz) for bandpass filter

    Returns:
        audio_data: Shape (channels, samples)
        sample_rate: Hz
    """
    if sf is None:
        raise ImportError("soundfile is required. Install with: pip install soundfile")
    audio_data, sample_rate = sf.read(input_file, always_2d=True)
    audio_data = audio_data.T  # (channels, samples)

    if duration is not None:
        start = 0 if duration[0] is None else int(duration[0] * sample_rate)
        end = audio_data.shape[1] if duration[1] is None else int(duration[1] * sample_rate)
        audio_data = audio_data[:, start:end]

    if bandpass is not None and bandpass[0] < bandpass[1]:
        nyq = sample_rate / 2.0
        low = max(1e-8, min(1 - 1e-8, bandpass[0] / nyq))
        high = max(low + 1e-8, min(1 - 1e-8, bandpass[1] / nyq))
        sos = butter(N=6, Wn=[low, high], btype="band", output="sos")
        audio_data = sosfiltfilt(sos, audio_data, axis=1)

    return audio_data, sample_rate


def get_sensor_positions_enu(
    baseline_m: float = 30.0,
) -> Tuple[np.ndarray, list]:
    """
    Sensor positions in ENU meters, origin at triangle centroid.
    3 vertices at 1 m, S4 at center at 2 m.

    Returns:
        positions: (4, 3) array
        sensor_ids: [1, 2, 3, 4]
    """
    r = baseline_m * math.sqrt(3) / 3
    S1 = (0.0, -r, 1.0)
    S2 = (r * math.sqrt(3) / 2, r / 2, 1.0)
    S3 = (-r * math.sqrt(3) / 2, r / 2, 1.0)
    S4 = (0.0, 0.0, 2.0)
    positions = np.array([S1, S2, S3, S4], dtype=float)
    return positions, [1, 2, 3, 4]


def calculate_speed_of_sound(
    temperature_c: float = 15.0,
    altitude_m: float = 200.0,
) -> float:
    """Speed of sound in m/s (approximate)."""
    return 331.3 + 0.606 * temperature_c - 0.006 * altitude_m / 1000.0
