#!/usr/bin/env python3
"""
Run TDOA-based localization (no TOA).
Uses detection from Gunshot_Simulation && Detection to get event time, then
extract_window -> gcc_phat -> compute_tdoa -> estimate_position.
"""
import argparse
import sys
from pathlib import Path
import numpy as np

# Paths: triangulation is sibling of "Gunshot_Simulation && Detection"
_triang_dir = Path(__file__).resolve().parent
_gunshot_root = _triang_dir.parent
_detection_root = _gunshot_root / "Gunshot_Simulation && Detection"
sys.path.insert(0, str(_gunshot_root))
sys.path.insert(0, str(_detection_root))

from triangulation.tdoa import extract_window, compute_tdoa, estimate_position


# Default WAV path (simulation output in detection project)
WAV_FILE = str(_detection_root / "gunshot_simulation" / "data" / "simulated_gunshot.wav")


def _first_event_time(toa_global_per_channel):
    """Get first event time as median of first TOA per channel (for stability)."""
    first_toas = []
    for ch_toas in toa_global_per_channel:
        if ch_toas:
            first_toas.append(ch_toas[0])
    if not first_toas:
        return None
    return float(np.median(first_toas))


def main():
    parser = argparse.ArgumentParser(description="TDOA-based gunshot localization (no TOA)")
    parser.add_argument(
        "--wav-file",
        type=str,
        default=WAV_FILE,
        help=f"WAV file path (default: {WAV_FILE})",
    )
    parser.add_argument(
        "--window-ms",
        type=float,
        default=50.0,
        help="Window duration in ms around event (default: 50)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable detection plots (always off when called from here)",
    )
    args = parser.parse_args()

    # Import detection after path is set (so config loads from detection project)
    from gunshot_detection.gunshot_detection import (
        detect_gunshots,
        load_sensor_positions,
        calculate_speed_of_sound,
    )

    # 1) Run detection (no TOA used for localization; only event time)
    print("Running detection to get gunshot event time...")
    results = detect_gunshots(
        input_wav_file=args.wav_file,
        trajectory_file=None,
        duration=None,
        bandpass=(1000.0, 24000.0),
        plot_cumulative=False,
        plot_spectrogram=False,
        plot_errors=False,
    )
    toa_global_per_channel = results["toa_global_per_channel"]
    audio_data = results["audio_data"]
    sample_rate = results["sample_rate"]

    event_time = _first_event_time(toa_global_per_channel)
    if event_time is None:
        print("No gunshot event detected. Cannot run TDOA localization.")
        return

    print(f"First event time (s): {event_time:.6f}")

    # 2) Sensor positions (ENU) and speed of sound
    sensor_positions, sensor_names, _ = load_sensor_positions()
    speed_of_sound = calculate_speed_of_sound()
    print(f"Speed of sound: {speed_of_sound:.2f} m/s")
    print(f"Sensors: {sensor_names}")

    # 3) Extract ~50 ms window around event
    window_duration_s = args.window_ms / 1000.0
    windows = extract_window(
        audio_data,
        sample_rate,
        event_time,
        window_duration_s=window_duration_s,
    )
    print(f"Extracted window: {window_duration_s*1000:.1f} ms, shape {windows.shape}")

    # 4) TDOA (mic 1 as reference)
    tdoa = compute_tdoa(windows, sample_rate, ref_channel=0)

    # Debug: TDOA values
    print("\n--- TDOA (s), ref = mic 1 ---")
    for i in range(len(tdoa)):
        print(f"  Mic {i+1} (ref={i==0}): Δt = {tdoa[i]:.6f} s")

    # 5) Multilateration
    p_est = estimate_position(
        sensor_positions,
        tdoa,
        speed_of_sound,
        ref_index=0,
    )

    # Debug: estimated position
    print("\n--- Estimated position (ENU, m) ---")
    print(f"  x (east):  {p_est[0]:.3f}")
    print(f"  y (north): {p_est[1]:.3f}")
    print(f"  z (up):    {p_est[2]:.3f}")
    print(f"  [x, y, z] = {p_est.tolist()}")

    return {
        "event_time_s": event_time,
        "tdoa_s": tdoa,
        "position_enu_m": p_est,
        "sensor_positions": sensor_positions,
        "speed_of_sound": speed_of_sound,
    }


if __name__ == "__main__":
    main()
