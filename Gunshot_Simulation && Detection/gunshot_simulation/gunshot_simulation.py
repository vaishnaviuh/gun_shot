"""
Gunshot Audio Simulation Module

Generates realistic simulated gunshot audio signals with broadband inverse
frequency and inverse time decay for testing and validation.
"""

import warnings
import sys
import importlib.util
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

# Load config from project root
_root = Path(__file__).resolve().parent.parent
_config_path = _root / "config.py"
_spec = importlib.util.spec_from_file_location("config", _config_path)
_cfg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cfg)
sys.modules["config"] = _cfg

# Add project root to path for imports
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import config
from shared_config import get_speed_of_sound
try:
    from src.visualisation.style import setup_plotting_style
except ImportError:
    def setup_plotting_style(**kwargs):
        return 'DejaVu Sans'

warnings.filterwarnings('ignore', category=UserWarning, message='.*FigureCanvasAgg is non-interactive.*')


def load_sensor_positions(baseline_m: Optional[float] = None) -> Tuple[np.ndarray, List[str]]:
    """
    Load sensor positions for simulation. Uses config.get_sensor_positions_enu when
    available so simulation and localization use the same ENU frame (origin at triangle center).
    """
    baseline = float(baseline_m) if baseline_m is not None else 150.0
    if getattr(config, "get_sensor_positions_enu", None):
        positions, ids = config.get_sensor_positions_enu(baseline)
        names = [f"Sensor_{int(i):02d}" for i in ids]
        return positions, names
    # Fallback: same formula as config
    r = baseline * (3 ** 0.5) / 3.0
    enu_sensor_offsets = [
        [0.0, -r, 1.0],
        [r * (3 ** 0.5) / 2.0, r / 2.0, 1.0],
        [-r * (3 ** 0.5) / 2.0, r / 2.0, 1.0],
        [0.0, 0.0, 2.0],
    ]
    sensor_positions = np.array(enu_sensor_offsets, dtype=float)
    sensor_names = [f"Sensor_{int(i):02d}" for i in [1, 2, 3, 4]]
    return sensor_positions, sensor_names


def generate_gunshot_audio(
    gunshot_positions: np.ndarray,
    gunshot_times: List[float],
    sensor_positions: np.ndarray,
    sensor_names: List[str],
    duration_s: float = 2.0,
    sample_rate: int = 16000,
    num_channels: int = 4,
    f_min_hz: float = 10.0,
    f_max_hz: float = 7990.0,
    df_hz: float = 10.0,
    t0_decay_s: float = 0.001,
    f_decay: float = 4000.0,
    t_decay: float = 0.5,
    speed_of_sound: float = None,  # If None, calculated from temperature/altitude
    background_noise_std: float = 1e-6,  # Open snow valley: low but non-zero ambient noise
    ref_distance: float = 100.0,
    random_seed: int = 42
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Generate simulated gunshot audio signals.
    
    Args:
        gunshot_positions: Array of shape (N, 3) with gunshot positions [x, y, z] in meters
        gunshot_times: List of gunshot times in seconds
        sensor_positions: Array of shape (M, 3) with sensor positions
        sensor_names: List of sensor names
        duration_s: Duration of recording in seconds
        sample_rate: Sample rate in Hz
        num_channels: Number of audio channels
        f_min_hz: Minimum frequency in Hz
        f_max_hz: Maximum frequency in Hz
        df_hz: Frequency step in Hz
        t0_decay_s: Small time offset to avoid 1/0 at t=0
        f_decay: Frequency decay coefficient
        t_decay: Time decay coefficient
        speed_of_sound: Speed of sound in m/s
        background_noise_std: Standard deviation of background noise
        ref_distance: Reference distance for attenuation calculation
        random_seed: Random seed for phase generation
    
    Returns:
        audio_data: Array of shape (num_channels, num_samples)
        gunshot_events: List of dictionaries with event information
    """
    # Calculate speed of sound if not provided (use same formula as processing)
    if speed_of_sound is None:
        speed_of_sound = get_speed_of_sound()
    
    num_samples = int(duration_s * sample_rate)
    t = np.arange(num_samples) / sample_rate
    dt = 1.0 / sample_rate
    audio_data = np.zeros((num_channels, num_samples))

    # Frequency bins
    freqs = np.arange(f_min_hz, f_max_hz + df_hz / 2, df_hz)

    gunshot_events = []

    print(f"Generating {duration_s:.1f}s at {sample_rate} Hz, broadband {f_min_hz}-{f_max_hz} Hz ({len(freqs)} bins)")
    
    for gunshot_idx, (gunshot_time, gunshot_position) in enumerate(zip(gunshot_times, gunshot_positions)):
        # Generate unique phases for this gunshot (same phases used across all channels)
        # Use deterministic seed based on random_seed + gunshot_idx for reproducibility
        np.random.seed(random_seed + gunshot_idx)
        phases = np.random.uniform(0, 2 * np.pi, size=len(freqs))
        
        print(f"Gunshot position: {gunshot_position}, time: {gunshot_time} s")
        toa_list = []

        for ch_idx in range(num_channels):
            sensor_pos = sensor_positions[ch_idx]
            distance = np.linalg.norm(gunshot_position - sensor_pos)
            arrival_time = gunshot_time + distance / speed_of_sound
            arrival_sample = int(arrival_time * sample_rate)

            toa_list.append((ch_idx, sensor_names[ch_idx], arrival_time, arrival_sample))
            # Use less aggressive attenuation - linear instead of quadratic for far distances
            # For distances > ref_distance, use linear falloff; for close distances, use quadratic
            if distance > ref_distance:
                # Linear attenuation for far distances (less aggressive)
                attenuation = ref_distance / distance
            else:
                # Quadratic for close distances (realistic near-field)
                attenuation = (ref_distance / distance) ** 2
            if distance <= 0:
                attenuation = 1.0

            channel_signal = np.zeros(num_samples)
            # Base amplitude multiplier to increase signal strength
            # .308 sniper, open snow‑capped valley (maximum clean setting)
            base_amplitude = 150000.0
            for i in range(num_samples):
                t_local = t[i] - arrival_time
                if t_local < 0:
                    continue
                t_decay_val = t_local + t0_decay_s
                for fi, f in enumerate(freqs):
                    amp = base_amplitude * (10**(-1.0 * (f / f_decay))) * (10**(-1.0 * (t_decay_val / t_decay))) * attenuation
                    channel_signal[i] += amp * np.sin(2.0 * np.pi * f * t_local + phases[fi])

            signal_power = np.mean(channel_signal ** 2)
            if signal_power > 0:
                # Add shot-synchronous noise at ~25 dB SNR (more realistic than 30 dB,
                # but still clearly detectable in synthetic tests).
                noise_power = signal_power / (10 ** (40.0 / 10.0))
                channel_signal += np.random.normal(0, np.sqrt(noise_power), num_samples)
            if background_noise_std > 0:
                channel_signal += np.random.normal(0, background_noise_std, num_samples)

            audio_data[ch_idx, :] += channel_signal
            if (ch_idx + 1) % 5 == 0:
                print(f"  Processed {ch_idx + 1}/{num_channels} channels for gunshot {gunshot_idx + 1}/{len(gunshot_times)}")

        # Print TOA table for this gunshot
        print(f"\nTime of Arrival (TOA) at each sensor for gunshot {gunshot_idx + 1} (t = {gunshot_time:.6f} s):")
        print(f"{'Sensor':<12}{'TOA (s)':>12}{'Sample #':>12}")
        for ch_idx, name, arrival_time, arrival_sample in toa_list:
            print(f"{name:<12}{arrival_time:12.6f}{arrival_sample:12d}")

        print(f"\nAudio shape: {audio_data.shape}, range [{np.min(audio_data):.6f}, {np.max(audio_data):.6f}]")

        gunshot_event = {
            'position': gunshot_position,
            'time': gunshot_time,
            'arrival_times': [float(arrival_time) for _, _, arrival_time, _ in toa_list],
        }
        gunshot_events.append(gunshot_event)

    return audio_data, gunshot_events


def save_audio_and_trajectory(
    audio_data: np.ndarray,
    gunshot_events: List[Dict[str, Any]],
    output_wav_file: str,
    trajectory_file: str,
    sample_rate: int
) -> None:
    """Save audio to WAV file and trajectory to log file."""
    # Ensure audio_data is a copy and proper dtype
    audio_data = np.array(audio_data, dtype=np.float32, copy=True)
    
    # Check for NaN or Inf values and replace with zeros
    if np.any(np.isnan(audio_data)) or np.any(np.isinf(audio_data)):
        print(f"   [WARN] Found NaN/Inf values in audio, replacing with zeros")
        audio_data = np.nan_to_num(audio_data, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Scale by fixed amplitude reference so amplitude up to AMPLITUDE_REF is preserved
    AMPLITUDE_REF = 300000.0
    audio_data = audio_data / AMPLITUDE_REF
    max_val = np.max(np.abs(audio_data))
    # If signal is very weak (e.g. far distance attenuation), normalize to 0.95 so it stays detectable
    if max_val > 0 and max_val < 0.01:
        audio_data = audio_data / max_val * 0.95
    # Ensure values are in valid range [-1.0, 1.0] for WAV format (clip if above 150000)
    audio_data = np.clip(audio_data, -1.0, 1.0)
    
    # Save WAV - ensure shape is (samples, channels) for soundfile
    os.makedirs(os.path.dirname(output_wav_file) or '.', exist_ok=True)
    # soundfile expects (samples, channels), so transpose if needed
    if audio_data.shape[0] < audio_data.shape[1]:
        # Already in (channels, samples) format, transpose to (samples, channels)
        sf.write(output_wav_file, audio_data.T, sample_rate, subtype='PCM_16')
    else:
        # Already in (samples, channels) format
        sf.write(output_wav_file, audio_data, sample_rate, subtype='PCM_16')
    print(f"Saved simulated audio to: {output_wav_file}")
    print(f"   Audio shape: {audio_data.shape}, range: [{np.min(audio_data):.6f}, {np.max(audio_data):.6f}]")

    # Write TOAs to file
    with open(trajectory_file, 'w') as f:
        for event in gunshot_events:
            f.write(f"{event['time']:.6f} {event['position'][0]:.2f} {event['position'][1]:.2f} "
                   f"{event['position'][2]:.2f} {event['arrival_times']}\n")


def plot_spectrogram_fn(
    audio_data: np.ndarray,
    sample_rate: int,
    num_channels: int
) -> None:
    """Plot spectrogram of generated audio."""
    nperseg = 1024
    sf_mono_font = setup_plotting_style(use_dark_theme=False)
    
    fig, axes = plt.subplots(num_channels, 1, figsize=(12, 4 * num_channels), dpi=150, sharex=True)
    if num_channels == 1:
        axes = [axes]

    for ch in range(num_channels):
        f_spec, t_spec, Sxx = spectrogram(audio_data[ch], sample_rate, nperseg=nperseg)
        Sxx_dB = 10 * np.log10(Sxx + 1e-12)
        im = axes[ch].pcolormesh(t_spec, f_spec, Sxx_dB, shading='gouraud', cmap='magma')
        axes[ch].set_ylabel('Frequency (Hz)', fontfamily=sf_mono_font, fontsize=12)
        axes[ch].set_title(f'CHANNEL {ch + 1}', fontfamily=sf_mono_font, fontsize=14)
        fig.colorbar(im, ax=axes[ch], label='dB')

    axes[-1].set_xlabel('Time (s)', fontfamily=sf_mono_font, fontsize=12)
    plt.tight_layout()
    plt.show()

# Backwards compatibility alias for external imports
plot_spectrogram = plot_spectrogram_fn


def plot_positions_2d(
    gunshot_events: List[Dict[str, Any]],
    sensor_positions: np.ndarray
) -> None:
    """Plot gunshot positions and sensors in 2D."""
    if not gunshot_events:
        print("Gunshot event data missing")
        return
    
    sf_mono_font = setup_plotting_style(use_dark_theme=False)
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    for i, event in enumerate(gunshot_events):
        pos = event['position']
        ax.scatter(pos[0], pos[1], color='red', s=30, label='Gunshots' if i == 0 else None, 
                  zorder=5, marker='+', linewidths=0.5)
        gunshot_z = pos[2]
        ax.text(
            pos[0] + 20, pos[1],
            f"{gunshot_z:.0f}m,{i+1}",
            color='orangered',
            fontfamily=sf_mono_font,
            fontsize=6,
            ha='left',
            va='bottom',
            zorder=10,
            bbox=dict(facecolor='black', alpha=0.2, edgecolor='none', boxstyle='round,pad=0.23')
        )

    ax.scatter(sensor_positions[:, 0], sensor_positions[:, 1], color='white', s=15, 
              label='Sensors', alpha=0.9, marker='x', linewidths=0.5)

    ax.set_xlabel('X (m)', fontfamily=sf_mono_font, fontsize=9)
    ax.set_ylabel('Y (m)', fontfamily=sf_mono_font, fontsize=9)
    ax.set_title('Positions', fontfamily=sf_mono_font, fontsize=9)
    ax.set_xlim(-1000, 1000)
    ax.set_ylim(-1000, 1000)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=7)
    ax.tick_params(axis='both', which='major', labelsize=7)
    plt.tight_layout()
    plt.show()


def plot_positions_3d(sensor_positions: np.ndarray) -> None:
    """Plot sensor positions in 3D."""
    sf_mono_font = setup_plotting_style(use_dark_theme=False)
    import itertools
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.05))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.05))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.05))

    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.line.set_color((0.5, 0.5, 0.5, 0.75))
        axis._axinfo["grid"]['color'] = (0.6, 0.6, 0.6, 0.5)

    for x, y, z in sensor_positions:
        ax.scatter(x, y, z, color='red', s=30, marker='x', linewidths=1.0, zorder=10)

    n_sensors = sensor_positions.shape[0]
    for i, j in itertools.combinations(range(n_sensors), 2):
        x_coords = [sensor_positions[i, 0], sensor_positions[j, 0]]
        y_coords = [sensor_positions[i, 1], sensor_positions[j, 1]]
        z_coords = [sensor_positions[i, 2], sensor_positions[j, 2]]
        ax.plot(x_coords, y_coords, z_coords, color='lime', linewidth=0.5, alpha=1.0, zorder=1)

    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(1, 2)

    ax.set_xlabel('X (m)', fontfamily=sf_mono_font, fontsize=9)
    ax.set_ylabel('Y (m)', fontfamily=sf_mono_font, fontsize=9)
    ax.set_zlabel('Z (m)', fontfamily=sf_mono_font, fontsize=9, labelpad=0)
    ax.set_title('Sensor Positions', fontfamily=sf_mono_font, fontsize=12)

    fig.subplots_adjust(right=0.88)
    plt.show()


def simulate_gunshots(
    gunshot_positions: np.ndarray,
    gunshot_times: List[float],
    output_wav_file: str,
    trajectory_file: str,
    duration_s: float = 2.0,
    sample_rate: int = 16000,
    baseline_m: Optional[float] = None,
    num_channels: int = 4,
    f_min_hz: float = 10.0,
    f_max_hz: float = 7990.0,
    df_hz: float = 10.0,
    t0_decay_s: float = 0.001,
    f_decay: float = 4000.0,
    t_decay: float = 0.5,
    temperature_c: float = 15.0,
    altitude_m: float = 200.0,
    background_noise_std: float = 1e-6,  # Open snow valley default (matches scenario parameters)
    ref_distance: float = 100.0,
    random_seed: int = 42,
    plot_spectrogram: bool = True,
    plot_2d: bool = True,
    plot_3d: bool = True
) -> Dict[str, Any]:
    """
    Main function to simulate gunshot audio.
    
    Args:
        gunshot_positions: Array of shape (N, 3) with gunshot positions [x, y, z] in meters
        gunshot_times: List of gunshot times in seconds
        output_wav_file: Path to output WAV file
        trajectory_file: Path to output trajectory log file
        duration_s: Duration of recording in seconds
        sample_rate: Sample rate in Hz
        num_channels: Number of audio channels
        f_min_hz: Minimum frequency in Hz
        f_max_hz: Maximum frequency in Hz
        df_hz: Frequency step in Hz
        t0_decay_s: Small time offset to avoid 1/0 at t=0
        f_decay: Frequency decay coefficient
        t_decay: Time decay coefficient
        temperature_c: Temperature in Celsius
        altitude_m: Altitude in meters
        background_noise_std: Standard deviation of background noise
        ref_distance: Reference distance for attenuation
        random_seed: Random seed for phase generation
        plot_spectrogram: Whether to plot spectrogram
        plot_2d: Whether to plot 2D positions
        plot_3d: Whether to plot 3D sensor positions
    
    Returns:
        Dictionary with simulation results:
            - 'audio_data': Generated audio data
            - 'gunshot_events': List of gunshot event dictionaries
            - 'sensor_positions': Sensor positions
            - 'sensor_names': Sensor names
    """
    # Load sensor positions (same baseline as localization when run from pipeline)
    sensor_positions, sensor_names = load_sensor_positions(baseline_m=baseline_m)
    print(f"Loaded {len(sensor_positions)} sensor positions")
    print(f"Sensor position range:")
    print(f"  X: {np.min(sensor_positions[:, 0]):.2f} to {np.max(sensor_positions[:, 0]):.2f} m")
    print(f"  Y: {np.min(sensor_positions[:, 1]):.2f} to {np.max(sensor_positions[:, 1]):.2f} m")
    print(f"  Z: {np.min(sensor_positions[:, 2]):.2f} to {np.max(sensor_positions[:, 2]):.2f} m")
    
    # Calculate speed of sound
    speed_of_sound = get_speed_of_sound(temperature_c, altitude_m)
    print(f"Speed of sound: {speed_of_sound:.2f} m/s")
    
    # Generate audio
    audio_data, gunshot_events = generate_gunshot_audio(
        gunshot_positions, gunshot_times, sensor_positions, sensor_names,
        duration_s, sample_rate, num_channels, f_min_hz, f_max_hz, df_hz,
        t0_decay_s, f_decay, t_decay, speed_of_sound, background_noise_std,
        ref_distance, random_seed
    )
    
    # Save files
    save_audio_and_trajectory(audio_data, gunshot_events, output_wav_file, 
                             trajectory_file, sample_rate)
    
    # Plotting
    if plot_spectrogram:
        plot_spectrogram_fn(audio_data, sample_rate, num_channels)
    
    if plot_2d:
        plot_positions_2d(gunshot_events, sensor_positions)
    
    if plot_3d:
        plot_positions_3d(sensor_positions)
    
    return {
        'audio_data': audio_data,
        'gunshot_events': gunshot_events,
        'sensor_positions': sensor_positions,
        'sensor_names': sensor_names
    }


if __name__ == "__main__":
    # Example usage
    gunshot_positions = np.array([
        [101.0, 175.0, 289.0],
        [45.0, 54.0, 121.0],
    ])
    gunshot_times = [0.1, 0.7]
    
    results = simulate_gunshots(
        gunshot_positions=gunshot_positions,
        gunshot_times=gunshot_times,
        output_wav_file="./data/2026-01-31_000000.wav",
        trajectory_file="./data/2026-01-31_000000.log",
        duration_s=2.0,
        sample_rate=16000
    )

