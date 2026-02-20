"""
Multi-Channel Gunshot Detection Module

Energy-based detection with CUSUM validation.
Works for both simulated and real-world audio.
Integrates with TDOA localization pipeline (GCC-PHAT runs separately).
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import soundfile as sf
from scipy.signal import butter, filtfilt
from scipy.stats import kurtosis
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Project root (parent of gunshot_detection package)
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from src.visualisation.style import setup_plotting_style
    _FONT = setup_plotting_style(use_dark_theme=False)
except Exception:
    _FONT = 'monospace'

# Default parameters: onset-based detection with STE + derivative + CUSUM
FRAME_SIZE_MS = 20.0
HOP_SIZE_MS = 5.0
BANDPASS_LOW = 500.0
BANDPASS_HIGH = 8000.0
NOISE_WINDOW_S = 0.5
NOISE_FLOOR_MULTIPLIER = 6.0
CUSUM_K = 0.1
CUSUM_H = 1.2
KURTOSIS_MIN = 2.0
TIME_TOLERANCE_MS = 120.0
CHUNK_WINDOW_BEFORE_MS = 20.0
CHUNK_WINDOW_AFTER_MS = 30.0
MIN_CHANNELS_DETECTED = 2
MIN_GAP_MS = 200.0


def bandpass_filter(signal: np.ndarray, fs: float,
                    low_hz: float = BANDPASS_LOW,
                    high_hz: float = BANDPASS_HIGH,
                    order: int = 4) -> np.ndarray:
    """
    Apply bandpass filter (Butterworth) to signal.

    Args:
        signal: 1D array (samples)
        fs: Sampling rate (Hz)
        low_hz: Lower cutoff (Hz)
        high_hz: Upper cutoff (Hz)
        order: Filter order

    Returns:
        Filtered signal
    """
    nyq = 0.5 * fs
    low = max(low_hz / nyq, 0.001)
    high = min(high_hz / nyq, 0.999)
    if low >= high:
        return signal
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal.astype(np.float64))


def compute_energy(signal: np.ndarray, fs: float,
                  frame_size_ms: float = FRAME_SIZE_MS,
                  hop_size_ms: float = HOP_SIZE_MS) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Short-Time Energy: E[n] = sum(frame^2).
    Returns normalized energy and time axis (center of each frame).

    Args:
        signal: 1D array
        fs: Sampling rate
        frame_size_ms: Frame duration (ms)
        hop_size_ms: Hop duration (ms)

    Returns:
        energy: Short-time energy
        time_axis: Time (s) for each frame
    """
    frame_samples = int(fs * frame_size_ms / 1000.0)
    hop_samples = int(fs * hop_size_ms / 1000.0)
    n_frames = max(1, (len(signal) - frame_samples) // hop_samples + 1)

    energy = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop_samples
        end = min(start + frame_samples, len(signal))
        frame = signal[start:end]
        if len(frame) < frame_samples:
            frame = np.pad(frame, (0, frame_samples - len(frame)), mode='constant')
        energy[i] = np.sum(frame ** 2)

    # Normalize energy (avoid div by zero)
    e_max = np.max(energy)
    if e_max > 0:
        energy = energy / e_max

    time_axis = (np.arange(n_frames) * hop_samples + frame_samples // 2) / fs
    return energy, time_axis


def compute_cusum(energy: np.ndarray, k: float = CUSUM_K, h: float = CUSUM_H
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """
    CUSUM on energy:
      s_pos[i] = max(0, s_pos[i-1] + (E[i] - k))
      s_neg[i] = min(0, s_neg[i-1] + (E[i] + k))

    Args:
        energy: Short-time energy sequence
        k: Reference (drift parameter)
        h: Threshold for detection

    Returns:
        s_pos: Positive CUSUM
        s_neg: Negative CUSUM
    """
    n = len(energy)
    s_pos = np.zeros(n)
    s_neg = np.zeros(n)
    for i in range(1, n):
        s_pos[i] = max(0, s_pos[i - 1] + (energy[i] - k))
        s_neg[i] = min(0, s_neg[i - 1] + (energy[i] + k))
    return s_pos, s_neg


def detect_peaks(energy: np.ndarray, threshold: float) -> np.ndarray:
    """Detect frame indices where energy > threshold."""
    return np.where(energy > threshold)[0]


def compute_ste_derivative(energy: np.ndarray) -> np.ndarray:
    """Compute derivative of STE (forward difference, same length as energy)."""
    der = np.zeros_like(energy)
    der[:-1] = np.diff(energy)
    return der


def find_onset_index(energy: np.ndarray, derivative: np.ndarray, peak_idx: int,
                     threshold: float, min_derivative_ratio: float = 0.1) -> int:
    """
    Find first rising edge (onset) before peak. Search backward from peak for first frame
    where energy exceeds threshold and derivative indicates sharp rise.
    Returns frame index of onset (use this for event time, not peak).
    """
    if peak_idx <= 0:
        return peak_idx
    der_at_peak = derivative[peak_idx]
    if der_at_peak <= 0:
        der_at_peak = 1e-12
    rise_thresh = max(min_derivative_ratio * der_at_peak, 1e-9)
    onset_idx = peak_idx
    for i in range(peak_idx - 1, -1, -1):
        if energy[i] < threshold * 0.3:
            break
        if derivative[i] > rise_thresh:
            onset_idx = i
        else:
            break
    return onset_idx


def _compute_kurtosis_per_frame(signal: np.ndarray, fs: float,
                                frame_size_ms: float, hop_size_ms: float) -> np.ndarray:
    """Compute kurtosis for each frame."""
    frame_samples = int(fs * frame_size_ms / 1000.0)
    hop_samples = int(fs * hop_size_ms / 1000.0)
    n_frames = max(1, (len(signal) - frame_samples) // hop_samples + 1)
    kurt = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop_samples
        end = min(start + frame_samples, len(signal))
        frame = signal[start:end]
        if len(frame) >= 4:
            kurt[i] = kurtosis(frame, fisher=True)  # Excess kurtosis
        else:
            kurt[i] = 0
    return kurt


def _deduplicate_per_channel(detections_per_channel: List[List[float]],
                             min_gap_s: float = 0.2) -> List[List[float]]:
    """Merge detections within the same channel that are within min_gap_s (one per gunshot)."""
    result = []
    for times in detections_per_channel:
        if not times:
            result.append([])
            continue
        times = sorted(times)
        merged = [times[0]]
        for t in times[1:]:
            if t - merged[-1] > min_gap_s:
                merged.append(t)
        result.append(merged)
    return result


def _cluster_detections(detections_per_channel: List[List[float]],
                        time_tolerance_s: float,
                        min_channels: int) -> List[List[Tuple[int, float]]]:
    """
    Cluster detections across channels. A cluster is valid if >= min_channels
    channels have detections within time_tolerance_s.
    """
    all_times = []
    for ch, times in enumerate(detections_per_channel):
        for t in times:
            all_times.append((ch, t))
    all_times.sort(key=lambda x: x[1])

    clusters = []
    used = set()
    for ch, t in all_times:
        if (ch, t) in used:
            continue
        cluster = [(ch, t)]
        used.add((ch, t))
        for ch2, t2 in all_times:
            if (ch2, t2) in used:
                continue
            if ch2 != ch and abs(t2 - t) <= time_tolerance_s:
                cluster.append((ch2, t2))
                used.add((ch2, t2))
        if len(set(c for c, _ in cluster)) >= min_channels:
            clusters.append(cluster)

    # Merge clusters that represent the same event (overlapping in time)
    merged_clusters = []
    for cluster in sorted(clusters, key=lambda c: np.mean([t for _, t in c])):
        t_center = np.mean([t for _, t in cluster])
        combined = False
        for mc in merged_clusters:
            mc_center = np.mean([t for _, t in mc])
            if abs(t_center - mc_center) <= 0.4:
                # Merge: take union of channels, keep earliest detection per channel
                ch_times = {c: t for c, t in mc}
                for c, t in cluster:
                    if c not in ch_times or t < ch_times[c]:
                        ch_times[c] = t
                merged_clusters.remove(mc)
                merged_clusters.append([(c, t) for c, t in ch_times.items()])
                combined = True
                break
        if not combined:
            merged_clusters.append(cluster)
    return merged_clusters


def detect_gunshots(
    audio: np.ndarray,
    fs: float,
    frame_size_ms: float = FRAME_SIZE_MS,
    hop_size_ms: float = HOP_SIZE_MS,
    bandpass_low: float = BANDPASS_LOW,
    bandpass_high: float = BANDPASS_HIGH,
    noise_floor_multiplier: float = NOISE_FLOOR_MULTIPLIER,
    cusum_k: float = CUSUM_K,
    cusum_h: float = CUSUM_H,
    kurtosis_min: float = KURTOSIS_MIN,
    time_tolerance_ms: float = TIME_TOLERANCE_MS,
    chunk_window_before_ms: float = CHUNK_WINDOW_BEFORE_MS,
    chunk_window_after_ms: float = CHUNK_WINDOW_AFTER_MS,
    chunk_window_ms: Optional[float] = None,
    min_channels: int = MIN_CHANNELS_DETECTED,
    use_kurtosis: bool = False,
    plot_results: bool = True,
    output_pdf: Optional[str] = None,
    channel_label: str = "CH1",
    pdf_append: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """
    Multi-channel gunshot detection.

    Args:
        audio: (num_channels, num_samples)
        fs: Sampling rate
        frame_size_ms: Frame duration (ms)
        hop_size_ms: Hop duration (ms)
        bandpass_low, bandpass_high: Bandpass filter (Hz)
        adaptive_k: T = mean + k*std
        cusum_k, cusum_h: CUSUM parameters
        kurtosis_min: Optional impulsiveness threshold
        time_tolerance_ms: Cluster tolerance across channels (ms)
        chunk_window_ms: ±window (ms) for chunk extraction
        min_channels: Min channels for valid event
        use_kurtosis: Apply kurtosis check
        plot_results: Generate plots
        output_pdf: Path to save PDF report (creates plots/ folder if needed)
        channel_label: Label for single-channel plot
        pdf_append: If provided (matplotlib PdfPages), append detection pages to this PDF instead of creating a new file

    Returns:
        List of {"event_time", "channel_times", "chunk"}
    """
    if audio.ndim == 1:
        audio = audio.reshape(1, -1)
    n_channels, n_samples = audio.shape
    duration_s = n_samples / fs

    # Preprocessing
    filtered = np.zeros_like(audio, dtype=np.float64)
    for ch in range(n_channels):
        filtered[ch] = bandpass_filter(audio[ch], fs, bandpass_low, bandpass_high)
        mx = np.max(np.abs(filtered[ch]))
        if mx > 0:
            filtered[ch] = filtered[ch] / mx

    # Per-channel detection: STE + derivative (onset) + CUSUM, combined logic
    detections_per_channel = [[] for _ in range(n_channels)]
    energy_per_channel = []
    time_axis_per_channel = []
    derivative_per_channel = []
    threshold_per_channel = []
    s_pos_per_channel = []

    time_tolerance_s = time_tolerance_ms / 1000.0
    min_gap_s = MIN_GAP_MS / 1000.0
    if chunk_window_ms is not None:
        chunk_before_samp = int(fs * chunk_window_ms / 2000.0)
        chunk_after_samp = int(fs * chunk_window_ms / 2000.0)
    else:
        chunk_before_samp = int(fs * chunk_window_before_ms / 1000.0)
        chunk_after_samp = int(fs * chunk_window_after_ms / 1000.0)

    for ch in range(n_channels):
        energy, time_axis = compute_energy(filtered[ch], fs, frame_size_ms, hop_size_ms)
        energy_per_channel.append(energy)
        time_axis_per_channel.append(time_axis)
        derivative = compute_ste_derivative(energy)
        derivative_per_channel.append(derivative)

        # Adaptive threshold: noise from first 0.5 sec, threshold = noise_floor * 4
        noise_frames = time_axis < NOISE_WINDOW_S
        noise_floor = np.mean(energy[noise_frames]) if np.any(noise_frames) else np.mean(energy)
        if noise_floor < 1e-12:
            noise_floor = 1e-6
        threshold = noise_floor * noise_floor_multiplier
        threshold_per_channel.append(threshold)

        k_ref = max(cusum_k, np.mean(energy) + 0.05)
        s_pos, _ = compute_cusum(energy, k_ref, cusum_h)
        s_pos_per_channel.append(s_pos)

        # Candidates: frames where STE > threshold
        above_thresh = np.where(energy > threshold)[0]
        kurt = None
        if use_kurtosis:
            kurt = _compute_kurtosis_per_frame(filtered[ch], fs, frame_size_ms, hop_size_ms)

        for idx in above_thresh:
            if s_pos[idx] <= cusum_h:
                continue
            if use_kurtosis and kurt is not None and idx < len(kurt) and kurt[idx] < kurtosis_min:
                continue
            if derivative[idx] <= 0:
                continue
            onset_idx = find_onset_index(energy, derivative, idx, threshold)
            t_event = time_axis[onset_idx]
            detections_per_channel[ch].append(t_event)

    # Deduplicate per channel (one detection per gunshot per channel)
    detections_per_channel = _deduplicate_per_channel(detections_per_channel)

    # Cluster across channels
    clusters = _cluster_detections(detections_per_channel, time_tolerance_s, min_channels)

    # Convert clusters to events (use earliest arrival for TDOA chunk - ensures all sensors in window)
    merged = []
    for cluster in clusters:
        times = [t for _, t in cluster]
        t_center = np.mean(times)
        t_earliest = min(times)
        merged.append({
            "event_time": t_center,
            "event_time_earliest": t_earliest,
            "channel_times": [None] * n_channels
        })
        for ch, t in cluster:
            merged[-1]["channel_times"][ch] = t

    # Chunk extraction: 20 ms before, 30 ms after event (onset)
    events = []
    for ev in merged:
        t_ev = ev["event_time_earliest"]
        center_sample = int(t_ev * fs)
        start_samp = max(0, center_sample - chunk_before_samp)
        end_samp = min(n_samples, center_sample + chunk_after_samp)
        chunk = filtered[:, start_samp:end_samp].copy()
        target_len = chunk_before_samp + chunk_after_samp
        if chunk.shape[1] < target_len:
            pad = np.zeros((n_channels, target_len - chunk.shape[1]))
            chunk = np.hstack([chunk, pad])
        events.append({
            "event_time": t_ev,
            "event_time_earliest": ev["event_time_earliest"],
            "channel_times": ev["channel_times"],
            "chunk": chunk
        })

    # Plotting
    if plot_results or output_pdf or pdf_append:
        _generate_report(
            filtered, fs, energy_per_channel, time_axis_per_channel,
            derivative_per_channel, threshold_per_channel, s_pos_per_channel, cusum_h,
            detections_per_channel, [e["event_time_earliest"] for e in events],
            output_pdf, n_channels, channel_label, pdf_append=pdf_append
        )

    return events


def _generate_report(
    filtered: np.ndarray,
    fs: float,
    energy_per_channel: List[np.ndarray],
    time_axis_per_channel: List[np.ndarray],
    derivative_per_channel: List[np.ndarray],
    threshold_per_channel: List[float],
    s_pos_per_channel: List[np.ndarray],
    cusum_h: float,
    candidate_times_per_channel: List[List[float]],
    final_times: List[float],
    output_pdf: Optional[str],
    n_channels: int,
    channel_label: str,
    pdf_append: Optional[Any] = None,
) -> None:
    """Create PDF report with Waveform, STE, derivative, CUSUM subplots (one page per channel).
    If pdf_append is a PdfPages object, write pages to it instead of creating a new file."""
    if pdf_append is not None:
        pdf = pdf_append
        save_to_file = False
    else:
        plots_dir = _ROOT / "plots"
        plots_dir.mkdir(exist_ok=True)
        pdf_path = output_pdf or str(plots_dir / "detection_report.pdf")
        pdf = PdfPages(pdf_path)
        save_to_file = True

    try:
        if pdf_append is not None:
            # Section title for combined report
            fig_title = plt.figure(figsize=(8, 2))
            fig_title.suptitle("Part 1: Gunshot Detection (energy + CUSUM, per-event chunk extraction)", fontsize=14)
            plt.tight_layout()
            pdf.savefig(fig_title, bbox_inches='tight', dpi=150)
            plt.close(fig_title)
        for ch in range(n_channels):
            fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
            t_sig = np.arange(filtered.shape[1]) / fs
            der_t = time_axis_per_channel[ch][:-1] if len(derivative_per_channel[ch]) < len(time_axis_per_channel[ch]) else time_axis_per_channel[ch]
            if len(derivative_per_channel[ch]) != len(der_t):
                der_t = time_axis_per_channel[ch][:len(derivative_per_channel[ch])]

            # Plot 1: Waveform
            ax1 = axes[0]
            ax1.plot(t_sig, filtered[ch], 'b-', linewidth=0.5, label='Signal')
            ax1.set_ylabel('Amplitude', fontfamily=_FONT, fontsize=10)
            ax1.set_title(f'{channel_label if n_channels == 1 else f"Channel {ch+1}"} - Waveform',
                         fontfamily=_FONT, fontsize=12)
            ax1.legend(loc='upper right', fontsize=8)
            ax1.grid(True, alpha=0.3)

            # Plot 2: Short-Time Energy
            ax2 = axes[1]
            ax2.plot(time_axis_per_channel[ch], energy_per_channel[ch], 'b-', linewidth=0.6,
                    label='STE')
            ax2.axhline(threshold_per_channel[ch], color='orange', linestyle='--', linewidth=1,
                       label=f'Threshold')
            cand_t = candidate_times_per_channel[ch]
            if len(cand_t) > 0:
                e_idx = [np.argmin(np.abs(time_axis_per_channel[ch] - t)) for t in cand_t]
                e_vals = [energy_per_channel[ch][min(i, len(energy_per_channel[ch])-1)] for i in e_idx]
                ax2.scatter(cand_t, e_vals, c='orange', s=20, marker='o', label='Candidates')
            for ft in final_times:
                ax2.axvline(ft, color='lime', linestyle='-', linewidth=0.8, alpha=0.7)
            ax2.set_ylabel('Energy', fontfamily=_FONT, fontsize=10)
            ax2.set_title('Short-Time Energy', fontfamily=_FONT, fontsize=12)
            ax2.legend(loc='upper right', fontsize=8)
            ax2.grid(True, alpha=0.3)

            # Plot 3: STE derivative (onset detection)
            ax3 = axes[2]
            der = derivative_per_channel[ch]
            t_der = time_axis_per_channel[ch][:len(der)]
            ax3.plot(t_der, der, 'g-', linewidth=0.6, label='d(STE)/dt')
            ax3.axhline(0, color='gray', linestyle=':', linewidth=0.8)
            for ft in final_times:
                ax3.axvline(ft, color='lime', linestyle='-', linewidth=0.8, alpha=0.7)
            ax3.set_ylabel('Derivative', fontfamily=_FONT, fontsize=10)
            ax3.set_title('STE Derivative (onset)', fontfamily=_FONT, fontsize=12)
            ax3.legend(loc='upper right', fontsize=8)
            ax3.grid(True, alpha=0.3)

            # Plot 4: CUSUM
            ax4 = axes[3]
            ax4.plot(time_axis_per_channel[ch], s_pos_per_channel[ch], 'b-', linewidth=0.6,
                    label='CUSUM+')
            ax4.axhline(cusum_h, color='red', linestyle='--', linewidth=1, label=f'h={cusum_h}')
            exceed_idx = np.where(s_pos_per_channel[ch] > cusum_h)[0]
            if len(exceed_idx) > 0:
                ex_t = [time_axis_per_channel[ch][i] for i in exceed_idx]
                ex_v = [s_pos_per_channel[ch][i] for i in exceed_idx]
                ax4.scatter(ex_t, ex_v, c='red', s=15, marker='x', label='Exceeds h')
            ax4.set_xlabel('Time (s)', fontfamily=_FONT, fontsize=10)
            ax4.set_ylabel('CUSUM+', fontfamily=_FONT, fontsize=10)
            ax4.set_title('CUSUM Curve', fontfamily=_FONT, fontsize=12)
            ax4.legend(loc='upper right', fontsize=8)
            ax4.grid(True, alpha=0.3)

            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight', dpi=150)
            plt.close()

    finally:
        if save_to_file:
            pdf.close()
            print(f"Detection report saved to: {pdf_path}")


def plot_results(
    audio: np.ndarray,
    fs: float,
    events: List[Dict[str, Any]],
    energy_per_channel: Optional[List[np.ndarray]] = None,
    time_axis_per_channel: Optional[List[np.ndarray]] = None,
    threshold_per_channel: Optional[List[float]] = None,
    s_pos_per_channel: Optional[List[np.ndarray]] = None,
    output_path: Optional[str] = None
) -> None:
    """
    Plot detection results (waveform, energy, CUSUM).
    If energy/time/threshold/s_pos not provided, they are computed.
    """
    if audio.ndim == 1:
        audio = audio.reshape(1, -1)
    n_channels = audio.shape[0]

    if energy_per_channel is None:
        filtered = np.zeros_like(audio, dtype=np.float64)
        for ch in range(n_channels):
            filtered[ch] = bandpass_filter(audio[ch], fs)
            mx = np.max(np.abs(filtered[ch]))
            if mx > 0:
                filtered[ch] /= mx
        energy_per_channel = []
        time_axis_per_channel = []
        derivative_per_channel = []
        threshold_per_channel = []
        s_pos_per_channel = []
        for ch in range(n_channels):
            e, t = compute_energy(filtered[ch], fs)
            energy_per_channel.append(e)
            time_axis_per_channel.append(t)
            derivative_per_channel.append(compute_ste_derivative(e))
            noise_f = np.mean(e[t < NOISE_WINDOW_S]) if np.any(t < NOISE_WINDOW_S) else np.mean(e)
            threshold_per_channel.append(noise_f * NOISE_FLOOR_MULTIPLIER)
            sp, _ = compute_cusum(e)
            s_pos_per_channel.append(sp)
    else:
        filtered = np.zeros_like(audio, dtype=np.float64)
        for ch in range(n_channels):
            filtered[ch] = bandpass_filter(audio[ch], fs)
            mx = np.max(np.abs(filtered[ch]))
            if mx > 0:
                filtered[ch] /= mx
        derivative_per_channel = [compute_ste_derivative(e) for e in energy_per_channel]

    detections_per_channel = [[] for _ in range(n_channels)]
    for ev in events:
        for ch, ct in enumerate(ev.get("channel_times", [])):
            if ct is not None:
                detections_per_channel[ch].append(ct)

    _generate_report(
        filtered, fs, energy_per_channel, time_axis_per_channel,
        derivative_per_channel, threshold_per_channel, s_pos_per_channel, CUSUM_H,
        detections_per_channel, [e.get("event_time_earliest", e["event_time"]) for e in events],
        output_path or str(_ROOT / "plots" / "detection_report.pdf"),
        n_channels, "CH1"
    )


def load_audio(path: str) -> Tuple[np.ndarray, int]:
    """Load WAV file, return (audio, fs). Audio shape: (channels, samples)."""
    data, fs = sf.read(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    elif data.ndim == 2 and data.shape[0] > data.shape[1]:
        # soundfile returns (samples, channels), convert to (channels, samples)
        data = data.T
    return data.astype(np.float64), fs


def run_detection_on_file(
    wav_path: str,
    output_pdf: Optional[str] = None,
    plot_results: bool = True,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Run detection on a WAV file.
    """
    audio, fs = load_audio(wav_path)
    plots_dir = _ROOT / "plots"
    plots_dir.mkdir(exist_ok=True)
    pdf_path = output_pdf or str(plots_dir / "detection_report.pdf")
    events = detect_gunshots(audio, fs, plot_results=plot_results, output_pdf=pdf_path, **kwargs)
    print("\n--- Detected event times (s) ---")
    for i, ev in enumerate(events):
        print(f"  Event {i+1}: {ev['event_time']:.4f}")
    return events
