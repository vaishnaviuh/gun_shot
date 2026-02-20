"""
Unified TDOA localization pipeline (no TOA in localization).
- simulate_or_load, compute_tdoa, estimate_position, plot_2d, plot_3d, plot_waveform, generate_report.
"""
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from .tdoa import (
    extract_window,
    compute_tdoa as tdoa_compute_tdoa,
    compute_tdoa_robust as tdoa_compute_tdoa_robust,
    apply_early_energy_weight,
    build_tdoa_matrix,
    estimate_position_with_gdop,
    validate_tdoa_ms,
    validate_tdoa_sign_consistency,
)


# ---------------------------------------------------------------------------
# Position naming and trajectory parsing
# ---------------------------------------------------------------------------

def _position_name(x: float, y: float, z: float) -> str:
    """Name for a position (used in WAV/log filenames)."""
    return f"pos_{x:.1f}_{y:.1f}_{z:.1f}".replace(".", "_")


def _positions_to_stem(positions: List[Tuple[float, float, float]]) -> str:
    """Build filename stem from positions, e.g. (300,200,2),(350,290,2) -> combined3002002_3502902."""
    parts = [f"{int(round(x))}{int(round(y))}{int(round(z))}" for (x, y, z) in positions]
    return "combined" + "_".join(parts)


def _stem_to_positions(stem: str) -> List[Tuple[float, float, float]]:
    """Parse ground truth positions from WAV stem, e.g. combined3002002_3502902_3902502 -> [(300,200,2), (350,290,2), (390,250,2)].
    Each block is xxxyyyz (x,y 3 digits, z rest). Returns [] if stem does not match."""
    if not stem.startswith("combined"):
        return []
    rest = stem[8:].strip("_")  # after "combined"
    if not rest:
        return []
    out = []
    for block in rest.split("_"):
        block = block.strip()
        if not block or not block.isdigit():
            continue
        # x=first 3, y=next 3, z=rest (e.g. 3002002 -> 300, 200, 2)
        if len(block) < 7:
            continue
        try:
            x, y = int(block[0:3]), int(block[3:6])
            z = int(block[6:]) if block[6:] else 0
            out.append((float(x), float(y), float(z)))
        except ValueError:
            continue
    return out


def _parse_trajectory_line(line: str) -> Optional[Tuple[float, float, float, float, List[float]]]:
    """Parse one line: event_time, x, y, z, [toa1, toa2, ...]. Returns (event_time, x, y, z, arrival_times) or None."""
    line = line.strip()
    if not line:
        return None
    parts = line.split()
    if len(parts) < 5:
        return None
    try:
        event_time = float(parts[0])
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        toas_str = " ".join(parts[4:]).strip("[]")
        arrival_times = [float(t) for t in toas_str.split(",")]
        return (event_time, x, y, z, arrival_times)
    except (ValueError, IndexError):
        return None


def _load_trajectory_toa(trajectory_file: str) -> Optional[Tuple[float, float, float, List[float]]]:
    """Load first event from trajectory .log. Returns (event_time, x, y, z, arrival_times_list) or None."""
    path = Path(trajectory_file)
    if not path.exists():
        return None
    with open(path, "r") as f:
        line = f.readline()
    parsed = _parse_trajectory_line(line)
    return (parsed[0], parsed[1], parsed[2], parsed[3], parsed[4]) if parsed else None


def _load_trajectory_all(trajectory_file: str) -> List[Tuple[float, float, float, float, List[float]]]:
    """Load all events from trajectory .log. Returns list of (event_time, x, y, z, arrival_times) per line."""
    path = Path(trajectory_file)
    if not path.exists():
        return []
    out = []
    with open(path, "r") as f:
        for line in f:
            parsed = _parse_trajectory_line(line)
            if parsed:
                out.append(parsed)
    return out


# ---------------------------------------------------------------------------
# simulate_or_load()
# ---------------------------------------------------------------------------

def simulate_or_load(
    positions: List[Tuple[float, float, float]],
    data_dir: str,
    duration_s: float = 2.0,
    sample_rate: int = 16000,
    skip_simulation: bool = False,
    detection_project_root: Optional[Path] = None,
    shot_times: Optional[List[float]] = None,
    baseline_m: float = 30.0,
) -> List[Dict[str, Any]]:
    """
    Simulate one combined WAV with all gunshots; each gunshot is localized separately via TDOA.
    One file: combined.wav (and combined.log). shot_times: one time in seconds per position (default 0.1 for all).
    Return list of items (one per position) with same wav_path; each has position, name (shot_1, shot_2, ...). Log may be written but is not used for localization (real-world mode).
    """
    if detection_project_root is None:
        detection_project_root = Path(__file__).resolve().parent.parent / "Gunshot_Simulation && Detection"
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    import sys
    if str(detection_project_root) not in sys.path:
        sys.path.insert(0, str(detection_project_root))
    from gunshot_simulation.gunshot_simulation import simulate_gunshots

    if shot_times is not None and len(shot_times) != len(positions):
        shot_times = None
    if shot_times is None:
        shot_times = [0.1] * len(positions)

    # Save WAV/log with position-based name under simulation data path
    stem = _positions_to_stem(positions)
    wav_path = data_path / f"{stem}.wav"
    log_path = data_path / f"{stem}.log"

    if not skip_simulation or not wav_path.exists():
        if not skip_simulation:
            gunshot_positions = np.array(positions, dtype=float)
            simulate_gunshots(
                gunshot_positions=gunshot_positions,
                gunshot_times=shot_times,
                output_wav_file=str(wav_path),
                trajectory_file=str(log_path),
                duration_s=duration_s,
                sample_rate=sample_rate,
                baseline_m=baseline_m,
                plot_spectrogram=False,
                plot_2d=False,
                plot_3d=False,
            )

    items = []
    for idx, (x, y, z) in enumerate(positions):
        items.append({
            "position": (x, y, z),
            "name": f"shot_{idx + 1}",
            "wav_path": str(wav_path),
        })
    return items


# ---------------------------------------------------------------------------
# Process one item: detect event, TDOA, estimate position (no TOA in localization)
# ---------------------------------------------------------------------------

def _first_event_time(toa_global_per_channel: List[List[float]]) -> Optional[float]:
    first_toas = [ch[0] for ch in toa_global_per_channel if ch]
    return float(np.median(first_toas)) if first_toas else None


def _deduplicate_events_keep_earliest(
    events: List[Dict[str, Any]], merge_gap_s: float = 0.45
) -> List[Dict[str, Any]]:
    """
    Merge events within merge_gap_s and keep only the earliest (direct arrival).
    Suppresses duplicate detections from reverberation.
    """
    if len(events) <= 1:
        return events
    times = [e.get("event_time_earliest") or e.get("event_time") for e in events]
    order = np.argsort(times)
    merged = []
    for i in order:
        t = times[i]
        # If within merge_gap of any kept event, skip (reverb)
        if any(abs(t - (m.get("event_time_earliest") or m.get("event_time"))) <= merge_gap_s for m in merged):
            continue
        merged.append(events[i])
    return merged


def _cluster_event_times(toa_global_per_channel: List[List[float]], cluster_gap_s: float = 0.3) -> List[float]:
    """
    From multi-channel TOAs (multiple events), cluster into N events and return sorted event times (median per cluster).
    """
    all_toas = []
    for ch_toas in toa_global_per_channel:
        all_toas.extend(ch_toas)
    if not all_toas:
        return []
    all_toas = sorted(set(all_toas))
    clusters = []
    current = [all_toas[0]]
    for t in all_toas[1:]:
        if t - current[-1] <= cluster_gap_s:
            current.append(t)
        else:
            clusters.append(float(np.median(current)))
            current = [t]
    clusters.append(float(np.median(current)))
    return sorted(clusters)


def process_item(
    item: Dict[str, Any],
    sensor_positions: np.ndarray,
    speed_of_sound: float,
    window_duration_s: float = 0.05,
    audio_data: Optional[np.ndarray] = None,
    sample_rate: Optional[int] = None,
    event_time: Optional[float] = None,
    chunk: Optional[np.ndarray] = None,
    baseline_m: Optional[float] = None,
    range_prior: float = 350.0,
    full_audio: Optional[np.ndarray] = None,
    chunk_duration_s: Optional[float] = None,
    t_earliest: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Run TDOA on chunk (GCC-PHAT) and estimate position. No TOA used for localization.
    Either (audio_data, sample_rate, event_time) or chunk must be provided.
    When chunk is provided: TDOA is computed on chunk only (extracted by detection pipeline).
    Returns item dict with event_time_s, tdoa_s, estimated_position, error_m, gdop, audio_data, sample_rate.
    """
    if chunk is not None:
        # TDOA on chunk only (from detection - each event separately)
        if sample_rate is None:
            raise ValueError("process_item with chunk requires sample_rate")
        # Gentle early weighting: tau=200ms preserves direct path, suppresses tail >150ms
        chunk_weighted = apply_early_energy_weight(chunk, sample_rate, tau_ms=200.0)
        item["audio_data"] = chunk_weighted
        item["sample_rate"] = sample_rate
        event_time = event_time or 0.5 * chunk.shape[1] / sample_rate
        ref_index = 0
        # Delay search: physical limit baseline/c with 10% margin
        max_lag_s = (baseline_m or 30.0) / speed_of_sound * 1.1 if speed_of_sound and speed_of_sound > 0 else 0.1
        tdoa_gcc = tdoa_compute_tdoa(
            chunk_weighted, sample_rate, ref_channel=ref_index,
            max_lag_s=max_lag_s, first_significant_peak=False,
        )
    elif audio_data is not None and sample_rate is not None and event_time is not None:
        # Legacy: extract window from full audio and compute TDOA
        item["audio_data"] = audio_data
        item["sample_rate"] = sample_rate
        ref_index = 0
        win_s = max(window_duration_s, 0.15)
        half_win = win_s / 2.0
        duration_s = audio_data.shape[1] / sample_rate
        max_lag_s = (baseline_m or 30.0) / speed_of_sound * 1.1 if speed_of_sound and speed_of_sound > 0 else None
        tdoa_gcc = tdoa_compute_tdoa_robust(
            audio_data, sample_rate, event_time, win_s,
            ref_channel=ref_index, max_lag_s=max_lag_s, n_windows=5, time_shift_s=0.01,
        )
    else:
        raise ValueError("process_item requires (audio_data, sample_rate, event_time) or (chunk, sample_rate, event_time)")

    ref_index = 0
    # GCC-PHAT: positive = sig_i delayed. Solver expects (d_i - d_ref) = c * tdoa with tdoa = TOA_i - TOA_ref.
    # When i closer, TOA_i < TOA_ref, tdoa negative. GCC returns neg when i earlier -> negate for solver.
    tdoa = -np.asarray(tdoa_gcc, dtype=float)
    tdoa[ref_index] = 0.0
    p_est, gdop, uncertainty_3d, confidence = estimate_position_with_gdop(
        sensor_positions, tdoa, speed_of_sound, ref_index=0,
        baseline_m=baseline_m, range_prior=range_prior,
    )
    # TDOA sign validation: reject if measured signs contradict geometry
    sign_ok, sign_msg = validate_tdoa_sign_consistency(
        tdoa, sensor_positions, p_est, speed_of_sound, ref_index
    )
    unc_norm = float(np.linalg.norm(uncertainty_3d))
    # Always try reflection-robust retries when chunk+full_audio available:
    # GCC may pick reflection peak -> wrong TDOA sign -> large error.
    # Direct path typically gives lower uncertainty (tighter fit).
    # Reject candidates with position near array centroid (noise gives ~zero TDOA -> false low unc).
    centroid = np.mean(sensor_positions, axis=0)
    min_range_m = 80.0  # Reject positions closer than this (sources at 100m+ should never localize near array)

    def _accept_candidate(p: np.ndarray, unc: np.ndarray) -> bool:
        r = np.linalg.norm(p[:2] - centroid[:2])
        return r >= min_range_m and not np.any(np.isnan(p))

    if chunk is not None:
        if _accept_candidate(p_est, uncertainty_3d):
            best_unc = unc_norm
            best = (tdoa, p_est, gdop, uncertainty_3d, confidence)
        else:
            best_unc = float("inf")
            best = None
        # 1) Try first-significant peak (may avoid reflection peak)
        tdoa_fb = tdoa_compute_tdoa(
            chunk_weighted, sample_rate, ref_channel=ref_index,
            max_lag_s=max_lag_s, first_significant_peak=True,
        )
        tdoa_fb = -np.asarray(tdoa_fb, dtype=float)
        tdoa_fb[ref_index] = 0.0
        p_fb, gdop_fb, unc_fb, conf_fb = estimate_position_with_gdop(
            sensor_positions, tdoa_fb, speed_of_sound, ref_index=0,
            baseline_m=baseline_m, range_prior=range_prior,
        )
        if _accept_candidate(p_fb, unc_fb) and np.linalg.norm(unc_fb) < best_unc:
            best_unc = np.linalg.norm(unc_fb)
            best = (tdoa_fb, p_fb, gdop_fb, unc_fb, conf_fb)
        # 2) Try alternative chunk centers (direct path may be before or near detected peak)
        if full_audio is not None and chunk_duration_s is not None and t_earliest is not None:
            for offset_s in [-0.35, -0.30, -0.25, -0.20, -0.15, -0.10, 0.0, 0.05, 0.10]:
                t_try = t_earliest + offset_s
                if t_try < 0.05:
                    continue
                chunk_try = extract_window(
                    full_audio, sample_rate, t_try, chunk_duration_s, center_at_onset=False
                )
                if chunk_try.shape[1] < 100:
                    continue
                chunk_try_w = apply_early_energy_weight(chunk_try, sample_rate, tau_ms=200.0)
                for use_first_peak in [False, True]:
                    tdoa_try = tdoa_compute_tdoa(
                        chunk_try_w, sample_rate, ref_channel=ref_index, max_lag_s=max_lag_s,
                        first_significant_peak=use_first_peak,
                    )
                    tdoa_try = -np.asarray(tdoa_try, dtype=float)
                    tdoa_try[ref_index] = 0.0
                    p_try, gdop_t, unc_t, conf_t = estimate_position_with_gdop(
                        sensor_positions, tdoa_try, speed_of_sound, ref_index=0,
                        baseline_m=baseline_m, range_prior=range_prior,
                    )
                    unc_t_norm = np.linalg.norm(unc_t)
                    if _accept_candidate(p_try, unc_t) and unc_t_norm < best_unc:
                        best_unc = unc_t_norm
                        best = (tdoa_try, p_try, gdop_t, unc_t, conf_t)
        # Replace when: we found a better accepted result, AND (initial was bad or we had no valid initial)
        if best is not None and best_unc < unc_norm:
            if unc_norm > 15.0 or not _accept_candidate(p_est, uncertainty_3d):
                tdoa, p_est, gdop, uncertainty_3d, confidence = best
    unc_norm = float(np.linalg.norm(uncertainty_3d))
    # Event-time refinement: only when initial uncertainty is high and we have full audio (not chunk-only).
    t_best = event_time
    if unc_norm > 20.0 and audio_data is not None:
        offsets_s = np.array([-0.10, -0.06, -0.03, 0.0, 0.03])
        best_unc = unc_norm
        best = (event_time, tdoa, p_est, gdop, uncertainty_3d, confidence)
        for dk in offsets_s:
            t_center = event_time + dk
            t_center = max(half_win, min(duration_s - half_win, t_center))
            tdoa_gcc = tdoa_compute_tdoa_robust(
                audio_data, sample_rate, t_center, win_s,
                ref_channel=ref_index, max_lag_s=max_lag_s, n_windows=5, time_shift_s=0.01,
            )
            tdoa_cand = -np.asarray(tdoa_gcc, dtype=float)
            tdoa_cand[ref_index] = 0.0
            p_cand, gdop_c, unc_c, conf_c = estimate_position_with_gdop(
                sensor_positions, tdoa_cand, speed_of_sound, ref_index=0,
                baseline_m=baseline_m, range_prior=range_prior,
            )
            unc_n = float(np.linalg.norm(unc_c))
            if unc_n < best_unc:
                best_unc = unc_n
                best = (t_center, tdoa_cand, p_cand, gdop_c, unc_c, conf_c)
        t_best, tdoa, p_est, gdop, uncertainty_3d, confidence = best
    tdoa_src = "GCC-PHAT"
    tdoa_matrix = build_tdoa_matrix(tdoa, ref_index=ref_index)
    tdoa_valid, tdoa_msg = validate_tdoa_ms(tdoa, ref_index=ref_index, max_abs_ms=100.0)
    name = item.get("name", "?")
    if abs(t_best - event_time) > 0.005:
        print(f"  [{name}] event_time={event_time:.4f} s -> refined {t_best:.4f} s, TDOA (s) = {[f'{t:.4f}' for t in tdoa]} ({tdoa_src}), TDOA validation: {tdoa_msg}")
    else:
        print(f"  [{name}] event_time={event_time:.4f} s, TDOA (s) = {[f'{t:.4f}' for t in tdoa]} ({tdoa_src}), TDOA validation: {tdoa_msg}")
    if not tdoa_valid:
        print(f"    WARNING: {tdoa_msg}")
    print(f"  [{name}] estimated (x,y,z) m = [{p_est[0]:.2f}, {p_est[1]:.2f}, {p_est[2]:.2f}], GDOP = {gdop:.4f}, uncertainty (m) = {uncertainty_3d}, confidence = {confidence:.3f}")

    gt = np.array(item["position"])
    err = float(np.linalg.norm(p_est - gt))

    item["event_time_s"] = t_best
    item["tdoa_s"] = tdoa
    item["tdoa_matrix"] = tdoa_matrix
    item["estimated_position"] = p_est
    item["error_m"] = err
    item["gdop"] = gdop
    item["uncertainty_3d"] = uncertainty_3d
    item["confidence"] = confidence
    return item


# ---------------------------------------------------------------------------
# Plotting (2D, 3D, waveform)
# ---------------------------------------------------------------------------

def _stats_from_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute avg error and mean GDOP over valid results."""
    errors = [r.get("error_m") for r in results if not np.isnan(r.get("error_m", float("nan")))]
    gdops = [r.get("gdop") for r in results if r.get("gdop") is not None and not np.isnan(r.get("gdop", float("nan")))]
    return {
        "avg_error_m": float(np.mean(errors)) if errors else float("nan"),
        "mean_gdop": float(np.mean(gdops)) if gdops else float("nan"),
        "errors": [r.get("error_m", float("nan")) for r in results],
        "gdops": [r.get("gdop", float("nan")) for r in results],
    }


def plot_2d(
    sensor_positions: np.ndarray,
    results: List[Dict[str, Any]],
    output_path: Optional[str] = None,
    stats: Optional[Dict[str, Any]] = None,
) -> plt.Figure:
    """2D map: sensors, ground truth, estimated positions, error lines; optional error/avg error/GDOP text."""
    if stats is None:
        stats = _stats_from_results(results)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_aspect("equal")
    ax.scatter(sensor_positions[:, 0], sensor_positions[:, 1], c="blue", s=80, label="Sensors", zorder=3)
    for i in range(len(sensor_positions)):
        ax.annotate(f"S{i+1}", (sensor_positions[i, 0], sensor_positions[i, 1]), fontsize=9, xytext=(5, 5), textcoords="offset points")
    for r in results:
        gt = r["position"]
        est = r.get("estimated_position")
        if est is not None and not np.any(np.isnan(est)):
            ax.scatter(gt[0], gt[1], c="green", s=60, marker="o", edgecolors="black", zorder=2)
            ax.scatter(est[0], est[1], c="red", s=60, marker="x", linewidths=2, zorder=2)
            ax.plot([gt[0], est[0]], [gt[1], est[1]], "k--", alpha=0.7, linewidth=1)
    ax.scatter([], [], c="green", s=60, marker="o", edgecolors="black", label="Ground truth")
    ax.scatter([], [], c="red", s=60, marker="x", linewidths=2, label="Estimated")
    ax.set_xlabel("X (East, m)")
    ax.set_ylabel("Y (North, m)")
    ax.set_title("2D localization: sensors, ground truth, estimated, error lines")
    err_txt = []
    for i, r in enumerate(results):
        e = r.get("error_m", float("nan"))
        g = r.get("gdop", float("nan"))
        err_txt.append(f"{r.get('name', i)}: err={e:.2f}m, GDOP={g:.3f}" if not np.isnan(e) and not np.isnan(g) else f"{r.get('name', i)}: err=N/A, GDOP=N/A")
    err_txt.append(f"Avg error: {stats.get('avg_error_m', float('nan')):.2f} m")
    err_txt.append(f"Mean GDOP: {stats.get('mean_gdop', float('nan')):.3f}")
    ax.text(0.02, 0.98, "\n".join(err_txt), transform=ax.transAxes, fontsize=8, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_3d(
    sensor_positions: np.ndarray,
    results: List[Dict[str, Any]],
    output_path: Optional[str] = None,
    stats: Optional[Dict[str, Any]] = None,
) -> plt.Figure:
    """3D map: sensors, ground truth, estimated positions; optional error/avg error/GDOP text."""
    if stats is None:
        stats = _stats_from_results(results)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(sensor_positions[:, 0], sensor_positions[:, 1], sensor_positions[:, 2], c="blue", s=80, label="Sensors")
    for r in results:
        gt = r["position"]
        est = r.get("estimated_position")
        if est is not None and not np.any(np.isnan(est)):
            ax.scatter(gt[0], gt[1], gt[2], c="green", s=60, marker="o")
            ax.scatter(est[0], est[1], est[2], c="red", s=60, marker="x")
    ax.set_xlabel("X (East, m)")
    ax.set_ylabel("Y (North, m)")
    ax.set_zlabel("Z (Up, m)")
    ax.set_title("3D localization: sensors, ground truth, estimated")
    err_txt = [f"Avg error: {stats.get('avg_error_m', float('nan')):.2f} m", f"Mean GDOP: {stats.get('mean_gdop', float('nan')):.3f}"]
    ax.text2D(0.02, 0.98, "\n".join(err_txt), transform=ax.transAxes, fontsize=9, verticalalignment="top",
              bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    ax.legend()
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_waveform(
    item: Dict[str, Any],
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Waveform: show channels and mark TOA for visualization only.
    - GT TOA from trajectory (simulated) if available.
    - TDOA-derived "TOA" = event_time + tdoa[i] (for visualization only; not used in localization).
    """
    audio = item.get("audio_data")
    sr = item.get("sample_rate")
    if audio is None or sr is None:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.set_title(f"{item.get('name', '')} (no audio)")
        return fig
    event_time = item.get("event_time_s")
    tdoa = item.get("tdoa_s")
    gt_toa = item.get("ground_truth_toa_per_ch")
    n_ch = audio.shape[0]
    duration_s = audio.shape[1] / sr
    t_axis = np.arange(audio.shape[1]) / sr
    # When audio is a short chunk (<0.2s), event is at center; else use event_time_s
    is_chunk = duration_s < 0.2
    event_time_viz = duration_s / 2.0 if is_chunk else event_time
    fig, axes = plt.subplots(n_ch, 1, sharex=True, figsize=(12, 2 * n_ch))
    if n_ch == 1:
        axes = [axes]
    for ch in range(n_ch):
        ax = axes[ch]
        ax.plot(t_axis, audio[ch], color="gray", alpha=0.8)
        # GT TOA (visualization only)
        if gt_toa and ch < len(gt_toa):
            ax.axvline(gt_toa[ch], color="green", linestyle="--", linewidth=1.5, label="TOA (GT)")
        # TDOA-derived "TOA" (visualization only; event_time_viz is center for chunk, else global event_time)
        if event_time is not None and tdoa is not None and ch < len(tdoa):
            toa_viz = event_time_viz + tdoa[ch]
            ax.axvline(toa_viz, color="red", linestyle=":", linewidth=1.5, label="TOA (from TDOA)")
        ax.set_ylabel(f"Ch {ch+1}")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Waveform: {item.get('name', '')} — TOA marks (visualization only, not used in localization)")
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    results: List[Dict[str, Any]],
    sensor_positions: np.ndarray,
    output_dir: str,
    fig_2d_path: Optional[str] = None,
    fig_3d_path: Optional[str] = None,
    waveform_fig_paths: Optional[List[str]] = None,
    figures_only_in_pdf: bool = True,
    report_name: Optional[str] = None,
    pdf_append: Optional[Any] = None,
) -> str:
    """
    Generate PDF report: table (position name, GT, estimated, error, GDOP), 2D plot, 3D plot, waveform plots.
    If report_name is given, save as report_{report_name}.pdf; otherwise report.pdf.
    If pdf_append is provided (PdfPages), append triangulation pages to it and do not create/close a new PDF.
    Returns path to report PDF.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report_filename = f"report_{report_name}.pdf" if report_name else "report.pdf"
    report_path = out / report_filename
    stats = _stats_from_results(results)

    def _add_pages(pdf, is_combined: bool = False):
        # Title (or section divider when combined with detection)
        if is_combined:
            fig = plt.figure(figsize=(8, 2))
            fig.suptitle("Part 2: TDOA Localization (GCC-PHAT on per-event chunks, no TOA)", fontsize=14)
        else:
            fig = plt.figure(figsize=(8, 2))
            fig.suptitle("TDOA Localization Report (localization uses GCC-PHAT TDOA only, no TOA)", fontsize=14)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Results table (Error, GDOP, Uncertainty, Confidence)
        fig, ax = plt.subplots(figsize=(16, max(4, 0.5 * len(results))))
        ax.axis("off")
        rows = [["Position name", "Ground truth (x,y,z) m", "Estimated (x,y,z) m", "Error (m)", "GDOP", "Uncertainty (m)", "Confidence"]]
        for r in results:
            name = r.get("name", "")
            gt = r["position"]
            est = r.get("estimated_position")
            err = r.get("error_m", float("nan"))
            gdop = r.get("gdop", float("nan"))
            unc = r.get("uncertainty_3d", np.full(3, float("nan")))
            conf = r.get("confidence", float("nan"))
            gt_s = f"({gt[0]:.2f}, {gt[1]:.2f}, {gt[2]:.2f})"
            est_s = f"({est[0]:.2f}, {est[1]:.2f}, {est[2]:.2f})" if est is not None and not np.any(np.isnan(est)) else "N/A"
            err_s = f"{err:.3f}" if not np.isnan(err) else "N/A"
            gdop_s = f"{gdop:.3f}" if not np.isnan(gdop) else "N/A"
            unc_s = f"({unc[0]:.2f},{unc[1]:.2f},{unc[2]:.2f})" if unc is not None and not np.any(np.isnan(unc)) else "N/A"
            conf_s = f"{conf:.3f}" if not np.isnan(conf) else "N/A"
            rows.append([name, gt_s, est_s, err_s, gdop_s, unc_s, conf_s])
        mean_conf = float(np.nanmean([r.get("confidence", float("nan")) for r in results]))
        rows.append(["Summary", "", "", f"Avg error: {stats.get('avg_error_m', float('nan')):.3f} m", f"Mean GDOP: {stats.get('mean_gdop', float('nan')):.3f}", "", f"Mean conf: {mean_conf:.3f}"])
        table = ax.table(cellText=rows, loc="center", cellLoc="center", colWidths=[0.15, 0.22, 0.22, 0.1, 0.08, 0.13, 0.1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2)
        ax.set_title("Localization results: error, avg error, GDOP")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # 2D and 3D: draw in-memory with error/GDOP on plot
        fig_2d = plot_2d(sensor_positions, results, output_path=None, stats=stats)
        pdf.savefig(fig_2d, bbox_inches="tight")
        plt.close(fig_2d)

        fig_3d = plot_3d(sensor_positions, results, output_path=None, stats=stats)
        pdf.savefig(fig_3d, bbox_inches="tight")
        plt.close(fig_3d)

        # Waveforms: one page per position (in-memory)
        for r in results:
            fig_wf = plot_waveform(r, output_path=None)
            pdf.savefig(fig_wf, bbox_inches="tight")
            plt.close(fig_wf)

    if pdf_append is not None:
        _add_pages(pdf_append, is_combined=True)
    else:
        with PdfPages(report_path) as pdf:
            _add_pages(pdf, is_combined=False)

    return str(report_path)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    positions: Optional[List[Tuple[float, float, float]]] = None,
    data_dir: Optional[str] = None,
    output_dir: str = "",
    skip_simulation: bool = False,
    detection_project_root: Optional[Path] = None,
    window_duration_s: float = 0.05,
    duration_s: float = 2.0,
    shot_times: Optional[List[float]] = None,
    baseline_m: float = 30.0,
    min_gap_s: float = 0.4,
    wav_file: Optional[str] = None,
    detect_threshold_percentile: float = 85.0,
    detect_prominence_ratio: float = 0.15,
) -> Dict[str, Any]:
    """
    Multi-shot TDOA localization. Detect gunshots -> extract per-event chunks -> GCC-PHAT TDOA on each chunk -> multilateration.
    Uses gunshot_detection for event detection; TDOA computed on short chunks only (no TOA for localization).
    """
    import sys
    from .utils import load_audio, get_sensor_positions_enu, calculate_speed_of_sound

    if detection_project_root is None:
        detection_project_root = Path(__file__).resolve().parent.parent / "Gunshot_Simulation && Detection"
    _det_root = str(detection_project_root)
    if _det_root not in sys.path:
        sys.path.insert(0, _det_root)
    from gunshot_detection.gunshot_detection import detect_gunshots

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data_path = Path(data_dir or ".")
    if data_dir:
        data_path.mkdir(parents=True, exist_ok=True)

    report_name_from_wav: Optional[str] = None
    tdoa_window_s = max(window_duration_s, 0.08)  # 80 ms min for GCC-PHAT on chunk

    # 1) Use saved WAV path (no regeneration) or simulate/load combined.wav
    if wav_file:
        wav_path = Path(wav_file)
        if not wav_path.is_absolute():
            # Relative: try cwd first, then repo root for "Gunshot_Simulation && Detection/..." style paths
            cwd_path = (Path.cwd() / wav_file.strip()).resolve()
            repo_path = (Path(__file__).resolve().parent.parent / wav_file.strip()).resolve()
            wav_path = cwd_path if cwd_path.exists() else repo_path
        else:
            wav_path = wav_path.resolve()
        # Logical stem from user path (for report and GT parsing), e.g. combined3002002_3502902_3902502
        logical_stem = Path(wav_file).stem
        if not wav_path.exists():
            # Fallback: some setups save as .wav.wav / .wav.log
            fallback = Path(str(wav_path) + ".wav")
            if fallback.exists():
                wav_path = fallback
            else:
                hint = ""
                parsed = _stem_to_positions(logical_stem)
                if parsed:
                    pos_str = " ".join(f"{x:.0f} {y:.0f} {z:.0f}" for (x, y, z) in parsed)
                    hint = f' Create it first: python run_pipeline.py --positions "{pos_str}"'
                raise FileNotFoundError(
                    f"WAV file not found: {wav_path}.{hint}"
                )
        report_name_from_wav = logical_stem  # report_combined3002002_3502902_3902502.pdf
        # Ground truth from WAV path when not provided (no need to write --positions every time)
        if not positions:
            positions = _stem_to_positions(logical_stem)
            if positions:
                print(f"Using ground truth from WAV filename ({len(positions)} position(s)): {positions}")
        # Load audio (WAV only, no simulation/detection)
        audio_data, sample_rate = load_audio(str(wav_path), duration=None, bandpass=(1000.0, 24000.0))
        print(f"Loaded audio from: {wav_path}")
        print(f"Audio shape (cut to None-None s): {audio_data.shape}, Sample rate: {sample_rate}")
        positions = positions or []
        items = []
        for idx in range(max(len(positions), 1)):  # at least 1 item
            x, y, z = (positions[idx] if idx < len(positions) else (0.0, 0.0, 0.0))
            name = f"pos_{x:.0f}_{y:.0f}_{z:.0f}" if (x or y or z) else f"shot_{idx + 1}"
            items.append({
                "position": (x, y, z),
                "name": name,
                "wav_path": str(wav_path),
            })
    else:
        positions = positions or []
        if not positions:
            raise ValueError("positions required when not using --wav-file")
        items = simulate_or_load(
            positions,
            data_dir,
            duration_s=duration_s,
            skip_simulation=skip_simulation,
            detection_project_root=detection_project_root,
            shot_times=shot_times,
            baseline_m=baseline_m,
        )

    # Sensor positions: equilateral triangle + center (from triangulation.utils, no external deps)
    sensor_positions, sensor_ids = get_sensor_positions_enu(baseline_m=baseline_m)
    names = [f"Sensor_{i:02d}" for i in sensor_ids]
    print(f"Loaded {len(sensor_positions)} sensor positions (ENU, origin at triangle center, baseline={baseline_m} m)")
    for i, row in enumerate(sensor_positions):
        print(f"  {names[i]}: [{row[0]:.2f}, {row[1]:.2f}, {row[2]:.2f}] m")
    speed_of_sound = calculate_speed_of_sound()

    # 2) Load audio (if not already done for wav_file path)
    if not wav_file:
        wav_path = Path(items[0]["wav_path"])
        audio_data, sample_rate = load_audio(str(wav_path), duration=None, bandpass=(1000.0, 24000.0))
        print(f"Loaded audio from: {wav_path}")
        print(f"Audio shape: {audio_data.shape}, Sample rate: {sample_rate}")

    # 3) Combined report: detection + triangulation, saved with position name (pure TDOA, no .log)
    report_name = report_name_from_wav or (_positions_to_stem(positions) if positions else "report")
    report_filename = f"report_{report_name}.pdf"
    report_path = out / report_filename
    time_tol_ms = max(10.0, min_gap_s * 500)

    with PdfPages(report_path) as pdf:
        # Part 1: Gunshot detection (adds pages to pdf, returns events)
        events = detect_gunshots(
            audio_data, sample_rate,
            chunk_window_ms=100,
            time_tolerance_ms=time_tol_ms,
            plot_results=True,
            output_pdf=None,
            pdf_append=pdf,
            use_kurtosis=False,
            min_channels=2,
            bandpass_low=500.0,
            bandpass_high=8000.0,
        )
        # Deduplicate: merge events within 0.45 s, keep earliest (direct arrival)
        events = _deduplicate_events_keep_earliest(events, merge_gap_s=0.45)
        # If more events than positions: keep N evenly spaced (avoids reverb duplicates)
        n_pos = len(positions) if positions else 0
        if n_pos > 0 and len(events) > n_pos:
            idx = np.linspace(0, len(events) - 1, n_pos, dtype=int)
            events = [events[i] for i in idx]
        n_events = len(events)
        if n_events > 0:
            print(f"  Detected {n_events} event(s)")
        if len(items) < n_events:
            for idx in range(len(items), n_events):
                items.append({
                    "position": (0.0, 0.0, 0.0),
                    "name": f"shot_{idx + 1}",
                    "wav_path": str(Path(items[0]["wav_path"])),
                })
        elif len(items) > n_events and positions:
            items = items[:n_events]
            print(f"  WARNING: Detected {n_events} event(s), using first {n_events} positions.")
        # Part 2: TDOA on chunks only (GCC-PHAT, no TOA)
        max_tdoa_s = baseline_m / speed_of_sound if speed_of_sound > 0 else 0.1
        chunk_duration_s = max(tdoa_window_s, max_tdoa_s + 0.08)
        # Use trajectory .log for chunk times when available (avoids detection reverb/assignment errors)
        wav_path = Path(items[0]["wav_path"]) if items else None
        log_path = wav_path.with_suffix(".log") if wav_path else None
        trajectory_events = _load_trajectory_all(str(log_path)) if log_path and log_path.exists() else []
        use_trajectory = len(trajectory_events) >= len(items) and len(positions) >= len(items)
        if use_trajectory:
            print(f"  Using trajectory {log_path.name} for chunk extraction")
        results = []
        for i, item in enumerate(items):
            ev = events[i] if i < len(events) else None
            event_time = ev["event_time"] if ev else None
            t_earliest = ev.get("event_time_earliest") if ev else None
            if use_trajectory and i < len(trajectory_events):
                _, _, _, _, toas = trajectory_events[i]
                t_earliest = min(toas)
                event_time = t_earliest
            elif t_earliest is None and event_time is not None:
                t_earliest = event_time - max_tdoa_s
            chunk = None
            if event_time is not None and t_earliest is not None:
                # Center so all direct arrivals fall in window
                chunk_center = t_earliest + max_tdoa_s * 0.5
                chunk = extract_window(
                    audio_data, sample_rate, chunk_center, chunk_duration_s,
                    center_at_onset=False,
                )
            if chunk is not None and event_time is not None:
                process_item(
                    item,
                    sensor_positions,
                    speed_of_sound,
                    sample_rate=sample_rate,
                    event_time=event_time,
                    chunk=chunk,
                    baseline_m=baseline_m,
                    range_prior=350.0,
                    full_audio=audio_data,
                    chunk_duration_s=chunk_duration_s,
                    t_earliest=t_earliest,
                )
            else:
                item["event_time_s"] = event_time
                item["estimated_position"] = np.array([float("nan"), float("nan"), float("nan")])
                item["error_m"] = float("nan")
                item["gdop"] = float("nan")
                item["uncertainty_3d"] = np.full(3, float("nan"))
                item["confidence"] = 0.0
                item["tdoa_s"] = None
                item["tdoa_matrix"] = None
                item["audio_data"] = audio_data
                item["sample_rate"] = sample_rate
            est = item.get("estimated_position")
            gt = item["position"]
            if est is not None and not np.any(np.isnan(est)) and gt[0] == 0 and gt[1] == 0 and gt[2] == 0:
                item["name"] = f"pos_{est[0]:.0f}_{est[1]:.0f}_{est[2]:.0f}"
            results.append(item)
        # Part 3: Triangulation pages
        generate_report(
            results,
            sensor_positions,
            output_dir,
            figures_only_in_pdf=True,
            report_name=None,
            pdf_append=pdf,
        )

    # Store estimated positions for each shot (no TOA used)
    estimated_positions = []
    for r in results:
        p = r.get("estimated_position")
        if p is not None and not np.any(np.isnan(p)):
            estimated_positions.append(np.array(p))
        else:
            estimated_positions.append(None)

    return {
        "results": results,
        "estimated_positions": estimated_positions,
        "sensor_positions": sensor_positions,
        "report_path": str(report_path),
        "plots_dir": str(out),
    }
