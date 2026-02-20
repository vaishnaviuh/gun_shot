"""
TDOA-based localization (no TOA).
- Event detection on one reference channel (no TOA).
- Extract short windows, GCC-PHAT for TDOA, multilateration via least_squares.
"""
import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from scipy.signal import find_peaks
from typing import List, Tuple, Optional


def detect_events_reference_channel(
    audio_data: np.ndarray,
    sample_rate: float,
    ref_channel: int = 0,
    min_gap_s: float = 0.1,
    window_s: float = 0.02,
    threshold_percentile: float = 85.0,
    min_prominence_ratio: float = 0.15,
) -> List[float]:
    """
    Detect multiple gunshot events using one reference channel only (no TOA).
    Uses short-time energy; enforces minimum gap between events (default 100 ms).
    Returns sorted list of event times in seconds.
    Tuned for real recordings (no .log): default threshold 85% and prominence 0.15 so weaker
    shots are detected. When a trajectory .log exists, the pipeline uses its event times instead.
    """
    ref_sig = audio_data[ref_channel]
    n = len(ref_sig)
    if n == 0:
        return []
    hop = max(1, int(0.005 * sample_rate))  # 5 ms hop
    win_len = max(1, int(window_s * sample_rate))
    n_win = max(1, (n - win_len) // hop + 1)
    energy = np.zeros(n_win)
    for i in range(n_win):
        start = i * hop
        end = min(start + win_len, n)
        energy[i] = np.sum(ref_sig[start:end] ** 2) / max(1, end - start)
    t_axis = (np.arange(n_win) * hop + win_len / 2) / sample_rate
    thresh = np.percentile(energy, threshold_percentile)
    prominence = max(np.ptp(energy) * min_prominence_ratio, 1e-12)
    peaks, props = find_peaks(energy, height=thresh, prominence=prominence, distance=max(1, int(min_gap_s * sample_rate / hop)))
    event_times = [float(t_axis[p]) for p in peaks]
    # Enforce min_gap_s: merge events that are too close (keep first in each group)
    if not event_times:
        return []
    event_times = sorted(event_times)
    out = [event_times[0]]
    for t in event_times[1:]:
        if t - out[-1] >= min_gap_s:
            out.append(t)
    return out


def extract_window(
    audio_data: np.ndarray,
    sample_rate: float,
    event_time_s: float,
    window_duration_s: float = 0.02,
    center_at_onset: bool = False,
    onset_offset_s: float = 0.0,
) -> np.ndarray:
    """
    Extract a short time window around the event for each channel.

    Args:
        audio_data: Shape (num_channels, num_samples).
        sample_rate: Sample rate in Hz.
        event_time_s: Center time (or start time if center_at_onset) in seconds.
        window_duration_s: Total window duration in seconds (default 20 ms).
        center_at_onset: If True, window starts at event_time_s (onset), not centered.
        onset_offset_s: Small offset before onset (s) to capture very start (default 0).

    Returns:
        windows: Shape (num_channels, window_samples).
    """
    num_channels, total_samples = audio_data.shape
    duration_s = total_samples / sample_rate
    if center_at_onset:
        # Start at onset (minus small offset), extend forward
        start_s = max(0.0, event_time_s - onset_offset_s)
        end_s = min(duration_s, start_s + window_duration_s)
        if end_s - start_s < window_duration_s * 0.5:
            start_s = max(0.0, end_s - window_duration_s)
    else:
        half = 0.5 * window_duration_s
        if event_time_s - half < 0:
            event_time_s = half
        elif event_time_s + half > duration_s:
            event_time_s = duration_s - half
        if event_time_s < half:
            event_time_s = half
        start_s = event_time_s - half
        end_s = event_time_s + half
    start_sample = int(start_s * sample_rate)
    end_sample = int(end_s * sample_rate)
    return audio_data[:, start_sample:end_sample].copy()


def _normalize_signal(sig: np.ndarray) -> np.ndarray:
    """Normalize to unit L2 norm so GCC is insensitive to level."""
    n = np.linalg.norm(sig)
    if n > 1e-12:
        return sig / n
    return sig


def apply_early_energy_weight(
    chunk: np.ndarray, sample_rate: float, tau_ms: float = 25.0
) -> np.ndarray:
    """
    Apply exponential decay to suppress late reflections.
    Earlier samples (direct path) retain more weight; later samples (reverb) are suppressed.
    w[n] = exp(-n / (tau_samples)).
    """
    n_samples = chunk.shape[1]
    tau_samples = max(1.0, sample_rate * tau_ms / 1000.0)
    t = np.arange(n_samples, dtype=np.float64)
    weight = np.exp(-t / tau_samples)
    return chunk * weight[np.newaxis, :]


def gcc_phat(
    sig_ref: np.ndarray,
    sig_i: np.ndarray,
    sample_rate: float,
    max_lag_samples: Optional[int] = None,
    first_significant_peak: bool = False,
    significance_threshold: float = 0.4,
) -> float:
    """
    Compute time delay between two signals using GCC-PHAT with sub-sample interpolation.
    Signals are normalized before GCC. Parabolic interpolation around the peak gives
    sub-sample accuracy. Delay in seconds: positive = sig_i delayed w.r.t. sig_ref.

    Args:
        first_significant_peak: If True, use first significant peak from zero lag
            (favors direct path over reflection peaks). If False, use global max.
        significance_threshold: Fraction of max |gcc| for "significant" (0.4 = 40%).
    """
    n = len(sig_ref)
    if n != len(sig_i) or n == 0:
        return 0.0
    if max_lag_samples is None:
        max_lag_samples = n // 2

    # Normalize so correlation is shape-based, not level-based
    sig_ref = _normalize_signal(sig_ref.astype(np.float64))
    sig_i = _normalize_signal(sig_i.astype(np.float64))

    # FFT
    n_fft = 2 ** int(np.ceil(np.log2(2 * n - 1)))
    R_ref = np.fft.rfft(sig_ref, n=n_fft)
    R_i = np.fft.rfft(sig_i, n=n_fft)
    cross = R_ref * np.conj(R_i)
    mag = np.abs(cross) + 1e-12
    G = cross / mag
    gcc = np.fft.irfft(G, n=n_fft)
    gcc = np.fft.fftshift(gcc)
    gcc = np.asarray(gcc, dtype=float)

    lags = np.arange(-n_fft // 2, n_fft // 2) if n_fft % 2 == 0 else np.arange(-(n_fft - 1) // 2, (n_fft + 1) // 2)
    lag_lim = min(max_lag_samples, len(lags) // 2)
    valid = np.abs(lags) <= lag_lim
    gcc_valid = np.where(valid, np.abs(gcc), -np.inf)

    if first_significant_peak:
        gcc_max = np.max(gcc_valid)
        thresh = significance_threshold * gcc_max
        # Find peaks (local maxima)
        peaks = []
        for i in range(1, len(gcc_valid) - 1):
            if gcc_valid[i] >= gcc_valid[i - 1] and gcc_valid[i] >= gcc_valid[i + 1]:
                if gcc_valid[i] >= thresh and valid[i]:
                    peaks.append((abs(lags[i]), i))
        # Sort by |lag| (closest to zero first) - direct path is typically earliest
        peaks.sort(key=lambda x: x[0])
        if peaks:
            idx_global = peaks[0][1]
        else:
            idx_global = np.argmax(gcc_valid)
    else:
        idx_global = int(np.argmax(gcc_valid))

    lag_samples_int = lags[idx_global]

    # Sub-sample: parabolic interpolation
    if idx_global > 0 and idx_global < len(gcc) - 1:
        y0 = gcc[idx_global - 1]
        y1 = gcc[idx_global]
        y2 = gcc[idx_global + 1]
        denom = 2.0 * (2.0 * y1 - y0 - y2)
        if abs(denom) > 1e-12:
            delta = (y2 - y0) / denom
            delta = np.clip(delta, -1.0, 1.0)
            lag_samples = float(lag_samples_int) + delta
        else:
            lag_samples = float(lag_samples_int)
    else:
        lag_samples = float(lag_samples_int)

    delay_s = lag_samples / sample_rate
    return delay_s


def tdoa_from_toa_list(
    arrival_times: List[float],
    ref_index: int = 0,
) -> np.ndarray:
    """
    Build TDOA vector from per-channel TOAs (e.g. from trajectory log).
    tdoa[i] = arrival_times[i] - arrival_times[ref_index] (seconds).
    """
    arrival_times = np.asarray(arrival_times, dtype=float)
    tdoa = arrival_times - arrival_times[ref_index]
    return tdoa


def compute_tdoa_robust(
    audio_data: np.ndarray,
    sample_rate: float,
    event_time_s: float,
    window_duration_s: float,
    ref_channel: int = 0,
    max_lag_s: Optional[float] = None,
    n_windows: int = 3,
    time_shift_s: float = 0.015,
) -> np.ndarray:
    """
    Compute TDOA using median over multiple windows for robustness to noise and overlap.
    Extracts n_windows centered at event_time_s ± k*time_shift_s, computes TDOA each, returns median.
    """
    shifts = np.linspace(-time_shift_s * (n_windows - 1) / 2, time_shift_s * (n_windows - 1) / 2, n_windows)
    tdoa_list = []
    for dt in shifts:
        windows = extract_window(audio_data, sample_rate, event_time_s + dt, window_duration_s)
        tdoa_list.append(compute_tdoa(windows, sample_rate, ref_channel, max_lag_s))
    tdoa_list = np.array(tdoa_list)
    return np.median(tdoa_list, axis=0)


def compute_tdoa(
    windows: np.ndarray,
    sample_rate: float,
    ref_channel: int = 0,
    max_lag_s: Optional[float] = None,
    first_significant_peak: bool = False,
) -> np.ndarray:
    """
    Compute TDOA for each channel relative to reference (GCC-PHAT).
    max_lag_s: max lag in seconds = baseline_m/speed_of_sound (physical limit).
    first_significant_peak: Use first significant peak to favor direct path.
    """
    num_channels = windows.shape[0]
    n = windows.shape[1]
    max_lag_samples = None
    if max_lag_s is not None and max_lag_s > 0:
        max_lag_samples = min(n // 2, int(max_lag_s * sample_rate))
    tdoa = np.zeros(num_channels)
    sig_ref = windows[ref_channel]
    for i in range(num_channels):
        if i == ref_channel:
            tdoa[i] = 0.0
        else:
            tdoa[i] = gcc_phat(
                sig_ref, windows[i], sample_rate,
                max_lag_samples=max_lag_samples,
                first_significant_peak=first_significant_peak,
            )
    return tdoa


def validate_tdoa_ms(tdoa: np.ndarray, ref_index: int = 0, max_abs_ms: float = 50.0) -> Tuple[bool, str]:
    """
    Validate TDOA values are in plausible ~ms range (e.g. ±50 ms for typical arrays).
    Returns (ok, message).
    """
    tdoa_no_ref = np.delete(tdoa, ref_index)
    max_abs_s = max_abs_ms / 1000.0
    if np.any(np.isnan(tdoa_no_ref)):
        return False, "TDOA contains NaN"
    if np.any(np.abs(tdoa_no_ref) > max_abs_s):
        bad = tdoa_no_ref[np.abs(tdoa_no_ref) > max_abs_s]
        return False, f"TDOA outside ±{max_abs_ms} ms: {bad * 1000}"
    return True, f"TDOA within ±{max_abs_ms} ms"


def validate_tdoa_sign_consistency(
    tdoa: np.ndarray,
    sensor_positions: np.ndarray,
    estimated_position: np.ndarray,
    speed_of_sound: float,
    ref_index: int = 0,
) -> Tuple[bool, str]:
    """
    Check that measured TDOA signs match expected signs from geometry.
    tdoa[i] = TOA_i - TOA_ref. Expected: (d_i - d_ref) / c has same sign.
    Reject if any channel has flipped sign (indicates reflection/localization error).
    """
    if np.any(np.isnan(estimated_position)) or np.any(np.isnan(tdoa)):
        return False, "NaN in position or TDOA"
    d_ref = np.linalg.norm(estimated_position - sensor_positions[ref_index])
    for i in range(len(tdoa)):
        if i == ref_index:
            continue
        d_i = np.linalg.norm(estimated_position - sensor_positions[i])
        expected_tdoa = (d_i - d_ref) / speed_of_sound
        # Sign must match (both same direction)
        if np.sign(tdoa[i]) != np.sign(expected_tdoa) and abs(expected_tdoa) > 0.001:
            return False, f"TDOA sign flip: ch{i} measured={tdoa[i]:.4f} expected={expected_tdoa:.4f}"
    return True, "Signs consistent"


def compute_tdoa_multi_ref(
    windows: np.ndarray,
    sample_rate: float,
    max_lag_s: float,
    first_significant_peak: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute TDOA using multiple reference channels; return median to reject outliers.
    Returns (tdoa_vector_ref0, tdoa_list_per_ref) - tdoa_vector_ref0 is ref_index=0,
    converted so ref has zero; we use median of consistent conversions.
    """
    num_channels = windows.shape[0]
    tdoa_per_ref = []
    for ref in range(num_channels):
        t = compute_tdoa(windows, sample_rate, ref_channel=ref, max_lag_s=max_lag_s,
                        first_significant_peak=first_significant_peak)
        # Convert to ref=0 basis: tdoa0[i] = tdoa_from_ref[i] - tdoa_from_ref[0]
        t0 = t - t[0]
        tdoa_per_ref.append(t0)
    tdoa_per_ref = np.array(tdoa_per_ref)
    tdoa_median = np.median(tdoa_per_ref, axis=0)
    tdoa_median[0] = 0.0
    return tdoa_median, tdoa_per_ref


def build_tdoa_matrix(tdoa_vector: np.ndarray, ref_index: int = 0) -> np.ndarray:
    """
    Build full TDOA matrix from ref-relative vector (no TOA used).
    tdoa_matrix[i,j] = TOA_j - TOA_i in seconds; diagonal zero.
    """
    n = len(tdoa_vector)
    tdoa_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # tdoa_vector[k] = TOA_k - TOA_ref => tdoa_matrix[i,j] = tdoa_vector[j] - tdoa_vector[i]
            tdoa_matrix[i, j] = float(tdoa_vector[j] - tdoa_vector[i])
    return tdoa_matrix


def estimate_position(
    sensor_positions: np.ndarray,
    tdoa: np.ndarray,
    speed_of_sound: float,
    ref_index: int = 0,
    x0: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Pure TDOA 3D multilateration; returns (x, y, z)."""
    p, _, _, _ = estimate_position_with_gdop(
        sensor_positions, tdoa, speed_of_sound, ref_index, x0
    )
    return p


def estimate_position_with_gdop(
    sensor_positions: np.ndarray,
    tdoa: np.ndarray,
    speed_of_sound: float,
    ref_index: int = 0,
    x0: Optional[np.ndarray] = None,
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    quality_weights: Optional[np.ndarray] = None,
    baseline_m: Optional[float] = None,
    range_prior: float = 350.0,
    far_field_baseline_threshold: float = 100.0,
) -> Tuple[np.ndarray, float, np.ndarray, float]:
    """
    Pure TDOA localization (no TOA). For far-field (baseline_m small, range >> baseline)
    uses direction-only LS to avoid collapse; else 3D TDOA least-squares for (x,y,z).
    Returns (position_3d, gdop, uncertainty_3d, confidence).
    """
    n = sensor_positions.shape[0]
    s1 = sensor_positions[ref_index]
    centroid = np.mean(sensor_positions, axis=0)

    non_ref = [i for i in range(n) if i != ref_index]
    tdoa_no_ref = np.array([tdoa[i] for i in non_ref], dtype=float)
    n_res = len(non_ref)

    use_far_field = (
        baseline_m is not None
        and baseline_m > 0
        and baseline_m <= far_field_baseline_threshold
        and n_res >= 2
    )
    if use_far_field:
        # Far-field: fix source height (z_prior). Solve for direction u, then choose range to minimize TDOA residual.
        z_prior = 2.0
        centroid_z = centroid[2]
        c = speed_of_sound
        A = np.array([sensor_positions[i] - s1 for i in non_ref], dtype=float)
        b = speed_of_sound * tdoa_no_ref

        def tdoa_residual_sq(r_prior: float) -> float:
            """Sum of squared TDOA residuals for a given range prior (used to set u_z)."""
            r_prior = max(50.0, min(2000.0, float(r_prior)))
            u_z = (z_prior - centroid_z) / r_prior
            A_xy = A[:, :2]
            b_xy = b - A[:, 2] * u_z
            try:
                u_xy, _, _, _ = np.linalg.lstsq(A_xy, b_xy, rcond=None)
            except Exception:
                return 1e20
            u_vec = np.array([float(u_xy[0]), float(u_xy[1]), u_z])
            norm_u = np.linalg.norm(u_vec)
            if norm_u < 1e-12:
                return 1e20
            u_vec = u_vec / norm_u
            if abs(u_vec[2]) < 1e-9:
                range_actual = r_prior
            else:
                range_actual = (z_prior - centroid_z) / abs(u_vec[2])
            p = centroid - range_actual * u_vec
            p[2] = z_prior
            d_ref = np.linalg.norm(p - s1)
            res_sq = 0.0
            for k, i in enumerate(non_ref):
                d_i = np.linalg.norm(p - sensor_positions[i])
                res_sq += ((d_i - d_ref) - c * tdoa_no_ref[k]) ** 2
            return res_sq

        # Optimize range prior over [150, 1000] m to minimize TDOA residual.
        try:
            opt = minimize_scalar(tdoa_residual_sq, bounds=(150.0, 1000.0), method="bounded")
            range_prior_opt = float(opt.x)
        except Exception:
            range_prior_opt = range_prior
        u_z = (z_prior - centroid_z) / range_prior_opt
        A_xy = A[:, :2]
        b_xy = b - A[:, 2] * u_z
        u_xy, res, _, _ = np.linalg.lstsq(A_xy, b_xy, rcond=None)
        u = np.array([float(u_xy[0]), float(u_xy[1]), u_z])
        norm_u = np.linalg.norm(u)
        if norm_u > 1e-12:
            u = u / norm_u
        else:
            u = np.array([1.0, 0.0, 0.0])
        if abs(u[2]) > 1e-9:
            range_actual = (z_prior - centroid_z) / abs(u[2])
        else:
            range_actual = range_prior_opt
        p = centroid - range_actual * u
        p[2] = z_prior

        # 3D refinement: for moderate range (100–600 m), far-field linearization error matters.
        # Run full nonlinear TDOA solve with x0=p to correct for it. Keep z fixed at z_prior (ground-level) to avoid drift from TDOA noise.
        range_from_centroid = np.linalg.norm(p - centroid)
        if 100.0 <= range_from_centroid <= 600.0:
            c = speed_of_sound

            def res_3d_xy(xy: np.ndarray) -> np.ndarray:
                q = np.array([xy[0], xy[1], z_prior])
                d1 = np.linalg.norm(q - s1)
                r = []
                for k, i in enumerate(non_ref):
                    di = np.linalg.norm(q - sensor_positions[i])
                    r.append((di - d1) - c * tdoa_no_ref[k])
                return np.array(r)

            xy0 = np.array([p[0], p[1]])
            lo = np.array([p[0] - 80, p[1] - 80])
            hi = np.array([p[0] + 80, p[1] + 80])
            try:
                res = least_squares(res_3d_xy, xy0, method="trf", loss="soft_l1", bounds=(lo, hi))
                res_norm_before = np.linalg.norm(res_3d_xy(xy0))
                res_norm_after = np.linalg.norm(res.fun)
                if res_norm_after <= res_norm_before * 1.01:
                    p = np.array([res.x[0], res.x[1], z_prior])
            except Exception:
                pass

        try:
            AtA_xy = A_xy.T @ A_xy
            if np.linalg.det(AtA_xy) > 1e-20:
                C_xy = np.linalg.inv(AtA_xy)
                gdop = float(np.sqrt(np.trace(C_xy)))
                res_norm = np.linalg.norm(A_xy @ u_xy - b_xy)
                sigma2 = (res_norm ** 2) / max(n_res - 2, 1)
                cov_xy = sigma2 * C_xy
                unc_xy = np.sqrt(np.maximum(np.diag(cov_xy), 0.0))
                range_actual = np.linalg.norm(p - centroid)
                uncertainty_3d = np.array([max(0.0, range_actual * unc_xy[0]), max(0.0, range_actual * unc_xy[1]), 0.0])
            else:
                gdop = float("nan")
                uncertainty_3d = np.full(3, float("nan"))
        except Exception:
            gdop = float("nan")
            uncertainty_3d = np.full(3, float("nan"))
        confidence = 1.0 / (1.0 + gdop) if not np.isnan(gdop) and gdop >= 0 else 0.0
        confidence = float(np.clip(confidence, 0.0, 1.0))
        return p, gdop, uncertainty_3d, confidence

    if quality_weights is None:
        quality_weights = np.ones(n_res, dtype=float)
    else:
        quality_weights = np.asarray(quality_weights, dtype=float)
        if len(quality_weights) != n_res:
            quality_weights = np.ones(n_res)
    quality_weights = np.maximum(quality_weights, 1e-12)

    if bounds is None:
        lo = np.array([-2000.0, -2000.0, 0.0])
        hi = np.array([2000.0, 2000.0, 500.0])
        bounds = (lo, hi)

    if x0 is None and len(non_ref) >= 2:
        A = np.array([sensor_positions[i] - s1 for i in non_ref], dtype=float)
        b = speed_of_sound * tdoa_no_ref
        try:
            u, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            norm_u = np.linalg.norm(u)
            if norm_u > 1e-6:
                u = u / norm_u
                x0 = centroid + 350.0 * u
            else:
                x0 = centroid + 350.0 * np.array([1.0, 0.0, 0.0])
        except Exception:
            x0 = centroid + 350.0 * np.array([1.0, 0.0, 0.0])
    elif x0 is None:
        x0 = centroid + 350.0 * np.array([1.0, 0.0, 0.0])
    x0 = np.clip(x0, bounds[0], bounds[1])

    c = speed_of_sound

    def residuals(p: np.ndarray) -> np.ndarray:
        d1 = np.linalg.norm(p - s1)
        res = []
        for k, i in enumerate(non_ref):
            di = np.linalg.norm(p - sensor_positions[i])
            r = (di - d1) - c * tdoa_no_ref[k]
            res.append(r * np.sqrt(quality_weights[k]))
        return np.array(res)

    result = least_squares(residuals, x0, method="trf", loss="soft_l1", bounds=bounds)
    p = result.x
    res_norm = np.linalg.norm(result.fun)

    J = result.jac
    uncertainty_3d = np.full(3, float("nan"))
    gdop = float("nan")
    try:
        JtJ = J.T @ J
        if np.linalg.det(JtJ) > 1e-20:
            C = np.linalg.inv(JtJ)
            gdop = float(np.sqrt(np.trace(C)))
            n_residuals = len(result.fun)
            if n_residuals > 3:
                sigma2 = res_norm ** 2 / max(n_residuals - 3, 1)
                cov = sigma2 * C
                uncertainty_3d = np.sqrt(np.maximum(np.diag(cov), 0.0))
            else:
                uncertainty_3d = np.sqrt(np.maximum(np.diag(C), 0.0))
    except Exception:
        pass

    confidence = 1.0 / (1.0 + gdop) if not np.isnan(gdop) and gdop >= 0 else 0.0
    confidence = float(np.clip(confidence, 0.0, 1.0))

    return p, gdop, uncertainty_3d, confidence
