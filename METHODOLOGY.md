# Gunshot Triangulation — Methodology

This document describes how the gunshot localization pipeline works, from audio input to estimated source positions.

---

## 1. Overview

The project performs **multi-shot gunshot localization** using **Time-Difference-of-Arrival (TDOA)** from a multi-channel microphone array. The pipeline consists of:

1. **Simulation** (optional) — Generate synthetic gunshot audio for given source positions and times  
2. **Detection** — Detect gunshot events in multi-channel audio  
3. **Chunk extraction** — Extract short time windows around each event for TDOA analysis  
4. **TDOA estimation** — GCC-PHAT cross-correlation to measure arrival time differences  
5. **Localization** — Multilateration from TDOA to estimate (x, y, z) source positions  
6. **Reporting** — Combined PDF report with detection plots and triangulation results  

**Important:** The system uses **TDOA only** for localization. No Time-of-Arrival (TOA) or trajectory logs are used — suitable for real-world recordings where ground truth is unknown.

---

## 2. Sensor Array Geometry

- **Layout:** Equilateral triangle + center (4 sensors)
- **Coordinate system:** East–North–Up (ENU), origin at triangle centroid
- **Baseline:** Configurable (default 30 m) — triangle side length
- **Positions (30 m baseline):**
  - S1: (0, −17.32, 1) m (south)
  - S2: (15, 8.66, 1) m (northeast)
  - S3: (−15, 8.66, 1) m (northwest)
  - S4: (0, 0, 2) m (center, elevated)

Source positions (300–500 m) are expressed in the same ENU frame.

---

## 3. Gunshot Simulation

**Module:** `Gunshot_Simulation && Detection/gunshot_simulation/gunshot_simulation.py`

### 3.1 Model

- **Broadband synthesis:** Multi-tone signal 10–7990 Hz with realistic frequency and time decay
- **Arrival model:** For each gunshot and each sensor:
  - TOA = gunshot_time + distance / speed_of_sound
  - Distance attenuation: quadratic near field, linear far field (ref 100 m)
  - Amplitude ∝ exp(−f/f_decay) × exp(−t/t_decay)
- **Noise:** Shot-synchronous noise (~40 dB SNR), low background ambient

### 3.2 Output

- Combined WAV file: all gunshots summed in a single multi-channel recording
- Trajectory .log (for reference only — **not used for localization**)

---

## 4. Gunshot Detection

**Module:** `Gunshot_Simulation && Detection/gunshot_detection/gunshot_detection.py`

### 4.1 Preprocessing

- **Bandpass filter:** 500–8000 Hz (Butterworth, order 4)
- **Per-channel normalization**

### 4.2 Onset-Based Detection Logic

| Stage | Description |
|-------|-------------|
| **Short-Time Energy (STE)** | Frame 20 ms, hop 5 ms; E[n] = Σ(frame²) |
| **Derivative** | Forward difference of STE — identifies rising edge (onset) |
| **Threshold** | Adaptive: noise floor from first 0.5 s × 6 |
| **CUSUM** | Change detection on energy; threshold h = 1.2 |
| **Combined logic** | Event when: STE > threshold **and** derivative > 0 **and** CUSUM > h |

### 4.3 Onset Timing

Instead of peak time, the system uses `find_onset_index()`: the first frame where energy crosses above threshold (rising edge). This aligns the extracted chunk with the direct arrival rather than reverberation.

### 4.4 Multi-Channel Clustering

- Detections are clustered across channels using a time tolerance (e.g. 120 ms)
- Event is valid only if ≥ 2 channels have detections within tolerance
- Clusters within 0.4 s are merged (same physical event)
- **Event time:** Uses `event_time_earliest` (minimum arrival across channels) for chunk placement to ensure all sensor arrivals fall inside the TDOA window

### 4.5 Chunk Extraction

- Window: 100 ms total (e.g. ±50 ms around event), large enough for max TDOA (~90 ms for 30 m baseline)
- Center: `t_earliest + max_tdoa/2` so both earliest and latest arrivals are inside the window

---

## 5. Post-Detection Deduplication (Pipeline)

**Module:** `triangulation/pipeline.py`

Reverberation can produce multiple detections per shot. Two steps reduce this:

1. **Merge by time:** Events within 0.45 s are merged; only the **earliest** (direct arrival) is kept.
2. **Evenly spaced subsampling:** If detected events > known positions, keep N events with the most even temporal spacing (e.g. indices 0, 2, 4 for 5 events → 3 positions).

---

## 6. TDOA Estimation (GCC-PHAT)

**Module:** `triangulation/tdoa.py`

### 6.1 Method

- **GCC-PHAT:** Generalized Cross-Correlation with Phase Transform
- Reference channel: Sensor 1 (index 0)
- For each channel i: delay = argmax of PHAT-weighted cross-correlation
- **Sub-sample refinement:** Parabolic interpolation around peak
- **Max lag:** `baseline_m / speed_of_sound × 1.1` (physical limit with margin)
- **Early energy weighting:** Optional exponential decay (τ=200 ms) on chunk to suppress late reflections
- **First-significant-peak option:** Can select peak closest to zero lag to favor direct path

### 6.2 Validation

- TDOA values checked to lie within ±100 ms (for 30 m baseline, max ≈ 88 ms)
- **Sign consistency:** After localization, measured TDOA signs are checked against geometry; rejection triggers retry

### 6.3 Reverberation Mitigation

When uncertainty > 50 m or sign validation fails:
1. Retry with first-significant-peak GCC
2. Retry with earlier chunk centers (t_earliest − 0.2 s, −0.15 s, −0.10 s) to capture direct path when detection latched onto reverb

---

## 7. Localization

**Module:** `triangulation/tdoa.py` — `estimate_position_with_gdop()`

### 7.1 Far-Field Mode (baseline ≤ 100 m)

For small arrays and distant sources (100–600 m):

1. **Direction estimate:** Linear LS on TDOA → direction vector u
2. **Range:** Search over 150–1000 m to minimize TDOA residual (fix height z = 2 m)
3. **Position:** p = centroid − range × u

### 7.2 Nonlinear Refinement

For 100–600 m range:

- Full nonlinear least-squares on (x, y) with z fixed
- Initial guess from far-field; bounds ±80 m from initial
- Soft-L1 loss for outlier robustness

### 7.3 Outputs

- **Position:** (x, y, z) in ENU meters
- **GDOP:** Geometric Dilution of Precision (lower = better geometry)
- **Uncertainty:** 3D uncertainty ellipse
- **Confidence:** 1/(1 + GDOP), clipped to [0, 1]

---

## 8. Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         run_pipeline.py (CLI)                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
    ┌──────────────────────────────────┼──────────────────────────────────┐
    ▼                                  ▼                                  ▼
┌──────────────┐            ┌──────────────────────┐            ┌─────────────────┐
│ Option A:    │            │ Option B:             │            │ Load WAV only   │
│ Simulate     │            │ Use existing WAV     │            │ (--wav-file)    │
│ (positions,  │            │ from data_dir        │            │ GT from filename │
│  shot_times) │            │ (skip_simulation)     │            │                  │
└──────┬───────┘            └──────────┬───────────┘            └────────┬────────┘
       │                               │                                  │
       └───────────────────────────────┼──────────────────────────────────┘
                                      ▼
                         ┌────────────────────────┐
                         │ load_audio (bandpass    │
                         │ 1000–24000 Hz)          │
                         └────────────┬───────────┘
                                      │
                                      ▼
                         ┌────────────────────────┐
                         │ detect_gunshots()      │
                         │ • STE + derivative     │
                         │ • CUSUM, clustering    │
                         │ • Per-event chunks     │
                         └────────────┬───────────┘
                                      │
                                      ▼
                         ┌────────────────────────┐
                         │ _deduplicate_events_    │
                         │ keep_earliest()        │
                         │ + evenly-spaced subsample│
                         └────────────┬───────────┘
                                      │
                                      ▼
                         ┌────────────────────────┐
                         │ For each event:       │
                         │ extract_window() →    │
                         │ compute_tdoa(GCC-PHAT) │
                         │ → estimate_position   │
                         └────────────┬───────────┘
                                      │
                                      ▼
                         ┌────────────────────────┐
                         │ generate_report()      │
                         │ PDF: detection + 2D +   │
                         │ 3D + waveforms         │
                         └────────────────────────┘
```

---

## 9. Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| **Detection** | | |
| frame_size_ms | 20 | STE frame length |
| hop_size_ms | 5 | STE hop |
| noise_floor_multiplier | 6 | Threshold = noise × 6 |
| time_tolerance_ms | 120 | Cluster tolerance across channels |
| chunk_window_ms | 100 | Total TDOA chunk length |
| min_channels | 2 | Minimum channels for valid event |
| **Pipeline** | | |
| merge_gap_s | 0.45 | Merge events within this gap |
| baseline_m | 30 | Array triangle side length |
| range_prior | 350 | Default range (m) for far-field |
| **Localization** | | |
| refinement_range | 100–600 m | Nonlinear refinement applied in this range |

---

## 10. Running the Pipeline

```bash
# Simulate 3 shots at given positions and times, then localize
python triangulation/run_pipeline.py \
  --positions "300 250 2 350 280 2 390 300 2" \
  --time 1.2 2.6 3.8 \
  --duration 6

# Use existing WAV (no simulation)
python triangulation/run_pipeline.py \
  --positions "300 250 2 350 280 2 390 300 2" \
  --time 1.2 2.6 3.8 \
  --duration 6 \
  --skip-simulation

# Localize from WAV only (positions from filename)
python triangulation/run_pipeline.py --wav-file "path/to/combined3002502_3502802_3903002.wav"
```

Output: `triangulation/plots/report_<stem>.pdf`

---

## 11. File Structure

```
gun_triangulation/
├── triangulation/
│   ├── pipeline.py         # Main pipeline, run_pipeline()
│   ├── tdoa.py             # GCC-PHAT, extract_window, estimate_position_with_gdop
│   ├── utils.py            # load_audio, get_sensor_positions_enu, speed_of_sound
│   ├── run_pipeline.py     # CLI entry point
│   └── plots/              # PDF reports, figures
├── Gunshot_Simulation && Detection/
│   ├── gunshot_simulation/
│   │   ├── gunshot_simulation.py   # generate_gunshot_audio, simulate_gunshots
│   │   └── data/                   # WAV files
│   ├── gunshot_detection/
│   │   └── gunshot_detection.py    # detect_gunshots
│   ├── config.py           # Sensor geometry
│   └── shared_config.py    # Speed of sound, etc.
└── METHODOLOGY.md          # This file
```
