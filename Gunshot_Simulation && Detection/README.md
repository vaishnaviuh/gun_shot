# Gunshot Simulation - Usage Guide

This project generates realistic simulated gunshot audio for testing and validation.

## Project Structure

```
├── gunshot_simulation/         # Simulation module
│   ├── __init__.py
│   ├── gunshot_simulation.py   # Main simulation module
│   ├── run_simulation.py       # Command-line script
│   └── data/                   # Output directory
├── gunshot_detection/          # Detection module
│   ├── __init__.py
│   ├── gunshot_detection.py    # Main detection module
│   └── run_detection.py        # Command-line script
├── plots/                      # Detection report PDFs
├── src/                        # Common source code
│   └── visualisation/          # Plotting utilities
├── config.py                   # Configuration (sensor positions, etc.)
├── shared_config.py            # Physics constants (speed of sound, etc.)
├── requirements.txt            # Python dependencies
├── init.sh                     # Setup script
└── README.md                   # This file
```

## Usage

### Gunshot Simulation

#### Command Line:
```bash
# Generate simulated gunshot at position (101, 175, 289) meters (from project root)
python gunshot_simulation/run_simulation.py --position 101 175 289 --time 0.1

# Or from simulation folder
cd gunshot_simulation
python run_simulation.py --position 101 175 289 --time 0.1

# Custom output files and duration
python gunshot_simulation/run_simulation.py --position 101 175 289 --time 0.1 \
    --output-wav ./data/my_simulation.wav \
    --trajectory ./data/my_simulation.log \
    --duration 2.0 \
    --sample-rate 16000

# Disable plots
python gunshot_simulation/run_simulation.py --position 101 175 289 --time 0.1 --no-plots

# Output named by position (saved to gunshot_simulation/data/)
python gunshot_simulation/run_simulation.py --position 101 175 289 --time 0.1 --name-by-position --no-plots
```

### Gunshot Detection

#### Command Line:
```bash
# Run detection on a WAV file (from project root)
python gunshot_detection/run_detection.py path/to/audio.wav

# Custom PDF output
python gunshot_detection/run_detection.py path/to/audio.wav --output-pdf plots/report.pdf

# Disable kurtosis check, adjust thresholds
python gunshot_detection/run_detection.py path/to/audio.wav --no-kurtosis --k 3 --cusum-h 2

# Disable plots
python gunshot_detection/run_detection.py path/to/audio.wav --no-plots
```

#### Python API:
```python
import sys
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from gunshot_simulation.gunshot_simulation import simulate_gunshots

gunshot_positions = np.array([
    [101.0, 175.0, 289.0],  # x, y, z in meters
    [200.0, 300.0, 150.0]   # Multiple gunshots supported
])
gunshot_times = [0.1, 0.5]  # Times in seconds

results = simulate_gunshots(
    gunshot_positions=gunshot_positions,
    gunshot_times=gunshot_times,
    output_wav_file="./data/simulated.wav",
    trajectory_file="./data/simulated.log",
    duration_s=2.0,
    sample_rate=16000,
    plot_spectrogram=True,
    plot_2d=True,
    plot_3d=True
)

# Access results
audio_data = results['audio_data']
gunshot_events = results['gunshot_events']
sensor_positions = results['sensor_positions']
```

## Available Functions

### Simulation Module (`gunshot_simulation/gunshot_simulation.py`)

- `simulate_gunshots()` - Main simulation function
- `generate_gunshot_audio()` - Generate audio signals
- `load_sensor_positions()` - Load sensor geometry
- `save_audio_and_trajectory()` - Save output files
- `plot_spectrogram()` - Visualize generated audio
- `plot_positions_2d()` - 2D position plot
- `plot_positions_3d()` - 3D sensor geometry plot

### Detection Module (`gunshot_detection/gunshot_detection.py`)

- `detect_gunshots()` - Main detection function
- `bandpass_filter()` - Bandpass filter (50–5000 Hz)
- `compute_energy()` - Short-time energy
- `compute_cusum()` - CUSUM validation
- `detect_peaks()` - Energy peak detection
- `load_audio()` - Load WAV file
- `plot_results()` - Generate PDF report

## Parameters

### Simulation Parameters

- `f_min_hz`, `f_max_hz`: Frequency range (default: 10.0–7990.0 Hz)
- `f_decay`: Frequency decay coefficient (default: 4000.0)
- `t_decay`: Time decay coefficient (default: 0.5)
- `background_noise_std`: Noise level (default: 4e-5)
- `temperature_c`, `altitude_m`: For speed of sound calculation

## Example

### Generate test data
```bash
python gunshot_simulation/run_simulation.py --position 101 175 289 --time 0.1
```
