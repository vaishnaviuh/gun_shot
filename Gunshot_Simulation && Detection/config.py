"""
Parameters for gunshot detection and triangulation.
Imports shared global params from repo root; defines deployment-specific params here.
"""
import sys
from pathlib import Path
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
from shared_config import (
    EARTH_RADIUS_M,
    TEMPERATURE_C,
    ALTITUDE_M,
    SENSOR_MOUNT_HEIGHT_M,
    get_speed_of_sound,
)

import math
import numpy as np

# Reference for the center of the tetrahedron base (set to your deployment location).
REFERENCE_LAT_DEG =  12.78403027111697
REFERENCE_LON_DEG = 77.65117318375009
REFERENCE_HEIGHT_M = 200.0

# The sensors are arranged as an equilateral triangle (S1-S3) with S4 at the centroid.
# Triangle side length: 150m
# S1, S2, S3: vertices at 1m altitude
# S4: centroid at 2m altitude
# Center of triangle (centroid) is at (REFERENCE_LAT_DEG, REFERENCE_LON_DEG, REFERENCE_HEIGHT_M).
# Coordinates below are (sensor_id, lat_deg, lon_deg, height_m) in ENU from that reference.

def _enu_to_geo(x_east, y_north, z_up):
    lat = REFERENCE_LAT_DEG + math.degrees(y_north / EARTH_RADIUS_M)
    lon = REFERENCE_LON_DEG + math.degrees(
        x_east / (EARTH_RADIUS_M * math.cos(math.radians(REFERENCE_LAT_DEG)))
    )
    return lat, lon, REFERENCE_HEIGHT_M + z_up

# Equilateral triangle configuration (baseline = side length)
# Use get_sensor_positions(baseline_m) for dynamic baseline (e.g. 30 m).

def get_sensor_positions_enu(baseline_m: float = 150.0):
    """
    Return sensor positions in ENU meters with origin at triangle center.
    Same geometry as get_sensor_positions: 3 vertices at 1 m, S4 at center at 2 m.
    Use this for simulation and localization so both use the same coordinate system.
    Returns (positions_xyx_m, sensor_ids): positions shape (4, 3), ids [1,2,3,4].
    """
    r = baseline_m * math.sqrt(3) / 3  # distance from centroid to each vertex
    S1 = (0.0, -r, 1.0)
    S2 = (r * math.sqrt(3) / 2, r / 2, 1.0)
    S3 = (-r * math.sqrt(3) / 2, r / 2, 1.0)
    S4 = (0.0, 0.0, 2.0)
    positions = np.array([S1, S2, S3, S4], dtype=float)
    return positions, [1, 2, 3, 4]


def get_sensor_positions(baseline_m: float = 150.0):
    """
    Return sensor coords for given baseline (triangle side length in meters).
    3 sensors at triangle vertices (1 m altitude), 1 at center (2 m altitude).
    Returns list of (sensor_id, lat_deg, lon_deg, height_m).
    """
    r = baseline_m * math.sqrt(3) / 3  # distance from centroid to each vertex
    S1 = (0.0, -r, 1.0)
    S2 = (r * math.sqrt(3) / 2, r / 2, 1.0)
    S3 = (-r * math.sqrt(3) / 2, r / 2, 1.0)
    S4 = (0.0, 0.0, 2.0)
    return [
        (1, *_enu_to_geo(*S1)),
        (2, *_enu_to_geo(*S2)),
        (3, *_enu_to_geo(*S3)),
        (4, *_enu_to_geo(*S4)),
    ]


# Default 150 m baseline (backward compatible)
_triangle_side = 150.0
SENSOR_COORDS = get_sensor_positions(_triangle_side)
