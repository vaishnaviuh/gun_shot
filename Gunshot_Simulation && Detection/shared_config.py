"""
Shared global parameters for Gunshot_Localization repo.
Change values here to affect both Triangulation_Engine and Gunshot_Simulation && Detection.
Keep pipeline-specific params (sensor geometry, reference lat/lon, file paths) in each project's config.
"""
# Physical / environment constants (single source of truth)
EARTH_RADIUS_M = 6371000.0
TEMPERATURE_C = 25.0   # Celsius – used for speed of sound
ALTITUDE_M = 200.0     # meters – used for speed of sound
SENSOR_MOUNT_HEIGHT_M = 1.0  # meters – sensor mast height above ground


def get_speed_of_sound(temperature_c=None, altitude_m=None):
    """Speed of sound in m/s. Uses module defaults if args are None."""
    if temperature_c is None:
        temperature_c = TEMPERATURE_C
    if altitude_m is None:
        altitude_m = ALTITUDE_M
    return 331.3 + 0.606 * temperature_c - 0.006 * altitude_m / 1000.0
