import numpy as np


def hamming_window(
    x_intensity: np.ndarray,
    y_intensity: np.ndarray,
    z_intensity: np.ndarray,
    duration_secs: float,
    sample_rate: int,
):
    # Tiempo de integración en ms (pasaje a muestras?)
    duration_samples = np.round(duration_secs * sample_rate, dtype=np.int64)
    duration_samples = duration_samples.astype(np.int64)

    # Peak scanning: keep only from the highest peak to the end of the signal
    x_intensity = x_intensity[np.argmax(x_intensity) :]
    y_intensity = y_intensity[np.argmax(y_intensity) :]
    z_intensity = z_intensity[np.argmax(z_intensity) :]
    
    # Truncate to the shortest signal
    if (len(x_intensity) < len(y_intensity)) and (len(x_intensity) < len(z_intensity)):
        y_intensity, z_intensity = (
            y_intensity[: len(x_intensity)],
            z_intensity[: len(x_intensity)],
        )
    elif len(y_intensity) < len(x_intensity) and len(y_intensity) < len(z_intensity):
        x_intensity, z_intensity = (
            x_intensity[: len(y_intensity)],
            z_intensity[: len(y_intensity)],
        )
    else:
        x_intensity, y_intensity = (
            x_intensity[: len(z_intensity)],
            y_intensity[: len(z_intensity)],
        )

    # Ventaneo y suma de amplitudes
    Ix_vent = []
    for i in range(0, len(x_intensity) - duration_samples):
        Ix_sep = x_intensity[i : i + duration_samples]
        ham = np.hamming(len(Ix_sep))
        Ix_ham = Ix_sep * ham
        Ix_ham = sum(Ix_ham)
        Ix_vent.append(Ix_ham)
    Ix_vent = np.array(Ix_vent)

    Iy_vent = []
    for i in range(0, len(y_intensity) - duration_samples):
        Iy_sep = y_intensity[i : i + duration_samples]
        ham = np.hamming(len(Iy_sep))
        Iy_ham = Iy_sep * ham
        Iy_ham = sum(Iy_ham)
        Iy_vent.append(Iy_ham)
    Iy_vent = np.array(Iy_vent)

    Iz_vent = []
    for i in range(0, len(z_intensity) - duration_samples):
        Iz_sep = z_intensity[i : i + duration_samples]
        ham = np.hamming(len(Iz_sep))
        Iz_ham = Iz_sep * ham
        Iz_ham = sum(Iz_ham)
        Iz_vent.append(Iz_ham)
    Iz_vent = np.array(Iz_vent)

    return Ix_vent, Iy_vent, Iz_vent


def convert_bformat_to_intensity(
    w_channel: np.ndarray,
    x_channel: np.ndarray,
    y_channel: np.ndarray,
    z_channel: np.ndarray,
    sample_rate: int,
    integration_time: int,
    cutoff_frequency: int = 5000,
):
    w_filtered, x_filtered, y_filtered, z_filtered = filter(
        cutoff_frequency, w_channel, x_channel, y_channel, z_channel
    )

    # Calculate intensity from directions
    intensity_x = w_filtered * x_filtered
    intensity_y = w_filtered * y_filtered
    intensity_z = w_filtered * z_filtered
    intensity_x_windowed, intensity_y_windowed, intensity_z_windowed = hamming_window(
        intensity_x,
        intensity_y,
        intensity_z,
        duration_secs=integration_time,
        sample_rate=sample_rate,
    )

    # Convert to total intensity, azimuth and elevation
    intensity = np.sqrt(
        intensity_x_windowed**2
        + intensity_y_windowed**2
        + intensity_z_windowed**2
    )
    azimuth = np.rad2deg(np.arctan(intensity_y_windowed / intensity_x_windowed))
    elevation = np.rad2deg(np.arctan(intensity_z_windowed / intensity))

    return intensity, azimuth, elevation
