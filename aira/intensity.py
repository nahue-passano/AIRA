from typing import Tuple

import numpy as np
from scipy.signal import kaiserord, lfilter, firwin


FILTER_CUTOFF = 5000
FILTER_TRANSITION_WIDTH_HZ = 250.0
FILTER_RIPPLE_DB = 60.0


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
    elif (len(y_intensity) < len(x_intensity)) and (
        len(y_intensity) < len(z_intensity)
    ):
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


def integrate_bformat(
    intensity: np.ndarray,
    duration_secs: float,
    sample_rate: int,
) -> np.ndarray:
    """Integrate the intensity signal with Hamming windows of length `duration_secs`.

    Args:
        intensity (np.ndarray): X, Y and Z intensity signals.
        duration_secs (float): the length of the window to apply, in seconds.
        sample_rate (int): sampling rate of the signal.

    Returns:
        np.ndarray: the integrated signal, of shape (3, ...)
    """
    if intensity.shape[0] == 4:
        intensity = intensity[1:, :]
    elif (intensity.shape[0] < 3) or (intensity.shape[0] > 4):
        raise ValueError(f"Unexpected input shape {intensity.shape}")

    # Convert integration time to samples
    duration_samples = np.round(duration_secs * sample_rate).astype(np.int64)
    duration_samples = duration_samples.astype(np.int64)

    # Peak scanning: keep only from the highest peak to the end of the signal
    earliest_peak_index = np.argmax(intensity, axis=1).min()
    intensity = intensity[:, earliest_peak_index:]

    # Windowing
    intensity = np.concatenate(
        [intensity, np.zeros((3, intensity.shape[1] % duration_samples))],
        axis=1,
    )  # Padding
    output_shape = (3, intensity.shape[1] // duration_samples)
    intensity_windowed = np.zeros(output_shape)
    window = np.hamming(duration_samples)
    for i in range(0, output_shape[1]):
        intensity_segment = intensity[:, i : i + duration_samples]
        intensity_windowed[:, i] = np.sum(
            intensity_segment * window,
            axis=1
        )  # TODO: verify that simple addition is ok. Might need to be energy...

    return intensity_windowed


def convert_bformat_channels_to_intensity(
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


def apply_low_pass_filter(
    signal: np.ndarray, cutoff_frequency: int, sample_rate: int
) -> np.ndarray:
    nyquist_rate = sample_rate / 2.0

    # Compute FIR filter parameters and apply to signal.
    transition_width_normalized = FILTER_TRANSITION_WIDTH_HZ / nyquist_rate
    filter_length, filter_beta = kaiserord(
        FILTER_RIPPLE_DB, transition_width_normalized
    )
    filter_coefficients = firwin(
        filter_length, cutoff_frequency / nyquist_rate, window=("kaiser", filter_beta)
    )

    return lfilter(filter_coefficients, 1.0, signal)


def convert_bformat_to_intensity(
    signal: np.ndarray,
    sample_rate: int,
    integration_time: int,
    cutoff_frequency: int = FILTER_CUTOFF,
) -> Tuple[np.ndarray]:
    """Integrate and compute intensities for a B-format Ambisonics recording.

    Args:
        signal (np.ndarray): input B-format Ambisonics signal.
        sample_rate (int): sampling rate of the signal.
        integration_time (int): integration time to apply, in seconds.
        cutoff_frequency (int, optional): cutoff frequency for the low-pass filter. Defaults to 5000 Hz.

    Returns:
        Tuple[np.ndarray]: integrated intensity, azimuth and elevation.
    """
    signal_filtered = apply_low_pass_filter(signal, cutoff_frequency, sample_rate)

    # Calculate intensity from directions
    signal_filtered[1:, :] = (
        signal_filtered[0, :] * signal_filtered[1:, :]
    )  # W channel * Other channels
    # TODO: shouldn't it be a correlation

    intensity_windowed = integrate_bformat(
        signal_filtered[1:, :],  # drop W
        duration_secs=integration_time,
        sample_rate=sample_rate,
    )

    # Convert to total intensity, azimuth and elevation
    intensity = np.sqrt((intensity_windowed**2).sum(axis=0)).squeeze()
    azimuth = np.rad2deg(np.arctan(intensity_windowed[1] / intensity_windowed[0]))
    elevation = np.rad2deg(np.arctan(intensity_windowed[2] / intensity))

    return intensity, azimuth, elevation
