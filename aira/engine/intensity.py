"""Functionality for intensity computation and related signal processing."""

from typing import Tuple

import numpy as np

from aira.engine.filtering import apply_low_pass_filter

FILTER_CUTOFF = 5000


def integrate_intensity_directions(
    intensity_directions: np.ndarray,
    duration_secs: float,
    sample_rate: int,
) -> np.ndarray:
    """Integrate the intensity signal with Hamming windows of length `duration_secs`.

    Args:
        intensity_directions (np.ndarray): X, Y and Z intensity signals.
        duration_secs (float): the length of the window to apply, in seconds.
        sample_rate (int): sampling rate of the signal.

    Returns:
        np.ndarray: the integrated signal, of shape (3, ...)
    """
    if intensity_directions.shape[0] == 4:
        intensity_directions = intensity_directions[1:, :]
    elif (intensity_directions.shape[0] < 3) or (intensity_directions.shape[0] > 4):
        raise ValueError(f"Unexpected input shape {intensity_directions.shape}")

    # Convert integration time to samples
    duration_samples = np.round(duration_secs * sample_rate).astype(np.int64)
    duration_samples = duration_samples.astype(np.int64)

    # Peak scanning: keep only from the leftmost highest peak to the end of the signal
    earliest_peak_index = np.argmax(np.abs(intensity_directions), axis=1).min()
    intensity_directions = intensity_directions[:, earliest_peak_index:]

    # Padding and windowing
    intensity_directions = np.concatenate(
        [
            intensity_directions,
            np.zeros((3, intensity_directions.shape[1] % duration_samples)),
        ],
        axis=1,
    )
    output_shape = (3, intensity_directions.shape[1] // duration_samples)
    intensity_windowed = np.zeros(output_shape)
    window = np.hamming(duration_samples)
    for i in range(0, output_shape[1]):
        intensity_segment = intensity_directions[:, i : i + duration_samples]
        intensity_windowed[:, i] = np.max(intensity_segment * window, axis=1)

    return intensity_windowed


def convert_bformat_to_intensity(
    signal: np.ndarray,
    sample_rate: int,
    integration_time: int,
    cutoff_frequency: int = FILTER_CUTOFF,
) -> Tuple[np.ndarray]:
    """Integrate and compute intensities for a B-format Ambisonics recording.

    Args:
        signal (np.ndarray): input B-format Ambisonics signal. Shape: (4, N).
        sample_rate (int): sampling rate of the signal.
        integration_time (int): integration time to apply, in seconds.
        cutoff_frequency (int, optional): cutoff frequency for the low-pass
        filter. Defaults to 5000 Hz.

    Returns:
        Tuple[np.ndarray]: integrated intensity, azimuth and elevation.
    """
    signal_filtered = apply_low_pass_filter(signal, cutoff_frequency, sample_rate)

    # Calculate intensity from directions
    intensity_directions = (
        signal_filtered[0, :] * signal_filtered[1:, :]
    )  # Intensity = pressure (W channel) * pressure gradient (XYZ channels)
    intensity_windowed = integrate_intensity_directions(
        intensity_directions,
        duration_secs=integration_time,
        sample_rate=sample_rate,
    )

    # Convert to total intensity, azimuth and elevation
    intensity = np.sqrt((intensity_windowed**2).sum(axis=0)).squeeze()
    azimuth = np.rad2deg(
        np.arctan(intensity_windowed[1] / intensity_windowed[0])
    ).squeeze()
    elevation = np.rad2deg(np.arctan(intensity_windowed[2] / intensity)).squeeze()

    return intensity, azimuth, elevation
