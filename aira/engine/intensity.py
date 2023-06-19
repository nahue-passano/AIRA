"""Functionality for intensity computation and related signal processing."""

from typing import Tuple

import numpy as np

from aira.engine.filtering import apply_low_pass_filter

FILTER_CUTOFF = 5000


def analysis_crop(
    analysis_length: float,
    sample_rate: int,
    intensity_directions: np.ndarray,
):
    """_summary_

    Parameters
    ----------
    analysis_length : float
        _description_
    sample_rate : int
        _description_
    intensity_directions : np.ndarray
        _description_

    Returns
    -------
    _type_
        _description_
    """
    # Get analysis length max index
    analysis_length_idx = int(analysis_length * sample_rate)

    # Slice from intensity max to analysis length from intensity max
    earliest_peak_index = np.argmax(np.abs(intensity_directions), axis=1).min()
    intensity_directions_cropped = intensity_directions[
        :, earliest_peak_index : earliest_peak_index + analysis_length_idx
    ]

    return intensity_directions_cropped


def intensity_thresholding(
    threshold: float,
    intensity: np.ndarray,
    azimuth: np.ndarray,
    elevation: np.ndarray,
    reflections: np.ndarray,
) -> Tuple[np.ndarray]:
    """_summary_

    Parameters
    ----------
    threshold : _type_
        _description_
    """
    reflex_to_direct = intensity_to_dB(intensity) - intensity_to_dB(intensity[0])
    thresholding_mask = reflex_to_direct > threshold
    return (
        reflex_to_direct[thresholding_mask],
        azimuth[thresholding_mask],
        elevation[thresholding_mask],
        reflections[thresholding_mask],
    )


def intensity_to_dB(intensity_array: np.ndarray) -> np.ndarray:
    """Converts intensity to dB scale using 1e-12 as intensity reference

    Parameters
    ----------
    intensity_array : np.ndarray
        Intensity array

    Returns
    -------
    np.ndarray
        Intensity array in dB scale
    """
    return 20 * np.log10(intensity_array / 1e-12)


def min_max_normalization(array: np.ndarray) -> np.ndarray:
    """Returns the input array normalized by its minimum and maximum value.

    Parameters
    ----------
    array : np.ndarray
        Array to be normalized

    Returns
    -------
    np.ndarray
        Array normalized
    """
    return (array - array.min()) / (array.max() - array.min())


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

    # Padding and windowing
    intensity_directions = np.concatenate(
        [
            intensity_directions,
            np.zeros((3, intensity_directions.shape[1] % duration_samples)),
        ],
        axis=1,
    )
    output_shape = (3, intensity_directions.shape[1] - duration_samples + 1)
    intensity_windowed = np.zeros(output_shape)
    window = np.hamming(duration_samples)
    for i in range(0, output_shape[1]):
        intensity_segment = intensity_directions[:, i : i + duration_samples]
        intensity_windowed[:, i] = (
            np.sum(intensity_segment * window, axis=1) / duration_samples
        )

    return intensity_windowed


def convert_bformat_to_intensity(
    signal: np.ndarray,
    sample_rate: int,
    cutoff_frequency: int = FILTER_CUTOFF,
) -> Tuple[np.ndarray]:
    """Integrate and compute intensities for a B-format Ambisonics recording.

    Args:
        signal (np.ndarray): input B-format Ambisonics signal. Shape: (4, N).
        sample_rate (int): sampling rate of the signal.
        cutoff_frequency (int, optional): cutoff frequency for the low-pass
        filter. Defaults to 5000 Hz.

    Returns:
        Tuple[np.ndarray]: integrated intensity, azimuth and elevation.
    """
    # signal_filtered = apply_low_pass_filter(signal, cutoff_frequency, sample_rate)
    signal_filtered = signal

    # Calculate intensity from directions
    intensity_directions = (
        signal_filtered[0, :] * signal_filtered[1:, :]
    )  # Intensity = pressure (W channel) * pressure gradient (XYZ channels)
    return intensity_directions


def get_intensity_polar_data(intensity_windowed: np.ndarray):
    """_summary_

    Parameters
    ----------
    intensity_windowed : np.ndarray
        _description_

    Returns
    -------
    _type_
        _description_
    """
    # Convert to total intensity, azimuth and elevation
    intensity = np.sqrt((intensity_windowed**2).sum(axis=0)).squeeze()
    azimuth = np.rad2deg(
        np.arctan(intensity_windowed[1] / intensity_windowed[0])
    ).squeeze()
    elevation = np.rad2deg(np.arccos(intensity_windowed[2] / intensity)).squeeze()
    return intensity, azimuth, elevation
