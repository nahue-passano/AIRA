"""Functionality for filtering signals."""

import numpy as np
import scipy.signal as sc
from scipy.signal import bilinear, firwin, kaiserord, lfilter

MIC2CENTER = 3
SOUND_SPEED = 340
FILTER_TRANSITION_WIDTH_HZ = 250.0
FILTER_RIPPLE_DB = 60.0


class NonCoincidentMicsCorrection:
    """Class for correct frequency response in Ambisonics B-format representation."""

    def __init__(
        self,
        sample_rate: int,
        mic2center: float = MIC2CENTER,
        sound_speed: float = SOUND_SPEED,
    ) -> None:
        self.sample_rate = sample_rate
        self.mic2center = mic2center / 100
        self.sound_speed = sound_speed
        self.delay2center = self.mic2center / self.sound_speed

    def _filter(
        # pylint: disable=invalid-name
        self,
        b: np.ndarray,
        a: np.ndarray,
        array: np.ndarray,
    ) -> np.ndarray:
        """Applies filter to array given numerator "b" and denominator "a" from
        analog filter frequency response

        Parameters
        ----------
        b : np.ndarray
            Array containing numerator's coefficients
        a : np.ndarray
            Array containing denominator's coefficients
        array : np.ndarray
            Array to be filtered

        Returns
        -------
        np.ndarray
            Filtered array
        """

        # Analog to digital filter conversion
        zeros, poles = bilinear(b, a, self.sample_rate)

        # Filtering
        array_filtered = lfilter(zeros, poles, array)

        return array_filtered

    def correct_axis(self, axis_signal: np.ndarray) -> np.ndarray:
        """Applies correction filter to axis array signal

        Parameters
        ----------
        axis_signal : np.ndarray
            Array containing axis signal

        Returns
        -------
        np.ndarray
            Axis array signal corrected
        """
        # Filter equations
        # pylint: disable=invalid-name
        b = np.sqrt(6) * np.array(
            [1, 1j * (1 / 3) * self.mic2center, -(1 / 3) * self.delay2center**2]
        )
        # pylint: disable=invalid-name
        a = np.array([1, 1j * (1 / 3) * self.delay2center])

        axis_corrected = self._filter(b, a, axis_signal)

        return axis_corrected

    def correct_omni(self, omni_signal: np.ndarray) -> np.ndarray:
        """Applies correction filter to omnidirectional array signal

        Parameters
        ----------
        omni_signal : np.ndarray
            Array containing omnidirectional signal

        Returns
        -------
        np.ndarray
            Omnidirectional array signal corrected
        """
        # Filter equations
        # pylint: disable=invalid-name
        b = np.array([1, 1j * self.delay2center, -(1 / 3) * self.delay2center**2])
        # pylint: disable=invalid-name
        a = np.array([1, 1j * (1 / 3) * self.delay2center])

        omni_corrected = self._filter(b, a, omni_signal)

        return omni_corrected


def apply_low_pass_filter(
    signal: np.ndarray, cutoff_frequency: int, sample_rate: int
) -> np.ndarray:
    """Filter a signal at the given cutoff with an optimized number of taps
    (order of the filter).

    Args:
        signal (np.ndarray): signal to filter.
        cutoff_frequency (int): cutoff frequency.
        sample_rate (int): sample rate of the signal.

    Returns:
        np.ndarray: filtered signal.
    """
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


def convolve(signal_1: np.ndarray, signal_2: np.ndarray) -> np.ndarray:
    """Applies convolution with scipy.signal.fftconvolve() function

    Parameters
    ----------
    signal_1 : np.ndarray
        First signal to be convolved
    signal_2 : np.ndarray
        Second signal to be convolved

    Returns
    -------
    np.ndarray
        Convolved signal
    """
    return sc.fftconvolve(signal_1, signal_2, mode="valid")
