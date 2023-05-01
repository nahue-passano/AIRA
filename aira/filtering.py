import numpy as np
import scipy.signal as sc

SOUND_SPEED = 340


def generate_log_sine_sweep(
    sample_rate: int, duration: float, frequency_min: float, frequency_max: float
) -> np.ndarray:
    """Generates Log-Sinesweep given minimum and maximum frequencies

    Parameters
    ----------
    sample_rate : int
        Sample rate of the Log-Sinesweep
    duration : int
        Lenght in seconds of the Log-Sinesweep
    frequency_min : float
        Minimum frequency at time=0
    frequency_max : float
        Maximum frequency at time=duration

    Returns
    -------
    np.ndarray
        Array with the Log-Sinesweep
    """

    # Time array
    time = np.arange(0, duration, 1 / sample_rate)

    log_sine_sweep = sc.chirp(
        t=time, f0=frequency_min, t1=duration, f1=frequency_max, method="logarithmic"
    )

    return log_sine_sweep


def generate_lss_inverse_filter(
    log_sine_sweep: np.ndarray,
    sample_rate: int,
    frequency_min: float,
    frequency_max: float,
) -> np.ndarray:
    """Generates the inverse filter given a Log-Sinesweep in time domain

    Parameters
    ----------
    log_sine_sweep : np.ndarray
        Array containing Log-Sinesweep
    sample_rate : int
        Sample rate of the Log-Sinesweep
    frequency_min : float
        Minimum frequency of the Log-Sinesweep
    frequency_max : float
        Maximum frequency of the Log-Sinesweep

    Returns
    -------
    np.ndarray
        Inverse filter of the Log-Sinesweep in time domain
    """

    # Time array
    time = np.arange(len(log_sine_sweep)) / sample_rate
    duration = time[-1]

    # Generate inverse filter from log sine sweep
    inv_lss = log_sine_sweep[::-1]
    modulation = 1 / (
        2 * np.pi * np.exp(time * np.log(frequency_max / frequency_min) / duration)
    )
    inv_lss_filter = inv_lss * modulation
    inv_lss_filter /= abs(inv_lss_filter).max()

    return inv_lss_filter


class NonCoincidentMicsCorrection:
    """Class for correct frequency response in Ambisonics B-format representation."""

    def __init__(
        self, mic2center: float, sample_rate: int, sound_speed: float = SOUND_SPEED
    ) -> None:
        self.mic2center = mic2center / 100
        self.sample_rate = sample_rate
        self.sound_speed = sound_speed
        self.delay2center = self.mic2center / self.sound_speed

    def _filter(self, b: np.ndarray, a: np.ndarray, array: np.ndarray) -> np.ndarray:
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
        zeros, poles = sc.bilinear(b, a, self.sample_rate)

        # Filtering
        array_filtered = sc.lfilter(zeros, poles, array)

        return array_filtered

    def correct_axis(self, axis_array: np.ndarray) -> np.ndarray:
        """Applies correction filter to axis array signal

        Parameters
        ----------
        axis_array : np.ndarray
            Array containing axis signal

        Returns
        -------
        np.ndarray
            Axis array signal corrected
        """
        # Filter equations
        b = np.sqrt(6) * np.array(
            [1, 1j * (1 / 3) * self.mic2center, -(1 / 3) * self.delay2center**2]
        )
        a = np.array([1, 1j * (1 / 3) * self.delay2center])

        axis_corrected = self._filter(b, a, axis_array)

        return axis_corrected

    def correct_omni(self, omni_array: np.ndarray) -> np.ndarray:
        """Applies correction filter to omnidirectional array signal

        Parameters
        ----------
        omni_array : np.ndarray
            Array containing omnidirectional signal

        Returns
        -------
        np.ndarray
            Omnidirectional array signal corrected
        """
        # Filter equations
        b = np.array([1, 1j * self.delay2center, -(1 / 3) * self.delay2center**2])
        a = np.array([1, 1j * (1 / 3) * self.delay2center])

        omni_corrected = self._filter(b, a, omni_array)

        return omni_corrected


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
