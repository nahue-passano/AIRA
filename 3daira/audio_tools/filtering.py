import numpy as np
import scipy.signal as sc


SOUND_VELOCITY = 340


def inverse_filter():
    pass


class NonCoincidentMicsCorrection:
    def __init__(self, mic2center: float, fs: int, c: float = SOUND_VELOCITY) -> None:
        self.mic2center = mic2center / 100
        self.fs = fs
        self.c = c
        self.delay2center = self.mic2center / self.c

    def _filter(self, b: np.ndarray, a: np.ndarray, array: np.ndarray) -> np.ndarray:
        """Applies filter to array given numerator "b" and denominator "a" from
        analog filter frequency response

        Parameters
        ----------
        b : np.ndarray
            _description_
        a : np.ndarray
            _description_
        array : np.ndarray
            _description_

        Returns
        -------
        np.ndarray
            _description_
        """

        # Analog to digital filter conversion
        zeros, poles = sc.bilinear(b, a, self.fs)

        # Filtering
        array_filtered = sc.lfilter(zeros, poles, array)

        return array_filtered

    def correct_axis(self, axis_array: np.ndarray) -> np.ndarray:
        """_summary_

        Parameters
        ----------
        axis_array : np.ndarray
            _description_

        Returns
        -------
        np.ndarray
            _description_
        """
        # Filter equations
        b = np.sqrt(6) * np.array(
            [1, 1j * (1 / 3) * self.mic2center, -(1 / 3) * self.delay2center**2]
        )
        a = np.array([1, 1j * (1 / 3) * self.delay2center])

        axis_corrected = self._filter(b, a, axis_array)

        return axis_corrected

    def correct_omni(self, omni_array: np.ndarray) -> np.ndarray:
        """_summary_

        Parameters
        ----------
        omni_array : np.ndarray
            _description_

        Returns
        -------
        np.ndarray
            _description_
        """
        # Filter equations
        b = np.array([1, 1j * self.delay2center, -(1 / 3) * self.delay2center**2])
        a = np.array([1, 1j * (1 / 3) * self.delay2center])

        omni_corrected = self._filter(b, a, omni_array)

        return omni_corrected


def convolve(signal_1: np.ndarray, signal_2: np.ndarray) -> np.ndarray:
    """_summary_

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
