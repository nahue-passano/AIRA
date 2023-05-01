import numpy as np
import scipy.signal as sc


def generate_log_sine_sweep(
    fs: int, samples: int, f_min: float, f_max: float
) -> np.ndarray:
    """_summary_

    Parameters
    ----------
    fs : int
        _description_
    samples : int
        _description_
    f_min : float
        _description_
    f_max : float
        _description_

    Returns
    -------
    np.ndarray
        _description_
    """

    # Time array
    time = np.arange(samples) / fs
    time_secs = time[-1]

    log_sine_sweep = sc.chirp(
        t=time, f0=f_min, f1=f_max, t1=time_secs, method="logarithmic"
    )

    return log_sine_sweep


def generate_lss_inverse_filter(
    log_sine_sweep: np.ndarray, fs: int, f_min: float, f_max: float
) -> np.ndarray:
    """_summary_

    Parameters
    ----------
    log_sine_sweep : np.ndarray
        _description_
    fs : int
        _description_
    f_min : float
        _description_
    f_max : float
        _description_

    Returns
    -------
    np.ndarray
        _description_
    """

    # Time array
    time = np.arange(len(log_sine_sweep)) / fs
    time_secs = time[-1]

    # Generate inverse filter from log sine sweep
    inv_lss = log_sine_sweep[::-1]
    modulation = 1 / (2 * np.pi * np.exp(time * np.log(f_max / f_min) / time_secs))
    inv_lss_filter = inv_lss * modulation
    inv_lss_filter /= abs(inv_lss_filter).max()

    return inv_lss_filter


def non_coincident_omni_correction(center2mic: float, c: float = 340):
    pass


def non_coincident_axes_correction(center2mic: float, c: float = 340):
    pass


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
