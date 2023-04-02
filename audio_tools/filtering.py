import numpy as np
import scipy.signal as sc

def inverse_filter():
    pass

def non_coincident_omni_correction(center2mic: float,
                                   c: float = 340):
    pass
    
    
def non_coincident_axes_correction(center2mic: float,
                                   c: float = 340):
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