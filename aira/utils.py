import numpy as np


def load_audio():
    pass


def write_audio():
    pass


def pad_to_target(array: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Pads an array to target's shape

    Parameters
    ----------
    array : np.ndarray
        Array to be padded
    target : np.ndarray
        Target to take pad's shape

    Returns
    -------
    np.ndarray
        Array padded
    """
    pad_len = len(target) - len(array)
    array_padded = np.pad(array=array, pad_width=(0, pad_len), constant_values=(0, 0))

    return array_padded
