import numpy as np

from aira.engine.filtering import moving_average_filter


def w_channel_preprocess(
    w_channel: np.ndarray, window_size: int, analysis_length: float, sample_rate: float
) -> np.ndarray:
    """_summary_

    Parameters
    ----------
    w_channel : np.ndarray
        _description_
    window_size : int
        _description_
    analysis_length : float
        _description_
    sample_rate : float
        _description_

    Returns
    -------
    np.ndarray
        _description_
    """
    w_channel_cropped = np.abs(
        analysis_crop_1d(w_channel, analysis_length, sample_rate)
    )
    w_channel_filtered = moving_average_filter(w_channel_cropped, int(window_size / 2))
    w_channel_filtered /= np.max(w_channel_filtered)
    return w_channel_filtered


def analysis_crop_1d(
    array: np.ndarray,
    analysis_length: float,
    sample_rate: int,
):
    """_summary_

    Parameters
    ----------
    analysis_length : float
        _description_
    sample_rate : int
        _description_
    array : np.ndarray
        _description_

    Returns
    -------
    _type_
        _description_
    """
    # Get analysis length max index
    analysis_length_idx = int(analysis_length * sample_rate)

    # Slice from intensity max to analysis length from intensity max
    earliest_peak_index = np.argmax(np.abs(array))
    array_cropped = array[
        earliest_peak_index : earliest_peak_index + analysis_length_idx
    ]

    return array_cropped
