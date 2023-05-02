import numpy as np
import soundfile as sf
from typing import List, Tuple


def read_audio(audio_paths: List[str]) -> Tuple[float, np.ndarray]:
    """Read audios given a list of audio paths

    Parameters
    ----------
    audio_paths : List[str]
        List of strings containing the audio paths to be loaded

    Returns
    -------
    Tuple[float,np.ndarray]
        Sample rate of the audios and audios loaded as rows of a np.ndarray
    """
    audio_array = []
    for audio_i in audio_paths:
        audio_array_i, sample_rate = sf.read(audio_i)
        audio_array.append(audio_array_i)

    return sample_rate, np.array(audio_array)


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
