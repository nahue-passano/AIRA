from functools import singledispatch
from pathlib import Path

import numpy as np
import soundfile as sf
from typing import List, Tuple, Union


def read_aformat(audio_paths: Union[str, Path, List[str]]) -> Tuple[np.ndarray, float]:
    """Read an A-format Ambisonics signal from audio paths. If a single path is given, it's
    expected to contain 4 channels; else, 4 paths are expected, one for each cardioid signal,
    in the following order:
        1. front left up
        2. front right down
        3. back right up
        4. back left_down

    Parameters
    ----------
    audio_paths : str | Path | List[str]
        Strings containing the audio paths to be loaded

    Returns
    -------
    Tuple[np.ndarray, float]
        Sample rate of the audios and audios loaded as rows of a np.ndarray
    """
    assert (isinstance(audio_paths, (str, Path, list))) or (
        len(audio_paths) in (1, 4)
    ), "One wave file with 4 channels or a list of 4 wave files is expected"

    audio_array = []
    for audio_i in audio_paths:
        audio_array_i, sample_rate = sf.read(audio_i)
        audio_array.append(audio_array_i)

    return np.array(audio_array), sample_rate


def read_aformat(
    front_left_up: Union[str, Path],
    front_right_down: Union[str, Path],
    back_right_up: Union[str, Path],
    back_left_down: Union[str, Path],
):
    ordered_aformat_channels = (
        "front_left_up",
        "front_right_down",
        "back_right_up",
        "back_left_down",
    )  # Assert the ordering is standardized across the project
    audio_paths = dict(
        zip(ordered_aformat_channels),
        (front_left_up, front_right_down, back_right_up, back_left_down),
    )
    audio_data = {
        cardioid_channel: dict(zip(("signal", "sample_rate"), sf.read(path)))
        for cardioid_channel, path in audio_paths.items()
    }
    audio_signals = [
        audio_data[channel_name]["signal"] for channel_name in ordered_aformat_channels
    ]
    sample_rates = [
        audio_data[channel_name]["sample_rate"]
        for channel_name in ("FLU", "FRD", "BRU", "BLD")
    ]
    assert len(set(sample_rates)) == 1, "Multiple different sample rates were found"

    signals_array = np.array(audio_signals)
    return signals_array, sample_rates[0]


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
