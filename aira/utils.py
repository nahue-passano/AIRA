from functools import singledispatch
from pathlib import Path
from traceback import print_exc

import numpy as np
import soundfile as sf
from typing import Dict, List, Tuple, Union


@singledispatch
def read_aformat(audio_path: Union[str, Path]) -> Tuple[np.ndarray, float]:
    """Read an A-format Ambisonics signal from a single audio path, which is expected
    to contain 4 channels.

    Parameters
    ----------
    audio_paths : str | Path
        Strings containing the audio paths to be loaded

    Returns
    -------
    Tuple[np.ndarray, float]
        Sample rate of the audios and audios loaded as rows of a np.ndarray
    """
    signal, sample_rate = sf.read(audio_path)
    signal = signal.T
    assert (
        signal.shape[0] == 4
    ), f"Audio file {str(audio_path)} with shape {signal.shape} does not contain 4 channels, so it cannot be A-format Ambisonics"
    return


@read_aformat.register(list)
def _(audio_paths: List[str]) -> Tuple[np.ndarray, float]:
    """Read an A-format Ambisonics signal from audio paths. 4 paths are expected,
    one for each cardioid signal, in the following order:
        1. front left up
        2. front right down
        3. back right up
        4. back left down

    Parameters
    ----------
    audio_paths : List[str]
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
        # TODO: implement try-catch for concrete exceptions of soundfile
        audio_array_i, sample_rate = sf.read(audio_i)
        audio_array.append(audio_array_i)

    return np.array(audio_array), sample_rate


@read_aformat.register(dict)
def _(audio_paths: Dict[str, str]):
    """Read an A-format Ambisonics signal from a dictionary with audio paths. 4 keys are expected,
    one for each cardioid signal:
        1. front_left_up
        2. front_right_down
        3. back_right_up
        4. back_left_down

    Parameters
    ----------
    audio_paths : Dict[str]
        Key-value pair containing the audio paths to be loaded for each FLU/FRD/BRU/BLD channel

    Returns
    -------
    Tuple[np.ndarray, float]
        Sample rate of the audios and audios loaded as rows of a np.ndarray
    """
    ordered_aformat_channels = (
        "front_left_up",
        "front_right_down",
        "back_right_up",
        "back_left_down",
    )  # Assert the ordering is standardized across the project
    try:
        audio_data = {
            cardioid_channel: dict(zip(("signal", "sample_rate"), sf.read(path)))
            for cardioid_channel, path in audio_paths.items()
        }

        # refactor from here
        audio_signals = [
            audio_data[channel_name]["signal"]
            for channel_name in ordered_aformat_channels
        ]
        sample_rates = [
            audio_data[channel_name]["sample_rate"]
            for channel_name in ordered_aformat_channels
        ]
        assert len(set(sample_rates)) == 1, "Multiple different sample rates were found"

        signals_array = np.array(audio_signals)
        return signals_array, sample_rates[0]
    except:
        # TODO: implement for concrete exceptions of soundfile
        print_exc()


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

    return
