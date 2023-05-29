"""Audio utilities"""

from functools import singledispatch
from pathlib import Path
from traceback import print_exc
from typing import Dict, List, Tuple, Union

import numpy as np
import soundfile as sf

# WARNING: Está codeado como el upite pero quiero probar algunas cosas.
# Por otro lado, creo que va a ser mejor pasar las seniales dentro de un diccionario,
# para poder identificar cada canal ambisonics, y ademas el filtro inverso


def read_signals_dict(signals_dict: dict) -> dict:
    """Read the signals contained in signals_dict and overwrites the paths with the arrays.

    Parameters
    ----------
    signals_dict : dict
        Dictionary with signals path.

    Returns
    -------
    dict
        Same signals_dict dictionary with the signals array overwritting signals path.
    """
    for key_i, path_i in signals_dict.items():
        try:  # a puro huevo
            signal_i, sample_rate = sf.read(path_i)
            signals_dict[key_i] = signal_i.T
        except:
            pass
    signals_dict["sample_rate"] = sample_rate

    if signals_dict["channels_per_file"] == 1:
        if signals_dict["input_mode"] == "bformat":
            bformat_keys = ["w_channel", "x_channel", "y_channel", "z_channel"]
            signals_dict["stacked_signals"] = stack_dict_arrays(
                signals_dict, bformat_keys
            )
        else:
            aformat_keys = [
                "front_left_up",
                "front_right_down",
                "back_right_up",
                "back_left_down",
            ]
            signals_dict["stacked_signals"] = stack_dict_arrays(
                signals_dict, aformat_keys
            )

    return signals_dict


def stack_dict_arrays(signals_dict_array: dict, keys: List[str]) -> np.ndarray:
    """Stacks arrays into single numpy.ndarray object given the dictionary and the keys
    to be stacked.

    Parameters
    ----------
    signals_dict_array : dict
        Dictionary containing the arrays to be stacked
    keys : List[str]
        Keys of signals_dict_array with the arrays to be stacked

    Returns
    -------
    np.ndarray
        Stacked arrays into single numpy.ndarray object
    """
    audio_array = []
    for key_i in keys:
        audio_array.append(signals_dict_array[key_i])

    return audio_array


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
    assert signal.shape[0] == 4, (
        f"Audio file {str(audio_path)} with shape {signal.shape} does not"
        f"contain 4 channels, so it cannot be A-format Ambisonics"
    )
    return signal, sample_rate


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
        try:
            audio_array_i, sample_rate = sf.read(audio_i)
            audio_array.append(audio_array_i)
        except sf.SoundFileError:
            print_exc()

    return np.array(audio_array), sample_rate


@read_aformat.register(dict)
def _(audio_paths: Dict[str, str]) -> Tuple[np.ndarray, float]:
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
    except sf.SoundFileError:
        print_exc()
