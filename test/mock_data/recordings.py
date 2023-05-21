from random import randint

import numpy as np
import pytest
from soundfile import read

from aira.formatter import convert_ambisonics_a_to_b


@pytest.fixture
def bformat_signal_random(sample_rate: int, duration_seconds: float) -> np.ndarray:
    """Return a random array of 4 rows corresponding to W, X, Y and Z channels."""
    return np.random.randint(
        -32767, 32768, size=(4, round(sample_rate * duration_seconds)), dtype=np.int16
    )


@pytest.fixture
def aformat_signal_and_samplerate() -> tuple:
    """Return a tuple with an array of FLU, FRD, BRU and BLD channels, one
    for each of the 4 rows, in the first element of the tuple, and the sample
    rate of the recording in the second element of the tuple."""
    ordered_aformat_channels = (
        "FLU",
        "FRD",
        "BRU",
        "BLD",
    )  # Assert the ordering is standardized across the project
    audio_paths = dict(
        FLU="./test/mock_data/soundfield_flu.wav",
        FRD="./test/mock_data/soundfield_frd.wav",
        BRU="./test/mock_data/soundfield_bru.wav",
        BLD="./test/mock_data/soundfield_bld.wav",
    )
    audio_data = {
        cardioid_channel: dict(zip(("signal", "sample_rate"), read(path)))
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


@pytest.fixture
def bformat_signal_and_samplerate(aformat_signal_and_samplerate: tuple):
    """Return a tuple with an array of W, X, Y and Z channels, one
    for each of the 4 rows, in the first element of the tuple, and the sample
    rate of the recording in the second element of the tuple."""
    aformat_signals, sample_rate = aformat_signal_and_samplerate
    aformat_signals = [aformat_signals[a_channel, :] for a_channel in range(4)]
    return convert_ambisonics_a_to_b(aformat_signals), sample_rate
