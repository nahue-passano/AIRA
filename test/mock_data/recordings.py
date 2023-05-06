from random import randint

import numpy as np
from soundfile import read


def create_mock_bformat_signal(sample_rate: int, duration_seconds: float) -> np.ndarray:
    return np.random.randint(
        -32767, 32768, size=(4, round(sample_rate * duration_seconds)), dtype=np.int16
    )


def load_mocked_bformat():
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
