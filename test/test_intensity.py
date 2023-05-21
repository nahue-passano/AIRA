from math import ceil

from aira.intensity import convert_bformat_to_intensity
from mock_data.recordings import (
    bformat_signal_and_samplerate,
    aformat_signal_and_samplerate,
)


def test_conversion_to_intensity(bformat_signal_and_samplerate: tuple):
    integration_time = 0.25
    signal_bformat, sample_rate = bformat_signal_and_samplerate
    intensity, _, _ = convert_bformat_to_intensity(
        signal_bformat, sample_rate, integration_time, 4000
    )

    expected_shape = (ceil(signal_bformat.shape[1] / (integration_time * sample_rate)),)
    assert all(
        list(map(lambda a, b: a >= (b * 0.6), intensity.shape, expected_shape))
    ), f"Output shape {intensity.shape} != expected shape {expected_shape}"
