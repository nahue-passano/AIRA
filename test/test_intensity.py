from math import ceil

from aira.intensity import convert_bformat_to_intensity
from mock_data.recordings import create_mock_bformat_signal, load_mocked_bformat


def test_conversion_to_intensity():
    integration_time = 0.25
    # sample_rate = 48000
    # signal_bformat = create_mock_bformat_signal(
    #     sample_rate=sample_rate, duration_seconds=5
    # )
    signal_bformat, sample_rate = load_mocked_bformat()
    intensity, azimuth, elevation = convert_bformat_to_intensity(
        signal_bformat, sample_rate, integration_time, 4000
    )

    expected_shape = (ceil(signal_bformat.shape[1] / (integration_time * sample_rate)),)
    assert all(
        list(map(lambda a, b: a >= (b * 0.6), intensity.shape, expected_shape))
    ), f"Output shape {intensity.shape} != expected shape {expected_shape}"
