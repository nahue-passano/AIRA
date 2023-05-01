from math import ceil

from aira.intensity import convert_bformat_to_intensity
from mock_data.recordings import create_mock_bformat_signal


def test_conversion_to_intensity():
    sample_rate = 48000
    integration_time = 0.25
    signal_b_format = create_mock_bformat_signal(sample_rate=sample_rate, duration_seconds=5)
    intensity, azimuth, elevation = convert_bformat_to_intensity(signal_b_format, sample_rate, integration_time, 4000)
    
    expected_shape = (ceil(signal_b_format.shape[1] / (integration_time * sample_rate)),)
    assert(intensity.shape >= (expected_shape * .9)), f"Output shape {intensity.shape} != expected shape {expected_shape}"
