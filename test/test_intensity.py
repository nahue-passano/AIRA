"""Unit tests for the intensity module."""

from math import ceil

from mock_data.recordings import (  # pylint: disable=unused-import
    aformat_signal_and_samplerate,
    bformat_signal_and_samplerate,
)

from aira.intensity import convert_bformat_to_intensity


def test_conversion_to_intensity(
    bformat_signal_and_samplerate: tuple,
):  # pylint: disable=redefined-outer-name
    """WHEN computing the intensity GIVEN a B-format signal, THEN the output
    is of reasonable shape.

    Args:
        bformat_signal_and_samplerate (tuple): a pytest fixture that returns a
        B-format array and its sample rate.
    """
    integration_time = 0.25
    (
        signal_bformat,
        sample_rate,
    ) = bformat_signal_and_samplerate  # pylint: disable=unpacking-non-sequence
    intensity, _, _ = convert_bformat_to_intensity(
        signal_bformat, sample_rate, integration_time, 4000
    )

    expected_shape = (ceil(signal_bformat.shape[1] / (integration_time * sample_rate)),)
    assert all(
        list(map(lambda a, b: a >= (b * 0.6), intensity.shape, expected_shape))
    ), f"Output shape {intensity.shape} != expected shape {expected_shape}"
