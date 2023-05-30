"""Mocked intensity data."""

import pytest

from mock_data.recordings import (  # pylint: disable=unused-import
    aformat_signal_and_samplerate,
    bformat_signal_and_samplerate,
)

from aira.engine.intensity import convert_bformat_to_intensity


@pytest.fixture
def intensity_azimuth_elevation(
    bformat_signal_and_samplerate: tuple,  # pylint: disable=redefined-outer-name
) -> tuple:
    """Return mocked data for smoothed intensity, azimuth and elevation, along
    with the sample rate and integration time.

    Parameters
        bformat_signal_and_samplerate (tuple). A pytest fixture automatically
        interpolated.

    Returns
        A tuple with intensity, azimuth, elevation, sample rate and integration
        time.
    """
    integration_time = 0.25
    (
        signal_bformat,
        sample_rate,
    ) = bformat_signal_and_samplerate  # pylint: disable=unpacking-non-sequence
    return (
        *convert_bformat_to_intensity(
            signal_bformat, sample_rate, integration_time, 4000
        ),
        sample_rate,
        integration_time,
    )
