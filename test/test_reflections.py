"""Unit tests for aira.reflections module."""

import numpy as np
import pytest
from mock_data.intensity import intensity_azimuth_elevation
from mock_data.recordings import (  # pylint: disable=unused-import
    aformat_signal_and_samplerate,
    bformat_signal_and_samplerate,
)

from aira.engine.intensity import convert_bformat_to_intensity
from aira.engine.reflections import (
    CorrelationReflectionDetectionStrategy,
    NeighborReflectionDetectionStrategy,
    ThresholdReflectionDetectionStrategy,
    get_hedgehog_arrays,
)


def test_correlation_reflection_detection_strategy(
    bformat_signal_and_samplerate: tuple,  # pylint: disable=redefined-outer-name
):
    """WHEN extracting reflections with the CorrelationReflectionDetectionStrategy
    GIVEN that the strategy is not implemented yet
    THEN raise a NotImplementedError.

    Parameters
        bformat_signal_and_samplerate: tuple. A PyTest fixture. It gets
        interpolated automatically.
    """
    signals, sample_rate = bformat_signal_and_samplerate
    intensity, _, _ = convert_bformat_to_intensity(signals, sample_rate, 0.01, 4000)
    intensity_magnitude = 20 * np.log(intensity)
    with pytest.raises(NotImplementedError):
        CorrelationReflectionDetectionStrategy.get_indeces_of_reflections(
            intensity_magnitude
        )


def test_threshold_reflection_detection(
    bformat_signal_and_samplerate: tuple,
):  # pylint: disable=redefined-outer-name
    """WHEN extracting reflections with the ThresholdReflectionDetectionStrategy
    GIVEN that the strategy is not implemented yet
    THEN raise a NotImplementedError.

    Parameters
        bformat_signal_and_samplerate: tuple. A PyTest fixture. It gets
        interpolated automatically.
    """
    signals, sample_rate = bformat_signal_and_samplerate
    intensity, _, _ = convert_bformat_to_intensity(signals, sample_rate, 0.01, 4000)
    intensity_magnitude = 20 * np.log(intensity)
    with pytest.raises(NotImplementedError):
        ThresholdReflectionDetectionStrategy.get_indeces_of_reflections(
            intensity_magnitude
        )


def test_neighbor_reflection_detection(
    bformat_signal_and_samplerate: tuple,
):  # pylint: disable=redefined-outer-name
    """WHEN getting the indeces of the reflections with the NeighborReflectionDetectionStrategy
    GIVEN a valid intensity
    THEN return an array with the reflections indeces

    Parameters
        bformat_signal_and_samplerate: tuple. A PyTest fixture. It gets
        interpolated automatically.
    """

    signals, sample_rate = bformat_signal_and_samplerate
    intensity, _, _ = convert_bformat_to_intensity(signals, sample_rate, 0.01, 4000)
    intensity_magnitude = 20 * np.log(intensity)

    reflections_indeces = (
        NeighborReflectionDetectionStrategy.get_indeces_of_reflections(
            intensity_magnitude
        )
    )
    assert (
        len(reflections_indeces) > 0
    ), """WHEN getting the indeces of the reflections with the NeighborReflectionDetectionStrategy
    GIVEN a valid intensity
    THEN return an array with the reflections indeces"""


def test_get_hedgehog_array(
    intensity_azimuth_elevation: tuple,  # pylint: disable=redefined-outer-name
):
    """WHEN creating the data for a hedgehog plot
    GIVEN valid intensity, azimuth and elevation arrays
    THEN lengths of the arrays must be equal

    Parameters
        intensity_azimuth_elevation: tuple. A PyTest fixture. It gets
        interpolated automatically.
    """
    (
        intensity,
        azimuth,
        elevation,
        _,
        _,
    ) = intensity_azimuth_elevation  # pylint: disable=unpacking-nono-sequence
    masked_intensity, masked_azimuth, masked_elevation = get_hedgehog_arrays(
        intensity, azimuth, elevation
    )

    assert len(masked_intensity) == len(
        masked_azimuth
    ), "Intensity and azimuth's length must be the same"
    assert len(masked_intensity) == len(
        masked_elevation
    ), "Intensity and elevation's length must be the same"
