"""Unit tests for aira.reflections module."""

import numpy as np
import pandas as pd
import plotly.express as px
import pytest

from mock_data.recordings import load_mocked_bformat

from aira.intensity import convert_bformat_to_intensity
from aira.reflections import (
    CorrelationReflectionDetectionStrategy,
    ThresholdReflectionDetectionStrategy,
    NeighborReflectionDetectionStrategy,
    get_hedgehog_arrays,
)


def test_correlation_reflection_detection_strategy():
    """WHEN extracting reflections with the CorrelationReflectionDetectionStrategy
    GIVEN that the strategy is not implemented yet
    THEN raise a NotImplementedError.
    """
    signals, sample_rate = load_mocked_bformat()
    intensity, _, _ = convert_bformat_to_intensity(signals, sample_rate, 0.01, 4000)
    intensity_magnitude = 20 * np.log(intensity)
    with pytest.raises(NotImplementedError):
        CorrelationReflectionDetectionStrategy.get_indeces_of_reflections(
            intensity_magnitude
        )


def test_threshold_reflection_detection():
    """WHEN extracting reflections with the ThresholdReflectionDetectionStrategy
    GIVEN that the strategy is not implemented yet
    THEN raise a NotImplementedError.
    """
    signals, sample_rate = load_mocked_bformat()
    intensity, _, _ = convert_bformat_to_intensity(signals, sample_rate, 0.01, 4000)
    intensity_magnitude = 20 * np.log(intensity)
    with pytest.raises(NotImplementedError):
        ThresholdReflectionDetectionStrategy.get_indeces_of_reflections(
            intensity_magnitude
        )


def test_neighbor_reflection_detection():
    """WHEN getting the indeces of the reflections with the NeighborReflectionDetectionStrategy
    GIVEN a valid intensity
    THEN return an array with the reflections indeces"""

    signals, sample_rate = load_mocked_bformat()
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


def test_get_hedgehog_array():
    """WHEN creating the data for a hedgehog plot
    GIVEN valid intensity, azimuth and elevation arrays
    THEN
    """

    signals, sample_rate = load_mocked_bformat()
    intensity, azimuth, elevation = convert_bformat_to_intensity(
        signals, sample_rate, 0.01, 4000
    )
    masked_intensity, masked_azimuth, masked_elevation = get_hedgehog_arrays(
        intensity, azimuth, elevation
    )

    assert len(masked_intensity) == len(
        masked_azimuth
    ), "Intensity and azimuth's length must be the same"
    assert len(masked_intensity) == len(
        masked_elevation
    ), "Intensity and elevation's length must be the same"

    time_axis = np.linspace(0, signals.shape[1] / sample_rate, masked_intensity.shape[0])
    plot_df = pd.DataFrame(
        {
            "time": time_axis,
            "intensity": masked_intensity,
            "azimuth": masked_azimuth,
            "elevation": masked_elevation,
        }
    )  # Plotly requires a dataframe input
    fig = px.line(plot_df, x="time", y="intensity")
    fig.show()
