import numpy as np
import pandas as pd
import plotly.express as px

from aira.intensity import convert_bformat_to_intensity
from aira.reflections import get_hedgehog_arrays
from mock_data.recordings import load_mocked_bformat


# TODO: add assertions for NotImplementedError for different `ReflectionDetectionStrategy`s


# TODO: add unit test for reflection extraction for NeighborReflectionDetectionStrategy


def test_get_hedgehog_array():
    signals, sample_rate = load_mocked_bformat()
    intensity, azimuth, elevation = convert_bformat_to_intensity(
        signals, sample_rate, 0.01, 4000
    )

    masked_intensity, masked_azimuth, masked_elevation = get_hedgehog_arrays(
        intensity, azimuth, elevation
    )
    t = np.linspace(0, signals.shape[1] / sample_rate, masked_intensity.shape[0])
    fig = px.line(t, masked_intensity)
    fig.show()